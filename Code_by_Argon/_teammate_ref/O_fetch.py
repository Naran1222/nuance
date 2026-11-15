from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.time import Time
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from scipy.interpolate import RegularGridInterpolator

from O_s3_bypass import LOCAL_CATALOG_PATH, build_offline_catalog, expand_s3_catalog_for_target

CUTOUT_CACHE_DIR = Path("./spherex_cache/cutouts")
TAP_CACHE_DIR = Path("./spherex_cache/tap")


def log(msg: str) -> None:
    """Виводить повідомлення в консоль без затримки буферизації."""
    print(msg, flush=True)


def _to_str(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    s = str(val).strip() if val is not None else ""
    return "" if s == "--" else s


def download_with_retry(
    url: str,
    retries: int = 2,
    base_delay: float = 1.0,
    timeout: int = 30,
    show_progress: bool = False,
) -> Path:
    """Завантажує файл із контролем таймауту та повторами при тимчасових збоях."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            path_str = download_file(url, cache=True, show_progress=show_progress, timeout=timeout)
            return Path(path_str)
        except (HTTPError, URLError, TimeoutError) as exc:
            last_exc = exc
            if isinstance(exc, HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt < retries:
                time.sleep(base_delay * attempt)
    raise RuntimeError(f"Збій завантаження {url}: {last_exc}")


def _search_via_siav2(ra: float, dec: float, radius_deg: float = 0.15) -> pd.DataFrame:
    """Швидкий пошук кадрів SPHEREx через мікросервіс IRSA SIAv2 (в обхід TAP/sync)."""
    params = urlencode({
        "COLLECTION": "SPHEREX",
        "POS": f"circle {ra} {dec} {radius_deg}",
        "RESPONSEFORMAT": "votable",
    })
    url = f"https://irsa.ipac.caltech.edu/SIA?{params}"
    log(f"Запит до NASA IRSA SIAv2 для RA={ra}, Dec={dec}...")

    with urlopen(url, timeout=15) as resp:
        raw_xml = resp.read()

    votable = parse(io.BytesIO(raw_xml))
    tables = list(votable.iter_tables())
    if not tables:
        return pd.DataFrame()

    df = tables[0].to_table().to_pandas()
    if df.empty:
        return df

    for col in ("obs_id", "access_url", "obs_collection"):
        if col in df.columns:
            df[col] = df[col].apply(_to_str)

    for col in ("s_ra", "s_dec", "t_min", "t_max", "em_min", "em_max"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[np.isfinite(df["t_min"]) & (df["access_url"] != "")].copy()
    return df.sort_values("t_min").reset_index(drop=True)


def search_spherex_epochs(ra: float, dec: float, radius_deg: float = 0.15, limit: int = 600) -> pd.DataFrame:
    """Шукає кадри SPHEREx по всьому небу: спершу через SIAv2, потім через індексований ADQL TAP."""
    TAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tap_key = hashlib.md5(f"v2_{ra:.4f}_{dec:.4f}_{radius_deg:.3f}_{limit}".encode()).hexdigest()
    tap_cache_file = TAP_CACHE_DIR / f"{tap_key}.pkl"

    if tap_cache_file.exists():
        log("Завантажено список спостережень із локального кешу.")
        return pd.read_pickle(tap_cache_file)

    # 1. Основний швидкий канал: IRSA SIAv2 (працює, навіть коли TAP/sync віддає 502)
    try:
        df = _search_via_siav2(ra, dec, radius_deg=radius_deg)
        if not df.empty:
            df.to_pickle(tap_cache_file)
            log(f"IRSA SIAv2 відповів! Знайдено експозицій: {len(df)}")
            return df
    except Exception as exc:
        log(f"SIAv2 не відповів ({exc}), пробуємо індексований TAP...")

    # 2. Резервний канал: оптимізований ADQL без OR і з точковим індексом CONTAINS
    adql = f"""
    SELECT TOP {limit}
        obs_id, obs_collection, calib_level, s_ra, s_dec,
        t_min, t_max, em_min, em_max, access_url
    FROM ivoa.obscore
    WHERE obs_collection = 'SPHEREX'
      AND dataproduct_type = 'image'
      AND CONTAINS(POINT('ICRS', {ra}, {dec}), s_region) = 1
    """
    Irsa.TIMEOUT = 15
    log(f"Запит до NASA IRSA TAP (індексований) для RA={ra}, Dec={dec}...")
    df = Irsa.query_tap(adql).to_table().to_pandas()
    if not df.empty:
        for col in ("obs_id", "access_url"):
            if col in df.columns:
                df[col] = df[col].apply(_to_str)
        df = df.sort_values("t_min").reset_index(drop=True)
        df.to_pickle(tap_cache_file)
    log(f"IRSA TAP відповів! Знайдено експозицій: {len(df)}")
    return df


def fetch_two_epoch_cube_slices(
    ra: float,
    dec: float,
    size_deg: float = 0.15,
    per_band: int = 2,
) -> dict:
    """Збирає різнохвильові cutout-кадри для Epoch 1 та Epoch 2 для будь-якої точки неба."""
    try:
        df = search_spherex_epochs(ra, dec, radius_deg=0.15, limit=600)
        if not df.empty:
            t1_min = float(df["t_min"].min())
            t2_max = float(df["t_min"].max())
            span = t2_max - t1_min

            if span >= 6.0:
                ep1_df = df[df["t_min"] <= t1_min + 3.0]
                ep2_df = df[df["t_min"] >= t2_max - 3.0]
            else:
                mid_t = (t1_min + t2_max) / 2.0
                ep1_df = df[df["t_min"] <= mid_t]
                ep2_df = df[df["t_min"] > mid_t]
                if ep2_df.empty:
                    ep2_df = ep1_df

            ep1_label = Time(t1_min, format="mjd").iso[:10]
            ep2_label = Time(float(ep2_df["t_min"].min()), format="mjd").iso[:10]

            log(f"Збір Епохи 1 (~{ep1_label}, кандидатів: {len(ep1_df)})...")
            slices_ep1 = _sample_window_slices(ep1_df, ra, dec, size_deg=size_deg, per_band=per_band)

            log(f"Збір Епохи 2 (~{ep2_label}, кандидатів: {len(ep2_df)})...")
            slices_ep2 = _sample_window_slices(ep2_df, ra, dec, size_deg=size_deg, per_band=per_band)

            if slices_ep1 and slices_ep2:
                return {
                    "epoch1": {
                        "date_label": ep1_label,
                        "mjd_mean": round(float(ep1_df["t_min"].mean()), 3),
                        "slices": slices_ep1,
                    },
                    "epoch2": {
                        "date_label": ep2_label,
                        "mjd_mean": round(float(ep2_df["t_min"].mean()), 3),
                        "slices": slices_ep2,
                    },
                }
    except Exception as exc:
        log(f"Помилка пошуку через IRSA ({exc}). Перевіряємо локальний S3-кеш...")

    return _fetch_from_s3_fallback(ra, dec, size_deg=size_deg, per_band=per_band)

def resolve_spherex_urls(datalink_url: str, ra: float, dec: float, size_deg: float = 0.15) -> dict:
    """Дістає з DataLink URL для cutout, статичного IRSA FITS та дзеркала AWS S3."""
    xml_path = download_with_retry(datalink_url, retries=2, timeout=20, show_progress=False)
    votable = parse(xml_path)
    vo_table = list(votable.iter_tables())[0]
    astropy_table = vo_table.to_table()

    field_map = {
        field.name.lower(): astropy_table.colnames[idx]
        for idx, field in enumerate(vo_table.fields)
        if field.name
    }

    full_fits_url = None
    cutout_path = None
    s3_url = None

    for row in astropy_table:
        sem = _to_str(row[field_map["semantics"]]) if "semantics" in field_map else ""
        if sem == "#this":
            if "access_url" in field_map:
                full_fits_url = _to_str(row[field_map["access_url"]])
            for col in astropy_table.colnames:
                val = _to_str(row[col])
                if val.startswith("{") and "nasa-irsa-spherex" in val:
                    try:
                        meta = json.loads(val)
                        aws = meta.get("aws", {})
                        if aws.get("bucket_name") and aws.get("key"):
                            s3_url = f"https://{aws['bucket_name']}.s3.amazonaws.com/{aws['key']}"
                    except Exception:
                        pass
        elif sem == "#cutout" and "path" in field_map:
            cutout_path = _to_str(row[field_map["path"]])

    cutout_url = None
    if cutout_path:
        params = urlencode({"ra": ra, "dec": dec, "size": size_deg, "path": cutout_path})
        cutout_url = f"https://irsa.ipac.caltech.edu/ibe/cutout?{params}"

    return {"cutout_url": cutout_url, "s3_url": s3_url, "full_url": full_fits_url}


def _crop_full_fits_locally(source_url: str, ra: float, dec: float, size_deg: float, out_path: Path) -> Path:
    """Скачує повний FITS із S3, вирізає повний квадрат (mode='partial') і чистить тимчасовий файл."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        bytes_read = 0
        with urlopen(source_url, timeout=45) as resp, open(tmp_path, "wb") as f_out:
            while chunk := resp.read(1024 * 1024):
                f_out.write(chunk)
                bytes_read += len(chunk)

        log(f"    ↓ Завантажено {Path(source_url).name} ({bytes_read / 1e6:.1f} МБ)")

        center = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
        size = u.Quantity((size_deg, size_deg), u.deg)

        with fits.open(tmp_path, memmap=False) as hdul:
            img_hdu = hdul["IMAGE"]
            wcs_full = WCS(img_hdu.header)
            cut_img = Cutout2D(
                img_hdu.data, position=center, size=size, wcs=wcs_full, mode="partial", fill_value=np.nan
            )

            # Відкидаємо крайові зачепи, де заповнено менше 25% площі квадрата
            if np.isfinite(cut_img.data).mean() < 0.60:
                raise NoOverlapError("Перекриття з детектором менше 60%.")

            cut_hdr = cut_img.wcs.to_header(relax=True)
            for key in ("DATE-OBS", "MJD-OBS", "BUNIT", "FILTER", "DETECTOR"):
                if key in img_hdu.header:
                    cut_hdr[key] = img_hdu.header[key]
            cut_hdr["LTV1"] = -float(cut_img.xmin_original)
            cut_hdr["LTV2"] = -float(cut_img.ymin_original)

            new_hdus = [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=cut_img.data.astype(np.float32), header=cut_hdr, name="IMAGE"),
            ]

            if "ZODI" in hdul:
                cut_zodi = Cutout2D(
                    hdul["ZODI"].data, position=center, size=size, wcs=wcs_full, mode="partial", fill_value=np.nan
                )
                new_hdus.append(fits.ImageHDU(data=cut_zodi.data.astype(np.float32), header=cut_hdr, name="ZODI"))

            if "WCS-WAVE" in hdul:
                ww = hdul["WCS-WAVE"]
                new_hdus.append(fits.BinTableHDU(data=ww.data.copy(), header=ww.header.copy(), name="WCS-WAVE"))

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fits.HDUList(new_hdus).writeto(out_path, overwrite=True)

        return out_path
    finally:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _is_full_square_fits(fits_path: Path, min_size: int = 50) -> bool:
    """Перевіряє, чи закешований FITS є повним квадратом, а не старою вузькою смужкою 13x88."""
    if not (fits_path.exists() and fits_path.stat().st_size > 1024):
        return False
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            if "IMAGE" not in hdul:
                return False
            hdr = hdul["IMAGE"].header
            nx, ny = int(hdr.get("NAXIS1", 0)), int(hdr.get("NAXIS2", 0))
            return nx >= min_size and ny >= min_size and abs(nx - ny) <= 2
    except Exception:
        return False


def fetch_cutout_resilient(obs_id: str, datalink_url: str, ra: float, dec: float, size_deg: float) -> Path:
    """Отримує cutout із локального кешу, швидкого ibe/cutout або через AWS S3 + Cutout2D."""
    CUTOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{obs_id}_{ra:.5f}_{dec:.5f}_{size_deg:.4f}".encode()).hexdigest()
    cached_file = CUTOUT_CACHE_DIR / f"{cache_key}.fits"

    if _is_full_square_fits(cached_file):
        return cached_file

    urls = resolve_spherex_urls(datalink_url, ra, dec, size_deg=size_deg)

    if urls["cutout_url"]:
        try:
            dl_path = download_with_retry(urls["cutout_url"], retries=1, timeout=10, show_progress=False)
            cached_file.write_bytes(dl_path.read_bytes())
            if _is_full_square_fits(cached_file):
                return cached_file
        except Exception:
            pass

    for fallback_url in (urls["s3_url"], urls["full_url"]):
        if fallback_url:
            try:
                return _crop_full_fits_locally(fallback_url, ra, dec, size_deg, cached_file)
            except Exception as exc:
                log(f"  Fallback збій ({fallback_url[:50]}...): {exc}")

    raise RuntimeError(f"Не вдалося отримати вирізку для {obs_id}")


def read_local_wavelength_um(fits_path: Path) -> float | None:
    """Обчислює реальну локальну довжину хвилі (мкм) центру вирізки з 2D-сітки WCS-WAVE."""
    with fits.open(fits_path, memmap=False) as hdul:
        if "WCS-WAVE" not in hdul or "IMAGE" not in hdul:
            return None
        hdr = hdul["IMAGE"].header
        ww_data = hdul["WCS-WAVE"].data
        if ww_data is None or len(ww_data) == 0:
            return None

        cx_cut = hdr.get("NAXIS1", 88) / 2.0
        cy_cut = hdr.get("NAXIS2", 88) / 2.0
        if "LTV1" in hdr and "LTV2" in hdr:
            x_det = cx_cut - float(hdr["LTV1"])
            y_det = cy_cut - float(hdr["LTV2"])
        else:
            x_det = 1024.5 - float(hdr.get("CRPIX1", cx_cut)) + cx_cut
            y_det = 1024.5 - float(hdr.get("CRPIX2", cy_cut)) + cy_cut

        row = ww_data[0]
        x_nodes = np.asarray(row[0], dtype=np.float64).ravel()
        y_nodes = np.asarray(row[1], dtype=np.float64).ravel()
        grid_raw = np.asarray(row[2], dtype=np.float64).ravel()

        nx, ny = len(x_nodes), len(y_nodes)
        if grid_raw.size == nx * ny * 2:
            lam_grid = grid_raw.reshape((ny, nx, 2))[:, :, 0]
            x_clamped = float(np.clip(x_det, x_nodes.min(), x_nodes.max()))
            y_clamped = float(np.clip(y_det, y_nodes.min(), y_nodes.max()))
            interp = RegularGridInterpolator((y_nodes, x_nodes), lam_grid, bounds_error=False, fill_value=None)
            lam = float(interp([[y_clamped, x_clamped]])[0])
            return lam * 1e6 if lam < 0.1 else lam

    return None


def _sample_window_slices(
    win_df: pd.DataFrame,
    ra: float,
    dec: float,
    size_deg: float,
    per_band: int = 2,
) -> list[dict]:
    """Витягує рівномірний набір спектральних зрізів у межах одного часового вікна."""
    if win_df.empty:
        return []

    win_df = win_df.copy()
    win_df["band_min_um"] = (win_df["em_min"] * 1e6).round(2)

    selected = []
    for b in sorted(win_df["band_min_um"].dropna().unique()):
        sub = win_df[win_df["band_min_um"] == b].sort_values("t_min")
        if len(sub) > per_band:
            idxs = np.linspace(0, len(sub) - 1, per_band, dtype=int)
            sub = sub.iloc[idxs]
        selected.append(sub)

    sample_df = pd.concat(selected) if selected else win_df.head(per_band * 6)
    slices = []
    for idx, (_, row) in enumerate(sample_df.iterrows(), start=1):
        obs_id = _to_str(row["obs_id"])
        access_url = _to_str(row["access_url"])
        try:
            log(f"  [{idx}/{len(sample_df)}] Отримуємо зріз {obs_id}...")
            p = fetch_cutout_resilient(obs_id, access_url, ra, dec, size_deg)
            lam = read_local_wavelength_um(p)
            if lam is not None:
                mjd_val = float(row["t_min"])
                slices.append({
                    "wave_um": round(lam, 4),
                    "mjd": mjd_val,
                    "date_obs": Time(mjd_val, format="mjd").iso[:19],
                    "path": p,
                })
        except Exception as exc:
            log(f"  Пропущено {obs_id}: {exc}")

    slices.sort(key=lambda x: x["wave_um"])
    return slices


def _fetch_from_s3_fallback(
    ra: float,
    dec: float,
    size_deg: float = 0.15,
    per_band: int = 2,
) -> dict:
    """Резервна витяжка двох епох напряму з AWS S3 з авторозширенням каталогу під ціль."""
    import re

    def _filter_near_target(df_all: pd.DataFrame, radius_deg: float = 1.72) -> pd.DataFrame:
        if df_all.empty or not {"s_ra", "s_dec", "t_min"}.issubset(df_all.columns):
            return pd.DataFrame()
        df_v = df_all[np.isfinite(df_all["t_min"]) & np.isfinite(df_all["s_ra"]) & np.isfinite(df_all["s_dec"])].copy()
        if df_v.empty:
            return df_v
        target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        cat_coords = SkyCoord(
            ra=df_v["s_ra"].to_numpy() * u.deg,
            dec=df_v["s_dec"].to_numpy() * u.deg,
            frame="icrs",
        )
        sep_deg = target.separation(cat_coords).deg
        return df_v[sep_deg <= radius_deg].sort_values("t_min").reset_index(drop=True)

    raw_df = pd.read_pickle(LOCAL_CATALOG_PATH) if LOCAL_CATALOG_PATH.exists() else build_offline_catalog()
    s3_df = _filter_near_target(raw_df, radius_deg=1.72)

    def _has_full_band_coverage(df_check: pd.DataFrame) -> bool:
        if len(df_check) < 20:
            return False
        df_s = df_check.sort_values("t_min").reset_index(drop=True)
        diffs_chk = np.diff(df_s["t_min"].to_numpy())
        if len(diffs_chk) == 0 or diffs_chk.max() <= 14.0:
            return False
        sp = int(np.argmax(diffs_chk)) + 1
        for part in (df_s.iloc[:sp], df_s.iloc[sp:]):
            dets = set(part["s3_url"].str.extract(r"(D[1-6])")[0].dropna())
            if len(dets & {"D1", "D2", "D3"}) < 2 or len(dets & {"D4", "D5", "D6"}) < 2:
                return False
        return True

    if not _has_full_band_coverage(s3_df):
        log(f"У локальному кеші неповний набір детекторів по епохах для RA={ra}, Dec={dec}. Доскановуємо сусідні витки на S3...")
        expanded_df = expand_s3_catalog_for_target(ra, dec, margin_px=50)
        s3_df = _filter_near_target(expanded_df, radius_deg=1.72)

    if s3_df.empty:
        raise ValueError(f"На AWS S3 не знайдено кадрів, що накривають RA={ra}, Dec={dec}.")

    CUTOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _process_one_row(row: pd.Series, label: str) -> dict | None:
        obs_id = str(row.get("obs_id") or Path(row["s3_url"]).stem)
        cache_key = hashlib.md5(f"{obs_id}_{ra:.5f}_{dec:.5f}_{size_deg:.4f}".encode()).hexdigest()
        cached_file = CUTOUT_CACHE_DIR / f"{cache_key}.fits"

        try:
            if not _is_full_square_fits(cached_file):
                log(f"  [{label}] Запит на S3: {Path(row['s3_url']).name}")
                _crop_full_fits_locally(row["s3_url"], ra, dec, size_deg, cached_file)

            lam = read_local_wavelength_um(cached_file)
            if lam is not None:
                mjd_val = float(row["t_min"])
                return {
                    "wave_um": round(lam, 4),
                    "mjd": mjd_val,
                    "date_obs": Time(mjd_val, format="mjd").iso[:19],
                    "path": cached_file,
                }
        except Exception as exc:
            log(f"  Пропущено S3-файл {obs_id}: {exc}")
        return None

    def _select_balanced_candidates(sub_df: pd.DataFrame) -> pd.DataFrame:
        sub = sub_df.copy()
        sub["det"] = sub["s3_url"].apply(
            lambda u_str: (m.group(0) if (m := re.search(r"D[1-6]", Path(str(u_str)).name)) else "D0")
        )
        picked = []
        for _, grp in sub.groupby("det"):
            grp_s = grp.sort_values("t_min")
            if len(grp_s) > per_band:
                idxs = np.linspace(0, len(grp_s) - 1, per_band, dtype=int)
                grp_s = grp_s.iloc[idxs]
            picked.append(grp_s)
        return pd.concat(picked) if picked else sub

    def _collect_s3_epoch(sub_df: pd.DataFrame, label: str) -> list[dict]:
        cand_df = _select_balanced_candidates(sub_df)
        slices = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_process_one_row, row, label) for _, row in cand_df.iterrows()]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    slices.append(res)
        slices.sort(key=lambda x: x["wave_um"])
        return slices

    s3_df = s3_df.sort_values("t_min").reset_index(drop=True)
    diffs = np.diff(s3_df["t_min"].to_numpy())
    if len(diffs) > 0 and diffs.max() > 14.0:
        split_idx = int(np.argmax(diffs)) + 1
        ep1_df = s3_df.iloc[:split_idx].copy()
        ep2_df = s3_df.iloc[split_idx:].copy()
    else:
        mid = max(1, len(s3_df) // 2)
        ep1_df, ep2_df = s3_df.iloc[:mid].copy(), s3_df.iloc[mid:].copy()

    ep1_label = Time(float(ep1_df["t_min"].min()), format="mjd").iso[:10]
    ep2_label = Time(float(ep2_df["t_min"].min()), format="mjd").iso[:10]

    log(f"S3-режим: збір Епохи 1 (~{ep1_label}, кандидатів: {len(ep1_df)})...")
    slices_ep1 = _collect_s3_epoch(ep1_df, "Ep1")

    log(f"S3-режим: збір Епохи 2 (~{ep2_label}, кандидатів: {len(ep2_df)})...")
    slices_ep2 = _collect_s3_epoch(ep2_df, "Ep2")

    return {
        "epoch1": {
            "date_label": ep1_label,
            "mjd_mean": round(float(ep1_df["t_min"].mean()), 3),
            "slices": slices_ep1,
        },
        "epoch2": {
            "date_label": ep2_label,
            "mjd_mean": round(float(ep2_df["t_min"].mean()), 3),
            "slices": slices_ep2,
        },
    }


def fetch_matched_wavelength_pair(
    ra: float,
    dec: float,
    size_deg: float = 0.15,
    min_time_gap_days: float = 1.0,
    max_wave_diff_um: float = 0.015,
    max_candidates: int = 20,
) -> tuple[Path, Path, float]:
    """Збережений метод для блінку: шукає два кадри одного каналу на різні дати."""
    df = search_spherex_epochs(ra, dec, radius_deg=0.2, limit=1000)
    if df.empty:
        raise ValueError("За вказаними координатами кадрів не знайдено.")

    df = df.copy()
    df["band_min_um"] = (df["em_min"] * 1e6).round(2)
    top_band = df["band_min_um"].value_counts().index[0]
    band_df = df[df["band_min_um"] == top_band].sort_values("t_min").copy()
    band_df["mjd_round"] = band_df["t_min"].round(3)
    band_df = band_df.drop_duplicates(subset=["mjd_round"]).head(max_candidates)

    candidates = []
    for _, row in band_df.iterrows():
        try:
            p = fetch_cutout_resilient(str(row["obs_id"]), row["access_url"], ra, dec, size_deg)
            lam = read_local_wavelength_um(p)
            if lam is None:
                continue
            new_cand = {"mjd": float(row["t_min"]), "wave_um": lam, "path": p}
            for prev in candidates:
                dt = abs(new_cand["mjd"] - prev["mjd"])
                dlam = abs(new_cand["wave_um"] - prev["wave_um"])
                if dt >= min_time_gap_days and dlam <= max_wave_diff_um:
                    c1, c2 = (prev, new_cand) if prev["mjd"] <= new_cand["mjd"] else (new_cand, prev)
                    return c1["path"], c2["path"], round((c1["wave_um"] + c2["wave_um"]) / 2.0, 4)
            candidates.append(new_cand)
        except Exception:
            continue

    raise RuntimeError("Не вдалося підібрати пару кадрів для блінку.")


def fetch_spectral_cube_slices(
    ra: float,
    dec: float,
    size_deg: float = 0.15,
    max_slices: int = 12,
) -> list[dict]:
    """Збережений метод для одиночного 3D-куба."""
    df = search_spherex_epochs(ra, dec, radius_deg=0.2, limit=1000)
    if df.empty:
        return []
    return _sample_window_slices(df, ra, dec, size_deg=size_deg, per_band=max(2, max_slices // 6))[:max_slices]