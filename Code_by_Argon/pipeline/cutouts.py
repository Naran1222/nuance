"""
КРОК 2. Завантаження вирізок FITS та їх читання.

Реалізує:
  - Чесний 2D Stratified Round-Robin відбір кадрів (select_balanced_indices) по всіх
    епохах та обраних смугах детекторів у межах ліміту (30, 60, 100, 180) зі строгим
    хронологічним сортуванням за MJD на виході.
  - Повну відмову від urllib.request.urlopen на користь потокобезпечної requests.Session.
  - Автоматичне перевикористання вже наявних у cache/cutouts/ FITS-файлів тієї ж експозиції
    (зокрема локальне вирізання меншої ділянки з уже завантаженого ширшого кешу).
  - Швидке перемикання на пряме завантаження повного FITS із AWS S3
    (https://nasa-irsa-spherex.s3.amazonaws.com/{s3_key}) та локальний Cutout2D при будь-якій
    помилці 502/503/таймауті серверів IRSA.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

from pipeline.catalog import (
    S3_BUCKET_URL,
    get_http_session,
    is_irsa_ibe_unavailable,
    resolve_target,
    set_irsa_ibe_unavailable,
)

# astropy дуже багато пише про альтернативні WCS, прибираємо цей шум
logging.getLogger("astropy").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
CUTOUT_CACHE_DIR = ROOT / "cache" / "cutouts"

WEB_LIMIT_OPTIONS = (30, 60, 100, 180)
DEFAULT_WEB_LIMIT = 100


# --------------------------------------------------------------------------
# Чесний 2D Stratified Round-Robin розподіл кадрів по епохах та смугах
# --------------------------------------------------------------------------
def _spread_order(n: int) -> List[int]:
    """
    Впорядковує індекси 0..n-1 методом двійкового поділу (midpoint subdivision):
    спочатку середина діапазону, далі чверті, восьмі тощо, щоб будь-який префікс
    рівномірно охоплював увесь спектральний діапазон смуги.
    """
    if n <= 0:
        return []
    if n == 1:
        return [0]

    result: List[int] = []
    queue: List[Tuple[int, int]] = [(0, n - 1)]
    while queue:
        lo, hi = queue.pop(0)
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        result.append(mid)
        if lo <= mid - 1:
            queue.append((lo, mid - 1))
        if mid + 1 <= hi:
            queue.append((mid + 1, hi))
    return result


def _assign_epoch_bins(mjd: np.ndarray, gap_days: float = 45.0) -> np.ndarray:
    """Розбиває спостереження на часові епохи за розривом > gap_days діб."""
    n = len(mjd)
    if n == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(mjd)
    sorted_mjd = mjd[order]
    bins_sorted = np.zeros(n, dtype=int)
    cur = 0
    for i in range(1, n):
        if np.isfinite(sorted_mjd[i]) and np.isfinite(sorted_mjd[i - 1]):
            if (sorted_mjd[i] - sorted_mjd[i - 1]) > gap_days:
                cur += 1
        bins_sorted[i] = cur
    out = np.empty(n, dtype=int)
    out[order] = bins_sorted
    return out


def select_balanced_indices(table: Table, limit: int, gap_days: float = 45.0) -> np.ndarray:
    """
    Відбирає рівно `limit` записів із таблиці каталогу методом 2D Stratified Round-Robin:
      1. Групує всі знайдені знімки у кошики (епоха, смуга_детектора).
      2. Усередині кожного кошика впорядковує кадри за midpoint subdivision (_spread_order).
      3. Циклічно бере по 1 кадру з кожного кошика до досягнення `limit`.
      4. Сортує фінальну вибірку строго хронологічно за MJD.
    """
    total = len(table)
    if limit is None or int(limit) <= 0 or int(limit) >= total:
        return np.arange(total, dtype=int)

    limit = int(limit)
    mjd = np.asarray(table["mjd"], dtype=float)
    bands = [str(b) for b in table["band"]] if "band" in table.colnames else ["ALL"] * total
    epochs = _assign_epoch_bins(mjd, gap_days=gap_days)

    buckets: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for idx in range(total):
        buckets[(int(epochs[idx]), bands[idx])].append(idx)

    ordered_queues: List[List[int]] = []
    for key in sorted(buckets.keys(), key=lambda k: (k[0], k[1])):
        idxs = sorted(buckets[key], key=lambda i: (np.isnan(mjd[i]), mjd[i], i))
        spread = _spread_order(len(idxs))
        ordered_queues.append([idxs[pos] for pos in spread])

    selected: List[int] = []
    pointers = [0] * len(ordered_queues)
    while len(selected) < limit:
        progressed = False
        for q_idx, queue in enumerate(ordered_queues):
            if pointers[q_idx] < len(queue):
                selected.append(queue[pointers[q_idx]])
                pointers[q_idx] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break

    selected.sort(key=lambda i: (np.isnan(mjd[i]), mjd[i], bands[i], i))
    return np.array(selected, dtype=int)


# --------------------------------------------------------------------------
# Перевірка FITS та локальна вирізка (Cutout2D)
# --------------------------------------------------------------------------
def _get_hdu(hdul, name: str, index: int):
    if name in hdul:
        return hdul[name]
    return hdul[index] if len(hdul) > index else None


def is_valid_fits(path) -> bool:
    """True, якщо файл відкривається і всі розширення читаються без помилок."""
    try:
        with fits.open(path, memmap=False) as hdul:
            if len(hdul) < 2:
                return False
            img_hdu = _get_hdu(hdul, "IMAGE", 1)
            if img_hdu is None or img_hdu.data is None or img_hdu.data.ndim != 2:
                return False
            ny, nx = img_hdu.data.shape
            if nx < 4 or ny < 4:
                return False
            for hdu in hdul:
                _ = hdu.data
        return True
    except Exception:
        return False


def local_path_for(url: str) -> Path:
    name = url.split("?")[0].rsplit("/", 1)[-1]
    if name.lower().endswith(".fits"):
        name = name[:-5]
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return CUTOUT_CACHE_DIR / f"{name}__{digest}.fits"


def _extract_s3_and_cutout_params(
    url: str,
) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float], Optional[float]]:
    """
    Витягує (s3_url, s3_key, ra, dec, size_deg) із URL вирізки IRSA або S3.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    ra = dec = size_deg = None

    if "center" in qs:
        c_str = qs["center"][0].replace("d", "").strip()
        parts = [p.strip() for p in c_str.split(",") if p.strip()]
        if len(parts) == 2:
            try:
                ra, dec = float(parts[0]), float(parts[1])
            except ValueError:
                pass
    elif "ra" in qs and "dec" in qs:
        try:
            ra, dec = float(qs["ra"][0]), float(qs["dec"][0])
        except ValueError:
            pass

    if "size" in qs:
        try:
            size_deg = float(qs["size"][0].replace("d", "").strip())
        except ValueError:
            pass

    if "amazonaws.com" in parsed.netloc:
        s3_key = parsed.path.lstrip("/")
        return f"{S3_BUCKET_URL}/{s3_key}", s3_key, ra, dec, size_deg

    m = re.search(r"/spherex/(.+?\.fits)$", parsed.path, re.IGNORECASE)
    if m:
        s3_key = m.group(1)
        return f"{S3_BUCKET_URL}/{s3_key}", s3_key, ra, dec, size_deg
    return None, None, ra, dec, size_deg


def _crop_hdul_to_file(
    hdul: fits.HDUList,
    ra: float,
    dec: float,
    size_deg: float,
    out_path: Path,
) -> bool:
    """
    Вирізає квадрат (ra, dec, size_deg) із відкритого FITS (повного знімка S3 або ширшої вирізки)
    і зберігає у валідний багаторозширений FITS (PRIMARY, IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE)
    з точним збереженням LTV1, LTV2, CRPIX1W, CRPIX2W.
    """
    img_hdu = _get_hdu(hdul, "IMAGE", 1)
    flags_hdu = _get_hdu(hdul, "FLAGS", 2)
    if img_hdu is None or img_hdu.data is None or flags_hdu is None or flags_hdu.data is None:
        return False

    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    size = u.Quantity((size_deg, size_deg), u.deg)
    wcs_in = WCS(img_hdu.header)

    cut_img = Cutout2D(
        img_hdu.data,
        position=center,
        size=size,
        wcs=wcs_in,
        mode="partial",
        fill_value=np.nan,
    )
    if np.isfinite(cut_img.data).mean() < 0.60:
        return False

    x0_orig = float(cut_img.xmin_original)
    y0_orig = float(cut_img.ymin_original)
    prev_ltv1 = float(img_hdu.header.get("LTV1", 0.0))
    prev_ltv2 = float(img_hdu.header.get("LTV2", 0.0))

    cut_hdr = img_hdu.header.copy()
    wcs_cards = cut_img.wcs.to_header(relax=True)
    for k, v in wcs_cards.items():
        cut_hdr[k] = v

    cut_hdr["LTV1"] = prev_ltv1 - x0_orig
    cut_hdr["LTV2"] = prev_ltv2 - y0_orig
    if "CRPIX1W" in cut_hdr:
        cut_hdr["CRPIX1W"] = float(cut_hdr["CRPIX1W"]) - x0_orig
    if "CRPIX2W" in cut_hdr:
        cut_hdr["CRPIX2W"] = float(cut_hdr["CRPIX2W"]) - y0_orig

    primary = fits.PrimaryHDU(header=hdul[0].header.copy())
    new_hdus = [
        primary,
        fits.ImageHDU(data=cut_img.data.astype(np.float32), header=cut_hdr, name="IMAGE"),
    ]

    cut_flags = Cutout2D(
        flags_hdu.data,
        position=center,
        size=size,
        wcs=wcs_in,
        mode="partial",
        fill_value=1,
    )
    new_hdus.append(
        fits.ImageHDU(data=cut_flags.data.astype(np.int32), header=cut_hdr.copy(), name="FLAGS")
    )

    var_hdu = _get_hdu(hdul, "VARIANCE", 3)
    if var_hdu is not None and var_hdu.data is not None:
        cut_var = Cutout2D(
            var_hdu.data,
            position=center,
            size=size,
            wcs=wcs_in,
            mode="partial",
            fill_value=np.nan,
        )
        new_hdus.append(
            fits.ImageHDU(data=cut_var.data.astype(np.float32), header=cut_hdr.copy(), name="VARIANCE")
        )

    if "ZODI" in hdul and hdul["ZODI"].data is not None:
        cut_zodi = Cutout2D(
            hdul["ZODI"].data,
            position=center,
            size=size,
            wcs=wcs_in,
            mode="partial",
            fill_value=np.nan,
        )
        new_hdus.append(
            fits.ImageHDU(data=cut_zodi.data.astype(np.float32), header=cut_hdr.copy(), name="ZODI")
        )

    if "WCS-WAVE" in hdul:
        ww = hdul["WCS-WAVE"]
        new_hdus.append(
            fits.BinTableHDU(data=ww.data.copy(), header=ww.header.copy(), name="WCS-WAVE")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(new_hdus).writeto(out_path, overwrite=True)
    return True


def _try_reuse_local_sibling_cutout(
    url: str,
    ra: Optional[float],
    dec: Optional[float],
    size_deg: Optional[float],
    target_path: Path,
) -> Optional[Path]:
    """
    Якщо в cache/cutouts/ уже лежить FITS цієї ж експозиції (наприклад, скачаний раніше з більшим
    --size 0.5 у demo.py або з координатами M101, що відрізняються на 0.5"),
    вирізає або копіює його локально без мережевих запитів.
    """
    if not CUTOUT_CACHE_DIR.exists() or ra is None or dec is None or size_deg is None:
        return None

    stem = url.split("?")[0].rsplit("/", 1)[-1]
    if stem.lower().endswith(".fits"):
        stem = stem[:-5]

    siblings = sorted(CUTOUT_CACHE_DIR.glob(f"{stem}__*.fits"))
    if not siblings:
        return None

    expected_px = int(round(float(size_deg) * 3600.0 / 6.15))
    for sib in siblings:
        if sib == target_path or not is_valid_fits(sib):
            continue
        try:
            with fits.open(sib, memmap=False) as hdul:
                img_hdu = _get_hdu(hdul, "IMAGE", 1)
                ny, nx = img_hdu.data.shape
                if abs(nx - expected_px) <= 3 and abs(ny - expected_px) <= 3:
                    shutil.copy2(sib, target_path)
                    return target_path
                if nx >= expected_px and ny >= expected_px:
                    tmp = target_path.with_name(f"{target_path.name}.{uuid.uuid4().hex[:8]}.part")
                    if _crop_hdul_to_file(hdul, ra, dec, size_deg, tmp) and is_valid_fits(tmp):
                        tmp.replace(target_path)
                        return target_path
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        except Exception:
            continue
    return None


def _download_full_fits_from_s3(s3_url: str, s3_key: Optional[str], dest_tmp: Path, timeout=(10, 60)) -> bool:
    """Качає повний FITS із публічного бакета AWS S3 через requests.Session або boto3 UNSIGNED."""
    session = get_http_session()
    try:
        with session.get(s3_url, stream=True, timeout=timeout) as resp:
            if resp.status_code == 200:
                with open(dest_tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                if dest_tmp.exists() and dest_tmp.stat().st_size > 1_000_000:
                    return True
    except Exception:
        pass

    if s3_key:
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
            s3.download_file("nasa-irsa-spherex", s3_key, str(dest_tmp))
            if dest_tmp.exists() and dest_tmp.stat().st_size > 1_000_000:
                return True
        except Exception:
            pass
    return False


def _crop_s3_fits_locally(
    s3_url: str,
    s3_key: Optional[str],
    ra: float,
    dec: float,
    size_deg: float,
    out_path: Path,
) -> bool:
    """Скачує повний FITS із AWS S3 у тимчасовий файл, вирізає Cutout2D і чистить тимчасовий файл."""
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not _download_full_fits_from_s3(s3_url, s3_key, tmp_path):
            return False
        with fits.open(tmp_path, memmap=False) as hdul:
            if not _crop_hdul_to_file(hdul, ra, dec, size_deg, out_path):
                return False
        return is_valid_fits(out_path)
    except Exception:
        return False
    finally:
        if tmp_path.exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def fetch_cutout(
    url: str,
    retries: int = 2,
    timeout=(8, 25),
    coord: Optional[SkyCoord] = None,
    size_deg: float = 0.1,
) -> Optional[Path]:
    """
    Завантажує одну вирізку у cache/cutouts/.
      1. Перевіряє кеш на диску (включно з локальними вирізками тієї ж експозиції).
      2. Якщо IRSA доступна — робить до 2 швидких спроб через requests.Session.
      3. Якщо IRSA повернула 502/503/таймаут — миттєво перемикається на AWS S3 + Cutout2D.
    """
    CUTOUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = local_path_for(url)

    if path.exists():
        if is_valid_fits(path):
            return path
        try:
            path.unlink()
        except OSError:
            pass

    s3_url, s3_key, u_ra, u_dec, u_size = _extract_s3_and_cutout_params(url)
    ra_val = u_ra if u_ra is not None else (float(coord.ra.deg) if coord is not None else None)
    dec_val = u_dec if u_dec is not None else (float(coord.dec.deg) if coord is not None else None)
    size_val = u_size if u_size is not None else float(size_deg)

    reused = _try_reuse_local_sibling_cutout(url, ra_val, dec_val, size_val, path)
    if reused is not None:
        return reused

    session = get_http_session()
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.part")
    is_direct_s3 = "amazonaws.com" in url

    if not is_direct_s3 and not is_irsa_ibe_unavailable():
        max_tries = min(int(retries), 2)
        for attempt in range(1, max_tries + 1):
            try:
                with session.get(url, stream=True, timeout=timeout) as resp:
                    if resp.status_code in (500, 502, 503, 504):
                        set_irsa_ibe_unavailable(True)
                        break
                    resp.raise_for_status()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 16):
                            if chunk:
                                fh.write(chunk)
                if is_valid_fits(tmp):
                    tmp.replace(path)
                    return path
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                if code in (500, 502, 503, 504):
                    set_irsa_ibe_unavailable(True)
                    break
                if 400 <= code < 500 and code not in (408, 429):
                    break
            except (requests.RequestException, OSError):
                if attempt >= max_tries:
                    set_irsa_ibe_unavailable(True)
            if attempt < max_tries:
                time.sleep(1.5 * attempt)

    try:
        tmp.unlink()
    except OSError:
        pass

    # Fallback: завантаження повного FITS із дзеркала AWS S3 та локальний Cutout2D
    if s3_url and ra_val is not None and dec_val is not None:
        if _crop_s3_fits_locally(s3_url, s3_key, ra_val, dec_val, size_val, tmp) and is_valid_fits(tmp):
            tmp.replace(path)
            return path
        try:
            tmp.unlink()
        except OSError:
            pass

    return None


def fetch_all(
    urls: List[str],
    workers: int = 6,
    verbose: bool = True,
    progress=None,
    coord: Optional[SkyCoord] = None,
    size_deg: float = 0.1,
) -> Dict[str, Optional[Path]]:
    """Паралельне завантаження. Повертає словник {url: шлях або None}."""
    results: Dict[str, Optional[Path]] = {}
    total = len(urls)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_cutout, url, 2, (8, 25), coord, size_deg): url
            for url in urls
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if progress is not None:
                progress(done, total)
            if verbose and (done % 5 == 0 or done == total):
                print(f"  [cutouts] завантажено {done}/{total} ({time.time() - started:.0f} с)")
    return results


# --------------------------------------------------------------------------
# Читання FITS-кадрів
# --------------------------------------------------------------------------
def _wavelength_from_wcswave_grid(hdul, header, x_pix: float, y_pix: float) -> float:
    """Резервне обчислення локальної довжини хвилі (мкм) з 2D-сітки WCS-WAVE через LTV1/LTV2."""
    try:
        if "WCS-WAVE" not in hdul:
            return float("nan")
        ww_data = hdul["WCS-WAVE"].data
        if ww_data is None or len(ww_data) == 0:
            return float("nan")

        if "LTV1" in header and "LTV2" in header:
            x_det = float(x_pix) - float(header["LTV1"])
            y_det = float(y_pix) - float(header["LTV2"])
        else:
            x_det = float(x_pix)
            y_det = float(y_pix)

        row = ww_data[0]
        x_nodes = np.asarray(row[0], dtype=np.float64).ravel()
        y_nodes = np.asarray(row[1], dtype=np.float64).ravel()
        grid_raw = np.asarray(row[2], dtype=np.float64).ravel()
        nx, ny = len(x_nodes), len(y_nodes)
        if grid_raw.size == nx * ny * 2:
            lam_grid = grid_raw.reshape((ny, nx, 2))[:, :, 0]
            xc = float(np.clip(x_det, x_nodes.min(), x_nodes.max()))
            yc = float(np.clip(y_det, y_nodes.min(), y_nodes.max()))
            interp = RegularGridInterpolator(
                (y_nodes, x_nodes), lam_grid, bounds_error=False, fill_value=None
            )
            lam = float(interp([[yc, xc]])[0])
            return lam * 1e6 if lam < 0.1 else lam
    except Exception:
        pass
    return float("nan")


def read_frame(path, coord: SkyCoord) -> Optional[dict]:
    """Читає FITS (вирізку або повний файл) і повертає кадр. None, якщо файл непридатний."""
    try:
        with fits.open(path, memmap=False) as hdul:
            image_hdu = _get_hdu(hdul, "IMAGE", 1)
            flags_hdu = _get_hdu(hdul, "FLAGS", 2)
            var_hdu = _get_hdu(hdul, "VARIANCE", 3)
            zodi_hdu = hdul["ZODI"] if "ZODI" in hdul else None
            if image_hdu is None or image_hdu.data is None or flags_hdu is None:
                return None

            header = image_hdu.header
            image = np.array(image_hdu.data, dtype=np.float32)
            flags = np.array(flags_hdu.data, dtype=np.int32)
            variance = None
            if var_hdu is not None and var_hdu.data is not None:
                variance = np.array(var_hdu.data, dtype=np.float32)
            zodi = None
            if zodi_hdu is not None and zodi_hdu.data is not None:
                zodi = np.array(zodi_hdu.data, dtype=np.float32)
                if zodi.shape != image.shape:
                    zodi = None

            wcs = WCS(header)
            x, y = wcs.world_to_pixel(coord)
            x, y = float(x), float(y)

            wavelength = float("nan")
            try:
                spectral_wcs = WCS(header, fobj=hdul, key="W")
                spectral_wcs.sip = None
                wl, _bandwidth = spectral_wcs.pixel_to_world(x, y)
                wavelength = float(wl.to(u.micrometer).value)
            except Exception:
                pass
            if not np.isfinite(wavelength) or not (0.5 <= wavelength <= 6.0):
                wavelength = _wavelength_from_wcswave_grid(hdul, header, x, y)

            mjd = header.get("MJD-AVG", header.get("MJD-OBS", float("nan")))
            ny, nx = image.shape
            det = header.get("DETECTOR")
            band_str = f"SPHEREx-D{int(det)}" if det is not None else ""
            return {
                "path": str(path),
                "image": image,
                "flags": flags,
                "variance": variance,
                "zodi": zodi,
                "wcs": wcs,
                "mjd": float(mjd) if mjd is not None else float("nan"),
                "wavelength_um": wavelength,
                "detector": det,
                "band": band_str,
                "exposure_s": header.get("XPOSURE"),
                "version": hdul[0].header.get("VERSION", ""),
                "x": x,
                "y": y,
                "inside": bool(0 <= x < nx and 0 <= y < ny),
            }
    except Exception as exc:
        print(f"  [cutouts] не вдалося прочитати {Path(path).name}: {type(exc).__name__}: {exc}")
        return None


def load_frames(
    table: Table,
    target,
    limit: int = 0,
    workers: int = 6,
    verbose: bool = True,
    progress=None,
) -> List[dict]:
    """
    Таблиця знімків -> список кадрів, відсортований строго за часом (MJD).
    limit > 0: відбирає рівно `limit` знімків за алгоритмом 2D Stratified Round-Robin
               (рівномірно по всіх епохах спостереження та обраних смугах детекторів).
               0 = завантажити всі знайдені знімки.
    """
    coord = resolve_target(target)
    if len(table) == 0:
        return []

    indices = select_balanced_indices(table, limit=limit)
    urls = [str(table["uri"][i]) for i in indices]
    size_deg = float(table.meta.get("size_deg", 0.1))

    if verbose:
        print(f"  [cutouts] завантажую {len(urls)} вирізок із {len(table)} (паралельно: {workers})...")
    paths = fetch_all(urls, workers=workers, verbose=verbose, progress=progress,
                      coord=coord, size_deg=size_deg)

    frames: List[dict] = []
    failed = 0
    for i, url in zip(indices, urls):
        path = paths.get(url)
        frame = read_frame(path, coord) if path else None
        if frame is None:
            failed += 1
            continue
        frame["url"] = url
        if not frame.get("band") and "band" in table.colnames:
            frame["band"] = str(table["band"][i])
        frames.append(frame)

    frames.sort(key=lambda f: (np.isnan(f["mjd"]), f["mjd"], str(f.get("band", ""))))
    if verbose:
        print(f"  [cutouts] успішно: {len(frames)}, не вдалося: {failed}")
    return frames


def frames_summary(frames: List[dict]) -> Table:
    """Підсумкова таблиця кадрів: коли, яка довжина хвилі, куди потрапив об'єкт."""
    def iso(m):
        return Time(m, format="mjd").iso[:19] if np.isfinite(m) else ""

    return Table(
        {
            "idx": list(range(len(frames))),
            "date_utc": [iso(f["mjd"]) for f in frames],
            "mjd": [round(f["mjd"], 5) for f in frames],
            "wavelength_um": [round(f["wavelength_um"], 4) for f in frames],
            "detector": [int(f["detector"]) if f["detector"] is not None else -1 for f in frames],
            "x_pix": [round(f["x"], 1) for f in frames],
            "y_pix": [round(f["y"], 1) for f in frames],
            "shape": [f"{f['image'].shape[1]}x{f['image'].shape[0]}" for f in frames],
            "file": [Path(f["path"]).name for f in frames],
        }
    )