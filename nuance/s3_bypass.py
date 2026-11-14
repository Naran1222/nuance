from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import astropy.units as u
import numpy as np
import pandas as pd
import urllib3
from astropy.config import get_cache_dir
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from astropy.wcs import WCS

S3_BUCKET_URL = "https://nasa-irsa-spherex.s3.amazonaws.com"
LOCAL_CATALOG_PATH = Path("./spherex_cache/s3_catalog.pkl")
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_HTTP_POOL = urllib3.PoolManager(num_pools=4, maxsize=64, retries=1, timeout=10.0)


def harvest_astropy_cache() -> pd.DataFrame:
    """Витягує збережені DataLink-маніфести SPHEREx із локального кешу astropy."""
    cache_dir = Path(get_cache_dir()) / "download" / "url"
    if not cache_dir.exists():
        return pd.DataFrame()

    records = []
    for entry in cache_dir.iterdir():
        contents = entry / "contents"
        if not contents.is_file():
            continue
        try:
            raw_bytes = contents.read_bytes()
            if b"VOTABLE" not in raw_bytes[:400] or b"nasa-irsa-spherex" not in raw_bytes:
                continue

            votable = parse(str(contents))
            vo_table = list(votable.iter_tables())[0]
            tbl = vo_table.to_table()
            fmap = {fld.name.lower(): tbl.colnames[i] for i, fld in enumerate(vo_table.fields) if fld.name}

            obs_id, s3_url, s_ra, s_dec = "", "", np.nan, np.nan
            for row in tbl:
                sem = str(row[fmap["semantics"]]).strip() if "semantics" in fmap else ""
                if sem == "#this":
                    obs_id = str(row[fmap["id"]]).strip() if "id" in fmap else ""
                    for col in tbl.colnames:
                        val = str(row[col]).strip()
                        if val.startswith("{") and "nasa-irsa-spherex" in val:
                            aws = json.loads(val).get("aws", {})
                            if aws.get("key"):
                                s3_url = f"{S3_BUCKET_URL}/{aws['key']}"
                elif sem == "#cutout":
                    for col in tbl.colnames:
                        val = str(row[col]).strip()
                        if re.match(r"^-?\d+\.\d+$", val):
                            num = float(val)
                            if np.isnan(s_ra):
                                s_ra = num
                            elif np.isnan(s_dec):
                                s_dec = num

            if s3_url:
                records.append({"obs_id": obs_id, "s_ra": s_ra, "s_dec": s_dec, "s3_url": s3_url})
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.drop_duplicates(subset=["s3_url"]).reset_index(drop=True)
    return df


def read_s3_fits_header_fast(s3_url: str, with_wcs: bool = False) -> dict | None:
    """Читає перші 20 КБ FITS-файлу з AWS S3 через пул з'єднань для отримання MJD-OBS, координат і WCS."""
    resp = _HTTP_POOL.request("GET", s3_url, headers={"Range": "bytes=0-20159"})
    if resp.status not in (200, 206):
        return None

    text = resp.data[2880:].decode("ascii", errors="ignore")
    end_pos = text.find("END     ")
    if end_pos == -1:
        return None

    hdr = fits.Header.fromstring(text[: end_pos + 80])
    t_min = float(hdr.get("MJD-OBS", np.nan))
    if not np.isfinite(t_min):
        return None

    meta = {
        "obs_id": Path(s3_url).stem,
        "s_ra": float(hdr.get("CRVAL1", np.nan)),
        "s_dec": float(hdr.get("CRVAL2", np.nan)),
        "t_min": t_min,
        "date_obs": str(hdr.get("DATE-OBS", "")),
        "s3_url": s3_url,
    }
    if with_wcs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta["wcs"] = WCS(hdr, naxis=2)
    return meta


def _discover_s3_week_prefixes() -> list[str]:
    """Знаходить тижневі директорії в AWS S3, залишаючи лише найсвіжішу ревізію кожного тижня."""
    candidates = ["", "level2/", "dr1/level2/", "qr1/level2/", "qr2/level2/"]

    if LOCAL_CATALOG_PATH.exists():
        try:
            old_df = pd.read_pickle(LOCAL_CATALOG_PATH)
            if not old_df.empty and "s3_url" in old_df.columns:
                sample_url = str(old_df["s3_url"].iloc[0])
                m = re.search(r"amazonaws\.com/(.+?)20\d{2}W\d{2}", sample_url)
                if m:
                    candidates.insert(0, m.group(1))
        except Exception:
            pass

    for pref in dict.fromkeys(candidates):
        url = f"{S3_BUCKET_URL}/?list-type=2&prefix={pref}&delimiter=/"
        try:
            with urlopen(url, timeout=15) as resp:
                xml_root = ET.fromstring(resp.read())
            sub_prefixes = [
                el.text for el in xml_root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS) if el.text
            ]
            weeks = [p for p in sub_prefixes if re.search(r"20\d{2}W\d{2}", p)]
            if weeks:
                latest_by_week: dict[str, str] = {}
                for w in sorted(weeks):
                    m = re.search(r"(20\d{2}W\d{2})", w)
                    if m:
                        latest_by_week[m.group(1)] = w
                return sorted(latest_by_week.values())
        except Exception:
            continue
    return []


def _list_fits_in_prefix(prefix: str, max_pages: int = 50) -> list[str]:
    """Отримує повний відсортований у часі список .fits ключів у тижневій папці S3."""
    keys = []
    token = None
    for _ in range(max_pages):
        url = f"{S3_BUCKET_URL}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={quote(token)}"
        with urlopen(url, timeout=20) as resp:
            w_xml = ET.fromstring(resp.read())
        for el in w_xml.findall(".//s3:Contents/s3:Key", S3_NS):
            if el.text and el.text.endswith(".fits"):
                keys.append(el.text)
        is_trunc = w_xml.findtext(".//s3:IsTruncated", default="false", namespaces=S3_NS)
        token = w_xml.findtext(".//s3:NextContinuationToken", default=None, namespaces=S3_NS)
        if is_trunc.lower() != "true" or not token:
            break
    return sorted(keys)


def expand_s3_catalog_for_target(ra: float, dec: float, margin_px: int = 50) -> pd.DataFrame:
    """
    Знаходить на AWS S3 експозиції, де ціль (ra, dec) фізично потрапляє всередину
    матриці 2048x2048 детектора (перевірка через точну WCS-проєкцію заголовка).
    """
    week_prefixes = _discover_s3_week_prefixes()
    if not week_prefixes:
        print("Не вдалося отримати список тижнів із AWS S3.", flush=True)
        return pd.read_pickle(LOCAL_CATALOG_PATH) if LOCAL_CATALOG_PATH.exists() else pd.DataFrame()

    ecl = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").geocentrictrueecliptic
    target_lon = float(ecl.lon.deg)
    target_lat = float(ecl.lat.deg)

    scored_ep1, scored_ep2 = [], []
    for wp in week_prefixes:
        m = re.search(r"(20\d{2})W(\d{2})", wp)
        if not m:
            continue
        week_num = int(m.group(2))
        doy = (week_num - 1) * 7 + 4
        sun_lon = ((doy - 80.0) / 365.25 * 360.0) % 360.0
        dlon_rad = np.radians(target_lon - sun_lon)
        cos_elong = abs(np.cos(np.radians(target_lat)) * np.cos(dlon_rad))
        if np.sin(dlon_rad) >= 0:
            scored_ep1.append((cos_elong, wp))
        else:
            scored_ep2.append((cos_elong, wp))

    scored_ep1.sort(key=lambda x: x[0])
    scored_ep2.sort(key=lambda x: x[0])
    target_weeks = [wp for _, wp in scored_ep1[:3]] + [wp for _, wp in scored_ep2[:3]]

    print(
        f"Пошук кадрів біля RA={ra:.2f}, Dec={dec:.2f} на AWS S3 у тижнях: "
        f"{[w.strip('/').split('/')[-1] for w in target_weeks]}...",
        flush=True,
    )

    target_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    dense_exposures: list[list[str]] = []

    for wp in target_weeks:
        try:
            all_keys = _list_fits_in_prefix(wp, max_pages=50)
        except Exception:
            continue

        groups: dict[str, list[str]] = {}
        for k in all_keys:
            base = re.sub(r"D[1-6](?:_spx|\.).*$", "", k)
            groups.setdefault(base, []).append(f"{S3_BUCKET_URL}/{k}")

        exp_list = list(groups.values())
        if not exp_list:
            continue

        # Етап 1: грубий прохід по кожній 18-й експозиції тижня
        coarse_indices = list(range(0, len(exp_list), 10))
        coarse_map = {
            next((u for u in exp_list[i] if "D1" in Path(u).name), exp_list[i][0]): i
            for i in coarse_indices
        }

        hit_indices = []
        with ThreadPoolExecutor(max_workers=48) as pool:
            futures = {pool.submit(read_s3_fits_header_fast, u, False): idx for u, idx in coarse_map.items()}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    meta = fut.result()
                    if meta and np.isfinite(meta["s_ra"]) and np.isfinite(meta["s_dec"]):
                        sep = target_coord.separation(
                            SkyCoord(ra=meta["s_ra"] * u.deg, dec=meta["s_dec"] * u.deg, frame="icrs")
                        ).deg
                        if sep <= 14.0:
                            hit_indices.append(idx)
                except Exception:
                    continue

        selected_idx_set = set()
        for h_idx in hit_indices:
            for j in range(max(0, h_idx - 75), min(len(exp_list), h_idx + 76)):
                selected_idx_set.add(j)

        if not selected_idx_set:
            selected_idx_set.update(range(0, len(exp_list), 5))

        for j in sorted(selected_idx_set):
            dense_exposures.append(exp_list[j])

    if not dense_exposures:
        return pd.read_pickle(LOCAL_CATALOG_PATH) if LOCAL_CATALOG_PATH.exists() else pd.DataFrame()

    # Етап 2: точна WCS-перевірка попадання в матрицю 2048x2048 окремо для поля A (D1-D3) і B (D4-D6)
    probe_to_group: dict[str, list[str]] = {}
    for urls in dense_exposures:
        short_urls = [u for u in urls if re.search(r"D[1-3](?:_spx|\.)", Path(u).name)]
        long_urls = [u for u in urls if re.search(r"D[4-6](?:_spx|\.)", Path(u).name)]
        if short_urls:
            probe_to_group[short_urls[0]] = short_urls
        if long_urls:
            probe_to_group[long_urls[0]] = long_urls
        if not short_urls and not long_urls:
            probe_to_group[urls[0]] = urls

    print(f"Точна WCS-перевірка проєкції у {len(probe_to_group)} напівполях S3...", flush=True)
    new_records = []

    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(read_s3_fits_header_fast, p_url, True): p_url for p_url in probe_to_group}
        for fut in as_completed(futures):
            p_url = futures[fut]
            try:
                meta = fut.result()
                if not meta or "wcs" not in meta:
                    continue
                px, py = meta["wcs"].world_to_pixel(target_coord)
                if np.isfinite(px) and np.isfinite(py) and (margin_px <= px <= 2048 - margin_px) and (margin_px <= py <= 2048 - margin_px):
                    for sib_url in probe_to_group[p_url]:
                        new_records.append({
                            "obs_id": Path(sib_url).stem,
                            "s_ra": meta["s_ra"],
                            "s_dec": meta["s_dec"],
                            "t_min": meta["t_min"],
                            "date_obs": meta["date_obs"],
                            "s3_url": sib_url,
                            "exact_hit": True,
                        })
            except Exception:
                continue

    print(f"Знайдено FITS-файлів зі 100% влучанням у матрицю: {len(new_records)}", flush=True)

    frames = []
    if LOCAL_CATALOG_PATH.exists():
        frames.append(pd.read_pickle(LOCAL_CATALOG_PATH))
    if new_records:
        frames.append(pd.DataFrame(new_records))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["s3_url"], keep="last").sort_values("t_min").reset_index(drop=True)
    LOCAL_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_pickle(LOCAL_CATALOG_PATH)
    return merged


def build_offline_catalog() -> pd.DataFrame:
    """Формує базовий локальний каталог метаданих із кешу astropy."""
    df = harvest_astropy_cache()
    if df.empty:
        return pd.DataFrame()

    enriched = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(read_s3_fits_header_fast, u, False): u for u in df["s3_url"]}
        for fut in as_completed(futures):
            try:
                meta = fut.result()
                if meta:
                    enriched.append(meta)
            except Exception:
                continue

    if not enriched:
        return pd.DataFrame()

    cat_df = pd.DataFrame(enriched).drop_duplicates(subset=["s3_url"]).sort_values("t_min").reset_index(drop=True)
    LOCAL_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_pickle(LOCAL_CATALOG_PATH)
    return cat_df