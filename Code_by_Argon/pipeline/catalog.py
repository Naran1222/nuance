"""
КРОК 1. Пошук знімків SPHEREx за координатами та обраними смугами детекторів.

Реалізує 3-рівневий відмовостійкий каскад пошуку (без використання urllib.request.urlopen):
  - Рівень 1: Прямий синхронний запит до IRSA TAP/sync (spherex.artifact + spherex.plane,
              або резервно ivoa.obscore) через потокобезпечну requests.Session.
  - Рівень 2: Швидкий мікросервіс IRSA SIAv2 + паралельний резолвінг DataLink у S3-ключі.
  - Рівень 3: Прямий пошук в AWS S3 (nasa-irsa-spherex.s3.amazonaws.com) через HTTP Range-запити
              до FITS-заголовків (логіка s3_bypass) та збір локальних кешів.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import re
import threading
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import astropy.units as u
import numpy as np
import requests
from astropy.config import get_cache_dir
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.votable import parse
from astropy.table import Table
from astropy.wcs import WCS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
CATALOG_CACHE_DIR = ROOT / "cache" / "catalog"
TARGETS_CACHE_FILE = ROOT / "cache" / "targets.json"
DATALINK_CACHE_FILE = ROOT / "cache" / "catalog" / "datalink_s3_map.json"
S3_INDEX_CACHE_FILE = ROOT / "cache" / "catalog" / "s3_headers_index.json"

S3_BUCKET_URL = "https://nasa-irsa-spherex.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

BANDS = [f"SPHEREx-D{i}" for i in range(1, 7)]
DEFAULT_BAND = "SPHEREx-D2"
MIN_SIZE_DEG = 0.005
MAX_SIZE_DEG = 0.5

# Глобальний прапорець стану проксі IRSA: якщо IRSA віддає 502/503, cutouts.py одразу качає з AWS S3
_IRSA_IBE_UNAVAILABLE = False
_SESSION_LOCK = threading.Lock()
_HTTP_SESSION: Optional[requests.Session] = None

# Вбудовані точні координати популярних цілей, щоб не залежати від збоїв Sesame
_BUILTIN_TARGETS: Dict[str, Tuple[float, float]] = {
    "M101": (210.80227, 54.34895),
    "NGC 5457": (210.80227, 54.34895),
    "NGC5457": (210.80227, 54.34895),
    "M44": (130.025, 19.66667),
    "M42": (83.82208, -5.39111),
    "M31": (10.68471, 41.26875),
    "M82": (148.96846, 69.67970),
    "NGC 253": (11.88806, -25.28880),
    "NGC253": (11.88806, -25.28880),
    "NGC 6543": (269.63926, 66.63320),
    "NGC6543": (269.63926, 66.63320),
}


def is_irsa_ibe_unavailable() -> bool:
    return _IRSA_IBE_UNAVAILABLE


def set_irsa_ibe_unavailable(flag: bool = True) -> None:
    global _IRSA_IBE_UNAVAILABLE
    _IRSA_IBE_UNAVAILABLE = bool(flag)

def inspect_s3_root():
    session = get_http_session()
    resp = session.get(
        f"{S3_BUCKET_URL}/",
        params={"list-type": "2", "delimiter": "/"},
        timeout=10,
    )
    print(resp.text)

def get_http_session() -> requests.Session:
    """Повертає потокобезпечну HTTP-сесію з пулом з'єднань, сертифікатами certifi та автоповторами."""
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION
    with _SESSION_LOCK:
        if _HTTP_SESSION is None:
            session = requests.Session()
            retry = Retry(
                total=2,
                connect=2,
                read=1,
                backoff_factor=0.4,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset(["GET", "POST", "HEAD"]),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64, max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) SPHERExPipeline/2.0"
                ),
                "Accept": "*/*",
                "Connection": "keep-alive",
            })
            try:
                import certifi
                session.verify = certifi.where()
            except ImportError:
                pass
            _HTTP_SESSION = session
    return _HTTP_SESSION


# --------------------------------------------------------------------------
# Нормалізація смуг та координат
# --------------------------------------------------------------------------
def normalize_band(band: Optional[str]) -> Optional[str]:
    if band is None:
        return None
    text = str(band).strip()
    if text.lower() in ("", "all", "any", "*", "усі", "всі"):
        return None
    text = text.upper().replace("SPHEREX-", "")
    if text.startswith("D"):
        text = text[1:]
    candidate = f"SPHEREx-D{text}"
    if candidate not in BANDS:
        raise ValueError(
            f"Невідома смуга '{band}'. Допустимі: D1..D6 (наприклад 'D2') або 'all'."
        )
    return candidate


def normalize_bands(bands: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    """
    Приймає рядок ('D1,D3,D5', 'D2', 'all') або список/кортеж смуг.
    Повертає відсортований список канонічних назв ['SPHEREx-D1', ...] або None (= усі 6 смуг).
    """
    if bands is None:
        return None
    if isinstance(bands, str):
        raw_items = [part.strip() for part in re.split(r"[,;\s]+", bands.strip()) if part.strip()]
    elif isinstance(bands, Iterable):
        raw_items = []
        for item in bands:
            if item is None:
                continue
            for sub in re.split(r"[,;\s]+", str(item).strip()):
                if sub:
                    raw_items.append(sub)
    else:
        raw_items = [str(bands).strip()]

    if not raw_items:
        return None

    normalized: List[str] = []
    for item in raw_items:
        b = normalize_band(item)
        if b is None:
            return None
        if b not in normalized:
            normalized.append(b)

    normalized.sort(key=lambda x: BANDS.index(x))
    if len(normalized) == len(BANDS):
        return None
    return normalized


def _bands_tag(bands_list: Optional[List[str]]) -> str:
    if not bands_list:
        return "ALL"
    if len(bands_list) == 1:
        return bands_list[0]
    return "-".join(b.replace("SPHEREx-", "") for b in bands_list)


def _infer_band_from_wavelength_um(lam_um: float) -> Optional[str]:
    if not np.isfinite(lam_um) or lam_um <= 0:
        return None
    if lam_um < 0.01:
        lam_um *= 1e6
    if lam_um < 1.08:
        return "SPHEREx-D1"
    if lam_um < 1.58:
        return "SPHEREx-D2"
    if lam_um < 2.35:
        return "SPHEREx-D3"
    if lam_um < 3.70:
        return "SPHEREx-D4"
    if lam_um < 4.35:
        return "SPHEREx-D5"
    if lam_um <= 5.50:
        return "SPHEREx-D6"
    return None


def _infer_band_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"SPHEREX-D([1-6])", text, re.IGNORECASE)
    if m:
        return f"SPHEREx-D{m.group(1)}"
    m2 = re.search(r"[_/0-9]D([1-6])(?:_spx|\b|_|\.)", text, re.IGNORECASE)
    if m2:
        return f"SPHEREx-D{m2.group(1)}"
    m3 = re.search(r"\bD([1-6])\b", text, re.IGNORECASE)
    if m3:
        return f"SPHEREx-D{m3.group(1)}"
    return None


def _load_target_cache() -> dict:
    if TARGETS_CACHE_FILE.exists():
        try:
            return json.loads(TARGETS_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_target_cache(data: dict) -> None:
    try:
        TARGETS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TARGETS_CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _resolve_via_sesame_requests(name: str) -> Optional[SkyCoord]:
    """Розпізнає назву об'єкта через CDS Sesame за допомогою requests.Session (без urllib.request)."""
    session = get_http_session()
    urls = [
        f"https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV?{quote(name)}",
        f"http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/SNV?{quote(name)}",
    ]
    for url in urls:
        try:
            resp = session.get(url, timeout=(6, 12))
            if resp.status_code == 200 and "<jradeg>" in resp.text and "<jdedeg>" in resp.text:
                m_ra = re.search(r"<jradeg>\s*([+-]?\d+(?:\.\d+)?)\s*</jradeg>", resp.text)
                m_dec = re.search(r"<jdedeg>\s*([+-]?\d+(?:\.\d+)?)\s*</jdedeg>", resp.text)
                if m_ra and m_dec:
                    ra = float(m_ra.group(1))
                    dec = float(m_dec.group(1))
                    return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        except Exception:
            continue
    return None


def resolve_target(target: Union[str, SkyCoord]) -> SkyCoord:
    if isinstance(target, SkyCoord):
        return target.icrs

    text = str(target).strip()
    if not text:
        raise ValueError("Порожнє введення: вкажи назву об'єкта або координати.")

    parts = text.replace(",", " ").split()
    if len(parts) == 2:
        try:
            ra, dec = float(parts[0]), float(parts[1])
        except ValueError:
            pass
        else:
            if not (0.0 <= ra < 360.0) or not (-90.0 <= dec <= 90.0):
                raise ValueError(
                    f"Координати поза межами: RA={ra}, Dec={dec} "
                    "(RA має бути 0..360, Dec -90..90 градусів)."
                )
            return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    key = re.sub(r"\s+", " ", text.upper())
    cache = _load_target_cache()
    if key in cache:
        ra, dec = cache[key]
        return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    if key in _BUILTIN_TARGETS:
        ra, dec = _BUILTIN_TARGETS[key]
        cache[key] = [ra, dec]
        _save_target_cache(cache)
        return SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    coord = _resolve_via_sesame_requests(text)
    if coord is None:
        try:
            coord = SkyCoord.from_name(text).icrs
        except Exception as exc:
            raise ValueError(
                f"Не вдалося знайти об'єкт '{text}'. Перевір назву або введи координати "
                f"у градусах, наприклад '210.80227 54.34895'. ({type(exc).__name__})"
            ) from exc

    cache[key] = [round(float(coord.ra.deg), 6), round(float(coord.dec.deg), 6)]
    _save_target_cache(cache)
    return coord


_RELEASE_RANK = {"dr1": 4, "qr3": 3, "qr2": 2, "qr1": 1}


def release_of(uri: str) -> str:
    m = re.search(r"/spherex/([A-Za-z0-9_]+)/", str(uri))
    if m:
        return m.group(1).lower()
    m_s3 = re.search(r"/(dr\d+|qr\d+)/", str(uri), re.IGNORECASE)
    return m_s3.group(1).lower() if m_s3 else "?"


def dedupe_rows(rows):
    best = {}
    for row in rows:
        key = (row[2], round(row[1], 5))
        rank = (_RELEASE_RANK.get(release_of(row[0]), 0), row[0])
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    return sorted((v[1] for v in best.values()), key=lambda r: (r[1], r[0]))


def _canonical_irsa_uri(s3_key_or_uri: str, ra: float, dec: float, size_deg: float) -> str:
    """Формує канонічний IRSA IBE cutout URI із S3-ключа або існуючого посилання."""
    s = str(s3_key_or_uri).split("?")[0].strip()
    m = re.search(r"/spherex/(.+?\.fits)$", s, re.IGNORECASE)
    if m:
        s3_key = m.group(1)
    elif "amazonaws.com/" in s:
        s3_key = s.split("amazonaws.com/", 1)[1].lstrip("/")
    else:
        s3_key = s.lstrip("/")
        if s3_key.startswith("ibe/data/spherex/"):
            s3_key = s3_key[len("ibe/data/spherex/"):]
    return f"https://irsa.ipac.caltech.edu/ibe/data/spherex/{s3_key}?center={ra:.6f},{dec:.6f}d&size={size_deg:g}"


# --------------------------------------------------------------------------
# Локальне кешування таблиць каталогу
# --------------------------------------------------------------------------
def _cache_path(coord: SkyCoord, bands_list: Optional[List[str]]) -> Path:
    tag = _bands_tag(bands_list)
    name = f"ra{coord.ra.deg:.5f}_dec{coord.dec.deg:+.5f}_{tag}.ecsv"
    return CATALOG_CACHE_DIR / name


def _parse_ecsv_coords_and_tag(path: Path) -> Optional[Tuple[float, float, str]]:
    m = re.match(
        r"^ra([0-9.]+)_dec([+-]?[0-9.]+)_(?:s[0-9.]+_)?(.+)\.ecsv$",
        path.name,
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2)), m.group(3)
    except ValueError:
        return None


def _find_cached_table(coord: SkyCoord, bands_list: Optional[List[str]], max_sep_deg: float = 0.02) -> Optional[Table]:
    """
    Шукає кешовану таблицю в cache/catalog/:
      1. Точний або близький за координатами (< 0.02°) файл із потрібним тегом або ALL.
      2. Якщо запитано декілька смуг (напр. D1, D3, D5), перевіряє, чи можна зібрати їх із наявних кешів.
    """
    if not CATALOG_CACHE_DIR.exists():
        return None

    exact = _cache_path(coord, bands_list)
    if exact.exists():
        try:
            return Table.read(exact, format="ascii.ecsv")
        except Exception:
            pass

    wanted_tag = _bands_tag(bands_list)
    short_tag = bands_list[0].replace("SPHEREx-", "") if bands_list and len(bands_list) == 1 else None

    near_files: List[Tuple[float, str, Path]] = []
    for p in CATALOG_CACHE_DIR.glob("ra*_dec*.ecsv"):
        parsed = _parse_ecsv_coords_and_tag(p)
        if not parsed:
            continue
        ra_p, dec_p, tag_p = parsed
        c_p = SkyCoord(ra_p * u.deg, dec_p * u.deg, frame="icrs")
        sep = float(coord.separation(c_p).deg)
        if sep <= max_sep_deg:
            near_files.append((sep, tag_p, p))

    near_files.sort(key=lambda x: x[0])

    for _sep, tag_p, p in near_files:
        if tag_p in (wanted_tag, "ALL") or (short_tag and tag_p == short_tag):
            try:
                tbl = Table.read(p, format="ascii.ecsv")
                if bands_list and "band" in tbl.colnames:
                    mask = np.isin([str(b) for b in tbl["band"]], bands_list)
                    tbl = tbl[mask]
                if len(tbl) > 0:
                    return tbl
            except Exception:
                continue

    # Якщо користувач обрав декілька смуг і кожна з них уже є в окремих файлах кешу
    if bands_list and len(bands_list) > 1:
        collected_rows = []
        covered_bands = set()
        for _sep, _tag_p, p in near_files:
            try:
                tbl = Table.read(p, format="ascii.ecsv")
                for r in tbl:
                    b_str = str(r["band"])
                    if b_str in bands_list:
                        collected_rows.append((str(r["uri"]), float(r["mjd"]), b_str))
                        covered_bands.add(b_str)
            except Exception:
                continue
        if covered_bands == set(bands_list) and collected_rows:
            deduped = dedupe_rows(collected_rows)
            return Table(rows=deduped, names=("uri", "mjd", "band"))

    return None


def _apply_cutout_params(table: Table, coord: SkyCoord, size_deg: float) -> Table:
    ra, dec = coord.ra.deg, coord.dec.deg
    new_uris = [
        _canonical_irsa_uri(str(u_str), ra, dec, size_deg)
        for u_str in table["uri"]
    ]
    out = table.copy()
    out["uri"] = new_uris
    out.meta["size_deg"] = float(size_deg)
    return out


# --------------------------------------------------------------------------
# РІВЕНЬ 1: IRSA TAP/sync (швидкий синхронний POST через requests.Session)
# --------------------------------------------------------------------------
def _build_tap_artifact_query(coord: SkyCoord, size_deg: float, bands_list: Optional[List[str]]) -> str:
    ra, dec = coord.ra.deg, coord.dec.deg
    if not bands_list:
        band_clause = ""
    elif len(bands_list) == 1:
        band_clause = f"AND p.energy_bandpassname = '{bands_list[0]}'"
    else:
        quoted = ", ".join(f"'{b}'" for b in bands_list)
        band_clause = f"AND p.energy_bandpassname IN ({quoted})"

    return f"""
        SELECT
            'https://irsa.ipac.caltech.edu/' || a.uri
                || '?center={ra:.6f},{dec:.6f}d&size={size_deg:g}' AS uri,
            p.time_bounds_lower AS mjd,
            p.energy_bandpassname AS band
        FROM spherex.artifact a
        JOIN spherex.plane p ON a.planeid = p.planeid
        WHERE 1 = CONTAINS(POINT('ICRS', {ra:.6f}, {dec:.6f}), p.poly)
            {band_clause}
        ORDER BY p.time_bounds_lower
    """


def _query_tap_sync(query: str, timeout: Tuple[int, int] = (8, 22)) -> Table:
    """Виконує синхронний ADQL-запит через /TAP/sync за допомогою requests.Session."""
    session = get_http_session()
    resp = session.post(
        "https://irsa.ipac.caltech.edu/TAP/sync",
        data={
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "votable",
            "QUERY": query,
        },
        timeout=timeout,
    )
    if resp.status_code in (502, 503, 504):
        set_irsa_ibe_unavailable(True)
    resp.raise_for_status()

    raw_bytes = resp.content
    if b"ERROR" in raw_bytes[:1500] and b"QUERY_STATUS=\"ERROR\"" in raw_bytes[:1500]:
        raise RuntimeError("TAP/sync повернув QUERY_STATUS=ERROR")

    votable = parse(io.BytesIO(raw_bytes))
    tables = list(votable.iter_tables())
    if not tables:
        return Table(names=("uri", "mjd", "band"), dtype=("U400", "f8", "U32"))
    return tables[0].to_table()


def _level1_tap_search(
    coord: SkyCoord,
    size_deg: float,
    bands_list: Optional[List[str]],
    verbose: bool = True,
) -> Table:
    """Рівень 1: TAP/sync запит до spherex.artifact, при збої — коротка спроба через pyvo run_sync."""
    query = _build_tap_artifact_query(coord, size_deg, bands_list)
    try:
        return _query_tap_sync(query, timeout=(8, 22))
    except Exception as sync_exc:
        err_str = str(sync_exc)
        if "502" in err_str or "503" in err_str or "504" in err_str:
            set_irsa_ibe_unavailable(True)
            raise RuntimeError(f"IRSA TAP/sync повернув помилку проксі: {sync_exc}") from sync_exc

        try:
            import pyvo
            service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP", session=get_http_session())
            return service.run_sync(query).to_table()
        except Exception as pyvo_exc:
            if "502" in str(pyvo_exc) or "503" in str(pyvo_exc):
                set_irsa_ibe_unavailable(True)
            raise RuntimeError(f"IRSA TAP недоступний: {pyvo_exc}") from pyvo_exc


# --------------------------------------------------------------------------
# РІВЕНЬ 2: IRSA SIAv2 + паралельний резолвінг DataLink (логіка O_fetch_2.py)
# --------------------------------------------------------------------------
def _load_datalink_cache() -> Dict[str, str]:
    if DATALINK_CACHE_FILE.exists():
        try:
            return json.loads(DATALINK_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_datalink_cache(mapping: Dict[str, str]) -> None:
    try:
        DATALINK_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATALINK_CACHE_FILE.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_s3_key_from_votable_bytes(raw_xml: bytes) -> Optional[str]:
    """Витягує точний S3-ключ (напр. dr1/level2/.../file.fits) із VOTable DataLink."""
    try:
        votable = parse(io.BytesIO(raw_xml))
        tables = list(votable.iter_tables())
        if not tables:
            return None
        vo_table = tables[0]
        tbl = vo_table.to_table()
        fmap = {fld.name.lower(): tbl.colnames[i] for i, fld in enumerate(vo_table.fields) if fld.name}

        for row in tbl:
            for col in tbl.colnames:
                val = str(row[col]).strip()
                if val.startswith("{") and "nasa-irsa-spherex" in val:
                    try:
                        aws = json.loads(val).get("aws", {})
                        if aws.get("key"):
                            return str(aws["key"]).lstrip("/")
                    except Exception:
                        pass

        for row in tbl:
            sem = str(row[fmap["semantics"]]).strip() if "semantics" in fmap else ""
            if sem == "#this" and "access_url" in fmap:
                acc = str(row[fmap["access_url"]]).strip()
                m = re.search(r"/spherex/(.+?\.fits)$", acc, re.IGNORECASE)
                if m:
                    return m.group(1)
            elif sem == "#cutout" and "path" in fmap:
                p_val = str(row[fmap["path"]]).strip().lstrip("/")
                if p_val.startswith("spherex/"):
                    p_val = p_val[len("spherex/"):]
                if p_val.endswith(".fits"):
                    return p_val
    except Exception:
        pass
    return None


def _resolve_single_datalink(datalink_url: str, timeout: Tuple[int, int] = (6, 14)) -> Optional[str]:
    session = get_http_session()
    try:
        resp = session.get(datalink_url, timeout=timeout)
        if resp.status_code in (502, 503, 504):
            set_irsa_ibe_unavailable(True)
            return None
        if resp.status_code != 200:
            return None
        return _extract_s3_key_from_votable_bytes(resp.content)
    except Exception:
        return None


def _level2_siav2_search(
    coord: SkyCoord,
    size_deg: float,
    bands_list: Optional[List[str]],
    verbose: bool = True,
) -> Table:
    """
    Рівень 2: Швидкий пошук через мікросервіс IRSA SIAv2 + паралельний резолвінг DataLink
    через requests.Session (як у O_fetch_2.py).
    """
    ra, dec = coord.ra.deg, coord.dec.deg
    radius_deg = max(float(size_deg) * 0.75, 0.15)
    session = get_http_session()

    raw_xml = None
    for coll in ("SPHEREX", "spherex_qr2", "spherex"):
        try:
            resp = session.get(
                "https://irsa.ipac.caltech.edu/SIA",
                params={
                    "COLLECTION": coll,
                    "POS": f"circle {ra:.6f} {dec:.6f} {radius_deg:.4f}",
                    "RESPONSEFORMAT": "votable",
                },
                timeout=(8, 18),
            )
            if resp.status_code in (502, 503, 504):
                set_irsa_ibe_unavailable(True)
                continue
            if resp.status_code == 200 and b"<TABLEDATA" in resp.content:
                raw_xml = resp.content
                break
        except Exception:
            continue

    if raw_xml is None:
        raise RuntimeError("Мікросервіс IRSA SIAv2 не відповів або повернув порожній результат.")

    votable = parse(io.BytesIO(raw_xml))
    tables = list(votable.iter_tables())
    empty = Table(names=("uri", "mjd", "band"), dtype=("U400", "f8", "U32"))
    if not tables:
        return empty

    tbl = tables[0].to_table()
    if len(tbl) == 0:
        return empty

    dl_cache = _load_datalink_cache()
    candidates = []

    for r in tbl:
        try:
            mjd_val = float(r["t_min"]) if "t_min" in tbl.colnames else float("nan")
        except Exception:
            mjd_val = float("nan")
        if not np.isfinite(mjd_val):
            continue

        obs_id = str(r["obs_id"]).strip() if "obs_id" in tbl.colnames else ""
        access_url = str(r["access_url"]).strip() if "access_url" in tbl.colnames else ""
        band_col_val = str(r["energy_bandpassname"]).strip() if "energy_bandpassname" in tbl.colnames else ""

        band_name = None
        if band_col_val and band_col_val != "--":
            try:
                band_name = normalize_band(band_col_val)
            except ValueError:
                band_name = None
        if not band_name:
            band_name = _infer_band_from_text(obs_id) or _infer_band_from_text(access_url)
        if not band_name and "em_min" in tbl.colnames:
            try:
                em_min_val = float(r["em_min"])
                band_name = _infer_band_from_wavelength_um(em_min_val * 1e6 if em_min_val < 0.01 else em_min_val)
            except Exception:
                band_name = None

        if bands_list and band_name and band_name not in bands_list:
            continue

        s3_key = None
        for col in tbl.colnames:
            val = str(r[col]).strip()
            if val.startswith("{") and "nasa-irsa-spherex" in val:
                try:
                    aws = json.loads(val).get("aws", {})
                    if aws.get("key"):
                        s3_key = str(aws["key"]).lstrip("/")
                        break
                except Exception:
                    pass

        if not s3_key:
            m_ibe = re.search(r"/spherex/(.+?\.fits)", access_url, re.IGNORECASE)
            if m_ibe:
                s3_key = m_ibe.group(1)
            elif access_url in dl_cache:
                s3_key = dl_cache[access_url]

        candidates.append({
            "mjd": mjd_val,
            "band": band_name,
            "access_url": access_url,
            "s3_key": s3_key,
        })

    unresolved = [c for c in candidates if not c["s3_key"] and c["access_url"]]
    if unresolved:
        # Швидка перевірка перших 2 посилань DataLink, щоб не чекати при повному падінні DataLink
        probe_batch = unresolved[:2]
        probe_ok = False
        for item in probe_batch:
            key = _resolve_single_datalink(item["access_url"])
            if key:
                item["s3_key"] = key
                dl_cache[item["access_url"]] = key
                probe_ok = True

        remaining = [c for c in unresolved[2:] if not c["s3_key"]]
        if probe_ok and remaining:
            with ThreadPoolExecutor(max_workers=24) as pool:
                fut_map = {pool.submit(_resolve_single_datalink, c["access_url"]): c for c in remaining}
                for fut in as_completed(fut_map):
                    item = fut_map[fut]
                    key = fut.result()
                    if key:
                        item["s3_key"] = key
                        dl_cache[item["access_url"]] = key

        if dl_cache:
            _save_datalink_cache(dl_cache)

    rows = []
    for c in candidates:
        s3_key = c["s3_key"]
        if not s3_key:
            continue
        band_final = _infer_band_from_text(s3_key) or c["band"] or DEFAULT_BAND
        if bands_list and band_final not in bands_list:
            continue
        uri = _canonical_irsa_uri(s3_key, ra, dec, size_deg)
        rows.append((uri, float(c["mjd"]), band_final))

    if not rows:
        raise RuntimeError("SIAv2 не зміг отримати ключі FITS-файлів (DataLink недоступний).")

    return Table(rows=rows, names=("uri", "mjd", "band"))


# --------------------------------------------------------------------------
# РІВЕНЬ 3: Прямий пошук на AWS S3 + локальні кеші (логіка O_s3_bypass_2.py)
# --------------------------------------------------------------------------
def _harvest_all_local_records() -> List[dict]:
    """Збирає всі відомі записи SPHEREx із локальних кешів проєкту, s3_catalog.pkl та кешу astropy."""
    records: List[dict] = []

    if S3_INDEX_CACHE_FILE.exists():
        try:
            records.extend(json.loads(S3_INDEX_CACHE_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass

    # 1. Існуючі .ecsv таблиці в cache/catalog/
    if CATALOG_CACHE_DIR.exists():
        for p in CATALOG_CACHE_DIR.glob("*.ecsv"):
            try:
                tbl = Table.read(p, format="ascii.ecsv")
                ra_m = float(tbl.meta.get("ra_deg", np.nan))
                dec_m = float(tbl.meta.get("dec_deg", np.nan))
                if not np.isfinite(ra_m) or not np.isfinite(dec_m):
                    parsed = _parse_ecsv_coords_and_tag(p)
                    if parsed:
                        ra_m, dec_m, _ = parsed
                for r in tbl:
                    u_str = str(r["uri"])
                    m_key = re.search(r"/spherex/(.+?\.fits)", u_str, re.IGNORECASE)
                    if not m_key:
                        continue
                    s3_key = m_key.group(1)
                    b_str = str(r["band"]) if "band" in tbl.colnames else (
                        _infer_band_from_text(s3_key) or DEFAULT_BAND
                    )
                    records.append({
                        "s3_key": s3_key,
                        "s_ra": ra_m,
                        "s_dec": dec_m,
                        "t_min": float(r["mjd"]),
                        "band": b_str,
                        "exact_target": True,
                    })
            except Exception:
                continue

    # 2. Кеш сокомандника spherex_cache/s3_catalog.pkl (якщо є)
    for pkl_path in (ROOT / "spherex_cache" / "s3_catalog.pkl", Path("./spherex_cache/s3_catalog.pkl")):
        if pkl_path.exists():
            try:
                import pandas as pd
                df = pd.read_pickle(pkl_path)
                for _, row in df.iterrows():
                    s3_url = str(row.get("s3_url", ""))
                    if "amazonaws.com/" in s3_url:
                        s3_key = s3_url.split("amazonaws.com/", 1)[1].lstrip("/")
                        records.append({
                            "s3_key": s3_key,
                            "s_ra": float(row.get("s_ra", np.nan)),
                            "s_dec": float(row.get("s_dec", np.nan)),
                            "t_min": float(row.get("t_min", np.nan)),
                            "band": _infer_band_from_text(s3_key) or DEFAULT_BAND,
                            "exact_target": bool(row.get("exact_hit", False)),
                        })
            except Exception:
                pass

    # 3. Кеш завантажень astropy (DataLink VOTable)
    try:
        astropy_dl_dir = Path(get_cache_dir()) / "download" / "url"
        if astropy_dl_dir.exists():
            for entry in astropy_dl_dir.iterdir():
                contents = entry / "contents"
                if not contents.is_file():
                    continue
                try:
                    raw_b = contents.read_bytes()
                    if b"VOTABLE" not in raw_b[:400] or b"nasa-irsa-spherex" not in raw_b:
                        continue
                    s3_key = _extract_s3_key_from_votable_bytes(raw_b)
                    if s3_key:
                        records.append({
                            "s3_key": s3_key,
                            "s_ra": np.nan,
                            "s_dec": np.nan,
                            "t_min": np.nan,
                            "band": _infer_band_from_text(s3_key) or DEFAULT_BAND,
                            "exact_target": False,
                        })
                except Exception:
                    continue
    except Exception:
        pass

    return records


def read_s3_fits_header_fast(s3_url: str, with_wcs: bool = False) -> Optional[dict]:
    """Читає перші 64 КБ FITS-файлу з AWS S3 (гарантовано захоплює повний заголовок і WCS)."""
    session = get_http_session()
    try:
        resp = session.get(s3_url, headers={"Range": "bytes=0-65535"}, timeout=(6, 15))
        if resp.status_code not in (200, 206) or len(resp.content) < 5760:
            return None

        text = resp.content.decode("ascii", errors="ignore")
        first_end = text.find("END     ")
        if first_end == -1:
            return None

        hdu1_offset = ((first_end + 80 + 2879) // 2880) * 2880
        sub_text = text[hdu1_offset:]
        second_end = sub_text.find("END     ")
        if second_end == -1:
            sub_text = text[:first_end + 80]
        else:
            sub_text = sub_text[:second_end + 80]

        hdr = fits.Header.fromstring(sub_text)
        t_min = float(hdr.get("MJD-AVG", hdr.get("MJD-OBS", np.nan)))
        ra_val = float(hdr.get("CRVAL1", np.nan))
        dec_val = float(hdr.get("CRVAL2", np.nan))
        if not (np.isfinite(t_min) and np.isfinite(ra_val) and np.isfinite(dec_val)):
            return None

        s3_key = s3_url.split("amazonaws.com/", 1)[-1].lstrip("/")
        det = hdr.get("DETECTOR")
        band_str = f"SPHEREx-D{int(det)}" if det is not None else (_infer_band_from_text(s3_key) or DEFAULT_BAND)
        meta = {
            "s3_key": s3_key,
            "s_ra": ra_val,
            "s_dec": dec_val,
            "t_min": t_min,
            "band": band_str,
            "s3_url": s3_url,
        }
        if with_wcs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta["wcs"] = WCS(hdr, naxis=2)
        return meta
    except Exception:
        return None


def _discover_s3_week_prefixes(known_records: List[dict]) -> List[str]:
    """Знаходить тижневі директорії в AWS S3 через requests.Session, залишаючи найсвіжішу ревізію тижня."""
    candidates = ["dr1/level2/", "qr3/level2/", "qr2/level2/", "qr1/level2/", "level2/", ""]
    for rec in known_records:
        key = str(rec.get("s3_key", ""))
        m = re.match(r"^(.+?)20\d{2}W\d{2}", key)
        if m:
            candidates.insert(0, m.group(1))

    session = get_http_session()
    latest_by_week: Dict[str, str] = {}

    for pref in dict.fromkeys(candidates):
        try:
            resp = session.get(
                f"{S3_BUCKET_URL}/",
                params={"list-type": "2", "prefix": pref, "delimiter": "/"},
                timeout=(6, 12),
            )
            if resp.status_code != 200:
                continue
            xml_root = ET.fromstring(resp.content)
            sub_prefixes = [
                el.text for el in xml_root.findall(".//s3:CommonPrefixes/s3:Prefix", S3_NS) if el.text
            ]
            weeks = [p for p in sub_prefixes if re.search(r"20\d{2}W\d{2}", p)]
            for w in sorted(weeks):
                m = re.search(r"(20\d{2}W\d{2})", w)
                if m:
                    w_code = m.group(1)
                    old = latest_by_week.get(w_code, "")
                    if not old or _RELEASE_RANK.get(release_of(f"/spherex/{w}"), 0) >= _RELEASE_RANK.get(release_of(f"/spherex/{old}"), 0):
                        latest_by_week[w_code] = w
            if latest_by_week:
                break
        except Exception:
            continue

    return sorted(latest_by_week.values())


def _list_fits_in_prefix(prefix: str, max_pages: int = 40) -> List[str]:
    session = get_http_session()
    keys: List[str] = []
    token: Optional[str] = None
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = session.get(f"{S3_BUCKET_URL}/", params=params, timeout=(8, 18))
        if resp.status_code != 200:
            break
        w_xml = ET.fromstring(resp.content)
        for el in w_xml.findall(".//s3:Contents/s3:Key", S3_NS):
            if el.text and el.text.endswith(".fits"):
                keys.append(el.text)
        is_trunc = w_xml.findtext(".//s3:IsTruncated", default="false", namespaces=S3_NS)
        token = w_xml.findtext(".//s3:NextContinuationToken", default=None, namespaces=S3_NS)
        if is_trunc.lower() != "true" or not token:
            break
    return sorted(keys)


def _level3_s3_direct_search(
    coord: SkyCoord,
    size_deg: float,
    bands_list: Optional[List[str]],
    verbose: bool = True,
) -> Table:
    """
    Рівень 3: Пошук безпосередньо в AWS S3 (nasa-irsa-spherex) та локальних індексах,
    коли сервери IRSA (TAP і SIAv2) недоступні (502/503).
    """
    set_irsa_ibe_unavailable(True)
    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    wanted_bands = set(bands_list) if bands_list else set(BANDS)

    records = _harvest_all_local_records()
    matched_Near: List[dict] = []
    known_target_weeks: List[str] = []

    for rec in records:
        r_ra, r_dec = rec.get("s_ra", np.nan), rec.get("s_dec", np.nan)
        t_min = rec.get("t_min", np.nan)
        if not np.isfinite(r_ra) or not np.isfinite(r_dec) or not np.isfinite(t_min):
            continue
        sep = float(coord.separation(SkyCoord(r_ra * u.deg, r_dec * u.deg, frame="icrs")).deg)
        if sep <= 2.5:
            matched_Near.append(rec)
            m_wk = re.match(r"^(.+?20\d{2}W\d{2}[^/]*/)", str(rec.get("s3_key", "")))
            if m_wk:
                known_target_weeks.append(m_wk.group(1))

    found_bands = {rec["band"] for rec in matched_Near if rec.get("band") in wanted_bands}
    if found_bands != wanted_bands:
        if verbose:
            print(f"  [catalog] сканування експозицій на AWS S3 для RA={ra:.4f}, Dec={dec:+.4f}...")
        week_prefixes = _discover_s3_week_prefixes(records)
        ecl = coord.geocentrictrueecliptic
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
        target_weeks = list(dict.fromkeys(
            known_target_weeks + [wp for _, wp in scored_ep1[:6]] + [wp for _, wp in scored_ep2[:6]]
        ))

        dense_exposures: List[List[str]] = []
        for wp in target_weeks:
            try:
                all_keys = _list_fits_in_prefix(wp, max_pages=60)
            except Exception:
                continue
            if not all_keys:
                continue

            groups: Dict[str, List[str]] = {}
            for k in all_keys:
                base = re.sub(r"D[1-6](?:_spx|\.).*$", "", k)
                groups.setdefault(base, []).append(f"{S3_BUCKET_URL}/{k}")

            exp_list = list(groups.values())
            if not exp_list:
                continue

            coarse_indices = list(range(0, len(exp_list), 8))
            coarse_map = {
                next((u_str for u_str in exp_list[i] if "D1" in Path(u_str).name or "D2" in Path(u_str).name), exp_list[i][0]): i
                for i in coarse_indices
            }

            hit_indices = []
            with ThreadPoolExecutor(max_workers=32) as pool:
                futures = {pool.submit(read_s3_fits_header_fast, u_str, False): idx for u_str, idx in coarse_map.items()}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    meta = fut.result()
                    if meta and np.isfinite(meta["s_ra"]) and np.isfinite(meta["s_dec"]):
                        sep = float(coord.separation(
                            SkyCoord(meta["s_ra"] * u.deg, meta["s_dec"] * u.deg, frame="icrs")
                        ).deg)
                        if sep <= 15.0:
                            hit_indices.append(idx)

            selected_idx_set = set()
            for h_idx in hit_indices:
                for j in range(max(0, h_idx - 85), min(len(exp_list), h_idx + 86)):
                    selected_idx_set.add(j)

            if not selected_idx_set:
                selected_idx_set.update(range(0, len(exp_list), 4))

            for j in sorted(selected_idx_set):
                dense_exposures.append(exp_list[j])

        need_short = bool(wanted_bands & {"SPHEREx-D1", "SPHEREx-D2", "SPHEREx-D3"})
        need_long = bool(wanted_bands & {"SPHEREx-D4", "SPHEREx-D5", "SPHEREx-D6"})

        probe_to_group: Dict[str, List[str]] = {}
        for urls in dense_exposures:
            short_urls = [u_str for u_str in urls if re.search(r"D[1-3](?:_spx|\.)", Path(u_str).name)]
            long_urls = [u_str for u_str in urls if re.search(r"D[4-6](?:_spx|\.)", Path(u_str).name)]
            if need_short and short_urls:
                probe_to_group[short_urls[0]] = short_urls
            if need_long and long_urls:
                probe_to_group[long_urls[0]] = long_urls

        new_records: List[dict] = []
        margin_px = 24
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {pool.submit(read_s3_fits_header_fast, p_url, True): p_url for p_url in probe_to_group}
            for fut in as_completed(futures):
                p_url = futures[fut]
                meta = fut.result()
                if not meta:
                    continue
                try:
                    is_hit = False
                    if "wcs" in meta:
                        px, py = meta["wcs"].world_to_pixel(coord)
                        px, py = float(px), float(py)
                        if np.isfinite(px) and np.isfinite(py) and (margin_px <= px <= 2048 - margin_px) and (margin_px <= py <= 2048 - margin_px):
                            is_hit = True
                    if not is_hit:
                        sep = float(coord.separation(
                            SkyCoord(meta["s_ra"] * u.deg, meta["s_dec"] * u.deg, frame="icrs")
                        ).deg)
                        if sep <= 2.3:
                            is_hit = True

                    if is_hit:
                        for sib_url in probe_to_group[p_url]:
                            s3_key = sib_url.split("amazonaws.com/", 1)[-1].lstrip("/")
                            rec_item = {
                                "s3_key": s3_key,
                                "s_ra": ra,
                                "s_dec": dec,
                                "t_min": float(meta["t_min"]),
                                "band": _infer_band_from_text(s3_key) or meta["band"],
                                "exact_target": True,
                            }
                            new_records.append(rec_item)
                            matched_Near.append(rec_item)
                except Exception:
                    continue

        if new_records:
            try:
                S3_INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                all_saved = records + new_records
                S3_INDEX_CACHE_FILE.write_text(json.dumps(all_saved[-5000:], indent=2), encoding="utf-8")
            except Exception:
                pass

    rows = []
    for rec in matched_Near:
        b_name = rec.get("band") or _infer_band_from_text(rec.get("s3_key", "")) or DEFAULT_BAND
        if b_name not in wanted_bands:
            continue
        uri = _canonical_irsa_uri(rec["s3_key"], ra, dec, size_deg)
        rows.append((uri, float(rec["t_min"]), b_name))

    if not rows:
        return Table(names=("uri", "mjd", "band"), dtype=("U400", "f8", "U32"))
    return Table(rows=rows, names=("uri", "mjd", "band"))


# --------------------------------------------------------------------------
# Головна точка входу пошуку: find_exposures
# --------------------------------------------------------------------------
def find_exposures(
    target: Union[str, SkyCoord],
    size_deg: float = 0.1,
    band: Optional[Union[str, Sequence[str]]] = DEFAULT_BAND,
    use_cache: bool = True,
    verbose: bool = True,
    bands: Optional[Union[str, Sequence[str]]] = None,
) -> Table:
    """
    Шукає знімки SPHEREx для заданої цілі та набору смуг детекторів:
      Кеш -> Рівень 1 (TAP/sync) -> Рівень 2 (SIAv2 + DataLink) -> Рівень 3 (AWS S3 напряму).
    """
    coord = resolve_target(target)
    raw_band_input = bands if bands is not None else band
    selected_bands = normalize_bands(raw_band_input)
    tag = _bands_tag(selected_bands)
    size_deg = float(size_deg)
    if not (MIN_SIZE_DEG <= size_deg <= MAX_SIZE_DEG):
        raise ValueError(
            f"Розмір вирізки {size_deg}° поза межами {MIN_SIZE_DEG}..{MAX_SIZE_DEG}°."
        )

    if use_cache:
        cached_tbl = _find_cached_table(coord, selected_bands)
        if cached_tbl is not None and len(cached_tbl) > 0:
            cached_tbl = _apply_cutout_params(cached_tbl, coord, size_deg)
            if verbose:
                print(f"  [catalog] взято з кешу ({len(cached_tbl)} знімків, смуги={tag})")
            return cached_tbl

    if verbose:
        label = ", ".join(selected_bands) if selected_bands else "усі"
        print(f"  [catalog] пошук знімків (RA={coord.ra.deg:.5f}, Dec={coord.dec.deg:+.5f}, "
              f"розмір={size_deg}°, смуга={label})...")

    started = time.time()
    raw: Optional[Table] = None

    # Рівень 1: IRSA TAP/sync
    try:
        raw = _level1_tap_search(coord, size_deg, selected_bands, verbose=verbose)
    except Exception as tap_exc:
        if verbose:
            print(f"  [catalog] Рівень 1 (TAP) недоступний ({tap_exc}) -> перехід на Рівень 2 (SIAv2)...")

    # Рівень 2: IRSA SIAv2 + паралельний резолвінг DataLink
    if raw is None or len(raw) == 0:
        try:
            raw = _level2_siav2_search(coord, size_deg, selected_bands, verbose=verbose)
        except Exception as sia_exc:
            if verbose:
                print(f"  [catalog] Рівень 2 (SIAv2) недоступний ({sia_exc}) -> перехід на Рівень 3 (AWS S3)...")

    # Рівень 3: Прямий пошук на AWS S3 + локальні кеші
    if raw is None or len(raw) == 0:
        raw = _level3_s3_direct_search(coord, size_deg, selected_bands, verbose=verbose)

    empty = Table(names=("uri", "mjd", "band"), dtype=("U400", "f8", "U32"))
    if raw is None or len(raw) == 0:
        if verbose:
            print("  [catalog] знімків не знайдено (спробуй інші смуги або більший розмір)")
        return empty

    time_col = "mjd" if "mjd" in raw.colnames else "time_bounds_lower"
    band_col = "band" if "band" in raw.colnames else "energy_bandpassname"

    uris = [
        _canonical_irsa_uri(str(x), float(coord.ra.deg), float(coord.dec.deg), size_deg)
        for x in raw["uri"]
    ]
    mjd = np.asarray(np.ma.filled(raw[time_col], np.nan), dtype=float)
    if band_col in raw.colnames:
        band_vals = [str(b) for b in raw[band_col]]
    else:
        default_b = selected_bands[0] if selected_bands and len(selected_bands) == 1 else DEFAULT_BAND
        band_vals = [_infer_band_from_text(u_str) or default_b for u_str in uris]

    order = sorted(range(len(uris)), key=lambda i: (mjd[i], uris[i]))
    seen, rows = set(), []
    for i in order:
        if uris[i] in seen:
            continue
        if selected_bands and band_vals[i] not in selected_bands:
            continue
        seen.add(uris[i])
        rows.append((uris[i], float(mjd[i]), band_vals[i]))

    before = len(rows)
    rows = dedupe_rows(rows)
    if len(rows) < before and verbose:
        print(f"  [catalog] прибрано дублікатів (різні версії одного знімка): {before - len(rows)}")

    table = Table(rows=rows, names=("uri", "mjd", "band"))
    table.meta.update({
        "ra_deg": float(coord.ra.deg),
        "dec_deg": float(coord.dec.deg),
        "size_deg": size_deg,
        "band": tag,
        "bands": selected_bands or BANDS,
    })

    save_path = _cache_path(coord, selected_bands)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    table.write(save_path, format="ascii.ecsv", overwrite=True)
    if verbose:
        print(f"  [catalog] знайдено {len(table)} знімків за {time.time() - started:.1f} с, "
              f"збережено в кеш: {save_path.name}")
    return table