"""
Самоперевірка кроків 1-2. Кожна перевірка пише [OK] або [ПОМИЛКА].

    python check.py                       усі перевірки (потрібен інтернет)
    python check.py --offline             лише ті, що не потребують мережі
    python check.py --sample шлях.fits    додатково перевірити читання повного FITS-файлу
"""
import argparse
import sys
import time
from pathlib import Path

PASSED, FAILED = [], []


def section(title):
    print(f"\n=== {title} ===")


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    mark = "OK" if condition else "ПОМИЛКА"
    print(f"  [{mark}] {name}" + (f"  ->  {detail}" if detail else ""))
    return bool(condition)


def raises(exc_type, func, *args):
    try:
        func(*args)
    except exc_type:
        return True
    except Exception:
        return False
    return False


# --------------------------------------------------------------------------
def test_environment():
    section("0. Середовище")
    print("  Python:", sys.executable)
    for mod in ["numpy", "astropy", "astroquery", "requests"]:
        try:
            m = __import__(mod)
            check(f"пакет {mod}", True, getattr(m, "__version__", ""))
        except Exception as exc:
            check(f"пакет {mod}", False, f"не встановлено ({exc})")
    for mod in ["reproject", "PIL", "cv2"]:   # потрібні лише на наступних кроках
        try:
            __import__(mod)
            print(f"  (для наступних кроків) {mod}: є")
        except Exception:
            print(f"  (для наступних кроків) {mod}: ще не встановлено")


def test_offline_logic():
    section("1. Логіка без мережі")
    import numpy as np
    from astropy.table import Table
    from pipeline.catalog import dedupe_rows, normalize_band, normalize_bands, release_of, resolve_target
    from pipeline.cutouts import WEB_LIMIT_OPTIONS, select_balanced_indices

    check("normalize_band('d2')", normalize_band("d2") == "SPHEREx-D2")
    check("normalize_band('2')", normalize_band("2") == "SPHEREx-D2")
    check("normalize_band('SPHEREx-D5')", normalize_band("SPHEREx-D5") == "SPHEREx-D5")
    check("normalize_band('all') -> None", normalize_band("all") is None)
    check("normalize_band('D9') -> помилка", raises(ValueError, normalize_band, "D9"))

    check("normalize_bands('D5, d1, D3') -> відсортований список",
          normalize_bands("D5, d1, D3") == ["SPHEREx-D1", "SPHEREx-D3", "SPHEREx-D5"])
    check("normalize_bands(['D2', 'D4'])",
          normalize_bands(["D2", "D4"]) == ["SPHEREx-D2", "SPHEREx-D4"])
    check("normalize_bands('D1,D2,D3,D4,D5,D6') -> None (усі)",
          normalize_bands("D1,D2,D3,D4,D5,D6") is None)
    check("normalize_bands('all') -> None", normalize_bands("all") is None)
    check("normalize_bands('D1,D9') -> помилка", raises(ValueError, normalize_bands, "D1,D9"))

    c = resolve_target("210.80227 54.34895")
    check("координати 'RA Dec' розпізнано",
          abs(c.ra.deg - 210.80227) < 1e-6 and abs(c.dec.deg - 54.34895) < 1e-6)
    c = resolve_target("210.80227, 54.34895")
    check("координати через кому розпізнано", abs(c.ra.deg - 210.80227) < 1e-6)
    check("RA=400 -> помилка", raises(ValueError, resolve_target, "400 10"))
    check("порожній рядок -> помилка", raises(ValueError, resolve_target, ""))

    base = "https://irsa.ipac.caltech.edu/ibe/data/spherex/{}/level2/x.fits"
    check("release_of: qr3 / dr1 / невідомий",
          release_of(base.format("qr3")) == "qr3" and release_of(base.format("dr1")) == "dr1"
          and release_of("https://example.com/file.fits") == "?")
    rows = [(base.format("qr3").replace("x.fits", "a_v27.fits"), 60805.71221, "SPHEREx-D2"),
            (base.format("dr1").replace("x.fits", "a_v30.fits"), 60805.71221, "SPHEREx-D2"),
            (base.format("qr3").replace("x.fits", "b.fits"), 60806.0, "SPHEREx-D2"),
            (base.format("qr3").replace("x.fits", "a_v27.fits"), 60805.71221, "SPHEREx-D3")]
    out = dedupe_rows(rows)
    d2 = [r for r in out if r[2] == "SPHEREx-D2" and abs(r[1] - 60805.71221) < 1e-6]
    check("dedupe_rows: одна версія на (смуга, час), новіший реліз виграє",
          len(out) == 3 and len(d2) == 1 and "/dr1/" in d2[0][0], f"{len(out)} рядки; обрано {release_of(d2[0][0])}")

    # Перевірка чесного 2D Stratified Round-Robin відбору для пресетів сайту (30, 60, 100, 180)
    synth_rows = []
    for ep_idx, ep_mjd in enumerate([60800.0, 60980.0]):
        for b_name in ("SPHEREx-D1", "SPHEREx-D3", "SPHEREx-D5"):
            for k in range(40):
                synth_rows.append((f"https://example.com/{ep_idx}_{b_name}_{k}.fits", ep_mjd + 0.1 * k, b_name))
    synth_tbl = Table(rows=synth_rows, names=("uri", "mjd", "band"))
    for lim in WEB_LIMIT_OPTIONS:
        idx_sel = select_balanced_indices(synth_tbl, limit=lim)
        sub_tbl = synth_tbl[idx_sel]
        counts_by_band = [int(np.sum(sub_tbl["band"] == b)) for b in ("SPHEREx-D1", "SPHEREx-D3", "SPHEREx-D5")]
        mjd_sel = np.asarray(sub_tbl["mjd"], dtype=float)
        ok_lim = (
            len(idx_sel) == lim
            and (max(counts_by_band) - min(counts_by_band) <= 1)
            and bool(np.all(np.diff(mjd_sel) >= 0))
        )
        check(f"2D Round-Robin limit={lim}: рівномірно по смугах і строго за часом",
              ok_lim, f"розподіл D1/D3/D5={counts_by_band}")


def test_sample_file(path):
    section(f"2. Читання повного FITS: {Path(path).name}")
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from pipeline.cutouts import read_frame

    with fits.open(path, memmap=False) as hdul:
        names = [h.name for h in hdul]
        header = hdul["IMAGE"].header
        wcs = WCS(header)
        crval, crpix = wcs.wcs.crval, wcs.wcs.crpix
    check("є розширення IMAGE, FLAGS, VARIANCE", all(n in names for n in ["IMAGE", "FLAGS", "VARIANCE"]), str(names))

    coord = SkyCoord(crval[0] * u.deg, crval[1] * u.deg)   # опорна точка самого файлу
    frame = read_frame(path, coord)
    if not check("read_frame повернув кадр", frame is not None):
        return
    check("image і flags одного розміру", frame["image"].shape == frame["flags"].shape,
          str(frame["image"].shape))
    check("піксель об'єкта збігається з CRPIX (WCS працює)",
          abs(frame["x"] - (crpix[0] - 1)) < 1.0 and abs(frame["y"] - (crpix[1] - 1)) < 1.0,
          f"x={frame['x']:.2f}, y={frame['y']:.2f}, очікувалось ~({crpix[0]-1:.1f}, {crpix[1]-1:.1f})")
    check("MJD правдоподібний (2025+)", 60600 < frame["mjd"] < 70000, f"{frame['mjd']}")
    wl = frame["wavelength_um"]
    check("довжина хвилі в межах 0.7..5.1 мкм", wl == wl and 0.7 < wl < 5.1, f"{wl:.4f} мкм")
    finite = (frame["image"] == frame["image"]).mean()
    check("більшість пікселів скінченні", finite > 0.5, f"{finite:.0%}")


def test_network():
    section("3. Мережа та IRSA")
    import requests
    try:
        r = requests.get("https://irsa.ipac.caltech.edu", timeout=20)
        check("irsa.ipac.caltech.edu доступний", r.status_code < 500, f"HTTP {r.status_code}")
    except Exception as exc:
        check("irsa.ipac.caltech.edu доступний", False, f"{type(exc).__name__}")
        print("  Підказка: спробуй з телефонного хотспоту; вимкни VPN; перевір антивірус.")
        return False

    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from pipeline.catalog import resolve_target
    try:
        c = resolve_target("M101")
        sep = c.separation(SkyCoord(210.80227 * u.deg, 54.34895 * u.deg)).deg
        check("назва 'M101' -> координати", sep < 0.05, f"RA={c.ra.deg:.4f}, Dec={c.dec.deg:.4f}")
    except Exception as exc:
        check("назва 'M101' -> координати", False, str(exc))
    return True


def test_catalog_and_cutouts():
    section("4. Крок 1: каталог знімків (M101, D2)")
    import numpy as np
    from pipeline.catalog import find_exposures

    t0 = time.time()
    table = find_exposures("210.80227 54.34895", size_deg=0.1, band="D2", use_cache=False, verbose=False)
    first = time.time() - t0
    check("знайдено знімки", len(table) >= 10, f"{len(table)} шт., {first:.1f} с")
    if len(table) == 0:
        return

    check("колонки uri, mjd, band", all(c in table.colnames for c in ("uri", "mjd", "band")))
    check("посилання ведуть на IRSA і містять center=",
          all(str(u_).startswith("https://irsa.ipac.caltech.edu/") and "center=" in str(u_) for u_ in table["uri"]))
    mjd = np.array(table["mjd"], dtype=float)
    check("час відсортований за зростанням", bool(np.all(np.diff(mjd) >= 0)))
    check("усі знімки зі смуги D2", all(str(b) == "SPHEREx-D2" for b in table["band"]))
    check("посилання унікальні", len(set(map(str, table["uri"]))) == len(table))

    t0 = time.time()
    again = find_exposures("210.80227 54.34895", size_deg=0.1, band="D2", use_cache=True, verbose=False)
    second = time.time() - t0
    check("кеш: повторний запит швидкий і дає те саме",
          second < 2.0 and list(again["uri"]) == list(table["uri"]), f"{second:.2f} с")

    section("5. Крок 2: одна вирізка")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from pipeline.cutouts import fetch_cutout, is_valid_fits, read_frame

    coord = SkyCoord(210.80227 * u.deg, 54.34895 * u.deg)
    row = table[len(table) // 2]
    path = fetch_cutout(str(row["uri"]))
    if not check("вирізка завантажена", path is not None):
        print("  Підказка: якщо тут помилка, а крок 4 пройшов, проблема з мережею під час завантаження.")
        return
    size_mb = path.stat().st_size / 1e6
    check("розмір вирізки малий (< 10 МБ)", size_mb < 10, f"{size_mb:.2f} МБ")
    check("файл цілий (is_valid_fits)", is_valid_fits(path))

    t0 = time.time()
    path2 = fetch_cutout(str(row["uri"]))
    check("кеш вирізок: повторно не качається", path2 == path and time.time() - t0 < 1.0)

    frame = read_frame(path, coord)
    if not check("read_frame повернув кадр", frame is not None):
        return
    ny, nx = frame["image"].shape
    check("кадр двовимірний і не порожній", frame["image"].ndim == 2 and nx > 10 and ny > 10, f"{nx}x{ny}")
    check("об'єкт у центрі вирізки (WCS правильний)",
          abs(frame["x"] - (nx - 1) / 2) <= 3 and abs(frame["y"] - (ny - 1) / 2) <= 3,
          f"x={frame['x']:.1f}, y={frame['y']:.1f}, центр=({(nx-1)/2:.1f}, {(ny-1)/2:.1f})")
    wl = frame["wavelength_um"]
    check("довжина хвилі в межах D2 (приблизно 1.0..1.7 мкм)", wl == wl and 0.9 < wl < 1.8, f"{wl:.4f} мкм")
    check("час кадру збігається з таблицею (±0.1 доби)", abs(frame["mjd"] - float(row["mjd"])) < 0.1,
          f"кадр={frame['mjd']:.4f}, таблиця={float(row['mjd']):.4f}")
    if frame["detector"] is not None:
        check("детектор = 2", int(frame["detector"]) == 2, str(frame["detector"]))

    section("6. Крок 2: паралельне завантаження 4 кадрів")
    from pipeline.cutouts import frames_summary, load_frames
    frames = load_frames(table, coord, limit=4, workers=4, verbose=False)
    check("отримано щонайменше 3 з 4 кадрів", len(frames) >= 3, f"{len(frames)} з 4")
    if frames:
        mjds = [f["mjd"] for f in frames]
        check("кадри відсортовані за часом", mjds == sorted(mjds))
        check("кадри мають різні довжини хвиль",
              len({round(f["wavelength_um"], 3) for f in frames}) > 1 or len(frames) == 1)
        frames_summary(frames).pprint(max_lines=-1, max_width=-1)


# --------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="без мережевих перевірок")
    parser.add_argument("--sample", help="шлях до повного FITS-файлу SPHEREx для перевірки читання")
    args = parser.parse_args()

    stages = [test_environment, test_offline_logic]
    if args.sample:
        stages.append(lambda: test_sample_file(args.sample))
    if not args.offline:
        stages.append(lambda: test_network() and test_catalog_and_cutouts())

    for stage in stages:
        try:
            stage()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            check(f"етап завершився без винятків ({getattr(stage, '__name__', 'stage')})", False, repr(exc))

    print(f"\nПідсумок: успішно {len(PASSED)}, помилок {len(FAILED)}")
    for name in FAILED:
        print("  ПОМИЛКА:", name)
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()