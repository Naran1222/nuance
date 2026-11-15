"""
Перевірка кроку 3 (render.py). Кожна перевірка пише [OK] або [ПОМИЛКА].

    python check_render.py             синтетичні тести + реальні дані (потрібен інтернет або кеш)
    python check_render.py --offline   лише синтетичні тести (без мережі)

Синтетичні тести будують штучні кадри з відомою "зорею" на небі, з різним поворотом і
зсувом WCS, і перевіряють, що після вирівнювання зоря опиняється в ТОМУ САМОМУ пікселі.
Це перевірка математики без жодних даних із мережі.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from pipeline.render import (BAD_MASK, OUTLIER_BIT, align_frames, default_fov, load_render,
                             render_all, stretch_to_uint8, target_wcs, wavelength_groups)

PASSED, FAILED, WARNINGS = [], [], []
PIX_SCALE_DEG = 6.15 / 3600.0      # реальний піксель SPHEREx ~ 6.15"


def section(title):
    print(f"\n=== {title} ===")


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'OK' if condition else 'ПОМИЛКА'}] {name}" + (f"  ->  {detail}" if detail else ""))
    return bool(condition)


def warn(name, condition, detail=""):
    """Не провал, а попередження: потрібна візуальна перевірка."""
    if not condition:
        WARNINGS.append(name)
    print(f"  [{'OK' if condition else 'УВАГА'}] {name}" + (f"  ->  {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Допоміжне: штучні кадри
# --------------------------------------------------------------------------
def make_wcs(coord, npix, rotation_deg=0.0, shift_pix=(0.0, 0.0)):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [coord.ra.deg, coord.dec.deg]
    w.wcs.crpix = [(npix + 1) / 2 + shift_pix[0], (npix + 1) / 2 + shift_pix[1]]
    th = np.deg2rad(rotation_deg)
    w.wcs.cd = PIX_SCALE_DEG * np.array([[-np.cos(th), np.sin(th)], [np.sin(th), np.cos(th)]])
    return w


def make_frame(coord, stars, npix=80, rotation=0.0, shift=(0.0, 0.0), bg=0.3, mjd=60800.0, wl=1.2, tag="a"):
    """Кадр: фон bg + гаусові зорі в заданих НЕБЕСНИХ координатах. zodi = фон (його треба відняти)."""
    w = make_wcs(coord, npix, rotation, shift)
    yy, xx = np.mgrid[0:npix, 0:npix]
    img = np.full((npix, npix), bg, dtype=np.float32)
    for star, amp in stars:
        x0, y0 = w.world_to_pixel(star)
        img += (amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.2 ** 2))).astype(np.float32)
    return {
        "image": img,
        "flags": np.zeros((npix, npix), dtype=np.int32),
        "zodi": np.full((npix, npix), bg, dtype=np.float32),
        "wcs": w, "mjd": mjd, "wavelength_um": wl, "detector": 2,
        "path": f"synthetic_{tag}.fits",
    }


def centroid(arr, x_guess, y_guess, half=16):
    a = np.nan_to_num(arr.astype(float) - np.nanmedian(arr))
    x0, y0 = int(round(float(x_guess))), int(round(float(y_guess)))
    sub = np.clip(a[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1], 0, None)
    yy, xx = np.mgrid[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    total = sub.sum()
    return (sub * xx).sum() / total, (sub * yy).sum() / total


# --------------------------------------------------------------------------
def test_synthetic(tmp):
    section("1. Вирівнювання на штучних кадрах (без мережі)")
    coord = SkyCoord(210.80227 * u.deg, 54.34895 * u.deg)
    star1 = coord.spherical_offsets_by(12 * u.arcsec, -9 * u.arcsec)
    star2 = coord.spherical_offsets_by(-20 * u.arcsec, 25 * u.arcsec)
    stars = [(star1, 5.0), (star2, 3.0)]

    size_deg = 80 * PIX_SCALE_DEG
    fov = default_fov(size_deg)
    npix = 256

    # три кадри: без повороту; поворот 37° + зсув; поворот -62° без зсуву
    frames = [
        make_frame(coord, stars, rotation=0.0, mjd=60800.0, wl=1.20, tag="a"),
        make_frame(coord, stars, rotation=37.0, shift=(3.0, -2.0), mjd=60810.0, wl=1.25, tag="b"),
        make_frame(coord, stars, rotation=-62.0, mjd=60820.0, wl=1.30, tag="c"),
    ]
    aligned = align_frames(frames, coord, fov, npix, verbose=False)
    imgs, cov = aligned["images"], aligned["coverage"]
    check("усі 3 кадри пройшли вирівнювання", len(aligned["kept"]) == 3 and imgs.shape == (3, npix, npix),
          str(imgs.shape))

    tw = target_wcs(coord, fov, npix)
    ok_all = True
    for name, star in (("зоря 1", star1), ("зоря 2", star2)):
        ex, ey = (float(v) for v in tw.world_to_pixel(star))
        errs = []
        for k in range(3):
            cx, cy = centroid(imgs[k], ex, ey)
            errs.append(float(np.hypot(cx - ex, cy - ey)))
        ok = max(errs) < 0.7
        ok_all &= ok
        check(f"{name}: на всіх кадрах у тому самому пікселі (похибка < 0.7 пікселя)", ok,
              "похибки: " + ", ".join(f"{e:.2f}" for e in errs))

    check("зодіакальне світло віднято (фон ~ 0)", all(abs(np.nanmedian(imgs[k])) < 0.02 for k in range(3)),
          f"медіани: {[round(float(np.nanmedian(imgs[k])), 4) for k in range(3)]}")
    no_zodi = align_frames(frames[:1], coord, fov, npix, subtract_zodi=False, verbose=False)["images"]
    check("без віднімання фон лишається ~0.3", abs(float(np.nanmedian(no_zodi[0])) - 0.3) < 0.02,
          f"{float(np.nanmedian(no_zodi[0])):.3f}")

    check("покриття повного квадрата при повороті без зсуву (c) > 99%", float(cov[2].mean()) > 0.99,
          f"{cov[2].mean():.1%}")
    check("покриття зі зсувом (b) > 90%", float(cov[1].mean()) > 0.90, f"{cov[1].mean():.1%}")

    # бікубічна інтерполяція (режим --order bicubic)
    cub = align_frames(frames, coord, fov, npix, order="bicubic", verbose=False)["images"]
    ex, ey = (float(v) for v in tw.world_to_pixel(star1))
    errs = [float(np.hypot(*(np.array(centroid(cub[k], ex, ey)) - [ex, ey]))) for k in range(3)]
    check("bicubic: зоря 1 у тому самому пікселі (< 0.7)", max(errs) < 0.7, ", ".join(f"{e:.2f}" for e in errs))

    # перевірка очищення 1-піксельного космічного променя (7 sigma) без зміни зір
    cr_frame = [dict(frames[0])]
    cr_frame[0]["image"] = frames[0]["image"].copy()
    cr_frame[0]["image"][20, 20] = 50.0  # ізольований 1-піксельний космічний промінь
    cr_img = align_frames(cr_frame, coord, fov, npix, verbose=False)["images"][0]
    check("1-піксельний космічний промінь (7 sigma) прибирається до репроєкції",
          float(np.nanmax(cr_img)) < 10.0, f"макс. після очищення: {float(np.nanmax(cr_img)):.2f}")

    # маски
    masked = [dict(f) for f in frames[:1]]
    masked[0]["flags"] = frames[0]["flags"].copy()
    masked[0]["flags"][38:42, 38:42] |= 1                  # біт TRANSIENT (поганий піксель)
    m_img = align_frames(masked, coord, fov, npix, verbose=False)["images"][0]
    check("поганий піксель (TRANSIENT) стає NaN", int(np.isnan(m_img[cov[0]]).sum()) > 0)
    m_cub = align_frames(masked, coord, fov, npix, order="bicubic", verbose=False)["images"][0]
    nan_cub = int(np.isnan(m_cub[cov[0]]).sum())
    check("bicubic: NaN лишається лише на місці поганих пікселів (не розповзається)",
          0 < nan_cub < 0.05 * cov[0].sum(), f"NaN: {nan_cub} пікселів")

    outl = [dict(f) for f in frames[:1]]
    outl[0]["flags"] = frames[0]["flags"].copy()
    outl[0]["flags"][38:42, 38:42] |= (1 << OUTLIER_BIT)
    o_def = align_frames(outl, coord, fov, npix, verbose=False)["images"][0]
    o_msk = align_frames(outl, coord, fov, npix, mask_outliers=True, verbose=False)["images"][0]
    check("OUTLIER за замовчуванням НЕ маскується", int(np.isnan(o_def[cov[0]]).sum()) == 0)
    check("OUTLIER маскується з mask_outliers=True", int(np.isnan(o_msk[cov[0]]).sum()) > 0)

    src = [dict(f) for f in frames[:1]]
    src[0]["flags"] = frames[0]["flags"].copy()
    src[0]["flags"][:, :] |= (1 << 21)                     # біт SOURCE на всіх пікселях
    s_img = align_frames(src, coord, fov, npix, verbose=False)["images"][0]
    check("SOURCE (зорі) ніколи не маскується", int(np.isnan(s_img[cov[0]]).sum()) == 0)
    check("BAD_MASK не містить бітів 19 і 21", ((BAD_MASK >> 19) & 1) == 0 and ((BAD_MASK >> 21) & 1) == 0)

    # --- PNG та файли ---
    section("2. PNG, manifest і stack на штучних кадрах")
    out_dir = render_all(frames, coord, size_deg, "SPHEREx-D2", npix=npix, out_root=tmp, verbose=False)
    pngs = sorted(out_dir.glob("frame_*.png"))
    check("створено 3 PNG", len(pngs) == 3, str([p.name for p in pngs]))

    from PIL import Image
    with Image.open(pngs[0]) as im:                    # закриваємо файл: на Windows інакше його не видалити
        im_mode, im_size = im.mode, im.size
        gray = np.array(im.getchannel("L"))
        alpha = np.array(im.getchannel("A"))
    check("PNG: режим LA, розмір 256x256", im_mode == "LA" and im_size == (npix, npix), f"{im_mode} {im_size}")
    check("PNG: повний динамічний діапазон (є і темне, і яскраве)", gray.min() < 30 and gray.max() > 220,
          f"{gray.min()}..{gray.max()}")
    check("PNG: повністю непрозорий (фон без порожніх кутів)", alpha.min() == 255)

    ex, ey = (float(v) for v in tw.world_to_pixel(star1))
    ey_png = npix - 1 - ey
    cx, cy = centroid(gray.astype(float), ex, ey_png)
    check("PNG: зоря 1 у очікуваному місці з урахуванням переворота (похибка < 1.0)",
          np.hypot(cx - ex, cy - ey_png) < 1.0, f"({cx:.1f}, {cy:.1f}) vs ({ex:.1f}, {ey_png:.1f})")
    # північ угору: зоря 2 має Dec вищий за центр -> у PNG вище (менший номер рядка)
    ex2, ey2 = (float(v) for v in tw.world_to_pixel(star2))
    check("PNG: північ угорі (зоря з більшим Dec вище центру)", (npix - 1 - ey2) < (npix - 1) / 2)

    manifest, stack = load_render(out_dir)
    check("manifest: 3 кадри, правильні поля",
          manifest["n_frames"] == 3 and all(k in manifest["frames"][0] for k in
                                            ("file", "mjd", "date_utc", "wavelength_um", "detector")))
    check("manifest: кадри за часом, дати і довжини хвиль збережені",
          [e["mjd"] for e in manifest["frames"]] == [60800.0, 60810.0, 60820.0]
          and [e["wavelength_um"] for e in manifest["frames"]] == [1.2, 1.25, 1.3],
          manifest["frames"][0]["date_utc"])
    check("stack.npz: масиви правильної форми",
          stack["images"].shape == (3, npix, npix) and stack["coverage"].shape == (3, npix, npix)
          and len(stack["mjd"]) == 3)
    check("contact_sheet.png створено", (out_dir / "contact_sheet.png").exists())
    json.dumps(manifest)    # manifest має серіалізуватись у валідний JSON
    check("manifest.json читається як JSON", True)

    # повторний запуск з меншою кількістю кадрів не залишає старих PNG
    render_all(frames[:2], coord, size_deg, "SPHEREx-D2", npix=npix, out_root=tmp, verbose=False)
    check("повторний рендер видаляє зайві старі PNG", len(list(out_dir.glob("frame_*.png"))) == 2)

    # нові поля manifest: покриття, групи за довжиною хвилі, WCS
    check("manifest: покриття кожного кадру записано (>= 0.98)",
          all(e["coverage"] >= 0.98 for e in manifest["frames"]),
          ", ".join(str(e["coverage"]) for e in manifest["frames"]))
    check("manifest: є блок wcs з crval, crpix, cdelt",
          all(k in manifest["wcs"] for k in ("crval_deg", "crpix_1based", "cdelt_deg")))
    check("manifest: кожен кадр має групу за довжиною хвилі",
          all(e["wl_group"] is not None for e in manifest["frames"]) and len(manifest["wl_groups"]) == 3,
          f"груп: {len(manifest['wl_groups'])}")

    ids, groups = wavelength_groups([1.20, 1.21, 1.22, 1.30, float("nan"), 2.0])
    check("wavelength_groups: 1.20-1.22 разом, 1.30 і 2.0 окремо, NaN поза групами",
          list(ids) == [0, 0, 0, 1, -1, 2] and [g["n_frames"] for g in groups] == [3, 1, 1], str(list(ids)))

    # кадр біля краю детектора: пропускається за замовчуванням, залишається при низькому порогу
    edge = make_frame(coord, stars, rotation=0.0, shift=(35.0, 0.0), mjd=60830.0, wl=1.35, tag="edge")
    with_edge = frames + [edge]
    d_skip = render_all(with_edge, coord, size_deg, "SPHEREx-D2", npix=npix, out_root=tmp, tag="edge_skip", verbose=False)
    m_skip = json.loads((d_skip / "manifest.json").read_text(encoding="utf-8"))
    check("кадр біля краю детектора відкинуто (покриття < 98%)",
          m_skip["n_frames"] == 3 and m_skip["n_skipped_low_coverage"] == 1,
          f"кадрів {m_skip['n_frames']}, відкинуто {m_skip['n_skipped_low_coverage']}")
    d_keep = render_all(with_edge, coord, size_deg, "SPHEREx-D2", npix=npix, out_root=tmp, tag="edge_keep",
                        min_coverage=0.3, verbose=False)
    m_keep = json.loads((d_keep / "manifest.json").read_text(encoding="utf-8"))
    cov_edge = [e["coverage"] for e in m_keep["frames"] if e["coverage"] < 0.98]
    check("з min_coverage=0.3 кадр лишається, а його покриття записано в manifest",
          m_keep["n_frames"] == 4 and len(cov_edge) == 1, f"покриття крайового кадру: {cov_edge}")

    # інші формати та параметр tag (різні налаштування -> різні папки)
    sizes = {}
    for fmt in ("png", "webp", "jpeg"):
        d = render_all(frames, coord, size_deg, "SPHEREx-D2", npix=npix, out_root=tmp,
                       fmt=fmt, tag=f"fmt_{fmt}", verbose=False)
        files = sorted(d.glob("frame_*.*"))
        sizes[fmt] = sum(f.stat().st_size for f in files)
        with Image.open(files[0]) as im_f:
            size_f = im_f.size
        ok = len(files) == 3 and d != out_dir and size_f == (npix, npix)
        check(f"формат {fmt}: 3 файли відкриваються, 256x256, окрема папка", ok,
              f"{files[0].suffix}, разом {sizes[fmt] / 1e3:.0f} КБ")
    check("webp і jpeg менші за png", sizes["webp"] < sizes["png"] and sizes["jpeg"] < sizes["png"])

    # розтяг: константа не ламає
    g, a = stretch_to_uint8(np.full((10, 10), 5.0, dtype=np.float32), np.ones((10, 10), bool))
    check("розтяг: однорідне зображення не падає", g.shape == (10, 10))
    g, a = stretch_to_uint8(np.full((10, 10), np.nan, dtype=np.float32), np.ones((10, 10), bool))
    check("розтяг: повністю порожнє зображення не падає", g.shape == (10, 10))


def test_real():
    section("3. Реальні дані M101 (смуга D2)")
    import astropy.units as u
    from pipeline.catalog import find_exposures
    from pipeline.cutouts import load_frames

    coord = SkyCoord(210.80227 * u.deg, 54.34895 * u.deg)
    try:
        table = find_exposures(coord, 0.1, "D2", verbose=False)
        frames = load_frames(table, coord, limit=6, verbose=False)
    except Exception as exc:
        check("отримано реальні кадри", False, f"{type(exc).__name__}: {exc}")
        return
    if not check("отримано щонайменше 4 реальні кадри", len(frames) >= 4, f"{len(frames)}"):
        return
    check("у кадрів є модель ZODI", all(f.get("zodi") is not None for f in frames))

    out_dir = render_all(frames, coord, 0.1, "SPHEREx-D2", npix=256, verbose=False)
    manifest, stack = load_render(out_dir)
    n = manifest["n_frames"]
    check("рендер зберіг усі кадри", n == len(frames), f"{n} з {len(frames)}")
    check("PNG існують", all((out_dir / e["file"]).exists() for e in manifest["frames"]))

    cov = stack["coverage"].mean(axis=(1, 2))
    check("кожен кадр покриває > 98% спільної сітки (без порожніх кутів)", bool(np.all(cov > 0.98)),
          "покриття: " + ", ".join(f"{c:.0%}" for c in cov))
    nan_frac = np.isnan(stack["images"]).mean(axis=(1, 2))
    check("частка поганих пікселів мала (< 5%)", bool(np.all(nan_frac < 0.05)),
          "NaN: " + ", ".join(f"{c:.1%}" for c in nan_frac))
    med = [float(np.nanmedian(im)) for im in stack["images"]]
    print("  Медіани кадрів після віднімання ZODI, MJy/sr: " + ", ".join(f"{m:.3f}" for m in med)
          + "  (поле заповнене світлом галактики, тому не нуль: це нормально)")

    try:
        import cv2

        def nucleus(img):
            """Центр яскравого ядра: максимум згладженого зображення + центроїд."""
            a = np.nan_to_num(img - np.nanmedian(img)).astype(np.float32)
            blur = cv2.GaussianBlur(a, (0, 0), 3)
            n_ = a.shape[0]
            c, r = n_ // 2, n_ // 6
            sub = blur[c - r:c + r, c - r:c + r]
            iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
            x0, y0 = int(ix + c - r), int(iy + c - r)
            h = 8
            win = a[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
            win = np.clip(win - np.percentile(win, 20), 0, None)
            yy, xx = np.mgrid[y0 - h:y0 + h + 1, x0 - h:x0 + h + 1]
            return float((win * xx).sum() / win.sum()), float((win * yy).sum() / win.sum())

        pos = np.array([nucleus(im) for im in stack["images"]])
        centre = (stack["images"].shape[1] - 1) / 2
        spread = np.hypot(*(pos - np.median(pos, axis=0)).T)
        print("  Положення ядра галактики на кадрах (x, y):",
              "; ".join(f"({x:.1f}, {y:.1f})" for x, y in pos), f"  центр сітки = {centre:.1f}")
        warn("ядро галактики на всіх кадрах у тому самому місці (розкид < 2.5 пікселя)",
             float(spread.max()) < 2.5, f"макс. відхилення від медіани: {spread.max():.2f} пікселя")
        warn("ядро поблизу центру (координати M101 = ядро; відхилення < 8 пікселів)",
             float(np.hypot(*(np.median(pos, axis=0) - centre))) < 8,
             f"{float(np.hypot(*(np.median(pos, axis=0) - centre))):.2f} пікселя")
    except ImportError:
        print("  (OpenCV не встановлено: перевірку положення ядра пропущено)")

    print(f"\n  ВІЗУАЛЬНА ПЕРЕВІРКА: відкрий  {out_dir / 'contact_sheet.png'}")
    print("  Зорі та галактика мають стояти на тих самих місцях на всіх мініатюрах")
    print("  (орієнтир: червоні риски по краях). Яскравість може відрізнятись: різна довжина хвилі.")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="лише синтетичні тести")
    args = parser.parse_args()

    for name, fn in (("synthetic", lambda: _run_synthetic()),
                     ("real", test_real if not args.offline else None)):
        if fn is None:
            continue
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            check(f"етап '{name}' завершився без винятків", False, repr(exc))

    print(f"\nПідсумок: успішно {len(PASSED)}, помилок {len(FAILED)}, попереджень {len(WARNINGS)}")
    for name in FAILED:
        print("  ПОМИЛКА:", name)
    for name in WARNINGS:
        print("  УВАГА:", name)
    if FAILED:
        sys.exit(1)


def _run_synthetic():
    kw = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
    with tempfile.TemporaryDirectory(**kw) as tmp:
        test_synthetic(Path(tmp))


if __name__ == "__main__":
    main()