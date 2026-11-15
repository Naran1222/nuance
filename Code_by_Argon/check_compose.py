"""
Перевірка кроку 3б (pipeline/compose.py). Не потребує ні мережі, ні астрономічних бібліотек.

    python check_compose.py

Будує штучне небо: блакитна зоря, червона зоря, нейтральна зоря та "астероїд", що в кожній епосі
стоїть в іншому місці. Для трьох епох створює кадри з різними довжинами хвиль і шумом, а потім
перевіряє: правильні кольори, що епохи розрізняються, що шум менший, що астероїд не потрапляє
в глибокий знімок, але видно в різниці епохи й глибокого знімка, а також роботу з 1, 2 і 3+ смугами.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from pipeline.compose import (assign_channels, build_color, channel_edges, find_epochs, load_renders, to_rgb8)

PASSED, FAILED = [], []
H = W = 96
BLUE_STAR, RED_STAR, NEUTRAL = (30, 30), (60, 35), (50, 70)   # (x, y)
MOVER = [(20, 80), (45, 80), (70, 80)]                         # положення "астероїда" в епохах 0,1,2


def section(t):
    print(f"\n=== {t} ===")


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'OK' if cond else 'ПОМИЛКА'}] {name}" + (f"  ->  {detail}" if detail else ""))
    return bool(cond)


def raises(exc, func, *a, **k):
    try:
        func(*a, **k)
    except exc:
        return True
    except Exception:
        return False
    return False


def star(x0, y0, amp, sigma=1.6):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))


def make_dir(root, name, band, wl_lo, wl_hi, epochs=3, per_epoch=12, seed=0, npix=H,
             skip_long_in_epoch=None, ra=130.0, dec=19.6, step_days=0.5, epoch_gap=180.0):
    """Створює папку у форматі кроку 3: manifest.json + stack.npz."""
    rng = np.random.default_rng(seed)
    imgs, mjd, wl = [], [], []
    for e in range(epochs):
        perm = rng.permutation(per_epoch)            # довжини хвиль у випадковому порядку, як у реальних даних
        for i in range(per_epoch):
            lam = wl_lo + (wl_hi - wl_lo) * (perm[i] + rng.uniform(-0.2, 0.2)) / max(per_epoch - 1, 1)
            lam = float(np.clip(lam, wl_lo, wl_hi))
            if skip_long_in_epoch == e and lam > wl_lo + 0.4 * (wl_hi - wl_lo):
                continue                                   # у цій епосі немає довгих хвиль
            t = lam / 1.3
            im = (star(*BLUE_STAR, 3.0 * t ** -2.0) + star(*RED_STAR, 3.0 * t ** 2.0)
                  + star(*NEUTRAL, 2.0) + star(*MOVER[e % len(MOVER)], 4.0))
            im = im + rng.normal(0, 0.10, im.shape)
            if i % 5 == 0:
                im[:3, :] = np.nan                       # край кадру без даних
            imgs.append(im.astype(np.float32))
            mjd.append(60800.0 + epoch_gap * e + step_days * i)
            wl.append(lam)
    d = Path(root) / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "stack.npz", images=np.stack(imgs), coverage=np.ones((len(imgs), npix, npix), bool),
                        mjd=np.array(mjd), wavelength_um=np.array(wl))
    (d / "manifest.json").write_text(json.dumps({
        "target": {"ra_deg": ra, "dec_deg": dec},
        "query": {"size_deg": 0.4, "band": band},
        "image": {"npix": npix, "fov_deg": 0.27, "pixel_scale_arcsec": 3.8},
        "n_frames": len(imgs), "frames": [],
    }), encoding="utf-8")
    return d


def img_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def px(rgb, pos):
    """Піксель PNG у позиції (x, y) масиву: PNG перевернутий по вертикалі."""
    x, y = pos
    return rgb[H - 1 - y, x].astype(int)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from PIL import Image

    section("1. Допоміжні функції")
    mjd = np.array([10, 11, 12, 200, 201, 400.0])
    check("find_epochs: 3 епохи при паузах > 45 діб", list(find_epochs(mjd)) == [0, 0, 0, 1, 1, 2])
    check("find_epochs: порядок кадрів не важливий", list(find_epochs(mjd[::-1])) == [2, 1, 1, 0, 0, 0])
    wl = np.linspace(1.0, 2.0, 30)
    e = channel_edges(wl)
    ch = assign_channels(wl, e)
    check("канали: по третині кадрів", [int((ch == c).sum()) for c in range(3)] == [10, 10, 10],
          str([int((ch == c).sum()) for c in range(3)]))
    check("канали: R = довгі хвилі, B = короткі", wl[ch == 0].min() > wl[ch == 2].max())

    weekly = np.arange(0, 65, 5.0)                      # поле знімають щотижня, пауз немає (як біля полюса екліптики)
    check("find_epochs: без обмеження безперервна серія = одна епоха", len(set(find_epochs(weekly))) == 1)
    capped = find_epochs(weekly, max_days=14)
    check("find_epochs: max_days=14 ділить безперервну серію на епохи по <= 14 діб",
          len(set(capped)) == 5 and all(np.ptp(weekly[capped == k]) <= 14 for k in set(capped)),
          f"епох: {len(set(capped))}")

    r = np.full((4, 4), 5.0, np.float32)
    g = np.full((4, 4), 5.0, np.float32)
    b = np.full((4, 4), 5.0, np.float32)
    r[0, 0] = np.nan                                     # у червоному каналі немає даних, у решті є
    rgb = to_rgb8([r, g, b], [(0.0, 5.0)] * 3)
    check("піксель без даних в одному каналі стає чорним (без кольорових країв)",
          rgb[::-1][0, 0].tolist() == [0, 0, 0] and rgb[::-1][1, 1].max() > 0, str(rgb[::-1][0, 0].tolist()))

    kw = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
    with tempfile.TemporaryDirectory(**kw) as tmp:
        tmp = Path(tmp)
        d2 = make_dir(tmp, "d2", "SPHEREx-D2", 1.1, 1.6, seed=1)

        section("2. Композит із трьох епох")
        out = build_color([d2], out_root=tmp, verbose=False)
        names = sorted(p.name for p in out.iterdir())
        check("створено deep_rgb, 3 епохи, overview, compose.json, compose_stack.npz",
              all(n in names for n in ["deep_rgb.png", "epoch_00_rgb.png", "epoch_01_rgb.png", "epoch_02_rgb.png",
                                       "overview.png", "compose.json", "compose_stack.npz"]), str(names))
        cj = json.loads((out / "compose.json").read_text(encoding="utf-8"))
        check("compose.json: 3 епохи з датами і кількістю кадрів по каналах",
              len(cj["epochs"]) == 3 and all(set(ep["n_frames"]) == {"R", "G", "B"} for ep in cj["epochs"]),
              cj["epochs"][0]["date_start"])
        check("compose.json: діапазони каналів зростають R > G > B",
              cj["channels"]["R"]["wavelength_um"][0] >= cj["channels"]["B"]["wavelength_um"][1] - 0.2)
        check("назва папки містить смугу і 'color'", "_D2_color_" in out.name, out.name)

        with Image.open(out / "deep_rgb.png") as im:
            deep = np.array(im)
        check("deep_rgb: RGB, розмір 96x96", deep.shape == (H, W, 3), str(deep.shape))
        b = px(deep, BLUE_STAR)
        r = px(deep, RED_STAR)
        check("блакитна зоря синіша за червону (B > R)", b[2] > b[0] + 20, f"RGB={b.tolist()}")
        check("червона зоря червоніша за синю (R > B)", r[0] > r[2] + 20, f"RGB={r.tolist()}")
        n = px(deep, NEUTRAL)
        check("нейтральна зоря майже сіра", abs(int(n[0]) - int(n[2])) < 60, f"RGB={n.tolist()}")

        with np.load(out / "compose_stack.npz") as z:      # закриваємо файл: на Windows його інакше не видалити
            ec, dc = z["epoch_channels"].copy(), z["deep_channels"].copy()
        check("compose_stack: форми (3 епохи, 3 канали, H, W)", ec.shape == (3, 3, H, W) and dc.shape == (3, H, W),
              f"{ec.shape}, {dc.shape}")

        # рухомий об'єкт
        g = 1  # канал G
        own = [float(ec[e, g][MOVER[e][1], MOVER[e][0]]) for e in range(3)]
        other = [float(ec[e, g][MOVER[(e + 1) % 3][1], MOVER[(e + 1) % 3][0]]) for e in range(3)]
        check("астероїд видно в 'своїй' епосі й немає в 'чужій'", all(o > 4 * max(x, 0.1) for o, x in zip(own, other)),
              f"свій: {[round(v, 2) for v in own]}, чужий: {[round(v, 2) for v in other]}")
        deep_at = [float(dc[g][MOVER[e][1], MOVER[e][0]]) for e in range(3)]
        check("у глибокому знімку астероїд придушено медіаною (< 40% від епохи)",
              all(d < 0.4 * o for d, o in zip(deep_at, own)), f"глибокий: {[round(v, 2) for v in deep_at]}")
        diff = ec[1, g] - dc[g]
        yy, xx = np.unravel_index(np.nanargmax(diff), diff.shape)
        check("різниця 'епоха - глибокий' показує астероїд у правильному місці",
              abs(int(xx) - MOVER[1][0]) <= 2 and abs(int(yy) - MOVER[1][1]) <= 2, f"максимум у ({xx}, {yy}), очікувалось {MOVER[1]}")

        # шум
        bg_deep = float(np.nanstd(dc[g][5:20, 60:90]))
        with np.load(d2 / "stack.npz") as zr:
            raw = zr["images"].copy()
        bg_raw = float(np.nanstd(raw[:, 5:20, 60:90]))
        check("сумування зменшує шум принаймні в 2 рази", bg_deep < bg_raw / 2, f"{bg_raw:.3f} -> {bg_deep:.3f}")
        check("NaN на краю кадру не ламає (край перекритий іншими кадрами)", bool(np.isfinite(dc[g][5:, :]).all()))

        section("3. Кілька смуг, формати, помилки")
        d3 = make_dir(tmp, "d3", "SPHEREx-D3", 1.6, 2.4, seed=2)
        out2 = build_color([d2, d3], out_root=tmp, verbose=False)
        cj2 = json.loads((out2 / "compose.json").read_text(encoding="utf-8"))
        check("D2+D3: назва містить 'D2-D3', діапазон хвиль ширший",
              "D2-D3" in out2.name and cj2["channels"]["R"]["wavelength_um"][1] > 2.0, out2.name)
        check("D2+D3: кадрів у каналах порівну (±1)",
              max(v["n_frames"] for v in cj2["channels"].values()) - min(v["n_frames"] for v in cj2["channels"].values()) <= 1)

        # Перевірка фолбеку для 1 вузької смуги (monochrome_fallback) та 2 дискретних смуг (bicolor_synth_g)
        d_mono = make_dir(tmp, "dmono", "SPHEREx-D1", 0.95, 0.95, epochs=2, per_epoch=4, seed=11)
        out_mono = build_color([d_mono], out_root=tmp, verbose=False)
        cj_mono = json.loads((out_mono / "compose.json").read_text(encoding="utf-8"))
        check("1 вузька довжина хвилі -> коректний monochrome_fallback без падіння",
              cj_mono.get("color_mode") == "monochrome_fallback" and (out_mono / "deep_rgb.png").exists())

        d_bi1 = make_dir(tmp, "dbi1", "SPHEREx-D1", 0.90, 0.90, epochs=2, per_epoch=3, seed=12)
        d_bi2 = make_dir(tmp, "dbi2", "SPHEREx-D5", 4.15, 4.15, epochs=2, per_epoch=3, seed=13)
        out_bi = build_color([d_bi1, d_bi2], out_root=tmp, verbose=False)
        cj_bi = json.loads((out_bi / "compose.json").read_text(encoding="utf-8"))
        check("2 дискретні смуги без середини -> bicolor_synth_g (G = 0.5*(R+B))",
              cj_bi.get("color_mode") == "bicolor_synth_g" and (out_bi / "deep_rgb.png").exists())

        for fmt, ext in (("webp", ".webp"), ("jpeg", ".jpg")):
            o = build_color([d2], out_root=tmp, fmt=fmt, tag=fmt, verbose=False)
            ok = (o / f"deep_rgb{ext}").exists() and img_size(o / f"epoch_00_rgb{ext}") == (W, H)
            check(f"формат {fmt}: файли створено й відкриваються", ok)

        d_missing = make_dir(tmp, "dm", "SPHEREx-D2", 1.1, 1.6, seed=3, skip_long_in_epoch=2)
        out3 = build_color([d_missing], out_root=tmp, verbose=False)
        cj3 = json.loads((out3 / "compose.json").read_text(encoding="utf-8"))
        check("епоха без одного з каналів пропускається", len(cj3["epochs"]) == 2, f"епох: {len(cj3['epochs'])}")

        d_bad = make_dir(tmp, "dbad", "SPHEREx-D3", 1.6, 2.4, seed=4, npix=H, ra=131.0)
        check("різні цілі в різних папках -> помилка з поясненням", raises(ValueError, load_renders, [d2, d_bad]))
        d_few = make_dir(tmp, "dfew", "SPHEREx-D2", 1.1, 1.6, epochs=1, per_epoch=4, seed=5)
        check("замало кадрів -> помилка", raises(ValueError, build_color, [d_few], tmp))
        check("невідомий формат -> помилка", raises(ValueError, build_color, [d2], tmp, 45.0, "gif"))
        check("порожній список папок -> помилка", raises(ValueError, load_renders, []))

        d_cont = make_dir(tmp, "dcont", "SPHEREx-D2", 1.1, 1.6, epochs=1, per_epoch=36, step_days=2.0, seed=7)
        one = json.loads((build_color([d_cont], out_root=tmp, verbose=False) / "compose.json").read_text(encoding="utf-8"))
        capped_dir = build_color([d_cont], out_root=tmp, max_epoch_days=24, verbose=False)
        many = json.loads((capped_dir / "compose.json").read_text(encoding="utf-8"))
        check("безперервна серія: без обмеження 1 епоха, з max_epoch_days=24 їх кілька",
              len(one["epochs"]) == 1 and len(many["epochs"]) >= 2, f"{len(one['epochs'])} -> {len(many['epochs'])}")
        check("різні параметри епох дають різні папки", capped_dir != build_color([d_cont], out_root=tmp, verbose=False))

        o_no = build_color([d2], out_root=tmp, overview=False, tag="noov", verbose=False)
        check("overview=False: overview.png не створюється", not (o_no / "overview.png").exists())

        # повторний запуск перезаписує, а не накопичує
        before = len(list(out.glob("*_rgb.*")))
        build_color([d2], out_root=tmp, verbose=False)
        check("повторний запуск не залишає зайвих файлів", len(list(out.glob("*_rgb.*"))) == before)

        # Windows: старий файл може бути відкритий іншою програмою (Провідник, сервер); перезапуск не має падати
        original_unlink = Path.unlink

        def locked(self, *a, **k):
            raise PermissionError("[WinError 32] файл зайнятий іншим процесом")

        Path.unlink = locked
        try:
            build_color([d2], out_root=tmp, verbose=False)
            survived = True
        except Exception as exc:
            survived = False
            print("   ", type(exc).__name__, exc)
        finally:
            Path.unlink = original_unlink
        check("перезапуск не падає, якщо старий файл заблоковано (імітація Windows)", survived)

    print(f"\nПідсумок: успішно {len(PASSED)}, помилок {len(FAILED)}")
    for n in FAILED:
        print("  ПОМИЛКА:", n)
    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()