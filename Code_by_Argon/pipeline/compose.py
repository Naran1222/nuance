"""
КРОК 3б. Кольорові композити та сумування кадрів за епохами.

Особливості:
  - Побудова композитів виключно на основі обраних користувачем смуг (1, 2, 3+ або всі 6 смуг).
  - Якщо присутні всі 3 спектральні канали: розподіл по квантилях (1/3, 2/3) з пропуском
    неповних епох. Якщо глобально доступні лише 2 канали (без проміжного) — побудова
    бі-колор композиту із синтетичним зеленим каналом G = 0.5 * (R + B). Якщо 1 вузький канал —
    монохромний RGB-фолбек без падіння.
  - Заповнення внутрішніх дірок (_fill_internal_nans), очищення космічних променів (_despeckle),
    віднімання п'єдесталу неба по 10-му перцентилю та Lupton-подібний asinh-розтяг зі спільними
    межами яскравості між епохами.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize
from PIL import Image, ImageDraw
from scipy.ndimage import label, median_filter

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"

CHANNELS = ("R", "G", "B")
FORMATS = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}
MIN_FRAMES = 6


def load_renders(dirs: Sequence) -> dict:
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]
    dirs = [Path(d) for d in dirs]
    if not dirs:
        raise ValueError("Не вказано жодної папки з результатом кроку 3.")

    images, mjd, wl, bands, meta = [], [], [], [], None
    for d in dirs:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        with np.load(d / "stack.npz") as z:
            im = z["images"]
            images.append(im)
            mjd.append(np.asarray(z["mjd"], dtype=float))
            wl.append(np.asarray(z["wavelength_um"], dtype=float))
            if "band" in z.files:
                bands.extend([str(b) for b in z["band"]])
            else:
                bands.extend([manifest["query"]["band"]] * len(im))

        if meta is None:
            meta = manifest
        else:
            a, b = meta, manifest
            same = (
                a["image"]["npix"] == b["image"]["npix"]
                and abs(a["image"]["fov_deg"] - b["image"]["fov_deg"]) < 1e-9
                and abs(a["target"]["ra_deg"] - b["target"]["ra_deg"]) < 1e-6
                and abs(a["target"]["dec_deg"] - b["target"]["dec_deg"]) < 1e-6
            )
            if not same:
                raise ValueError(
                    f"Папки {dirs[0].name} і {d.name} мають різну сітку (ціль, поле зору або розмір "
                    "картинки). Для композиту потрібна та сама область і ті самі --size, --npix, --fov."
                )
    return {
        "images": np.concatenate(images).astype(np.float32),
        "mjd": np.concatenate(mjd),
        "wl": np.concatenate(wl),
        "band": bands,
        "meta": meta,
        "sources": [d.name for d in dirs],
    }


def find_epochs(mjd: np.ndarray, gap_days: float = 45.0, max_days: Optional[float] = None) -> np.ndarray:
    mjd = np.asarray(mjd, dtype=float)
    order = np.argsort(mjd)
    sorted_mjd = mjd[order]
    ids_sorted = np.zeros(len(mjd), dtype=int)
    current, start = 0, sorted_mjd[0] if len(mjd) else 0.0
    for i in range(1, len(sorted_mjd)):
        too_long = max_days is not None and (sorted_mjd[i] - start) > max_days
        if (sorted_mjd[i] - sorted_mjd[i - 1]) > gap_days or too_long:
            current += 1
            start = sorted_mjd[i]
        ids_sorted[i] = current
    epoch = np.empty(len(mjd), dtype=int)
    epoch[order] = ids_sorted
    return epoch


def channel_edges(wl: np.ndarray):
    e1, e2 = np.quantile(wl, [1 / 3, 2 / 3])
    return float(e1), float(e2)


def assign_channels(wl: np.ndarray, edges) -> np.ndarray:
    e1, e2 = edges
    return np.where(wl > e2, 0, np.where(wl > e1, 1, 2))


def _fill_internal_nans(img: np.ndarray) -> np.ndarray:
    """Заповнює внутрішні NaN гаусовим ядром (як у process.py), не чіпаючи зовнішні краї кадру."""
    nan_mask = ~np.isfinite(img)
    if not nan_mask.any() or nan_mask.all():
        return img

    labeled, _ = label(nan_mask)
    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]
    ]))) - {0}

    internal_holes = nan_mask & ~np.isin(labeled, list(border_labels))
    if not internal_holes.any():
        return img

    healed = interpolate_replace_nans(img.astype(np.float64), Gaussian2DKernel(x_stddev=1.0))
    out = img.copy()
    out[internal_holes] = healed[internal_holes].astype(np.float32)
    return out


def _despeckle(img: np.ndarray, threshold_sigma: float = 7.0) -> np.ndarray:
    """Прибирає як гарячі, так і чорні биті пікселі (двосторонній sigma-фільтр із process.py)."""
    filled = np.nan_to_num(img, nan=np.nanmedian(img))
    med3 = median_filter(filled, size=3)
    residual = filled - med3
    _, _, std = sigma_clipped_stats(residual, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return img
    outliers = np.abs(residual) > (threshold_sigma * std)
    cleaned = img.copy()
    cleaned[outliers] = med3[outliers]
    return cleaned


def coadd(images: np.ndarray, method: str = "median") -> Optional[np.ndarray]:
    if len(images) == 0:
        return None
    func = np.nanmedian if method == "median" else np.nanmean
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = func(images, axis=0).astype(np.float32)
    res = _fill_internal_nans(res)
    res = _despeckle(res)
    valid = res[np.isfinite(res)]
    if valid.size >= 10:
        bg10 = float(np.nanpercentile(valid, 10.0))
        res = (res - bg10).astype(np.float32)
    return res


def _limits(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.4):
    """Межі контрасту 1.0%..99.4% як у save_false_color_rgb_comparison (process.py)."""
    valid = np.isfinite(img)
    if valid.sum() < 10:
        return 0.0, 1.0
    vals = img[valid]
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if not hi > lo:
        hi = lo + 1.0
    return lo, hi


def to_rgb8(channels: Sequence[np.ndarray], limits: Sequence, a: float = 0.06,
            saturation: float = 1.25) -> np.ndarray:
    """
    Незалежний поканальний AsinhStretch(a=0.06) у стилі process.py:
    глибокий чорний фон без сірої вуалі та максимальна насиченість спектральних каналів.
    """
    cleaned = [_fill_internal_nans(np.asarray(img, dtype=np.float64)) for img in channels]
    valid = np.all([np.isfinite(img) for img in cleaned], axis=0)

    rgb_layers = []
    for img, (vmin, vmax) in zip(cleaned, limits):
        if not vmax > vmin:
            vmax = vmin + 1.0
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=a), clip=True)
        clean_ch = np.where(np.isfinite(img), img, vmin)
        rgb_layers.append(np.asarray(norm(clean_ch), dtype=np.float32))

    rgb = np.stack(rgb_layers, axis=-1)

    if saturation != 1.0:
        lum = rgb.mean(axis=-1, keepdims=True)
        rgb = np.clip(lum + saturation * (rgb - lum), 0.0, 1.0)

    rgb[~valid] = 0.0
    return np.ascontiguousarray((rgb[::-1] * 255.0).astype(np.uint8))


def save_rgb(rgb8: np.ndarray, path, fmt: str = "png", quality: int = 90) -> Path:
    path = Path(path)
    img = Image.fromarray(rgb8)
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.save(buf, "JPEG", quality=quality, optimize=True)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=quality, method=4)
    else:
        img.save(buf, "PNG", optimize=True)
    data = buf.getvalue()
    try:
        path.write_bytes(data)
        return path
    except OSError:
        alt = path.with_stem(f"{path.stem}_new")
        alt.write_bytes(data)
        return alt


def _mjd_to_str(mjd: float) -> str:
    base = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return (base + timedelta(days=float(mjd))).strftime("%Y-%m-%d")


def _overview(out_dir: Path, items: List[tuple], thumb=220, cols=4):
    cols = min(cols, len(items))
    rows = math.ceil(len(items) / cols)
    label_h = 20
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), (20, 20, 28))
    draw = ImageDraw.Draw(sheet)
    for k, (fname, label_txt) in enumerate(items):
        with Image.open(out_dir / fname) as src:
            im = src.convert("RGB").resize((thumb, thumb))
        r, c = divmod(k, cols)
        x0, y0 = c * thumb, r * (thumb + label_h)
        sheet.paste(im, (x0, y0))
        draw.text((x0 + 4, y0 + thumb + 4), label_txt, fill=(230, 230, 230))
    path = out_dir / "overview.png"
    sheet.save(path)
    return path


def _complete_channels(
    raw_chs: List[Optional[np.ndarray]],
    global_mode: str,
) -> Optional[List[np.ndarray]]:
    """
    Заповнює канали відповідно до глобального режиму набору:
      - 'rgb' (присутні всі 3 канали): вимагає наявності всіх 3 каналів в епосі (інакше None -> пропуск епохи).
      - 'bicolor_synth_g' (глобально є лише 2 канали): синтезує G = 0.5 * (R + B).
      - 'monochrome_fallback' (глобально є лише 1 канал): дублює канал у R, G, B.
    """
    if global_mode == "rgb":
        if any(c is None for c in raw_chs):
            return None
        return [raw_chs[0], raw_chs[1], raw_chs[2]]

    present = [c for c in raw_chs if c is not None]
    if global_mode == "bicolor_synth_g":
        if len(present) < 2:
            return None
        r_ch = raw_chs[0] if raw_chs[0] is not None else present[0]
        b_ch = raw_chs[2] if raw_chs[2] is not None else present[-1]
        g_ch = (0.5 * (r_ch + b_ch)).astype(np.float32)
        return [r_ch, g_ch, b_ch]

    if global_mode == "monochrome_fallback":
        if not present:
            return None
        mono = present[0]
        return [mono, mono.copy(), mono.copy()]

    return None


def build_color(
    dirs: Sequence,
    out_root=OUT_DIR,
    epoch_gap_days: float = 45.0,
    fmt: str = "png",
    quality: int = 90,
    method: str = "median",
    tag: str = "",
    verbose: bool = True,
    max_epoch_days: Optional[float] = None,
    overview: bool = True,
) -> Path:
    if fmt not in FORMATS:
        raise ValueError(f"Невідомий формат '{fmt}'. Допустимі: {', '.join(FORMATS)}.")
    data = load_renders(dirs)

    ok = np.isfinite(data["mjd"]) & np.isfinite(data["wl"])
    images, mjd, wl = data["images"][ok], data["mjd"][ok], data["wl"][ok]
    bands = [b for b, keep in zip(data["band"], ok) if keep]
    if len(images) < MIN_FRAMES:
        raise ValueError(f"Замало кадрів для композиту: {len(images)} (потрібно щонайменше {MIN_FRAMES}).")

    edges = channel_edges(wl)
    chan = assign_channels(wl, edges)
    epoch = find_epochs(mjd, epoch_gap_days, max_epoch_days)

    # Якщо всі хвилі потрапили лише у 2 кластери (наприклад 2 вузькі смуги без середини),
    # розкладаємо довші у R (0), коротші у B (2), а G синтезуємо як 0.5 * (R + B).
    global_counts = [int((chan == c).sum()) for c in range(3)]
    n_non_empty = sum(1 for cnt in global_counts if cnt > 0)
    if n_non_empty == 3:
        global_mode = "rgb"
    elif n_non_empty == 2:
        global_mode = "bicolor_synth_g"
        mid_wl = 0.5 * (float(wl.min()) + float(wl.max()))
        chan = np.where(wl >= mid_wl, 0, 2)
    else:
        global_mode = "monochrome_fallback"

    meta = data["meta"]
    raw_band_tokens = []
    for b in bands:
        for tok in str(b).replace("SPHEREx-", "").split("-"):
            tok = tok.strip()
            if tok and tok != "ALL":
                raw_band_tokens.append(tok)
    band_names = sorted(set(raw_band_tokens)) if raw_band_tokens else ["ALL"]
    digest = hashlib.md5(("|".join(sorted(data["sources"])) + f"|{epoch_gap_days}|{max_epoch_days}|{method}").encode()).hexdigest()[:6]
    name = (f"ra{meta['target']['ra_deg']:.5f}_dec{meta['target']['dec_deg']:+.5f}"
            f"_s{meta['query']['size_deg']:g}_{'-'.join(band_names)}_color_{digest}")
    if tag:
        name += f"_{tag}"
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("*_rgb.*")) + [out_dir / n for n in
                                                 ("overview.png", "compose.json", "compose_stack.npz")]:
        try:
            old.unlink()
        except OSError:
            pass

    deep_raw = [coadd(images[chan == c], method) for c in range(3)]
    deep = _complete_channels(deep_raw, global_mode)
    if deep is None:
        raise ValueError("Один з каналів не має кадрів: занадто вузький діапазон довжин хвиль.")
    limits = [_limits(d) for d in deep]
    ext = FORMATS[fmt]
    save_rgb(to_rgb8(deep, limits), out_dir / f"deep_rgb{ext}", fmt, quality)

    epoch_entries, epoch_arrays, epoch_mjd = [], [], []
    for e in sorted(set(epoch.tolist())):
        sel = epoch == e
        counts = [int((sel & (chan == c)).sum()) for c in range(3)]
        if global_mode == "rgb" and min(counts) == 0:
            if verbose:
                print(f"  [compose] епоха {_mjd_to_str(mjd[sel].min())}: немає кадрів у каналі "
                      f"{CHANNELS[counts.index(0)]}, пропускаю (кадрів R/G/B = {counts})")
            continue
        chs_raw = [coadd(images[sel & (chan == c)], method) for c in range(3)]
        chs = _complete_channels(chs_raw, global_mode)
        if chs is None:
            continue
        k = len(epoch_entries)
        fname = f"epoch_{k:02d}_rgb{ext}"
        save_rgb(to_rgb8(chs, limits), out_dir / fname, fmt, quality)
        epoch_entries.append({
            "index": k,
            "file": fname,
            "date_start": _mjd_to_str(mjd[sel].min()),
            "date_end": _mjd_to_str(mjd[sel].max()),
            "mjd_start": float(mjd[sel].min()),
            "mjd_end": float(mjd[sel].max()),
            "n_frames": {CHANNELS[c]: counts[c] for c in range(3)},
        })
        epoch_arrays.append(np.stack(chs))
        epoch_mjd.append(float(mjd[sel].mean()))

    np.savez_compressed(
        out_dir / "compose_stack.npz",
        epoch_channels=np.stack(epoch_arrays) if epoch_arrays else np.zeros((0, 3) + images.shape[1:], np.float32),
        deep_channels=np.stack(deep),
        epoch_mjd=np.array(epoch_mjd),
        channel_edges_um=np.array(edges),
    )

    channels_meta = {}
    for c in range(3):
        mask_c = chan == c
        if mask_c.any():
            channels_meta[CHANNELS[c]] = {
                "wavelength_um": [float(wl[mask_c].min()), float(wl[mask_c].max())],
                "n_frames": int(mask_c.sum()),
            }
        else:
            channels_meta[CHANNELS[c]] = {
                "wavelength_um": [float(wl.min()), float(wl.max())],
                "n_frames": 0,
            }

    manifest = {
        "kind": "color_composite",
        "color_mode": global_mode,
        "target": meta["target"],
        "query": {"size_deg": meta["query"]["size_deg"], "bands": band_names},
        "image": {**meta["image"], "format": fmt, "alpha": False, "channels": list(CHANNELS)},
        "channels": channels_meta,
        "deep": {"file": f"deep_rgb{ext}", "n_frames": int(len(images))},
        "epochs": epoch_entries,
        "processing": {
            "coadd": method,
            "sky_pedestal": "10th_percentile",
            "epoch_gap_days": float(epoch_gap_days),
            "max_epoch_days": max_epoch_days,
            "stretch": "noise-anchored asinh (a=22, sat=1.6, gamma=0.75)",
            "false_color": "longest wavelengths = red, shortest = blue",
        },
        "sources": data["sources"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "compose.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    items = [(f"deep_rgb{ext}", f"deep, {len(images)} frames")] + [
        (e["file"], f"{e['date_start']}  {e['n_frames']['R']}/{e['n_frames']['G']}/{e['n_frames']['B']}")
        for e in epoch_entries]
    if overview:
        _overview(out_dir, items)

    if verbose:
        print(f"  [compose] режим: {global_mode} | канали (мкм): R > {edges[1]:.3f}, G {edges[0]:.3f}..{edges[1]:.3f}, B <= {edges[0]:.3f}")
        print(f"  [compose] епох: {len(epoch_entries)}; результат: {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) >= 2:
        dirs = sys.argv[1:]
    else:
        candidates = sorted(
            [p.parent for p in OUT_DIR.glob("*/stack.npz")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit("У папці out/ немає жодної папки зі stack.npz. Спочатку запусти main.py.")
        dirs = [candidates[0]]
        print(f"Автовибір останньої папки: {dirs[0].name}")

    print(build_color(dirs))