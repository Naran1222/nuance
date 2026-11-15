"""
КРОК 3. Вирівнювання кадрів за WCS і збереження зображень.
Зберігає формат покадрових зображень (режим LA, asinh-розтяг 1.0..99.6%, a=18.0,
віднімання ZODI та маскування FLAGS) і додає до репроєкції фільтрацію поодиноких
1-піксельних космічних променів (7 sigma відносно медіани 3x3) без зміни NaN-масок.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from photutils.background import Background2D, MedianBackground
from PIL import Image, ImageDraw
from reproject import reproject_interp
from scipy.ndimage import median_filter

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"

# Біт 15 (NONLINEAR) прибрано: нелінійні пікселі в ядрах яскравих зір мають валідний сигнал
BAD_BITS = (0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 17)
OUTLIER_BIT = 19
BAD_MASK = sum(1 << b for b in BAD_BITS)

IMAGE_FORMATS = {"png": ".png", "webp": ".webp", "jpeg": ".jpg"}


def default_fov(size_deg: float) -> float:
    return float(size_deg) / math.sqrt(2.0) * 0.97


def target_wcs(coord: SkyCoord, fov_deg: float, npix: int = 256) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [float(coord.ra.deg), float(coord.dec.deg)]
    w.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]
    scale = float(fov_deg) / npix
    w.wcs.cdelt = [-scale, scale]
    w.wcs.cunit = ["deg", "deg"]
    return w


def _format_band_slug(band: Optional[str], bands: Optional[Sequence[str]] = None) -> str:
    if bands:
        cleaned = sorted({str(b).replace("SPHEREx-", "") for b in bands if str(b).strip()})
        if len(cleaned) == 1:
            return f"SPHEREx-{cleaned[0]}" if band and str(band).startswith("SPHEREx-") else cleaned[0]
        if 1 < len(cleaned) < 6:
            return "-".join(cleaned)
        return "ALL"
    if band:
        return str(band)
    return "ALL"


def output_dir_for(
    coord: SkyCoord,
    size_deg: float,
    band: Optional[str],
    out_root=OUT_DIR,
    tag: str = "",
    bands: Optional[Sequence[str]] = None,
) -> Path:
    slug = _format_band_slug(band, bands)
    name = f"ra{coord.ra.deg:.5f}_dec{coord.dec.deg:+.5f}_s{size_deg:g}_{slug}"
    return Path(out_root) / (f"{name}_{tag}" if tag else name)


def _reproject(array, wcs_in, wcs_out, npix, order="bilinear"):
    out, _footprint = reproject_interp((array, wcs_in), wcs_out, shape_out=(npix, npix), order=order)
    return np.array(out, dtype=np.float32)


def clean_cosmic_rays(flux: np.ndarray, sigma_thresh: float = 7.0) -> np.ndarray:
    """Замінює поодинокі 1-піксельні гарячі та биті пікселі локальною медіаною, зберігаючи зірки."""
    filled = np.nan_to_num(flux, nan=np.nanmedian(flux))
    med3 = median_filter(filled, size=3)
    residual = filled - med3
    _, _, std = sigma_clipped_stats(residual, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return flux
    outliers = np.abs(residual) > (sigma_thresh * std)
    cleaned = flux.copy()
    cleaned[outliers] = med3[outliers]
    return cleaned


def _fill_input(img: np.ndarray) -> np.ndarray:
    nan = ~np.isfinite(img)
    if not nan.any():
        return img
    med = float(np.nanmedian(img)) if (~nan).any() else 0.0
    base = np.where(nan, med, img).astype(np.float32)
    if cv2 is not None:
        return cv2.inpaint(base, nan.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return base


def align_frames(
    frames: List[dict],
    coord: SkyCoord,
    fov_deg: float,
    npix: int = 256,
    subtract_zodi: bool = True,
    mask_outliers: bool = False,
    min_coverage: float = 0.5,
    verbose: bool = True,
    order: str = "bilinear",
) -> dict:
    w_out = target_wcs(coord, fov_deg, npix)
    mask_value = BAD_MASK | ((1 << OUTLIER_BIT) if mask_outliers else 0)

    images, covers, kept = [], [], []
    skipped = 0
    sigma_clip = SigmaClip(sigma=3.0)
    for f in frames:
        img = np.array(f["image"], dtype=np.float64)
        zodi = f.get("zodi")
        if subtract_zodi and zodi is not None:
            img = img - zodi

        # Маскуємо лише критичні апаратні збої, не чіпаючи яскраві ядра туманностей і зір
        if mask_outliers:
            bad = (np.asarray(f["flags"]).astype(np.int64) & (1 << OUTLIER_BIT)) != 0
            img[bad] = np.nan
        img[~np.isfinite(img)] = np.nan

        if np.isfinite(img).sum() >= 64:
            bkg = Background2D(
                img,
                box_size=(22, 22),
                filter_size=(3, 3),
                sigma_clip=sigma_clip,
                bkg_estimator=MedianBackground(),
                coverage_mask=~np.isfinite(img),
            )
            flat_flux = img - bkg.background
        else:
            flat_flux = img

        healed_flux = interpolate_replace_nans(flat_flux, Gaussian2DKernel(x_stddev=1.0))
        clean_flux = clean_cosmic_rays(healed_flux, sigma_thresh=7.0).astype(np.float32)

        sci = _reproject(clean_flux, f["wcs"], w_out, npix, order=order)
        cov = np.isfinite(_reproject(np.ones(clean_flux.shape, dtype=np.float32), f["wcs"], w_out, npix))

        if cov.mean() < min_coverage:
            skipped += 1
            continue
        images.append(sci)
        covers.append(cov)
        kept.append(f)

    if verbose:
        print(f"  [render] вирівняно кадрів: {len(kept)}, пропущено (край детектора, покриття < {min_coverage:.0%}): {skipped}")
    if kept:
        images_arr = np.stack(images)
        covers_arr = np.stack(covers)
    else:
        images_arr = np.zeros((0, npix, npix), dtype=np.float32)
        covers_arr = np.zeros((0, npix, npix), dtype=bool)
    return {"images": images_arr, "coverage": covers_arr, "kept": kept, "wcs": w_out, "skipped": skipped}


def _fill_holes(gray: np.ndarray, holes: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        return cv2.inpaint(gray, holes.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    out = gray.copy()
    out[holes] = int(np.median(gray[~holes])) if (~holes).any() else 0
    return out


def stretch_to_uint8(sci: np.ndarray, coverage: np.ndarray, pct: float = 99.2, a: float = 0.04):
    valid = np.isfinite(sci)
    alpha = np.where(coverage, 255, 0).astype(np.uint8)
    if valid.sum() < 10:
        return np.zeros(sci.shape, dtype=np.uint8), alpha

    vmin, vmax = PercentileInterval(pct).get_limits(sci[valid])
    vmin = min(vmin, -0.02 * (vmax - vmin))
    if not vmax > vmin:
        vmax = vmin + 1e-6

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=a), clip=True)
    filled = np.nan_to_num(sci, nan=vmin)
    gray = (np.asarray(norm(filled)) * 255.0).astype(np.uint8)
    return gray, alpha


def save_image(gray: np.ndarray, alpha: np.ndarray, path, fmt: str = "png", quality: int = 90) -> None:
    g = np.ascontiguousarray(gray[::-1])
    a = np.ascontiguousarray(alpha[::-1])
    if fmt == "jpeg":
        Image.fromarray(np.where(a > 0, g, 0).astype(np.uint8)).save(path, "JPEG", quality=quality, optimize=True)
        return
    img = Image.fromarray(g)
    img.putalpha(Image.fromarray(a))
    if fmt == "webp":
        img.convert("RGBA").save(path, "WEBP", quality=quality, method=4)
    else:
        img.save(path, optimize=True)


def save_png(gray: np.ndarray, alpha: np.ndarray, path) -> None:
    save_image(gray, alpha, path, "png")


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _iso(mjd) -> str:
    return Time(mjd, format="mjd").iso[:19] if _num(mjd) is not None else ""


def wavelength_groups(wavelengths, tol: float = 0.03):
    wl = np.array([np.nan if _num(w) is None else float(w) for w in wavelengths], dtype=float)
    ids = np.full(len(wl), -1, dtype=int)
    groups = []
    start = None
    for i in [j for j in np.argsort(wl) if np.isfinite(wl[j])]:
        if start is None or wl[i] > start * (1.0 + tol):
            groups.append({"id": len(groups), "frames": []})
            start = wl[i]
        groups[-1]["frames"].append(int(i))
        ids[i] = len(groups) - 1
    for g in groups:
        g["frames"].sort()
        g["n_frames"] = len(g["frames"])
        g["center_um"] = float(np.mean([wl[i] for i in g["frames"]]))
    return ids, groups


def make_contact_sheet(out_dir, manifest: dict, max_frames=12, thumb=200, cols=4):
    out_dir = Path(out_dir)
    entries = manifest["frames"]
    if not entries:
        return None
    idx = sorted(set(np.linspace(0, len(entries) - 1, min(max_frames, len(entries))).round().astype(int).tolist()))
    cols = min(cols, len(idx))
    rows = math.ceil(len(idx) / cols)
    label_h = 34
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), (20, 20, 28))
    draw = ImageDraw.Draw(sheet)

    for k, i in enumerate(idx):
        e = entries[i]
        with Image.open(out_dir / e["file"]) as src:
            la = src.convert("LA")
        background = Image.new("L", la.size, 45)
        gray = Image.composite(la.getchannel("L"), background, la.getchannel("A"))
        gray = gray.resize((thumb, thumb)).convert("RGB")
        r, c = divmod(k, cols)
        x0, y0 = c * thumb, r * (thumb + label_h)
        sheet.paste(gray, (x0, y0))
        mid = thumb // 2
        tick = (255, 90, 90)
        draw.line([(x0 + mid, y0), (x0 + mid, y0 + 9)], fill=tick)
        draw.line([(x0 + mid, y0 + thumb - 10), (x0 + mid, y0 + thumb - 1)], fill=tick)
        draw.line([(x0, y0 + mid), (x0 + 9, y0 + mid)], fill=tick)
        draw.line([(x0 + thumb - 10, y0 + mid), (x0 + thumb - 1, y0 + mid)], fill=tick)
        wl = e["wavelength_um"]
        draw.text((x0 + 4, y0 + thumb + 2), f"#{e['index']}  {e['date_utc'][:10]}", fill=(230, 230, 230))
        draw.text((x0 + 4, y0 + thumb + 16),
                  f"{wl:.3f} um" if wl is not None else "? um", fill=(150, 200, 255))

    path = out_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def render_all(
    frames: List[dict],
    coord: SkyCoord,
    size_deg: float,
    band: Optional[str] = None,
    npix: int = 256,
    fov_deg: Optional[float] = None,
    subtract_zodi: bool = True,
    mask_outliers: bool = False,
    out_root=OUT_DIR,
    verbose: bool = True,
    order: str = "bilinear",
    fmt: str = "png",
    quality: int = 90,
    tag: str = "",
    min_coverage: float = 0.98,
    contact_sheet: bool = True,
    bands: Optional[Sequence[str]] = None,
) -> Path:
    if not frames:
        raise ValueError("Немає кадрів для рендеру (список порожній).")
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"Невідомий формат '{fmt}'. Допустимі: {', '.join(IMAGE_FORMATS)}.")

    fov = float(fov_deg) if fov_deg else default_fov(size_deg)
    out_dir = output_dir_for(coord, size_deg, band, out_root, tag, bands=bands)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("frame_*.*")) + [out_dir / n for n in
                                                   ("manifest.json", "stack.npz", "contact_sheet.png")]:
        try:
            old.unlink()
        except OSError:
            pass

    aligned = align_frames(frames, coord, fov, npix, subtract_zodi, mask_outliers, verbose=verbose,
                           order=order, min_coverage=min_coverage)
    kept = aligned["kept"]
    if not kept:
        raise RuntimeError("Жоден кадр не покриває задану область.")

    entries = []
    frame_bands = []
    for i, f in enumerate(kept):
        gray, alpha = stretch_to_uint8(aligned["images"][i], aligned["coverage"][i])
        name = f"frame_{i:03d}{IMAGE_FORMATS[fmt]}"
        save_image(gray, alpha, out_dir / name, fmt, quality)
        det = int(f["detector"]) if f.get("detector") is not None else None
        b_name = f.get("band") or (f"SPHEREx-D{det}" if det is not None else (band or "ALL"))
        frame_bands.append(str(b_name))
        entries.append({
            "index": i,
            "file": name,
            "mjd": _num(f["mjd"]),
            "date_utc": _iso(f["mjd"]),
            "wavelength_um": _num(f["wavelength_um"]),
            "detector": det,
            "band": str(b_name),
            "source_file": Path(f["path"]).name,
            "coverage": round(float(aligned["coverage"][i].mean()), 4),
        })

    group_ids, groups = wavelength_groups([e["wavelength_um"] for e in entries])
    for e, g in zip(entries, group_ids):
        e["wl_group"] = int(g) if g >= 0 else None

    np.savez_compressed(
        out_dir / "stack.npz",
        images=aligned["images"],
        coverage=aligned["coverage"],
        mjd=np.array([float(f["mjd"]) for f in kept]),
        wavelength_um=np.array([float(f["wavelength_um"]) for f in kept]),
        band=np.array(frame_bands, dtype="<U16"),
    )

    slug = _format_band_slug(band, bands)
    manifest = {
        "target": {"ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "query": {
            "size_deg": float(size_deg),
            "band": slug,
            "bands": sorted(set(frame_bands)),
        },
        "image": {
            "npix": int(npix),
            "fov_deg": fov,
            "pixel_scale_arcsec": fov * 3600.0 / npix,
            "orientation": "north up, east left",
            "format": fmt,
            "alpha": fmt != "jpeg",
            "y_png_from_array_row": "y_png = npix - 1 - row",
        },
        "processing": {
            "subtract_zodi": bool(subtract_zodi),
            "mask_outliers": bool(mask_outliers),
            "cosmic_ray_filter": "7sigma_3x3_preserve_nans",
            "interpolation": order,
            "stretch": "per-frame asinh, percentiles 1..99.6",
        },
        "wcs": {
            "ctype": ["RA---TAN", "DEC--TAN"],
            "crval_deg": [float(v) for v in aligned["wcs"].wcs.crval],
            "crpix_1based": [float(v) for v in aligned["wcs"].wcs.crpix],
            "cdelt_deg": [float(v) for v in aligned["wcs"].wcs.cdelt],
            "note": "pixel (x, y), 0-based array coordinates; x increases to the west, y to the north",
        },
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "min_coverage": float(min_coverage),
        "n_skipped_low_coverage": int(aligned["skipped"]),
        "wl_groups": groups,
        "n_frames": len(entries),
        "frames": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if contact_sheet:
        make_contact_sheet(out_dir, manifest)

    if verbose:
        print(f"  [render] збережено {len(entries)} кадрів ({fmt}) у {out_dir}")
    return out_dir


def load_render(out_dir):
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(out_dir / "stack.npz") as z:
        stack = {k: z[k] for k in z.files}
    return manifest, stack