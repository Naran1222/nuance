import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import io
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, median_filter

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from reproject import reproject_interp
from astropy.visualization import AsinhStretch, ManualInterval
from fetch import fetch_two_epoch_cube_slices


def clean_cosmic_rays(flux: np.ndarray, sigma_thresh: float = 7.0) -> np.ndarray:
    """Замінює поодинокі 1-піксельні артефакти/космічні частинки локальною медіаною, зберігаючи зірки."""
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


def load_spherex_cutout(fits_path: Path) -> dict:
    """Зчитує потік, віднімає ZODI + 2D-фон фільтра, лікує NaN та 1-піксельні викиди."""
    with fits.open(fits_path) as hdul:
        img_hdr = hdul["IMAGE"].header
        image = hdul["IMAGE"].data.astype(np.float64)
        zodi = hdul["ZODI"].data.astype(np.float64) if "ZODI" in hdul else np.zeros_like(image)
        wcs = WCS(img_hdr)

    raw_flux = image - zodi

    sigma_clip = SigmaClip(sigma=3.0)
    bkg = Background2D(
        raw_flux,
        box_size=(22, 22),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
        coverage_mask=~np.isfinite(raw_flux),
    )
    flat_flux = raw_flux - bkg.background
    healed_flux = interpolate_replace_nans(flat_flux, Gaussian2DKernel(x_stddev=1.0))
    clean_flux = clean_cosmic_rays(healed_flux, sigma_thresh=7.0)

    return {
        "flux": clean_flux,
        "wcs": wcs,
        "date_obs": img_hdr.get("DATE-OBS", ""),
        "mjd": float(img_hdr.get("MJD-OBS", 0.0)),
    }


def detect_stars(flux: np.ndarray, wcs: WCS, fwhm_px: float = 2.0, snr_threshold: float = 5.0) -> list[dict]:
    """Знаходить центроїди зірок у пікселях і переводить їх у небесні координати (RA, Dec)."""
    mean, median, std = sigma_clipped_stats(flux, sigma=3.0)
    if not np.isfinite(std) or std <= 0:
        return []

    finder = DAOStarFinder(fwhm=fwhm_px, threshold=snr_threshold * std)
    sources = finder(flux - median)
    if sources is None or len(sources) == 0:
        return []

    x_col = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    y_col = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"

    sky_coords = wcs.pixel_to_world(sources[x_col], sources[y_col])
    stars = []
    for idx, row in enumerate(sources):
        stars.append({
            "id": idx + 1,
            "x": float(row[x_col]),
            "y": float(row[y_col]),
            "ra": float(sky_coords[idx].ra.deg),
            "dec": float(sky_coords[idx].dec.deg),
            "flux": float(row["flux"]),
            "peak": float(row["peak"]),
            "sharpness": float(row["sharpness"]),
        })
    return stars


def match_epochs(stars_t1: list[dict], stars_t2: list[dict], wcs_ref: WCS, match_radius_arcsec: float = 3.5) -> list[dict]:
    """Зіставляє зірки між двома епохами та виявляє зміщення або нові/зниклі об'єкти."""
    if not stars_t1:
        return []

    coords1 = wcs_ref.pixel_to_world([s["x"] for s in stars_t1], [s["y"] for s in stars_t1])

    if not stars_t2:
        return [{**s, "status": "epoch1_only", "shift_arcsec": None, "flux_ratio": None} for s in stars_t1]

    coords2 = wcs_ref.pixel_to_world([s["x"] for s in stars_t2], [s["y"] for s in stars_t2])
    idx2, sep2d, _ = coords1.match_to_catalog_sky(coords2)

    catalog = []
    matched_t2_indices = set()

    for i, s1 in enumerate(stars_t1):
        sep_arcsec = float(sep2d[i].arcsec)
        j = int(idx2[i])
        s2 = stars_t2[j]

        if sep_arcsec <= match_radius_arcsec:
            matched_t2_indices.add(j)
            is_moving = sep_arcsec > 1.5
            flux_ratio = (s2["flux"] / s1["flux"]) if s1["flux"] > 0 else None
            catalog.append({
                **s1,
                "x_t2": s2["x"],
                "y_t2": s2["y"],
                "status": "moving" if is_moving else "stationary",
                "shift_arcsec": round(sep_arcsec, 3),
                "flux_ratio": round(flux_ratio, 3) if flux_ratio else None,
            })
        else:
            catalog.append({
                **s1,
                "x_t2": None,
                "y_t2": None,
                "status": "epoch1_only",
                "shift_arcsec": None,
                "flux_ratio": None,
            })

    for j, s2 in enumerate(stars_t2):
        if j not in matched_t2_indices:
            catalog.append({
                "id": len(catalog) + 1,
                "x": s2["x"],
                "y": s2["y"],
                "x_t2": s2["x"],
                "y_t2": s2["y"],
                "ra": s2["ra"],
                "dec": s2["dec"],
                "flux": s2["flux"],
                "peak": s2["peak"],
                "sharpness": s2["sharpness"],
                "status": "epoch2_only",
                "shift_arcsec": None,
                "flux_ratio": None,
            })

    return catalog


def normalize_pair_uint8(img1: np.ndarray, img2: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Нормалізує два вирівняні кадри у спільній шкалі [0..255]."""
    combined_valid = np.concatenate([img1[valid_mask], img2[valid_mask]])
    vmin, vmax = PercentileInterval(99.2).get_limits(combined_valid)
    vmin = min(vmin, -0.02 * (vmax - vmin))

    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.04), clip=True)
    f1 = np.nan_to_num(img1, nan=vmin)
    f2 = np.nan_to_num(img2, nan=vmin)

    return (norm(f1) * 255).astype(np.uint8), (norm(f2) * 255).astype(np.uint8)


def process_blink_pair(fits_path1: Path, fits_path2: Path, wave_um: float | None = None) -> dict:
    """Повний цикл: репроєкція Кадру 2 на WCS Кадру 1, нормалізація та трекінг."""
    ep1 = load_spherex_cutout(fits_path1)
    ep2 = load_spherex_cutout(fits_path2)

    aligned_flux2, footprint = reproject_interp(
        (ep2["flux"], ep2["wcs"]),
        ep1["wcs"],
        shape_out=ep1["flux"].shape,
        order="bilinear",
    )

    valid_mask = np.isfinite(ep1["flux"]) & np.isfinite(aligned_flux2) & (footprint > 0.5)
    inner_mask = binary_erosion(valid_mask, iterations=3)

    flux1_common = np.where(inner_mask, ep1["flux"], np.nan)
    flux2_common = np.where(inner_mask, aligned_flux2, np.nan)

    stars_t1 = detect_stars(flux1_common, ep1["wcs"], fwhm_px=2.0, snr_threshold=5.0)
    stars_t2 = detect_stars(flux2_common, ep1["wcs"], fwhm_px=2.0, snr_threshold=5.0)
    catalog = match_epochs(stars_t1, stars_t2, ep1["wcs"], match_radius_arcsec=3.5)

    u1, u2 = normalize_pair_uint8(ep1["flux"], aligned_flux2, valid_mask)
    pixel_scale_arcsec = float(np.mean(proj_plane_pixel_scales(ep1["wcs"])) * 3600.0)

    return {
        "frame_t1": u1,
        "frame_t2": u2,
        "valid_mask": valid_mask,
        "diff": flux2_common - flux1_common,
        "wcs": ep1["wcs"],
        "meta": {
            "wave_um": wave_um,
            "date_t1": ep1["date_obs"],
            "date_t2": ep2["date_obs"],
            "delta_days": round(ep2["mjd"] - ep1["mjd"], 3),
            "pixel_scale_arcsec": round(pixel_scale_arcsec, 3),
            "shape": list(u1.shape),
        },
        "catalog": catalog,
    }


def build_spectral_cube(slices_meta: list[dict]) -> dict:
    """Збирає вирівняний 3D-куб (N_lambda, H, W) із серії різнохвильових кадрів одного регіону."""
    if not slices_meta:
        raise ValueError("Список зрізів порожній.")

    ref_ep = load_spherex_cutout(slices_meta[0]["path"])
    ref_wcs = ref_ep["wcs"]
    shape = ref_ep["flux"].shape

    cube_layers = []
    wavelengths = []

    for item in slices_meta:
        ep = load_spherex_cutout(item["path"])
        aligned, _ = reproject_interp((ep["flux"], ep["wcs"]), ref_wcs, shape_out=shape, order="bilinear")
        cube_layers.append(aligned)
        wavelengths.append(item["wave_um"])

    cube = np.stack(cube_layers, axis=0)  # shape: (N_lambda, H, W)
    return {
        "cube": cube,
        "wavelengths_um": np.array(wavelengths),
        "wcs": ref_wcs,
    }


def extract_pixel_spectrum(cube_data: dict, x: int, y: int, aperture_radius: int = 1) -> list[dict]:
    """Витягує спектр (потік vs довжина хвилі) у точці (x, y) із 3D-куба."""
    cube = cube_data["cube"]
    waves = cube_data["wavelengths_um"]
    _, h, w = cube.shape

    x0, x1 = max(0, x - aperture_radius), min(w, x + aperture_radius + 1)
    y0, y1 = max(0, y - aperture_radius), min(h, y + aperture_radius + 1)

    spectrum = []
    for k, lam in enumerate(waves):
        patch = cube[k, y0:y1, x0:x1]
        val = float(np.nansum(patch)) if np.any(np.isfinite(patch)) else None
        spectrum.append({"wave_um": float(lam), "flux": round(val, 4) if val is not None else None})
    return spectrum


def save_debug_visuals(result: dict, cube_data: dict | None = None, out_dir: Path = Path("./debug_out")) -> None:
    """Зберігає контрольний графік блінку, GIF-анімацію та зрізи 3D-куба."""
    out_dir.mkdir(parents=True, exist_ok=True)
    u1, u2 = result["frame_t1"], result["frame_t2"]
    catalog = result["catalog"]
    wave_str = f" | λ ≈ {result['meta']['wave_um']} мкм" if result["meta"].get("wave_um") else ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
    axes[0].imshow(u1, cmap="gray", origin="lower")
    axes[0].set_title(f"Epoch 1{wave_str}\n{result['meta']['date_t1']}")

    axes[1].imshow(u2, cmap="gray", origin="lower")
    axes[1].set_title(f"Epoch 2 (+{result['meta']['delta_days']} d)\n{result['meta']['date_t2']}")

    diff = np.nan_to_num(result["diff"], nan=0.0)
    lim = np.percentile(np.abs(diff[result["valid_mask"]]), 98)
    axes[2].imshow(diff, cmap="coolwarm", origin="lower", vmin=-lim, vmax=lim)
    axes[2].set_title("Difference (T2 - T1) & Catalog")

    colors = {"stationary": "#00ff88", "moving": "#ff3355", "epoch1_only": "#ffaa00", "epoch2_only": "#00ccff"}
    for star in catalog:
        c = colors.get(star["status"], "white")
        axes[2].plot(star["x"], star["y"], marker="o", markersize=9, markerfacecolor="none", markeredgecolor=c, lw=1.4)

    for ax in axes:
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

    plt.tight_layout()
    plt.savefig(out_dir / "blink_check.png")
    plt.close(fig)

    scale = 4
    pil1 = Image.fromarray(np.flipud(u1)).resize((u1.shape[1] * scale, u1.shape[0] * scale), Image.Resampling.NEAREST)
    pil2 = Image.fromarray(np.flipud(u2)).resize((u2.shape[1] * scale, u2.shape[0] * scale), Image.Resampling.NEAREST)
    pil1.save(out_dir / "blink.gif", save_all=True, append_images=[pil2], duration=450, loop=0)

    if cube_data is not None:
        cube = cube_data["cube"]
        waves = cube_data["wavelengths_um"]
        n_show = min(6, len(waves))
        idxs = np.linspace(0, len(waves) - 1, n_show, dtype=int)
        fig_c, axes_c = plt.subplots(1, n_show, figsize=(3 * n_show, 3.2), dpi=130)
        for ax, idx in zip(axes_c, idxs):
            layer = np.nan_to_num(cube[idx], nan=0.0)
            vmin, vmax = np.percentile(layer, [5, 99])
            ax.imshow(layer, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"λ = {waves[idx]:.2f} мкм")
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / "spectral_cube_slices.png")
        plt.close(fig_c)

    print(f"Усі контрольні файли збережено у {out_dir.resolve()}")

def build_matched_dual_cubes(
    dual_slices: dict,
    max_wave_diff_um: float = 0.04,
    min_wave_step_um: float = 0.05,
) -> dict:
    """
    Збирає два 3D-куби (N_lambda, H, W) для Епохи 1 та Епохи 2,
    узгоджені по WCS та унікальних довжинах хвиль.
    """
    s1_list = dual_slices["epoch1"]["slices"]
    s2_list = dual_slices["epoch2"]["slices"]

    if not s1_list or not s2_list:
        raise ValueError("Недостатньо зрізів в одній із епох для побудови куба.")

    candidate_pairs = []
    for i, s1 in enumerate(s1_list):
        for j, s2 in enumerate(s2_list):
            dlam = abs(s1["wave_um"] - s2["wave_um"])
            candidate_pairs.append((dlam, i, j))
    candidate_pairs.sort(key=lambda x: x[0])

    def _select_pairs(thresh: float) -> list[tuple[dict, dict]]:
        used_i, used_j, pairs = set(), set(), []
        for dlam, i, j in candidate_pairs:
            if dlam > thresh:
                break
            if i in used_i or j in used_j:
                continue
            avg_w = (s1_list[i]["wave_um"] + s2_list[j]["wave_um"]) / 2.0
            if any(abs(avg_w - (p[0]["wave_um"] + p[1]["wave_um"]) / 2.0) < min_wave_step_um for p in pairs):
                continue
            used_i.add(i)
            used_j.add(j)
            pairs.append((s1_list[i], s2_list[j]))
        pairs.sort(key=lambda p: (p[0]["wave_um"] + p[1]["wave_um"]) / 2.0)
        return pairs

    matched_pairs = _select_pairs(max_wave_diff_um)
    if len(matched_pairs) < 4:
        matched_pairs = _select_pairs(0.55)

    if not matched_pairs:
        raise RuntimeError("Не вдалося зіставити жодної спільної довжини хвилі між епохами.")

    loaded_ep1 = [load_spherex_cutout(s1["path"]) for s1, _ in matched_pairs]
    best_ref = max(loaded_ep1, key=lambda ep: int(np.isfinite(ep["flux"]).sum()))
    ref_wcs = best_ref["wcs"]
    shape = best_ref["flux"].shape

    cube_t1_layers, cube_t2_layers, wavelengths_um, wave_details = [], [], [], []

    for s1, s2 in matched_pairs:
        ep1 = load_spherex_cutout(s1["path"])
        ep2 = load_spherex_cutout(s2["path"])

        aligned1, _ = reproject_interp((ep1["flux"], ep1["wcs"]), ref_wcs, shape_out=shape, order="bilinear")
        aligned2, _ = reproject_interp((ep2["flux"], ep2["wcs"]), ref_wcs, shape_out=shape, order="bilinear")

        cube_t1_layers.append(aligned1)
        cube_t2_layers.append(aligned2)

        avg_lam = round((s1["wave_um"] + s2["wave_um"]) / 2.0, 4)
        wavelengths_um.append(avg_lam)
        wave_details.append({
            "wave_um": avg_lam,
            "wave_t1_um": s1["wave_um"],
            "wave_t2_um": s2["wave_um"],
            "delta_wave_um": round(abs(s2["wave_um"] - s1["wave_um"]), 4),
            "date_t1": s1["date_obs"],
            "date_t2": s2["date_obs"],
        })

    cube_t1 = np.stack(cube_t1_layers, axis=0)
    cube_t2 = np.stack(cube_t2_layers, axis=0)
    delta_days = round(dual_slices["epoch2"]["mjd_mean"] - dual_slices["epoch1"]["mjd_mean"], 2)

    return {
        "cube_t1": cube_t1,
        "cube_t2": cube_t2,
        "wavelengths_um": np.array(wavelengths_um),
        "wave_details": wave_details,
        "wcs": ref_wcs,
        "meta": {
            "epoch1_date": dual_slices["epoch1"]["date_label"],
            "epoch2_date": dual_slices["epoch2"]["date_label"],
            "delta_days": delta_days,
            "cube_shape": list(cube_t1.shape),
        },
    }

def extract_dual_spectrum(dual_cubes: dict, x: int, y: int, aperture_radius: int = 1) -> list[dict]:
    """Витягує потік об'єкта в точці (x, y) по всіх узгоджених довжинах хвиль для обох епох."""
    c1, c2 = dual_cubes["cube_t1"], dual_cubes["cube_t2"]
    waves = dual_cubes["wavelengths_um"]
    _, h, w = c1.shape

    x0, x1 = max(0, x - aperture_radius), min(w, x + aperture_radius + 1)
    y0, y1 = max(0, y - aperture_radius), min(h, y + aperture_radius + 1)

    rows = []
    for k, lam in enumerate(waves):
        p1 = c1[k, y0:y1, x0:x1]
        p2 = c2[k, y0:y1, x0:x1]
        f1 = round(float(np.nansum(p1)), 4) if np.any(np.isfinite(p1)) else None
        f2 = round(float(np.nansum(p2)), 4) if np.any(np.isfinite(p2)) else None
        rows.append({"wave_um": float(lam), "flux_t1": f1, "flux_t2": f2})
    return rows

def _save_png_safe(out_path: Path, data: bytes) -> Path:
    """Зберігає PNG на диск; якщо файл заблокований переглядачем у Windows, пише поруч із суфіксом _new."""
    try:
        out_path.write_bytes(data)
        return out_path
    except OSError:
        alt_path = out_path.with_stem(f"{out_path.stem}_new")
        alt_path.write_bytes(data)
        return alt_path
    
def save_side_by_side_cubes(dual_cubes: dict, out_dir: Path = Path("./debug_out"), max_rows: int = 8) -> None:
    """Рендерить side-by-side порівняння зрізів обох кубів на однакових довжинах хвиль."""
    out_dir.mkdir(parents=True, exist_ok=True)
    c1 = dual_cubes["cube_t1"]
    c2 = dual_cubes["cube_t2"]
    details = dual_cubes["wave_details"]
    meta = dual_cubes["meta"]

    n_total = len(details)
    idxs = np.linspace(0, n_total - 1, min(max_rows, n_total), dtype=int)
    n_rows = len(idxs)

    fig, axes = plt.subplots(n_rows, 2, figsize=(8.5, 3.2 * n_rows), dpi=135, squeeze=False)

    for row_idx, k in enumerate(idxs):
        info = details[k]
        s1 = c1[k]
        s2 = c2[k]

        valid = np.concatenate([s1[np.isfinite(s1)], s2[np.isfinite(s2)]])
        if valid.size > 0:
            vmin, vmax = np.percentile(valid, [3, 99.2])
            if vmax <= vmin:
                vmax = vmin + 1e-3
        else:
            vmin, vmax = 0.0, 1.0

        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.05), clip=True)

        ax1, ax2 = axes[row_idx, 0], axes[row_idx, 1]
        ax1.imshow(norm(np.nan_to_num(s1, nan=vmin)), cmap="gray", origin="lower")
        ax1.set_title(
            f"Epoch 1 ({meta['epoch1_date']}) | λ = {info['wave_t1_um']:.3f} мкм",
            fontsize=9.5,
        )
        ax1.set_ylabel(f"λ ≈ {info['wave_um']:.2f} мкм\nY (px)", fontsize=9)

        ax2.imshow(norm(np.nan_to_num(s2, nan=vmin)), cmap="gray", origin="lower")
        ax2.set_title(
            f"Epoch 2 ({meta['epoch2_date']}, +{meta['delta_days']} d) | λ = {info['wave_t2_um']:.3f} мкм",
            fontsize=9.5,
        )

        for ax in (ax1, ax2):
            ax.set_xlabel("X (px)", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    out_path = _save_png_safe(out_dir.resolve() / "cubes_side_by_side.png", buf.getvalue())
    print(f"Збережено side-by-side зрізи кубів: {out_path}", flush=True)

def save_false_color_rgb_comparison(
    dual_cubes: dict,
    out_dir: str | Path = "debug_out",
    target_rgb_um: tuple[float, float, float] = (4.15, 2.00, 1.00),
) -> Path:
    """Будує калібрований астрономічний False-Color RGB для двох епох + часовий композит змін."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    waves = np.asarray(dual_cubes["wavelengths_um"], dtype=np.float64)
    c1 = dual_cubes["cube_t1"]
    c2 = dual_cubes["cube_t2"]
    meta = dual_cubes["meta"]

    # Якщо шарів >= 3, беремо найдовшу (R), середню (G) та найкоротшу (B) хвилі; інакше найближчі
    if len(waves) >= 3:
        idx_r = len(waves) - 1
        idx_g = len(waves) // 2
        idx_b = 0
    else:
        idx_r = int(np.argmin(np.abs(waves - target_rgb_um[0])))
        idx_g = int(np.argmin(np.abs(waves - target_rgb_um[1])))
        idx_b = int(np.argmin(np.abs(waves - target_rgb_um[2])))
    ch_indices = (idx_r, idx_g, idx_b)

    rgb_ep1 = np.zeros((*c1.shape[1:], 3), dtype=np.float32)
    rgb_ep2 = np.zeros((*c2.shape[1:], 3), dtype=np.float32)

    for ch_out, layer_idx in enumerate(ch_indices):
        img1 = c1[layer_idx].astype(np.float64)
        img2 = c2[layer_idx].astype(np.float64)

        # Віднімаємо 10-й перцентиль як фон (для туманностей це краще за медіану, щоб не зрізати газ)
        bg1 = float(np.nanpercentile(img1, 10)) if np.isfinite(img1).any() else 0.0
        bg2 = float(np.nanpercentile(img2, 10)) if np.isfinite(img2).any() else 0.0
        sub1 = img1 - bg1
        sub2 = img2 - bg2

        combined_valid = np.concatenate([sub1[np.isfinite(sub1)], sub2[np.isfinite(sub2)]])
        vmin = float(np.percentile(combined_valid, 1.0))
        vmax = float(np.percentile(combined_valid, 99.4))
        if vmax <= vmin:
            vmax = vmin + 1.0

        # ImageNormalize застосовує спочатку інтервал [vmin, vmax], а потім AsinhStretch
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=0.06), clip=True)

        clean1 = np.where(np.isfinite(sub1), sub1, vmin)
        clean2 = np.where(np.isfinite(sub2), sub2, vmin)

        rgb_ep1[..., ch_out] = np.asarray(norm(clean1), dtype=np.float32)
        rgb_ep2[..., ch_out] = np.asarray(norm(clean2), dtype=np.float32)

    blink_rgb = np.stack([rgb_ep2[..., 1], rgb_ep1[..., 1], rgb_ep1[..., 1]], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)

    axes[0].imshow(rgb_ep1, origin="lower")
    axes[0].set_title(
        f"Epoch 1 ({meta['epoch1_date']})\n"
        f"R={waves[idx_r]:.2f}, G={waves[idx_g]:.2f}, B={waves[idx_b]:.2f} мкм",
        fontsize=10,
    )

    axes[1].imshow(rgb_ep2, origin="lower")
    axes[1].set_title(
        f"Epoch 2 ({meta['epoch2_date']}, +{meta['delta_days']} d)\n"
        f"R={waves[idx_r]:.2f}, G={waves[idx_g]:.2f}, B={waves[idx_b]:.2f} мкм",
        fontsize=10,
    )

    axes[2].imshow(blink_rgb, origin="lower")
    axes[2].set_title(
        f"Change Map (λ ≈ {waves[idx_g]:.2f} мкм)\n"
        "Cyan = Epoch 1 | Red = Epoch 2 | White = Static",
        fontsize=10,
    )

    for ax in axes:
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)

    out_path = _save_png_safe(out_dir.resolve() / "rgb_false_color_comparison.png", buf.getvalue())
    print(f"Збережено калібрований RGB-композит: {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    TEST_RA, TEST_DEC = 161.26500, -59.68440

    dual_slices = fetch_two_epoch_cube_slices(TEST_RA, TEST_DEC, size_deg=0.15, per_band=6)
    dual_cubes = build_matched_dual_cubes(dual_slices, max_wave_diff_um=0.035)
    save_false_color_rgb_comparison(dual_cubes)

    print("\n=== Метадані пари 3D-кубів ===")
    print(dual_cubes["meta"])
    print("\nУзгоджені спектральні шари:")
    for idx, w in enumerate(dual_cubes["wave_details"]):
        print(
            f"  Шар [{idx:2d}]: λ ≈ {w['wave_um']:.4f} мкм "
            f"(T1={w['wave_t1_um']:.4f}, T2={w['wave_t2_um']:.4f}, Δλ={w['delta_wave_um']:.4f})"
        )

    # Перевіряємо детекцію зірок на першому спільному зрізі та виводимо спектр найяскравішої
    stars = detect_stars(dual_cubes["cube_t1"][0], dual_cubes["wcs"], fwhm_px=2.0, snr_threshold=5.0)
    if stars:
        brightest = max(stars, key=lambda s: s["flux"])
        spec = extract_dual_spectrum(dual_cubes, int(round(brightest["x"])), int(round(brightest["y"])))
        print(f"\nСпектр зірки (x={brightest['x']:.1f}, y={brightest['y']:.1f}) по двох епохах:")
        for pt in spec:
            print(f"  λ={pt['wave_um']:.3f} мкм -> F1={pt['flux_t1']} | F2={pt['flux_t2']}")

    save_side_by_side_cubes(dual_cubes)