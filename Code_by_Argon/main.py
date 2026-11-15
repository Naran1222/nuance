"""
Точка входу для кроків 1-3 (та опціонально кроку 3б — кольорових композитів):
  пошук знімків (IRSA TAP/sync -> SIAv2 -> AWS S3) -> 2D Round-Robin відбір у межах ліміту ->
  завантаження вирізок (IRSA / AWS S3) -> вирівнювання за WCS -> PNG та кольорові композити.

Запуск з терміналу:
    python main.py                                           (M101, смуга D2, 100 кадрів)
    python main.py --target "M101" --size 0.3 --bands D1,D3,D5 --limit 100 --npix 512 --color
    python main.py --target "NGC 253" --band D3 --limit 30
    python main.py --target "210.80227 54.34895" --size 0.14 --limit 0   (0 = усі кадри)
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence, Union

from pipeline.catalog import (
    DEFAULT_BAND,
    find_exposures,
    normalize_band,
    normalize_bands,
    release_of,
    resolve_target,
)
from pipeline.cutouts import (
    DEFAULT_WEB_LIMIT,
    WEB_LIMIT_OPTIONS,
    frames_summary,
    load_frames,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"


def validate_limit(limit: int, strict_web_presets: bool = False) -> int:
    val = int(limit)
    if strict_web_presets and val not in WEB_LIMIT_OPTIONS:
        raise ValueError(
            f"Недопустимий ліміт {val}. На сайті дозволені лише значення: {WEB_LIMIT_OPTIONS}."
        )
    if val < 0:
        raise ValueError("Ліміт кадрів не може бути від'ємним.")
    return val


def run(
    target="M101",
    size_deg=0.1,
    band: Optional[Union[str, Sequence[str]]] = DEFAULT_BAND,
    limit: int = DEFAULT_WEB_LIMIT,
    workers: int = 6,
    use_cache: bool = True,
    render: bool = True,
    npix: int = 256,
    fov_deg: Optional[float] = None,
    subtract_zodi: bool = True,
    mask_outliers: bool = False,
    order: str = "bilinear",
    fmt: str = "png",
    quality: int = 90,
    min_coverage: float = 0.98,
    bands: Optional[Union[str, Sequence[str]]] = None,
    color: bool = False,
    max_epoch_days: Optional[float] = None,
    strict_web_presets: bool = False,
):
    """Повертає (таблиця знімків, список кадрів, папка з результатом кроку 3 або None)."""
    limit = validate_limit(limit, strict_web_presets=strict_web_presets)
    coord = resolve_target(target)
    raw_bands = bands if bands is not None else band
    selected_bands = normalize_bands(raw_bands)
    single_band_name = selected_bands[0] if selected_bands and len(selected_bands) == 1 else (
        normalize_band(raw_bands) if isinstance(raw_bands, str) and "," not in raw_bands else None
    )
    bands_label = ", ".join(b.replace("SPHEREx-", "") for b in selected_bands) if selected_bands else "ALL"
    print(f"Об'єкт: {target}  ->  RA={coord.ra.deg:.5f}°, Dec={coord.dec.deg:+.5f}°  |  Смуги: {bands_label}")

    print("\nКрок 1. Пошук знімків")
    table = find_exposures(coord, size_deg=size_deg, bands=selected_bands, use_cache=use_cache)
    if len(table) == 0:
        print("Знімків не знайдено.")
        return table, [], None
    print(f"  Усього знімків: {len(table)}  (MJD від {table['mjd'][0]:.2f} до {table['mjd'][-1]:.2f})")
    releases = Counter(release_of(str(u)) for u in table["uri"])
    print("  Релізи даних: " + ", ".join(f"{k}: {v}" for k, v in sorted(releases.items())))

    print("\nКрок 2. Завантаження та читання вирізок")
    frames = load_frames(table, coord, limit=limit, workers=workers)
    summary = frames_summary(frames)
    summary.pprint(max_lines=-1, max_width=-1)
    OUT_DIR.mkdir(exist_ok=True)

    if not render or not frames:
        summary.write(OUT_DIR / "frames_summary.csv", format="ascii.csv", overwrite=True)
        return table, frames, None

    print("\nКрок 3. Вирівнювання за WCS та PNG")
    from pipeline.render import render_all
    tag = f"n{limit or 'all'}_px{npix}_{order}"
    if not subtract_zodi:
        tag += "_nozodi"
    if mask_outliers:
        tag += "_mo"
    if fov_deg:
        tag += f"_fov{fov_deg:g}"
    out_dir = render_all(
        frames,
        coord,
        size_deg,
        band=single_band_name,
        bands=selected_bands,
        npix=npix,
        fov_deg=fov_deg,
        subtract_zodi=subtract_zodi,
        mask_outliers=mask_outliers,
        order=order,
        fmt=fmt,
        quality=quality,
        tag=tag,
        min_coverage=min_coverage,
    )
    summary.write(out_dir / "frames_summary.csv", format="ascii.csv", overwrite=True)

    if color:
        print("\nКрок 3б. Кольоровий композит за обраними смугами")
        from pipeline.compose import build_color
        try:
            color_dir = build_color([out_dir], fmt=fmt, quality=quality, max_epoch_days=max_epoch_days)
            print(f"  Кольоровий результат: {color_dir}")
        except Exception as exc:
            print(f"  [compose] не вдалося побудувати кольоровий композит: {exc}")

    print(f"\nГотово. Результат: {out_dir}")
    print(f"Для швидкої візуальної перевірки відкрий: {out_dir / 'contact_sheet.png'}")
    return table, frames, out_dir


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="SPHEREx: кроки 1-3 + кольорові композити")
    parser.add_argument("--target", default="M101", help="назва об'єкта або 'RA Dec' у градусах")
    parser.add_argument("--size", type=float, default=0.15, help="розмір вирізки, градуси")
    parser.add_argument("--band", default="D2", help="одна або декілька смуг D1..D6 (наприклад 'D1,D3,D5') або 'all'")
    parser.add_argument("--bands", default=None, help="довільний список смуг через кому (має пріоритет над --band)")
    parser.add_argument("--limit", type=int, default=DEFAULT_WEB_LIMIT,
                        help=f"максимальна кількість кадрів (пресети сайту: {WEB_LIMIT_OPTIONS}; 0 = усі)")
    parser.add_argument("--strict-web-limit", action="store_true",
                        help="дозволяти лише фіксовані веб-пресети ліміту (30, 60, 100, 180)")
    parser.add_argument("--color", action="store_true",
                        help="побудувати кольоровий композит виключно за обраними смугами")
    parser.add_argument("--workers", type=int, default=6, help="кількість паралельних завантажень")
    parser.add_argument("--no-cache", action="store_true", help="ігнорувати кеш списку знімків")
    parser.add_argument("--no-render", action="store_true", help="зупинитись після кроку 2")
    parser.add_argument("--npix", type=int, default=256, help="розмір вихідної картинки, пікселів")
    parser.add_argument("--fov", type=float, default=None,
                        help="поле зору результату, градуси (за замовчуванням size/sqrt(2))")
    parser.add_argument("--order", default="bilinear", choices=["bilinear", "bicubic"],
                        help="інтерполяція: bilinear (без ореолів) або bicubic")
    parser.add_argument("--format", default="png", choices=["png", "webp", "jpeg"],
                        help="формат кадрів: png (без втрат), webp або jpeg (менші файли)")
    parser.add_argument("--quality", type=int, default=90, help="якість для webp/jpeg, 1-100")
    parser.add_argument("--min-coverage", type=float, default=0.98,
                        help="відкидати кадри, де покриття поля нижче цієї частки (кадри біля краю детектора)")
    parser.add_argument("--no-zodi", action="store_true", help="не віднімати зодіакальне світло")
    parser.add_argument("--mask-outliers", action="store_true", help="маскувати піксели-викиди (біт OUTLIER)")
    args = parser.parse_args()

    run(
        target=args.target,
        size_deg=args.size,
        band=args.band,
        limit=args.limit,
        workers=args.workers,
        use_cache=not args.no_cache,
        render=not args.no_render,
        npix=args.npix,
        fov_deg=args.fov,
        subtract_zodi=not args.no_zodi,
        mask_outliers=args.mask_outliers,
        order=args.order,
        fmt=args.format,
        quality=args.quality,
        min_coverage=args.min_coverage,
        bands=args.bands,
        color=args.color,
        strict_web_presets=args.strict_web_limit,
    )


if __name__ == "__main__":
    main()