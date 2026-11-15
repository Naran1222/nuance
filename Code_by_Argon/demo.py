"""
Демонстраційний набір: БІЛЬШЕ поле зору, БІЛЬШЕ кадрів, більша картинка + кольорові знімки за епохами.

    python demo.py                          усі цілі зі списку TARGETS
    python demo.py M101                     лише одна ціль
    python demo.py M44 --bands D1,D3,D5     обрані смуги детектора
    python demo.py M44 --limit 60           загальний ліміт кадрів (пресети сайту: 30, 60, 100, 180; 0 = усі)
    python demo.py --no-color               без кольорових композитів

Використовує чесний 2D Round-Robin розподіл ліміту кадрів між обраними смугами та епохами,
будує хронологічний набір покадрових зображень і кольорові композити тільки з обраних смуг.
"""
import argparse
import sys
import time

from main import run
from pipeline.catalog import normalize_bands
from pipeline.compose import build_color
from pipeline.cutouts import DEFAULT_WEB_LIMIT, WEB_LIMIT_OPTIONS

# (назва для Sesame, розмір вирізки в градусах, макс. тривалість епохи в добах або None, опис поля)
TARGETS = [
    ("M44", 0.4, None, "скупчення Ясла: багато зір; лежить майже на екліптиці, де часто пролітають астероїди"),
    ("M101", 0.5, None, "галактика-вертушка повністю (спіральні рукави та яскраве ядро)"),
    ("M42", 0.4, None, "туманність Оріона: складна газопилова структура"),
    ("NGC 6543", 0.3, 14, "туманність Котяче Око майже на північному полюсі екліптики: поле знімають часто, "
                          "тому епохи обмежені 14 добами"),
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="SPHEREx: демонстраційний набір")
    parser.add_argument("name", nargs="?", help="назва цілі зі списку TARGETS (за замовчуванням усі)")
    parser.add_argument("--limit", type=int, default=DEFAULT_WEB_LIMIT,
                        help=f"загальний ліміт кадрів на ціль (пресети: {WEB_LIMIT_OPTIONS}, 0 = усі)")
    parser.add_argument("--bands", default="D2,D3",
                        help="смуги детектора через кому (наприклад D1,D3,D5 або D2,D3 або all)")
    parser.add_argument("--npix", type=int, default=512, help="розмір картинки, пікселів")
    parser.add_argument("--order", default="bilinear", choices=["bilinear", "bicubic"],
                        help="інтерполяція (bilinear за замовчуванням — без ореолів навколо зір)")
    parser.add_argument("--format", default="png", choices=["png", "webp", "jpeg"])
    parser.add_argument("--min-coverage", type=float, default=0.98,
                        help="відкидати кадри біля краю детектора (покриття поля нижче цієї частки)")
    parser.add_argument("--no-color", action="store_true", help="не будувати кольорові композити")
    parser.add_argument("--max-epoch-days", type=float, default=None,
                        help="обмежити тривалість епохи (за замовчуванням береться зі списку TARGETS)")
    args = parser.parse_args()

    targets = [t for t in TARGETS if args.name is None or t[0].lower() == args.name.lower()]
    if not targets:
        print(f"Цілі '{args.name}' немає у списку TARGETS. Доступні: {', '.join(t[0] for t in TARGETS)}")
        return

    selected_bands = normalize_bands(args.bands)
    bands_desc = ", ".join(b.replace("SPHEREx-", "") for b in selected_bands) if selected_bands else "ALL (D1..D6)"

    results = []
    for name, size, target_max_epoch, note in targets:
        print("\n" + "=" * 78)
        print(f"{name}: {note}")
        print(f"Обрані смуги: {bands_desc} | Ліміт кадрів: {args.limit or 'усі'}")
        print("=" * 78)
        started = time.time()
        out_dirs, errors = [], []
        try:
            _table, _frames, out_dir = run(
                name,
                size_deg=size,
                bands=selected_bands,
                limit=args.limit,
                npix=args.npix,
                order=args.order,
                fmt=args.format,
                min_coverage=args.min_coverage,
            )
            if out_dir is not None:
                out_dirs.append(out_dir)
        except Exception as exc:
            print(f"\n[ПОМИЛКА] {name}: {type(exc).__name__}: {exc}")
            errors.append(str(exc))

        color_dir = None
        if out_dirs and not args.no_color:
            print(f"\n--- {name}: кольоровий композит за обраними смугами ({bands_desc}) ---")
            try:
                color_dir = build_color(
                    out_dirs,
                    fmt=args.format,
                    max_epoch_days=args.max_epoch_days or target_max_epoch,
                )
            except Exception as exc:
                print(f"\n[ПОМИЛКА] композит {name}: {type(exc).__name__}: {exc}")
                errors.append(f"композит: {exc}")
        results.append((name, out_dirs, color_dir, time.time() - started, errors))

    print("\n" + "=" * 78)
    print("ПІДСУМОК")
    for name, out_dirs, color_dir, secs, errors in results:
        print(f"  {name}  ({secs:.0f} с)")
        for d in out_dirs:
            print(f"     кадри:   {d / 'contact_sheet.png'}")
        if color_dir is not None:
            print(f"     колір:   {color_dir / 'overview.png'}")
        for e in errors:
            print(f"     ПОМИЛКА: {e}")


if __name__ == "__main__":
    main()