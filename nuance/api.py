import json
import subprocess
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="Nuance Sky Explorer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "Code_by_Argon"
SCRIPT_PATH = BACKEND_DIR / "main.py"
BASE_OUT_DIR = BACKEND_DIR / "out"


def get_latest_run_dir() -> Path:
    """Знаходить найсвіжішу папку результатів у Code_by_Argon/out"""
    if not BASE_OUT_DIR.exists():
        raise HTTPException(status_code=500, detail="Папка out не існує")
    
    # Шукаємо всі compose.json усередині out та підпапок
    compose_files = list(BASE_OUT_DIR.rglob("compose.json"))
    if not compose_files:
        raise HTTPException(status_code=500, detail="compose.json не знайдено в папці out")
    
    # Беремо найсвіжіший файл за часом зміни
    latest_compose = max(compose_files, key=lambda p: p.stat().st_mtime)
    return latest_compose.parent


@app.get("/api/get-anomaly")
async def get_anomaly(
    target: str = "210.802429, 54.34875",
    band: str = "D1, D3, D5",
    limit: str = "60"
):
    # Прибираємо зайві пробіли: "D1, D3, D5" -> "D1,D3,D5"
    clean_bands = ",".join(b.strip() for b in band.split(",") if b.strip())

    cmd = [
        sys.executable, str(SCRIPT_PATH),
        "--target", target,
        "--bands", clean_bands,
        "--limit", limit,
        "--size", "0.35",
        "--color",
        "--format", "png"
    ]

    try:
        subprocess.run(cmd, cwd=str(BACKEND_DIR), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Помилка скрипту: {e.stderr}")

    run_dir = get_latest_run_dir()
    with open(run_dir / "compose.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    ra = data.get("target", {}).get("ra_deg", 0)
    dec = data.get("target", {}).get("dec_deg", 0)
    bands_list = data.get("query", {}).get("bands", [])
    epochs = data.get("epochs", [])

    return {
        "name": f"Target {target}",
        "coordinates": f"RA {ra:.4f}, DEC {dec:.4f}",
        "band": ", ".join(bands_list) if bands_list else band,
        "total_frames": len(epochs),
        "epochs": epochs
    }


@app.get("/api/frame/{frame_id}")
async def get_frame_image(frame_id: int, is_color: bool = False):
    run_dir = get_latest_run_dir()
    with open(run_dir / "compose.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    epochs = data.get("epochs", [])
    if 0 <= frame_id < len(epochs):
        rgb_filename = epochs[frame_id]["file"]
        
        if is_color:
            target_file = run_dir / rgb_filename
            if target_file.exists():
                return FileResponse(target_file)
        else:
            base_name = rgb_filename.replace("_rgb", "")
            for bw_file in [
                run_dir / rgb_filename.replace("_rgb", "_bw"),
                run_dir / rgb_filename.replace("_rgb", "_gray"),
                run_dir / base_name,
            ]:
                if bw_file.exists():
                    return FileResponse(bw_file)

    raise HTTPException(status_code=404, detail=f"Кадр {frame_id} не знайдено")