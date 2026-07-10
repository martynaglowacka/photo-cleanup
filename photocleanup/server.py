import io
import shutil
import tempfile
from pathlib import Path
import PIL.Image
import PIL.ImageOps
import pillow_heif
import uvicorn
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from .clustering import cluster
from .hashing import hash_folder
from .scoring import allocate, reasons, score_cluster

pillow_heif.register_heif_opener()  # let PIL open HEIC

DEFAULT_THRESHOLD = 15
THUMB_SIZE = (400, 400)
STATIC = Path(__file__).resolve().parent / "static"
WORKDIR = Path(tempfile.gettempdir()) / "photocleanup_session"

app = FastAPI()
STATE = {"clusters": [], "paths": {}, "thumbs": {}, "source": None, "mode": "upload"}
def _save_upload(upload, dest_dir):
    name = Path(upload.filename).name
    raw = upload.file.read()
    if Path(name).suffix.lower() in (".heic", ".heif"):  # convert so cv2 can read it
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(io.BytesIO(raw))).convert("RGB")
        img.save(dest_dir / (Path(name).stem + ".jpg"), "JPEG", quality=92)
    else:
        (dest_dir / name).write_bytes(raw)


def build_clusters(folder, total_keep, threshold=DEFAULT_THRESHOLD):
    items = hash_folder(folder)
    by_name = {p.name: p for p, _ in items}
    groups = cluster([(p.name, h) for p, h in items], threshold)
    multi = [g for g in groups if len(g) > 1]
    singles = [g for g in groups if len(g) == 1]
    keeps_per = allocate([len(g) for g in multi], total_keep)
    out = []
    for members, k in zip(multi, keeps_per):
        rows = score_cluster([by_name[m] for m in members])
        photos = [
            {"name": r["path"].name, "score": round(r["score"], 3),
             "recommendation": "keep" if i < k else "delete",
             "reasons": reasons(r["signals"]) if i < k else []}
            for i, r in enumerate(rows)
        ]
        out.append({"size": len(members), "photos": photos})
    for g in singles:
        out.append({"size": 1, "photos": [
            {"name": g[0], "score": None, "recommendation": "maybe", "reasons": []}]})
    return by_name, out


def _load(folder, total_keep, source, mode):
    paths, clusters = build_clusters(folder, total_keep=total_keep)
    STATE.update(paths=paths, clusters=clusters, source=source, mode=mode,
                 thumbs={n: _thumb_bytes(p) for n, p in paths.items()})


def _thumb_bytes(path):
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert("RGB")
    img.thumbnail(THUMB_SIZE)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()



@app.get("/")
def index():
    return FileResponse(STATIC / "upload.html")


@app.get("/results")
def results():
    return FileResponse(STATIC / "results.html")


@app.post("/api/analyze")
def analyze(keep: int = Form(4), files: list[UploadFile] = File(...)):
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    for f in files:
        _save_upload(f, WORKDIR)
    _load(WORKDIR, keep, source=None, mode="upload")
    return JSONResponse({"photos": len(STATE["paths"]), "clusters": len(STATE["clusters"])})


@app.post("/api/analyze_folder")
def analyze_folder(payload: dict = Body(...)):
    folder = Path(payload["path"]).expanduser()
    if not folder.is_dir():
        return JSONResponse({"error": f"not a folder: {folder}"}, status_code=400)
    _load(folder, int(payload.get("keep", 4)), source=folder, mode="folder")
    return JSONResponse({"photos": len(STATE["paths"]), "clusters": len(STATE["clusters"])})


@app.get("/api/clusters")
def api_clusters():
    return {"clusters": STATE["clusters"], "mode": STATE["mode"], "source": str(STATE["source"] or "")}


@app.get("/thumb/{name}")
def thumb(name: str):
    data = STATE["thumbs"].get(name)
    return Response(data, media_type="image/jpeg") if data else Response(status_code=404)


@app.get("/photo/{name}")
def photo(name: str):
    path = STATE["paths"].get(name)
    if path is None:
        return Response(status_code=404)
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.post("/api/approve")
def approve(payload: dict = Body(...)):
    decisions = payload["decisions"]  # {name: "keep" | "delete"}
    if payload.get("action") == "in_place" and STATE["source"] is not None:
        dest = STATE["source"] / "delete_candidates"
        dest.mkdir(exist_ok=True)
        moved = 0
        for name, decision in decisions.items():
            src = STATE["paths"].get(name)
            if decision != "keep" and src is not None and src.exists():
                shutil.move(str(src), str(dest / name))  # move, never erase
                moved += 1
        return JSONResponse({"moved": moved, "folder": str(dest)})

    out = Path(tempfile.gettempdir()) / "photocleanup_result"
    if out.exists():
        shutil.rmtree(out)
    (out / "keep").mkdir(parents=True)
    (out / "delete_candidates").mkdir(parents=True)
    for name, decision in decisions.items():
        src = STATE["paths"].get(name)
        if src is None:
            continue
        folder = "keep" if decision == "keep" else "delete_candidates"
        shutil.copy2(src, out / folder / name)
    archive = shutil.make_archive(str(out), "zip", out)
    return FileResponse(archive, filename="photocleanup_result.zip", media_type="application/zip")
def main():
    print("PhotoCleanup -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
