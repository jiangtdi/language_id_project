"""Run the local web demo for OWSM language identification."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.inference_service import language_id_service

FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

app = FastAPI(title="OWSM Language Identification System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model_on_startup():
    language_id_service.load()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info():
    try:
        return language_id_service.model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        return language_id_service.predict_bytes(content, suffix=suffix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def index():
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index)
    return HTMLResponse(
        """
        <main style="font-family:system-ui;padding:32px;line-height:1.7">
          <h1>FastAPI 后端已启动</h1>
          <p>React 前端请在另一个终端运行：</p>
          <pre>cd frontend
npm install
npm run dev</pre>
          <p>然后打开 <a href="http://127.0.0.1:5173">http://127.0.0.1:5173</a></p>
          <p>如果已经执行 <code>npm run build</code>，FastAPI 会自动托管 <code>frontend/dist</code>。</p>
        </main>
        """
    )


if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
