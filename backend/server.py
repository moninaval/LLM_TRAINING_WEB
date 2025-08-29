# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os, json, glob, re, time
from fastapi.responses import FileResponse
from typing import Any, Dict
import os, json, tempfile, pathlib, time
import uvicorn
from fastapi import FastAPI, Body, Query, HTTPException, UploadFile, File, Form

app = FastAPI()

# CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = "../frontend"

# sanity log at startup
if not os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
    print(f"[WARN] index.html not found at {FRONTEND_DIR}")

# re-mount static with html=True (so /static/ serves index.html)
app.mount("/LearnLLMLive", StaticFiles(directory=FRONTEND_DIR, html=True), name="LearnLLMLive")

# ---------- Config ----------
LOG_DIR = os.environ.get("LOG_DIR", "LLM/log/current")  # where trainer writes logs
CFG_PATH = os.environ.get("LOG_DIR", "LLM/log/current/config")  # where trainer writes logs
VIEWS_FILE = "views.json"

# Load existing count if file exists
if os.path.exists(VIEWS_FILE):
    with open(VIEWS_FILE, "r") as f:
        data = json.load(f)
        total_views = data.get("total_views", 0)
else:
    total_views = 0

@app.get("/views")
def get_views():
    global total_views
    total_views += 1
    with open(VIEWS_FILE, "w") as f:
        json.dump({"total_views": total_views}, f)
    return {"total_views": total_views}
# ---------- Training state (simple mock) ----------
training_active = False
current_step = 0
class TrainConfig(BaseModel):
    metrics: List[str] = []
    max_steps: int = 50

# ---------- Small file helpers ----------
def _safe_mtime(path: Optional[str]) -> Optional[float]:
    try:
        return os.path.getmtime(path) if path else None
    except Exception:
        return None

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix="config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, path)  # atomic on same filesystem
    finally:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass

def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

def _read_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    out.extend(obj)
                else:
                    out.append(obj)
            except json.JSONDecodeError:
                # ignore a partial/in-flight line
                continue
    return out

# ---------- Loaders for each artifact ----------
def load_tokenization() -> Tuple[List[dict], Optional[str], Optional[float]]:
    """Return (records, path, mtime). Prefer JSONL."""
    candidates = [
        os.path.join(LOG_DIR, "tokenization.jsonl"),
        os.path.join(LOG_DIR, "tokenization.json"),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith(".jsonl"):
                recs = _read_jsonl(p)
            else:
                obj = _read_json(p)
                recs = obj if isinstance(obj, list) else [obj]
            if recs:
                return recs, p, _safe_mtime(p)
        except Exception:
            continue
    return [], None, None

def load_embeddings() -> Tuple[Optional[dict], Optional[str], Optional[float]]:
    """Return (object, path, mtime). Expected single JSON object."""
    p = os.path.join(LOG_DIR, "embedding.json")
    print("Found ",p)
    if not os.path.exists(p):
        return None, None, None
    try:
        obj = _read_json(p)
        return obj, p, _safe_mtime(p)
    except Exception:
        return None, None, None

def list_decoder_layers() -> List[dict]:
    """Find decoder_layer_*.json and return sorted layer metadata."""
    files = glob.glob(os.path.join(LOG_DIR, "decoder_layer_*.json"))
    layers = []
    for fp in files:
        m = re.search(r"decoder_layer_(\d+)\.json$", fp)
        if not m:
            continue
        li = int(m.group(1))
        layers.append({"layer": li, "file": fp, "updated_unix": _safe_mtime(fp)})
    layers.sort(key=lambda x: x["layer"])
    return layers

def load_decoder_layer(layer_idx: int) -> Tuple[Optional[dict], Optional[str], Optional[float]]:
    p = os.path.join(LOG_DIR, f"decoder_layer_{layer_idx}.json")
    if not os.path.exists(p):
        return None, None, None
    try:
        obj = _read_json(p)
        return obj, p, _safe_mtime(p)
    except Exception:
        return None, None, None

def extract_head_from_layer(layer_obj: dict, head_idx: int) -> Optional[dict]:
    """Return a minimal payload for a single head from a layer JSON."""
    heads = layer_obj.get("heads", [])
    if not heads:
        return None
    # prefer matching by explicit 'head' field; else fallback to index
    for h in heads:
        if isinstance(h, dict) and int(h.get("head", -1)) == int(head_idx):
            return h
    if 0 <= head_idx < len(heads):
        return heads[head_idx]
    return None

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok", "log_dir": LOG_DIR}

# ---------- Training controls (mock) ----------
@app.post("/start_training")
def start_training(config: TrainConfig):
    global training_active, current_step
    training_active = True
    current_step = 0
    # simulate step work
    time.sleep(20)
    return {"status": "started", "config": config.dict()}

@app.get("/training_step")
def training_step():
    global current_step
    if not training_active:
        return {"status": "stopped"}
    # simulate step work
    time.sleep(0.2)
    current_step += 1
    # you can enrich this later with real loss from logs
    import random
    return {
        "status": "ok",
        "step": current_step,
        "loss": round(random.uniform(1.0, 5.0) / (current_step + 1), 4),
    }

@app.post("/stop_training")
def stop_training():
    global training_active
    training_active = False
    return {"status": "stopped"}

# ---------- Live: Tokenization ----------
@app.get("/tokenization")
def get_tokenization():
    print("NAVAL token")
    records, path, mtime = load_tokenization()
    if not records:
        return JSONResponse(
            {"status": "error", "msg": "No tokenization data found", "log_dir": LOG_DIR},
            status_code=404,
        )
    print("NAVAL returned 200")    
    return {"status": "ok", "file": path, "updated_unix": mtime, "records": records}

# ---------- Live: Embeddings ----------
@app.get("/embeddings")
def get_embeddings():
    print("NAVAL embediing")
    obj, path, mtime = load_embeddings()
    if obj is None:
        return JSONResponse(
            {"status": "error", "msg": "No embeddings data found", "log_dir": LOG_DIR},
            status_code=404,
        )
    return {"status": "ok", "file": path, "updated_unix": mtime, "data": obj}

# ---------- Live: Decoder (layers & heads) ----------
@app.get("/decoder")
def decoder_summary():
    print("NAVAL decoder")
    layers = list_decoder_layers()
    if not layers:
        return JSONResponse(
            {"status": "error", "msg": "No decoder layer logs found", "log_dir": LOG_DIR},
            status_code=404,
        )
    return {"status": "ok", "layers": layers, "num_layers": len(layers)}

@app.get("/decoder/layers")
def decoder_layers():
    print("NAVAL decoder 1")
    layers = list_decoder_layers()
    return {"status": "ok", "layers": layers, "num_layers": len(layers)}

@app.get("/decoder/layers/{layer_idx}")
def decoder_layer(layer_idx: int):
    print("NAVAL decoder laye0")
    obj, path, mtime = load_decoder_layer(layer_idx)
    if obj is None:
        return JSONResponse(
            {"status": "error", "msg": f"decoder_layer_{layer_idx}.json not found", "log_dir": LOG_DIR},
            status_code=404,
        )
    # pass-through the layer JSON and add file/meta
    return {"status": "ok", "file": path, "updated_unix": mtime, "data": obj}

@app.get("/decoder/layers/{layer_idx}/heads/{head_idx}")
def decoder_head(layer_idx: int, head_idx: int):
    obj, path, mtime = load_decoder_layer(layer_idx)
    if obj is None:
        return JSONResponse(
            {"status": "error", "msg": f"decoder_layer_{layer_idx}.json not found", "log_dir": LOG_DIR},
            status_code=404,
        )
    head = extract_head_from_layer(obj, head_idx)
    if head is None:
        return JSONResponse(
            {"status": "error", "msg": f"Head {head_idx} not found in layer {layer_idx}"},
            status_code=404,
        )
    header = {
        "layer_index": obj.get("layer_index", layer_idx),
        "sample_index": obj.get("sample_index"),
        "token_index": obj.get("token_index"),
        "num_heads": obj.get("num_heads"),
        "head_dim": obj.get("head_dim"),
        "rope": obj.get("rope"),
        "formula": obj.get("formula"),
    }
    return {
        "status": "ok",
        "file": path,
        "updated_unix": mtime,
        "layer": layer_idx,
        "head": head_idx,
        "meta": header,
        "data": head,  # includes ordered 'trace' if you log it that way
    }

def _read_output_json(filename: str) -> JSONResponse:
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"bad json in {filename}: {e}")
    return JSONResponse(
        content=data,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )
@app.get("/output")
def get_manifest():
    # Return the manifest itself (no wrapper)
    print("NAVAL output ")
    return _read_output_json("output.json")

@app.get("/output_full_pre.json")
def get_full_pre():
    print("NAVAL output_full_pre ")
    return _read_output_json("output_full_pre.json")

@app.get("/output_full_final.json")
def get_full_final():
    print("NAVAL output_output_full_final ")
    return _read_output_json("output_full_final.json")

@app.get("/output_full_logits.json")
def get_full_logits():
    print("NAVAL output_output_full_logits ")
    return _read_output_json("output_full_logits.json")

@app.get("/FFN/{layer_idx}") 
def read_output(layer_idx):
    print("NAVAL FFN output")
    fname = f"decoder_layer_{layer_idx}_FFN.json"
    path =  os.path.join(LOG_DIR,fname)
    if not os.path.exists(path):
        print("fetching output 1")
        return JSONResponse({"status": "error", "msg": "no output.json yet"}, status_code=404)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"status": "ok", "data": data, "file": path, "updated_unix": os.path.getmtime(path)}

@app.get("/param_wq")
def read_loss():
    print("NAVAL param_wq implimented")
    path = os.path.join(LOG_DIR, "attn_qkv_Wq.json")
    if not os.path.exists(path):
        return {"status": "error", "msg": "no loss.json yet"}
    with open(path, "r") as f:
        data = json.load(f)
    return {"status": "ok", "data": data, "file": path, "updated_unix": os.path.getmtime(path)}
@app.get("/loss")
def read_loss():
    print("NAVAL no loss implimented")
    path = os.path.join(LOG_DIR, "loss.json")
    if not os.path.exists(path):
        return {"status": "error", "msg": "no loss.json yet"}
    with open(path, "r") as f:
        data = json.load(f)
    return {"status": "ok", "data": data, "file": path, "updated_unix": os.path.getmtime(path)}

@app.get("/optim")
def read_optim():
    print("NAVAL optim implimented")
    path = os.path.join(LOG_DIR, "optim.json")
    if not os.path.exists(path):
        return {"status": "error", "msg": "no optim.json yet"}
    with open(path, "r") as f:
        data = json.load(f)
    return {"status": "ok", "data": data, "file": path, "updated_unix": os.path.getmtime(path)}
@app.get("/config")
def read_optim():
    print("NAVAL no config implimented")
    path = os.path.join(LOG_DIR, "config.json")
    if not os.path.exists(path):
        return {"status": "error", "msg": "no optim.json yet"}
    with open(path, "r") as f:
        data = json.load(f)
    return {"status": "ok", "data": data, "file": path, "updated_unix": os.path.getmtime(path)}
@app.post("/config", status_code=201)
async def write_config(
    payload: Dict[str, Any] = Body(..., description="Full config JSON or partial if merge=true"),
    merge: bool = Query(False, description="Merge into existing config instead of replacing")
):
    print("NAVAL post config")
    current: Dict[str, Any] = {}
    if merge and os.path.exists(CFG_PATH):
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            current = json.load(f)
        new_cfg = deep_merge(current, payload)
    else:
        new_cfg = payload
    atomic_write_json(CFG_PATH, new_cfg)
    return {"status": "ok", "file": CFG_PATH, "merged": merge, "updated_unix": os.path.getmtime(CFG_PATH)}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render gives $PORT
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
