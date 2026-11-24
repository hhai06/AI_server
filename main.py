from __future__ import annotations

import hmac
import io
import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import httpx
import numpy as np
import onnxruntime as ort
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("ai-service")
load_dotenv()

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("AI_MODEL_PATH", str(ROOT / "forWeb.onnx"))).expanduser()
MODEL_META_PATH = Path(os.getenv("AI_MODEL_META_PATH", str(ROOT / "forWeb_meta.json"))).expanduser()
MODEL_VERSION = os.getenv("AI_MODEL_VERSION", "forWeb-onnx-v1")
THREADS = int(os.getenv("AI_THREADS", "0")) or min(os.cpu_count() or 1, 4)
MAX_PENDING = int(os.getenv("AI_MAX_PENDING", "20"))
# Số view chạy mỗi lượt ONNX (chia nhỏ 6 view để giảm đỉnh RAM activation).
BATCH_SIZE = max(1, int(os.getenv("AI_BATCH", "3")))
# Tắt CPU arena + mem-pattern của ONNX Runtime để không giữ RAM thừa.
LOW_MEMORY = os.getenv("AI_LOW_MEMORY", "true").lower() != "false"
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_PIXELS = 40_000_000

CROP_SPECS = (
    (1.0, 0.5, 0.5),
    (0.8, 0.0, 0.0),
    (0.8, 1.0, 0.0),
    (0.8, 0.0, 1.0),
    (0.8, 1.0, 1.0),
    (0.8, 0.5, 0.5),
)

model_name = "onnx-model"
model_threshold = 0.5
input_size = 224
mean = np.zeros(3, np.float32)
std = np.ones(3, np.float32)
session: ort.InferenceSession | None = None
input_name, output_name = "input", "output"
fixed_batch: int | None = None
ready = False
load_error: str | None = None
accepted_jobs: set[str] = set()
jobs_lock = threading.Lock()

s3 = boto3.client(
    "s3", region_name=os.getenv("S3_REGION", "us-east-1"), endpoint_url=os.getenv("S3_ENDPOINT") or None,
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"), aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=5, read_timeout=20,
                  s3={"addressing_style": "path"} if os.getenv("S3_FORCE_PATH_STYLE", "").lower() == "true" else None),
)
bucket = os.getenv("S3_BUCKET", "")
service_key = os.getenv("AI_SERVICE_KEY", "").encode()
callback_secret = os.getenv("AI_CALLBACK_SECRET", "")
callback_url = os.getenv("AI_CALLBACK_URL", "").strip()


class AnalysisJob(BaseModel):
    analysis_id: str
    case_id: str
    image_key: str = Field(min_length=1, max_length=500)
    mime_type: str = Field(pattern=r"^image/(jpeg|png|webp)$")
    gradcam_key: str = Field(min_length=1, max_length=500)

job_queue: queue.Queue[AnalysisJob | None] = queue.Queue(maxsize=MAX_PENDING)


def configured_callback_url() -> str:
    parsed = urlparse(callback_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise RuntimeError("AI_CALLBACK_URL phải là URL HTTP(S) hợp lệ của NestJS")
    return callback_url


def validate_runtime_config() -> None:
    required = {"AI_SERVICE_KEY": service_key, "AI_CALLBACK_SECRET": callback_secret, "S3_BUCKET": bucket,
                "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY", ""), "S3_SECRET_KEY": os.getenv("S3_SECRET_KEY", "")}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Thiếu cấu hình: {', '.join(missing)}")
    if len(service_key) < 32 or len(callback_secret) < 32:
        raise RuntimeError("AI_SERVICE_KEY và AI_CALLBACK_SECRET phải dài ít nhất 32 ký tự")
    if hmac.compare_digest(service_key, callback_secret.encode()):
        raise RuntimeError("AI_SERVICE_KEY và AI_CALLBACK_SECRET phải khác nhau")
    configured_callback_url()


def load_model() -> None:
    global input_name, output_name, input_size, mean, std, model_name, model_threshold, session, fixed_batch
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Không tìm thấy model ONNX: {MODEL_PATH}")
    if not MODEL_META_PATH.is_file():
        raise RuntimeError(f"Không tìm thấy metadata model: {MODEL_META_PATH}")
    meta = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    size, threshold = int(meta.get("size", 0)), float(meta.get("threshold", -1))
    input_mean, input_std = np.asarray(meta.get("mean"), dtype=np.float32), np.asarray(meta.get("std"), dtype=np.float32)
    if size <= 0 or not 0 <= threshold <= 1 or input_mean.shape != (3,) or input_std.shape != (3,):
        raise RuntimeError("Metadata preprocessing/threshold không hợp lệ")
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = THREADS, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # Không để worker của ORT quay vòng (spin) ăn CPU khi quota container nhỏ.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if LOW_MEMORY:
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
    loaded = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
    inputs, outputs = loaded.get_inputs(), loaded.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[-3:] != [3, size, size]:
        raise RuntimeError("Model ONNX không có input [batch, 3, size, size] và một output")
    batch_dim = inputs[0].shape[0]
    fixed_batch = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
    session, input_name, output_name = loaded, inputs[0].name, outputs[0].name
    input_size, model_threshold, mean, std = size, threshold, input_mean, input_std
    model_name = f"{meta.get('arch', 'model')}-onnx"
    LOGGER.info("Loaded %s, input=%s, threshold=%.4f, threads=%d, batch=%s, low_memory=%s",
                model_name, inputs[0].shape, model_threshold, THREADS, fixed_batch or BATCH_SIZE, LOW_MEMORY)


def make_crops(image: Image.Image) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    width, height = image.size
    side = min(width, height)
    crops = []
    for scale, anchor_x, anchor_y in CROP_SPECS:
        length = max(1, min(side, round(side * scale)))
        left = round((width - length) * anchor_x)
        top = round((height - length) * anchor_y)
        box = (left, top, left + length, top + length)
        crops.append((image.crop(box), box))
    return crops


def preprocess_into(image: Image.Image, dest: np.ndarray, scratch: np.ndarray) -> None:

    image = image.resize((input_size, input_size), Image.Resampling.BICUBIC)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    out.seek(0)
    with Image.open(out) as decoded:
        pixels = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    np.copyto(scratch, pixels)         
    np.divide(scratch, 255.0, out=scratch)
    np.subtract(scratch, mean, out=scratch)
    np.divide(scratch, std, out=scratch)
    dest[...] = scratch.transpose(2, 0, 1)


def predict(image: Image.Image) -> tuple[float, dict[str, Any]]:
    if session is None:
        raise RuntimeError("Model chưa sẵn sàng")
    views = make_crops(image)
    batch = np.empty((len(views), 3, input_size, input_size), np.float32)
    scratch = np.empty((input_size, input_size, 3), np.float32)
    for index, (crop, _) in enumerate(views):
        preprocess_into(crop, batch[index], scratch)

    step = fixed_batch or BATCH_SIZE
    parts = []
    for start in range(0, len(batch), step):
        chunk = batch[start:start + step]
        real = len(chunk)
        if fixed_batch and real < fixed_batch:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], fixed_batch - real, axis=0)])
        output = session.run([output_name], {input_name: np.ascontiguousarray(chunk)})[0]
        parts.append(np.asarray(output).reshape(-1)[:real])
    logits = np.concatenate(parts)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    crops = sorted(
        ({"box": list(box), "probability": float(prob), "label": int(float(prob) >= model_threshold)}
         for (_crop, box), prob in zip(views, probabilities, strict=True)), key=lambda item: item["probability"], reverse=True,
    )
    return crops[0]["probability"], {"best_crop": crops[0], "crops": crops, "aggregation": "max_over_square_crops"}


def _load_in_background() -> None:
    global ready, load_error
    try:
        validate_runtime_config()
        load_model()
        predict(Image.fromarray(np.full((input_size, input_size, 3), 127, np.uint8)))
        ready = True
        LOGGER.info("Model ready")
    except Exception as exc:  # noqa: BLE001
        load_error = str(exc)
        LOGGER.exception("Model load failed")


def read_image(key: str) -> Image.Image:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        data = body.read(MAX_IMAGE_BYTES + 1)
    finally:
        body.close()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Ảnh vượt quá giới hạn xử lý")
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > MAX_PIXELS:
            raise ValueError("Kích thước ảnh vượt giới hạn xử lý")
        return ImageOps.exif_transpose(source).convert("RGB")


def send_callback(payload: dict[str, Any]) -> None:
    with httpx.Client(timeout=20.0, headers={"X-AI-Key": callback_secret}) as client:
        for attempt in range(3):
            try:
                client.post(configured_callback_url(), json=payload).raise_for_status()
                return
            except Exception:  # noqa: BLE001
                LOGGER.warning("Callback attempt %d failed: %s", attempt + 1, payload.get("analysis_id"))
                if attempt < 2:
                    time.sleep(2 ** attempt)
    LOGGER.error("Callback gave up: %s", payload.get("analysis_id"))


def process_job(job: AnalysisJob) -> None:
    started = time.perf_counter()
    try:
        LOGGER.info("Processing analysis %s for %s", job.analysis_id, job.image_key)
        probability, details = predict(read_image(job.image_key))
        payload: dict[str, Any] = {
            "analysis_id": job.analysis_id, "status": "completed", "model_name": model_name, "model_version": MODEL_VERSION,
            "predicted_label": "malignant" if probability >= model_threshold else "benign",
            "probabilities": {"malignant": probability, "benign": 1.0 - probability}, "risk_score": round(probability * 100, 4),
            "raw_output": {"threshold": model_threshold, **details},
        }
        LOGGER.info("Analysis %s done in %.2fs", job.analysis_id, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed: %s", job.analysis_id)
        payload = {"analysis_id": job.analysis_id, "status": "failed", "error_message": str(exc)[:1000]}
    try:
        send_callback(payload)
    finally:
        with jobs_lock:
            accepted_jobs.discard(job.analysis_id)


def inference_loop() -> None:
    while True:
        job = job_queue.get()
        if job is None:
            return
        process_job(job)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_load_in_background, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    yield
    job_queue.put(None)


app = FastAPI(title="NPC image analysis", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if not ready:
        raise HTTPException(status_code=503, detail=load_error or "Model đang được nạp")
    return {"status": "ok", "model": model_name, "device": "cpu", "format": "onnx"}


def verify_key(value: str | None) -> None:
    if not service_key or value is None or not hmac.compare_digest(value.encode(), service_key):
        raise HTTPException(status_code=401, detail="Không được phép")


@app.post("/jobs", status_code=202)
async def submit_job(job: AnalysisJob, x_ai_key: str | None = Header(default=None)) -> dict[str, str]:
    verify_key(x_ai_key)
    if not ready:
        raise HTTPException(status_code=503, detail=load_error or "Model chưa sẵn sàng")
    with jobs_lock:
        if job.analysis_id in accepted_jobs:
            LOGGER.info("Duplicate analysis accepted: %s", job.analysis_id)
            return {"status": "accepted", "analysis_id": job.analysis_id}
        try:
            job_queue.put_nowait(job)
        except queue.Full:
            raise HTTPException(status_code=429, detail="Hàng đợi đầy, thử lại sau") from None
        accepted_jobs.add(job.analysis_id)
    LOGGER.info("Accepted analysis: %s", job.analysis_id)
    return {"status": "accepted", "analysis_id": job.analysis_id}
