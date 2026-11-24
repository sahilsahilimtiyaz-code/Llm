"""
LocalCore AI — FastAPI + SSE Local IPC Streaming Server (127.0.0.1:8765)
Bridges the Kotlin Android Native Bridge & Hardware-Accelerated WebView UI to
the Python llama.cpp GGUF Engine and SQLite3 + FTS5 Database.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from database import LocalDatabase
from gguf_engine import (
    CorruptedGGUFError,
    GGUFEngine,
    GGUFEngineError,
    MODEL_MANIFEST,
    OOMSafetyError,
)

app = FastAPI(title="LocalCore AI On-Device Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = LocalDatabase()
engine = GGUFEngine(db=db)


class TelemetryPayload(BaseModel):
    totalRamGb: float
    freeRamGb: float
    gpuRenderer: str
    vulkanSupported: bool = True
    openclSupported: bool = True
    batteryTempC: float = 36.0
    thermalStatus: str = "nominal"
    primePerformanceCores: int = 6
    storageFreeGb: float = 64.0


class DownloadRequest(BaseModel):
    modelId: str
    quant: str = "Q4_K_M"
    hfUrl: str
    sha256: str = ""
    expectedSizeBytes: int


class LoadModelRequest(BaseModel):
    modelId: str
    filePath: str
    quant: str = "Q4_K_M"
    gpuConfig: Dict[str, Any]
    contextLength: int = 4096
    overrideOomLock: bool = False


class StreamChatRequest(BaseModel):
    conversationId: str
    parentId: Optional[str] = None
    prompt: str
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "loadedModelId": engine._loaded_model_id,
        "loadedQuant": engine._loaded_quant,
        "batteryTempC": engine.hardware.battery_temp_c,
        "freeRamGb": engine.hardware.free_ram_gb,
    }


@app.post("/api/hardware/telemetry")
def sync_telemetry(payload: TelemetryPayload) -> Dict[str, Any]:
    hw = engine.update_hardware_telemetry(payload.model_dump())
    return {"status": "synced", "hardware": hw.__dict__}


@app.post("/api/hardware/trim-memory")
def on_trim_memory(level: int) -> Dict[str, Any]:
    return engine.handle_android_trim_memory(level)


@app.get("/api/models")
def list_models() -> Dict[str, Any]:
    installed = {row["model_id"]: row for row in db.list_downloaded_models()}
    return {
        "manifest": MODEL_MANIFEST,
        "installed": installed,
        "loadedModelId": engine._loaded_model_id,
        "loadedQuant": engine._loaded_quant,
    }


@app.post("/api/models/download")
async def download_model(req: DownloadRequest) -> Dict[str, Any]:
    try:
        state = await engine.start_download(
            model_id=req.modelId,
            quant=req.quant,
            hf_url=req.hfUrl,
            expected_sha256=req.sha256,
            expected_size_bytes=req.expectedSizeBytes,
        )
        return {"status": state.status, "destPath": state.dest_path}
    except GGUFEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/download/{model_id}/{action}")
def control_download(model_id: str, action: str) -> Dict[str, Any]:
    if action == "pause":
        engine.pause_download(model_id)
    elif action == "resume":
        engine.resume_download(model_id)
    elif action == "cancel":
        engine.cancel_download(model_id)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown download action: {action}")
    return {"modelId": model_id, "action": action}


@app.post("/api/models/import-saf")
def import_saf_model(filePath: str, displayName: Optional[str] = None) -> Dict[str, Any]:
    try:
        return engine.register_saf_gguf(filePath, displayName)
    except CorruptedGGUFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/models/load")
def load_model(req: LoadModelRequest) -> Dict[str, Any]:
    try:
        return engine.load_model(
            model_id=req.modelId,
            file_path=req.filePath,
            quant=req.quant,
            gpu_config=req.gpuConfig,
            context_length=req.contextLength,
            override_oom_lock=req.overrideOomLock,
        )
    except OOMSafetyError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except CorruptedGGUFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GGUFEngineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/models/unload")
def unload_model() -> Dict[str, Any]:
    return engine.unload_model()


@app.get("/api/conversations")
def list_conversations(q: Optional[str] = None) -> List[Dict[str, Any]]:
    if q and q.strip():
        return db.search_fts5(q)
    return db.list_conversations()


@app.post("/api/conversations")
def create_conversation(title: str = "New Offline Chat") -> Dict[str, Any]:
    return db.create_conversation(title=title, model_id=engine._loaded_model_id)


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(conversation_id: str) -> List[Dict[str, Any]]:
    return db.get_messages(conversation_id)


@app.post("/api/chat/abort")
def abort_chat() -> Dict[str, Any]:
    engine.abort_generation()
    return {"aborted": True}


@app.post("/api/chat/stream")
async def stream_chat(req: StreamChatRequest, request: Request) -> StreamingResponse:
    user_msg = db.add_message(
        conversation_id=req.conversationId,
        parent_id=req.parentId,
        role="user",
        content=req.prompt,
        attachments=req.attachments,
    )

    history = db.get_messages(req.conversationId)
    llm_messages: List[Dict[str, str]] = []
    system_prompt = req.params.get("systemPrompt", "You are a helpful local Android AI assistant.")
    llm_messages.append({"role": "system", "content": system_prompt})

    for m in history[-16:]:
        content = m["content"]
        if m.get("attachments"):
            attached_text = "\n\n".join(
                f"--- FILE: {a['name']} ---\n{a['content']}" for a in m["attachments"]
            )
            content = f"{attached_text}\n\n{content}"
        llm_messages.append({"role": m["role"], "content": content})

    async def event_generator():
        yield f"data: {json.dumps({'type': 'user_saved', 'message': user_msg})}\n\n"
        full_response = []
        ttft_ms = 0
        final_tps = 0.0
        final_tokens = 0
        started_ms = int(time.time() * 1000)

        try:
            async for ev in engine.stream_chat(llm_messages, req.params):
                if await request.is_disconnected():
                    engine.abort_generation()
                    break
                if ev["type"] == "token":
                    full_response.append(ev["delta"])
                elif ev["type"] == "ttft":
                    ttft_ms = ev["ttftMs"]
                elif ev["type"] == "done":
                    final_tps = ev["decodeTokPerSec"]
                    final_tokens = ev["tokenCount"]
                yield f"data: {json.dumps(ev)}\n\n"
        finally:
            assembled = "".join(full_response).strip()
            if assembled:
                assistant_msg = db.add_message(
                    conversation_id=req.conversationId,
                    parent_id=user_msg["id"],
                    role="assistant",
                    content=assembled,
                    think_ms=int(time.time() * 1000) - started_ms if "<think>" in assembled else None,
                    model_id=engine._loaded_model_id,
                    quant=engine._loaded_quant,
                    decode_tok_per_sec=final_tps,
                    ttft_ms=ttft_ms,
                    token_count=final_tokens,
                )
                yield f"data: {json.dumps({'type': 'assistant_saved', 'message': assistant_msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("LOCALCORE_PORT", "8765"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
