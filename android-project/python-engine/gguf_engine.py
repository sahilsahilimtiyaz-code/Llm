"""
LocalCore AI — Python GGUF Inference, Resumable Downloader & Mobile GPU Engine
Supports llama.cpp (llama-cpp-python / arm64-v8a libllama.so) with Vulkan, OpenCL,
ARM NEON/dotprod/i8mm, Flash Attention, KV-Cache quantization (FP16/Q8_0/Q4_0),
mmap streaming, thermal guard, explicit llama_free_model() memory cleanup, and abort signals.
"""

from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import httpx

try:
    from llama_cpp import Llama, llama_cpp as llama_c_api
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    llama_c_api = None  # type: ignore
    LLAMA_CPP_AVAILABLE = False

from database import LocalDatabase

logger = logging.getLogger("localcore.gguf_engine")

# GGUF binary magic header: ASCII "GGUF" = 0x46554747
GGUF_MAGIC = b"GGUF"

KV_CACHE_TYPE_MAP = {
    "FP16": 1,   # GGML_TYPE_F16
    "Q8_0": 8,   # GGML_TYPE_Q8_0 (50% memory reduction)
    "Q4_0": 2,   # GGML_TYPE_Q4_0 (75% memory reduction)
}


@dataclass
class HardwareTelemetry:
    total_ram_gb: float = 8.0
    free_ram_gb: float = 4.6
    gpu_renderer: str = "Adreno (TM) 740"
    vulkan_supported: bool = True
    opencl_supported: bool = True
    battery_temp_c: float = 36.2
    thermal_status: str = "nominal"
    prime_cores: int = 6
    storage_free_gb: float = 64.0


@dataclass
class DownloadState:
    model_id: str
    quant: str
    url: str
    dest_path: str
    expected_sha256: str
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_mbps: float = 0.0
    status: str = "idle"  # idle | downloading | paused | verifying | installed | error
    error: Optional[str] = None
    pause_event: threading.Event = field(default_factory=threading.Event)
    cancel_flag: bool = False


MODEL_MANIFEST: List[Dict[str, Any]] = [
    # Level 1: Ultra-Light Mobile / Edge (0.5B - 1.7B)
    {
        "id": "qwen2.5-0.5b-instruct",
        "name": "Qwen2.5-0.5B-Instruct",
        "level": 1,
        "params_b": 0.5,
        "layers": 24,
        "default_quant": "Q5_K_M",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename_tpl": "qwen2.5-0.5b-instruct-{quant}.gguf",
        "q4km_gb": 0.39,
    },
    {
        "id": "qwen2.5-1.5b-instruct",
        "name": "Qwen2.5-1.5B-Instruct",
        "level": 1,
        "params_b": 1.5,
        "layers": 28,
        "default_quant": "Q4_K_M",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename_tpl": "qwen2.5-1.5b-instruct-{quant}.gguf",
        "q4km_gb": 0.98,
    },
    {
        "id": "llama-3.2-1b-instruct",
        "name": "Llama-3.2-1B-Instruct",
        "level": 1,
        "params_b": 1.24,
        "layers": 16,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename_tpl": "Llama-3.2-1B-Instruct-{quant}.gguf",
        "q4km_gb": 0.81,
    },
    {
        "id": "smollm2-1.7b-instruct",
        "name": "SmolLM2-1.7B-Instruct",
        "level": 1,
        "params_b": 1.7,
        "layers": 24,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
        "filename_tpl": "SmolLM2-1.7B-Instruct-{quant}.gguf",
        "q4km_gb": 1.06,
    },
    # Level 2: Balanced Mobile Daily Driver (2B - 3.8B)
    {
        "id": "llama-3.2-3b-instruct",
        "name": "Llama-3.2-3B-Instruct",
        "level": 2,
        "params_b": 3.2,
        "layers": 28,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename_tpl": "Llama-3.2-3B-Instruct-{quant}.gguf",
        "q4km_gb": 2.02,
    },
    {
        "id": "qwen2.5-3b-instruct",
        "name": "Qwen2.5-3B-Instruct",
        "level": 2,
        "params_b": 3.1,
        "layers": 36,
        "default_quant": "Q4_K_M",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename_tpl": "qwen2.5-3b-instruct-{quant}.gguf",
        "q4km_gb": 1.93,
    },
    {
        "id": "gemma-2-2b-it",
        "name": "Gemma-2-2B-IT",
        "level": 2,
        "params_b": 2.6,
        "layers": 26,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/gemma-2-2b-it-GGUF",
        "filename_tpl": "gemma-2-2b-it-{quant}.gguf",
        "q4km_gb": 1.71,
    },
    {
        "id": "phi-3.5-mini-instruct",
        "name": "Phi-3.5-mini-instruct",
        "level": 2,
        "params_b": 3.8,
        "layers": 32,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename_tpl": "Phi-3.5-mini-instruct-{quant}.gguf",
        "q4km_gb": 2.39,
    },
    # Level 3: Pro Flagship Smartphone (7B - 8B)
    {
        "id": "llama-3.1-8b-instruct",
        "name": "Llama-3.1-8B-Instruct",
        "level": 3,
        "params_b": 8.03,
        "layers": 32,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename_tpl": "Meta-Llama-3.1-8B-Instruct-{quant}.gguf",
        "q4km_gb": 4.92,
    },
    {
        "id": "qwen2.5-7b-instruct",
        "name": "Qwen2.5-7B-Instruct",
        "level": 3,
        "params_b": 7.6,
        "layers": 28,
        "default_quant": "Q4_K_M",
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "filename_tpl": "qwen2.5-7b-instruct-{quant}.gguf",
        "q4km_gb": 4.68,
    },
    {
        "id": "qwen2.5-coder-7b-instruct",
        "name": "Qwen2.5-Coder-7B-Instruct",
        "level": 3,
        "params_b": 7.6,
        "layers": 28,
        "default_quant": "Q4_K_M",
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename_tpl": "qwen2.5-coder-7b-instruct-{quant}.gguf",
        "q4km_gb": 4.68,
    },
    {
        "id": "deepseek-r1-distill-llama-8b",
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "level": 3,
        "params_b": 8.03,
        "layers": 32,
        "default_quant": "Q4_K_M",
        "repo": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename_tpl": "DeepSeek-R1-Distill-Llama-8B-{quant}.gguf",
        "q4km_gb": 4.92,
        "supports_think": True,
    },
    # Level 4: Ultra Workstation / Tablet (14B+)
    {
        "id": "qwen2.5-14b-instruct",
        "name": "Qwen2.5-14B-Instruct",
        "level": 4,
        "params_b": 14.7,
        "layers": 48,
        "default_quant": "IQ3_M",
        "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename_tpl": "Qwen2.5-14B-Instruct-{quant}.gguf",
        "q4km_gb": 8.99,
    },
    {
        "id": "deepseek-r1-distill-qwen-14b",
        "name": "DeepSeek-R1-Distill-Qwen-14B",
        "level": 4,
        "params_b": 14.7,
        "layers": 48,
        "default_quant": "IQ3_M",
        "repo": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "filename_tpl": "DeepSeek-R1-Distill-Qwen-14B-{quant}.gguf",
        "q4km_gb": 8.99,
        "supports_think": True,
    },
]


class GGUFEngineError(RuntimeError):
    """Base error for GGUF loading, OOM protection, or file verification failures."""


class OOMSafetyError(GGUFEngineError):
    """Raised when loading a model would trigger Android's Low-Memory Killer."""


class CorruptedGGUFError(GGUFEngineError):
    """Raised when a .gguf header magic or SHA-256 digest fails verification."""


class GGUFEngine:
    """Manages on-device GGUF downloading, verification, GPU loading, streaming, and cleanup."""

    def __init__(self, db: LocalDatabase, models_dir: Optional[str] = None) -> None:
        base_dir = os.environ.get("ANDROID_FILES_DIR") or str(Path.home() / ".localcore_ai")
        self.models_dir = Path(models_dir or (Path(base_dir) / "models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.db = db
        self.hardware = HardwareTelemetry()
        self._model_lock = threading.RLock()
        self._llm: Optional[Any] = None
        self._loaded_model_id: Optional[str] = None
        self._loaded_quant: Optional[str] = None
        self._loaded_config: Dict[str, Any] = {}
        self._abort_event = threading.Event()
        self._downloads: Dict[str, DownloadState] = {}

    # ── Hardware Telemetry & Memory Safety ────────────────────────────────────

    def update_hardware_telemetry(self, payload: Dict[str, Any]) -> HardwareTelemetry:
        self.hardware = HardwareTelemetry(
            total_ram_gb=float(payload.get("totalRamGb", self.hardware.total_ram_gb)),
            free_ram_gb=float(payload.get("freeRamGb", self.hardware.free_ram_gb)),
            gpu_renderer=str(payload.get("gpuRenderer", self.hardware.gpu_renderer)),
            vulkan_supported=bool(payload.get("vulkanSupported", self.hardware.vulkan_supported)),
            opencl_supported=bool(payload.get("openclSupported", self.hardware.opencl_supported)),
            battery_temp_c=float(payload.get("batteryTempC", self.hardware.battery_temp_c)),
            thermal_status=str(payload.get("thermalStatus", self.hardware.thermal_status)),
            prime_cores=int(payload.get("primePerformanceCores", self.hardware.prime_cores)),
            storage_free_gb=float(payload.get("storageFreeGb", self.hardware.storage_free_gb)),
        )
        return self.hardware

    def evaluate_ram_safety(
        self,
        file_size_gb: float,
        context_length: int = 4096,
        kv_cache_quant: str = "Q8_0",
        mmap: bool = True,
    ) -> Dict[str, Any]:
        kv_mult = {"FP16": 1.0, "Q8_0": 0.52, "Q4_0": 0.27}.get(kv_cache_quant, 0.52)
        kv_gb = (0.45 * (context_length / 4096.0)) * kv_mult
        weights_gb = file_size_gb * (1.02 if mmap else 1.26)
        total_req_gb = round(weights_gb + kv_gb + 0.35, 2)
        headroom_gb = round(self.hardware.free_ram_gb - total_req_gb, 2)

        if headroom_gb >= 1.15:
            tier = "smooth"
            label = "Smooth on Your Phone (GPU Ready)"
        elif headroom_gb >= 0.25:
            tier = "tight"
            label = "Tight Fit (Close Background Apps)"
        else:
            tier = "oom"
            label = "Risk of Android OOM Crash"

        return {
            "tier": tier,
            "label": label,
            "totalRequiredGb": total_req_gb,
            "freeRamGb": self.hardware.free_ram_gb,
            "headroomGb": headroom_gb,
            "requiresOverride": tier == "oom",
        }

    # ── GGUF File Validation & SHA-256 ────────────────────────────────────────

    @staticmethod
    def validate_gguf_header(file_path: Path) -> None:
        if not file_path.exists():
            raise CorruptedGGUFError(f"GGUF file not found: {file_path}")
        if file_path.stat().st_size < 1024:
            raise CorruptedGGUFError("GGUF file is truncated or empty (<1KB).")
        with file_path.open("rb") as fh:
            magic = fh.read(4)
            if magic != GGUF_MAGIC:
                raise CorruptedGGUFError(
                    f"Invalid GGUF header magic {magic!r}; expected {GGUF_MAGIC!r}."
                )

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as fh:
            while chunk := fh.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    # ── Resumable Chunked HTTP Range-Request Downloader ───────────────────────

    async def start_download(
        self,
        model_id: str,
        quant: str,
        hf_url: str,
        expected_sha256: str,
        expected_size_bytes: int,
        progress_cb: Optional[Callable[[DownloadState], None]] = None,
    ) -> DownloadState:
        # Storage free-space pre-check (require file size + 512MB safety buffer)
        disk_usage = shutil.disk_usage(self.models_dir)
        required_disk = expected_size_bytes + 512 * 1024 * 1024
        if disk_usage.free < required_disk:
            free_gb = disk_usage.free / (1024 ** 3)
            need_gb = required_disk / (1024 ** 3)
            raise GGUFEngineError(
                f"Insufficient Android storage: {free_gb:.2f} GB free, need {need_gb:.2f} GB."
            )

        filename = f"{model_id}-{quant}.gguf"
        dest_path = self.models_dir / filename
        part_path = self.models_dir / f"{filename}.part"

        state = self._downloads.get(model_id)
        if state and state.status == "downloading":
            return state

        state = DownloadState(
            model_id=model_id,
            quant=quant,
            url=hf_url,
            dest_path=str(dest_path),
            expected_sha256=expected_sha256,
            total_bytes=expected_size_bytes,
            status="downloading",
        )
        state.pause_event.set()
        self._downloads[model_id] = state

        try:
            existing_bytes = part_path.stat().st_size if part_path.exists() else 0
            state.downloaded_bytes = existing_bytes
            headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}

            async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
                async with client.stream("GET", hf_url, headers=headers) as resp:
                    if resp.status_code not in (200, 206):
                        raise GGUFEngineError(f"Hugging Face HTTP {resp.status_code} for {hf_url}")

                    if resp.status_code == 200 and existing_bytes > 0:
                        existing_bytes = 0
                        state.downloaded_bytes = 0

                    content_len = int(resp.headers.get("content-length", 0))
                    if content_len > 0:
                        state.total_bytes = existing_bytes + content_len

                    mode = "ab" if resp.status_code == 206 and existing_bytes > 0 else "wb"
                    window_start = time.monotonic()
                    window_bytes = 0

                    with part_path.open(mode) as fh:
                        async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                            if state.cancel_flag:
                                state.status = "idle"
                                if part_path.exists():
                                    part_path.unlink(missing_ok=True)
                                return state

                            while not state.pause_event.is_set():
                                state.status = "paused"
                                state.speed_mbps = 0.0
                                if progress_cb:
                                    progress_cb(state)
                                await asyncio.sleep(0.25)
                                if state.cancel_flag:
                                    part_path.unlink(missing_ok=True)
                                    state.status = "idle"
                                    return state

                            state.status = "downloading"
                            fh.write(chunk)
                            chunk_len = len(chunk)
                            state.downloaded_bytes += chunk_len
                            window_bytes += chunk_len

                            elapsed = time.monotonic() - window_start
                            if elapsed >= 0.5:
                                state.speed_mbps = round((window_bytes / (1024 * 1024)) / elapsed, 2)
                                window_start = time.monotonic()
                                window_bytes = 0
                                if progress_cb:
                                    progress_cb(state)

            # Verify GGUF header + SHA-256
            state.status = "verifying"
            if progress_cb:
                progress_cb(state)

            self.validate_gguf_header(part_path)
            actual_sha256 = await asyncio.to_thread(self.compute_sha256, part_path)
            if expected_sha256 and len(expected_sha256) == 64 and actual_sha256.lower() != expected_sha256.lower():
                part_path.unlink(missing_ok=True)
                raise CorruptedGGUFError(
                    f"SHA-256 mismatch: expected {expected_sha256[:12]}..., got {actual_sha256[:12]}..."
                )

            part_path.rename(dest_path)
            meta = next((m for m in MODEL_MANIFEST if m["id"] == model_id), None)
            self.db.upsert_downloaded_model(
                model_id=model_id,
                name=meta["name"] if meta else model_id,
                level=int(meta["level"]) if meta else 2,
                quant=quant,
                file_path=str(dest_path),
                file_size_bytes=dest_path.stat().st_size,
                sha256=actual_sha256,
                verified=True,
                is_custom_import=False,
            )
            state.status = "installed"
            state.speed_mbps = 0.0
            if progress_cb:
                progress_cb(state)
            return state
        except Exception as exc:
            state.status = "error"
            state.error = str(exc)
            state.speed_mbps = 0.0
            if progress_cb:
                progress_cb(state)
            raise

    def pause_download(self, model_id: str) -> None:
        if state := self._downloads.get(model_id):
            state.pause_event.clear()
            state.status = "paused"

    def resume_download(self, model_id: str) -> None:
        if state := self._downloads.get(model_id):
            state.pause_event.set()
            state.status = "downloading"

    def cancel_download(self, model_id: str) -> None:
        if state := self._downloads.get(model_id):
            state.cancel_flag = True
            state.pause_event.set()

    # ── SAF Direct GGUF Import (Zero-Copy Path Registration) ──────────────────

    def register_saf_gguf(self, file_path_str: str, display_name: Optional[str] = None) -> Dict[str, Any]:
        path = Path(file_path_str)
        self.validate_gguf_header(path)
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        level = 1 if size_gb <= 1.3 else 2 if size_gb <= 3.0 else 3 if size_gb <= 6.2 else 4
        model_id = f"custom-{hashlib.sha1(str(path).encode()).hexdigest()[:10]}"
        name = display_name or path.stem
        self.db.upsert_downloaded_model(
            model_id=model_id,
            name=name,
            level=level,
            quant="Q4_K_M",
            file_path=str(path),
            file_size_bytes=size_bytes,
            sha256="saf-local-verified",
            verified=True,
            is_custom_import=True,
        )
        return {"modelId": model_id, "name": name, "level": level, "fileSizeGb": round(size_gb, 2)}

    # ── Model Loading, GPU Layer Offloading & Explicit Unload ─────────────────

    def unload_model(self) -> Dict[str, Any]:
        """Aborts generation, invokes native llama_free_model(), and forces GC + malloc_trim."""
        self.abort_generation()
        with self._model_lock:
            previous = self._loaded_model_id
            if self._llm is not None:
                try:
                    if hasattr(self._llm, "close"):
                        self._llm.close()
                    elif llama_c_api is not None and getattr(self._llm, "model", None):
                        llama_c_api.llama_free_model(self._llm.model)
                finally:
                    self._llm = None
            self._loaded_model_id = None
            self._loaded_quant = None
            self._loaded_config = {}
            gc.collect()
            try:
                libc = ctypes.CDLL("libc.so")
                if hasattr(libc, "malloc_trim"):
                    libc.malloc_trim(0)
            except OSError:
                pass
        return {"unloadedModelId": previous, "status": "freed"}

    def load_model(
        self,
        model_id: str,
        file_path: str,
        quant: str,
        gpu_config: Dict[str, Any],
        context_length: int = 4096,
        override_oom_lock: bool = False,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        self.validate_gguf_header(path)
        size_gb = path.stat().st_size / (1024 ** 3)

        safety = self.evaluate_ram_safety(
            file_size_gb=size_gb,
            context_length=context_length,
            kv_cache_quant=gpu_config.get("kvCacheQuant", "Q8_0"),
            mmap=bool(gpu_config.get("mmap", True)),
        )
        if safety["requiresOverride"] and not override_oom_lock:
            raise OOMSafetyError(
                f"Blocked load to prevent Android OOM crash: requires {safety['totalRequiredGb']} GB "
                f"but only {safety['freeRamGb']} GB free RAM available."
            )

        # Explicitly free any previously loaded model before allocating the next one
        self.unload_model()

        backend = gpu_config.get("backend", "vulkan")
        n_gpu_layers = 0 if backend == "cpu_arm64" else int(gpu_config.get("nGpuLayers", 28))
        n_threads = int(gpu_config.get("nThreads", self.hardware.prime_cores))
        batch_size = int(gpu_config.get("batchSize", 512))

        # Apply thermal throttling guard before allocation
        if gpu_config.get("thermalGuardEnabled", True) and self.hardware.battery_temp_c >= float(
            gpu_config.get("thermalThresholdC", 43.0)
        ):
            batch_size = max(64, batch_size // 2)

        kv_type = KV_CACHE_TYPE_MAP.get(gpu_config.get("kvCacheQuant", "Q8_0"), 8)

        # Configure backend environment hints for native libllama.so
        if backend == "vulkan":
            os.environ["GGML_VK_DISABLE_F16"] = "0"
        elif backend == "opencl":
            os.environ["GGML_OPENCL_PLATFORM"] = "0"

        if not LLAMA_CPP_AVAILABLE:
            raise GGUFEngineError(
                "llama-cpp-python / libllama.so is not installed in this Python runtime."
            )

        with self._model_lock:
            try:
                self._llm = Llama(
                    model_path=str(path),
                    n_ctx=context_length,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    n_threads_batch=n_threads,
                    n_batch=batch_size,
                    use_mmap=bool(gpu_config.get("mmap", True)),
                    use_mlock=False,
                    flash_attn=bool(gpu_config.get("flashAttention", True)),
                    type_k=kv_type,
                    type_v=kv_type,
                    verbose=False,
                )
            except MemoryError as mem_err:
                self.unload_model()
                raise OOMSafetyError(f"Native allocation OOM: {mem_err}") from mem_err
            except Exception as exc:
                self.unload_model()
                raise GGUFEngineError(f"Failed to load GGUF model: {exc}") from exc

            self._loaded_model_id = model_id
            self._loaded_quant = quant
            self._loaded_config = {**gpu_config, "batchSize": batch_size, "contextLength": context_length}

        self.db.save_gpu_settings(model_id, gpu_config)
        return {
            "loadedModelId": model_id,
            "quant": quant,
            "backend": backend,
            "nGpuLayers": n_gpu_layers,
            "safety": safety,
        }

    # ── Async Token Streaming + Native Abort + TrimMemory Handler ─────────────

    def abort_generation(self) -> None:
        self._abort_event.set()

    def handle_android_trim_memory(self, level: int) -> Dict[str, Any]:
        """Invoked by Kotlin ComponentCallbacks2.onTrimMemory() to prevent LMK termination."""
        self.abort_generation()
        # TRIM_MEMORY_RUNNING_CRITICAL (15) or TRIM_MEMORY_COMPLETE (80) -> full unload
        if level >= 15:
            return self.unload_model()
        # Moderate memory pressure -> clear KV cache only
        with self._model_lock:
            if self._llm is not None and hasattr(self._llm, "reset"):
                self._llm.reset()
        return {"status": "kv_cache_cleared", "trimLevel": level}

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        if self._llm is None:
            raise GGUFEngineError("No GGUF model loaded. Load a model from the Model Hub first.")

        self._abort_event.clear()
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        token_count = 0

        stream = self._llm.create_chat_completion(
            messages=messages,
            temperature=float(params.get("temperature", 0.65)),
            top_p=float(params.get("topP", 0.9)),
            top_k=int(params.get("topK", 40)),
            min_p=float(params.get("minP", 0.05)),
            repeat_penalty=float(params.get("repeatPenalty", 1.1)),
            max_tokens=int(params.get("maxTokens", 1024)),
            stream=True,
        )

        for chunk in stream:
            if self._abort_event.is_set():
                yield {"type": "aborted", "tokenCount": token_count}
                return

            delta = chunk["choices"][0]["delta"].get("content", "")
            if not delta:
                continue

            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
                ttft_ms = int((first_token_time - start_time) * 1000)
                yield {"type": "ttft", "ttftMs": ttft_ms}

            token_count += 1
            decode_elapsed = max(0.001, now - (first_token_time or start_time))
            decode_tps = round(token_count / decode_elapsed, 1)

            yield {
                "type": "token",
                "delta": delta,
                "tokenCount": token_count,
                "decodeTokPerSec": decode_tps,
            }
            await asyncio.sleep(0)

        total_elapsed = max(0.001, time.perf_counter() - (first_token_time or start_time))
        yield {
            "type": "done",
            "tokenCount": token_count,
            "decodeTokPerSec": round(token_count / total_elapsed, 1),
            "ttftMs": int(((first_token_time or start_time) - start_time) * 1000),
        }
