from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import traceback
import unicodedata
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = APP_ROOT / "vendor" / "Irodori-TTS"
FFMPEG_BIN_DIR = VENDOR_ROOT / "ffmpeg" / "bin"
if FFMPEG_BIN_DIR.is_dir():
    os.environ["PATH"] = f"{FFMPEG_BIN_DIR};{os.environ.get('PATH', '')}"
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(FFMPEG_BIN_DIR))

import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field, model_validator

if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav
from irodori_tts.text_normalization import normalize_text

DEFAULT_CHECKPOINT_REPO = "Aratako/Irodori-TTS-500M-v3"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_REF_NORMALIZE_DB = -16.0
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs" / "irodori_api"
DEFAULT_EMBED_PROFILE_DIR = APP_ROOT / "irodori_embed_profiles"
BRACKET_PAIRS = {
    "\u300c": "\u300d",
    "\u3010": "\u3011",
    "\uff08": "\uff09",
    "(": ")",
    "[": "]",
    "{": "}",
}
PREFERRED_BOUNDARIES = {"\u3002", "\uff01", "\uff1f", "!", "?", "\u2026", "\n"}
SECONDARY_BOUNDARIES = {"\u3001", "\uff0c", ","}
SHIFTABLE_LEADING_CHARS = {
    "\u266a",
    "\u266b",
    "\u266c",
    "\u2669",
    "\u266d",
    "\u266f",
    "\u2606",
    "\u2605",
    "\u2661",
    "\u2665",
    "\u2764",
    "\u2b50",
    "\u2728",
    "\ud83c\udf1f",
    "\u26a1",
    "\ud83d\udd25",
    "\ud83d\udca6",
}
PUNCT_TO_PERIOD_PATTERN = re.compile(r"[!\uff01\?\uff1f]+")
ELLIPSIS_PATTERN = re.compile(r"(?:\u2026|\.{3,}|…{2,})+")
REPEATED_PERIOD_PATTERN = re.compile(r"\u3002{2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
LOGGER = logging.getLogger("irodori_perfect_api")
API_LOG_PATH = APP_ROOT / "irodori_api.log"
if not any(
    isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == API_LOG_PATH
    for handler in LOGGER.handlers
):
    file_handler = logging.FileHandler(API_LOG_PATH, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(file_handler)
LOGGER.propagate = True


@dataclass
class ChunkPlan:
    index: int
    text: str
    reason: str


@dataclass
class ChunkEvaluation:
    chunk_index: int
    boundary_reason: str
    original_text: str
    sanitized_text: str
    normalized_text: str
    chunk_char_len: int
    skipped_reason: str | None


class HealthResponse(BaseModel):
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    vram_total_mb: float | None
    vram_reserved_mb: float | None
    vram_allocated_mb: float | None
    runtime_loaded: bool
    checkpoint: str
    codec_repo: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    startup_error: str | None = None


class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    ref_wav: str | None = None
    ref_latent: str | None = None
    ref_embed: str | None = None
    speaker_name: str | None = None
    profile_path: str | None = None
    embed_path: str | None = None
    no_ref: bool = False
    output_path: str | None = None
    caption: str | None = None
    num_steps: int = 40
    seed: int | None = None
    device: str = DEFAULT_DEVICE
    chunk_soft_chars: int = 140
    chunk_hard_chars: int = 220
    chunk_min_chars: int = 48
    duration_scale: float = 1.0
    context_kv_cache: bool = True
    ref_normalize_db: float | None = DEFAULT_REF_NORMALIZE_DB
    decode_mode: str = "sequential"
    trim_tail: bool = True
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    seconds: float | None = None
    max_ref_seconds: float | None = 30.0
    max_seconds: float = 30.0
    target_chunk_seconds: float = 22.0
    hard_chunk_seconds: float = 26.0
    max_chunk_chars: int = 110
    auto_split_by_estimated_duration: bool = True
    retry_split_on_truncation: bool = True
    max_split_retries: int = 1
    append_silence_ms: float = 0.0
    chunk_gap_ms: int = 0
    sanitize_symbols: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> "VoiceRequest":
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0.")
        if self.chunk_min_chars <= 0:
            raise ValueError("chunk_min_chars must be > 0.")
        if self.chunk_soft_chars < self.chunk_min_chars:
            raise ValueError("chunk_soft_chars must be >= chunk_min_chars.")
        if self.chunk_hard_chars < self.chunk_soft_chars:
            raise ValueError("chunk_hard_chars must be >= chunk_soft_chars.")
        if self.device not in {"cuda:0", "cpu"}:
            raise ValueError("device must be either 'cuda:0' or 'cpu'.")
        if self.decode_mode not in {"sequential", "batch"}:
            raise ValueError("decode_mode must be 'sequential' or 'batch'.")
        if self.duration_scale <= 0:
            raise ValueError("duration_scale must be > 0.")
        if self.max_seconds <= 0:
            raise ValueError("max_seconds must be > 0.")
        if self.target_chunk_seconds <= 0:
            raise ValueError("target_chunk_seconds must be > 0.")
        if self.hard_chunk_seconds < self.target_chunk_seconds:
            raise ValueError("hard_chunk_seconds must be >= target_chunk_seconds.")
        if self.max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars must be > 0.")
        if self.max_split_retries < 0:
            raise ValueError("max_split_retries must be >= 0.")
        if self.append_silence_ms < 0:
            raise ValueError("append_silence_ms must be >= 0.")
        if self.chunk_gap_ms < 0:
            raise ValueError("chunk_gap_ms must be >= 0.")
        ref_count = sum(
            1
            for value in [
                self.ref_wav,
                self.ref_latent,
                self.ref_embed or self.profile_path or self.embed_path,
            ]
            if value
        )
        if self.speaker_name and ref_count > 0:
            raise ValueError("speaker_name cannot be combined with ref_embed/profile_path/embed_path/ref_wav/ref_latent.")
        if self.no_ref and ref_count > 0:
            raise ValueError("no_ref=true cannot be combined with reference inputs.")
        return self


class RuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[RuntimeKey, InferenceRuntime] = {}
        self._startup_error: str | None = None
        self._startup_traceback: str | None = None
        self._checkpoint_path: str | None = None

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    def _precision_for_device(self, device: str) -> str:
        return "bf16" if device.startswith("cuda") else "fp32"

    def _resolve_checkpoint_path(self) -> str:
        if self._checkpoint_path is None:
            self._checkpoint_path = hf_hub_download(
                repo_id=DEFAULT_CHECKPOINT_REPO,
                filename="model.safetensors",
            )
            LOGGER.info("Resolved checkpoint: %s", self._checkpoint_path)
        return self._checkpoint_path

    def make_key(self, device: str) -> RuntimeKey:
        precision = self._precision_for_device(device)
        return RuntimeKey(
            checkpoint=self._resolve_checkpoint_path(),
            model_device=device,
            codec_repo=DEFAULT_CODEC_REPO,
            model_precision=precision,
            codec_device=device,
            codec_precision=precision,
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )

    def get_runtime(self, device: str) -> tuple[InferenceRuntime, bool]:
        key = self.make_key(device)
        with self._lock:
            runtime = self._runtimes.get(key)
            if runtime is not None:
                LOGGER.info("Runtime cache hit for device=%s", device)
                return runtime, False
            LOGGER.info("Loading runtime for device=%s", device)
            runtime = InferenceRuntime.from_key(key)
            self._runtimes[key] = runtime
            return runtime, True

    def warmup_runtime(self, device: str) -> None:
        runtime, created = self.get_runtime(device)
        if not created:
            LOGGER.info("Warmup skipped because runtime already exists for device=%s", device)
            return
        warmup_req = SamplingRequest(
            text="\u3053\u3093\u306b\u3061\u306f\u3002",
            no_ref=True,
            num_steps=4,
            duration_scale=1.0,
            context_kv_cache=True,
            ref_normalize_db=DEFAULT_REF_NORMALIZE_DB,
            decode_mode="sequential",
            trim_tail=True,
            cfg_scale_text=3.0,
            cfg_scale_speaker=5.0,
            cfg_guidance_mode="independent",
            seconds=None,
            max_ref_seconds=30.0,
        )
        LOGGER.info("Running warmup synthesize on device=%s", device)
        runtime.synthesize(warmup_req, log_fn=LOGGER.info)
        LOGGER.info("Warmup completed on device=%s", device)

    def load_default_runtime(self) -> None:
        try:
            self.warmup_runtime(DEFAULT_DEVICE)
            self._startup_error = None
            self._startup_traceback = None
        except Exception as exc:
            self._startup_error = str(exc)
            self._startup_traceback = traceback.format_exc()
            LOGGER.exception("Failed to load default runtime on %s", DEFAULT_DEVICE)

    def default_runtime_status(self) -> tuple[RuntimeKey | None, InferenceRuntime | None]:
        checkpoint = self._checkpoint_path
        if checkpoint is None:
            return None, None
        key = self.make_key(DEFAULT_DEVICE)
        return key, self._runtimes.get(key)


RUNTIME_MANAGER = RuntimeManager()


def _shiftable_prefix_length(text: str) -> int:
    length = 0
    for char in text:
        if char in SHIFTABLE_LEADING_CHARS or unicodedata.category(char).startswith("S"):
            length += 1
            continue
        if char in {"!", "！", "?", "？", "…"}:
            length += 1
            continue
        break
    return length


def _strip_and_log_chunk(text: str) -> str:
    return text.strip()


def sanitize_text_for_tts(text: str) -> tuple[str, bool]:
    sanitized = text.replace("\r\n", "\n").replace("\r", "\n")
    sanitized = ELLIPSIS_PATTERN.sub("\u3002", sanitized)
    sanitized = PUNCT_TO_PERIOD_PATTERN.sub("\u3002", sanitized)
    sanitized = REPEATED_PERIOD_PATTERN.sub("\u3002", sanitized)
    sanitized = WHITESPACE_PATTERN.sub(" ", sanitized)
    sanitized = sanitized.strip()
    return sanitized, sanitized != text.strip()


def _has_meaningful_content(text: str) -> bool:
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith(("L", "N")):
            return True
    return False


def evaluate_chunk(
    *,
    chunk_index: int,
    boundary_reason: str,
    original_text: str,
    sanitize_symbols: bool,
) -> ChunkEvaluation:
    if sanitize_symbols:
        sanitized_text, _ = sanitize_text_for_tts(original_text)
    else:
        sanitized_text = original_text.strip()
    normalized_text = normalize_text(sanitized_text).strip()
    skipped_reason = None
    if sanitized_text == "":
        skipped_reason = "empty_after_sanitize"
    elif normalized_text == "":
        skipped_reason = "empty_after_normalize"
    elif not _has_meaningful_content(normalized_text):
        skipped_reason = "punctuation_only_chunk"
    return ChunkEvaluation(
        chunk_index=chunk_index,
        boundary_reason=boundary_reason,
        original_text=original_text,
        sanitized_text=sanitized_text,
        normalized_text=normalized_text,
        chunk_char_len=len(original_text),
        skipped_reason=skipped_reason,
    )


def _find_split_index(text: str, start: int, soft_limit: int, hard_limit: int) -> tuple[int, str]:
    stack: list[str] = []
    preferred_idx: int | None = None
    secondary_idx: int | None = None
    newline_idx: int | None = None
    text_len = len(text)
    max_scan = min(text_len, start + hard_limit + soft_limit)
    idx = start
    while idx < max_scan:
        char = text[idx]
        if char in BRACKET_PAIRS:
            stack.append(BRACKET_PAIRS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        if stack:
            idx += 1
            continue
        if char == "\n":
            newline_idx = idx + 1
            if idx - start >= soft_limit:
                return newline_idx, "newline_boundary"
        elif char in PREFERRED_BOUNDARIES:
            preferred_idx = idx + 1
            if idx - start >= soft_limit:
                return preferred_idx, f"sentence_boundary:{repr(char)}"
        elif char in SECONDARY_BOUNDARIES or char.isspace():
            secondary_idx = idx + 1
        if idx - start + 1 >= hard_limit and preferred_idx is not None:
            return preferred_idx, "hard_limit_sentence_boundary"
        idx += 1
    if preferred_idx is not None:
        return preferred_idx, "trailing_sentence_boundary"
    if newline_idx is not None:
        return newline_idx, "trailing_newline_boundary"
    if secondary_idx is not None:
        return secondary_idx, "secondary_boundary"
    return min(text_len, start + hard_limit), "forced_hard_limit"


def split_text_into_chunks(
    text: str,
    *,
    soft_limit: int,
    hard_limit: int,
    min_chars: int,
) -> list[ChunkPlan]:
    source = text.strip()
    if not source:
        return []
    chunks: list[ChunkPlan] = []
    start = 0
    index = 1
    while start < len(source):
        remaining = len(source) - start
        if remaining <= hard_limit:
            chunk_text = _strip_and_log_chunk(source[start:])
            if chunk_text:
                chunks.append(ChunkPlan(index=index, text=chunk_text, reason="final_chunk"))
            break
        cut, reason = _find_split_index(source, start, soft_limit, hard_limit)
        chunk_text = _strip_and_log_chunk(source[start:cut])
        if len(chunk_text) < min_chars and cut < len(source):
            next_cut, next_reason = _find_split_index(source, cut, soft_limit, hard_limit)
            extended = _strip_and_log_chunk(source[start:next_cut])
            if extended:
                chunk_text = extended
                cut = next_cut
                reason = f"{reason}+merge_small_chunk->{next_reason}"
        if not chunk_text:
            cut = min(len(source), start + hard_limit)
            chunk_text = _strip_and_log_chunk(source[start:cut])
            reason = "forced_nonempty_chunk"
        chunks.append(ChunkPlan(index=index, text=chunk_text, reason=reason))
        start = cut
        index += 1

    for idx in range(1, len(chunks)):
        prefix_len = _shiftable_prefix_length(chunks[idx].text)
        if prefix_len <= 0:
            continue
        prefix = chunks[idx].text[:prefix_len]
        chunks[idx - 1].text = f"{chunks[idx - 1].text}{prefix}"
        chunks[idx].text = chunks[idx].text[prefix_len:].lstrip()
        chunks[idx].reason = f"{chunks[idx].reason}+shifted_prefix"
    chunks = [chunk for chunk in chunks if chunk.text]
    for new_index, chunk in enumerate(chunks, start=1):
        chunk.index = new_index
        LOGGER.info("Chunk %s boundary=%s text=%r", chunk.index, chunk.reason, chunk.text)
    return chunks


def resolve_output_paths(output_path: str | None) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if output_path is None:
        base_dir = DEFAULT_OUTPUT_DIR / stamp
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "final.wav", base_dir / "chunks"
    target = Path(output_path).expanduser()
    if target.suffix.lower() == ".wav":
        target.parent.mkdir(parents=True, exist_ok=True)
        chunk_dir = target.parent / f"{target.stem}_chunks"
        return target, chunk_dir
    target.mkdir(parents=True, exist_ok=True)
    return target / "final.wav", target / "chunks"


def get_device_vram(device_name: str) -> tuple[float | None, float | None, float | None, str | None]:
    if not torch.cuda.is_available():
        return None, None, None, None
    device = torch.device(device_name)
    props = torch.cuda.get_device_properties(device)
    total_mb = props.total_memory / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    return total_mb, reserved_mb, allocated_mb, props.name


def build_health() -> HealthResponse:
    key, runtime = RUNTIME_MANAGER.default_runtime_status()
    device_name = DEFAULT_DEVICE
    total_mb, reserved_mb, allocated_mb, cuda_name = get_device_vram(device_name)
    if key is None:
        precision = "bf16" if DEFAULT_DEVICE.startswith("cuda") else "fp32"
        key = RuntimeKey(
            checkpoint=RUNTIME_MANAGER.checkpoint_path or DEFAULT_CHECKPOINT_REPO,
            model_device=DEFAULT_DEVICE,
            codec_repo=DEFAULT_CODEC_REPO,
            model_precision=precision,
            codec_device=DEFAULT_DEVICE,
            codec_precision=precision,
        )
    return HealthResponse(
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device_name=cuda_name,
        vram_total_mb=round(total_mb, 2) if total_mb is not None else None,
        vram_reserved_mb=round(reserved_mb, 2) if reserved_mb is not None else None,
        vram_allocated_mb=round(allocated_mb, 2) if allocated_mb is not None else None,
        runtime_loaded=runtime is not None,
        checkpoint=key.checkpoint,
        codec_repo=key.codec_repo,
        model_device=key.model_device,
        codec_device=key.codec_device,
        model_precision=key.model_precision,
        codec_precision=key.codec_precision,
        startup_error=RUNTIME_MANAGER.startup_error,
    )


def _measure_audio_seconds(audio: torch.Tensor, sample_rate: int) -> float:
    return float(audio.shape[-1]) / float(sample_rate)


def _append_silence(audio: torch.Tensor, sample_rate: int, silence_ms: float) -> torch.Tensor:
    if silence_ms <= 0:
        return audio
    silence_samples = max(1, int(sample_rate * (silence_ms / 1000.0)))
    silence = torch.zeros((audio.shape[0], silence_samples), dtype=audio.dtype)
    return torch.cat([audio, silence], dim=-1)


def _extract_predicted_seconds(messages: list[str]) -> float | None:
    for message in messages:
        match = re.search(r"\((\d+(?:\.\d+)?)s\)", message)
        if match:
            return float(match.group(1))
    return None


def derive_chunk_limits(req: VoiceRequest) -> tuple[int, int]:
    if not req.auto_split_by_estimated_duration:
        return req.chunk_soft_chars, req.chunk_hard_chars
    chars_per_second = 5.0
    target_chars = max(req.chunk_min_chars, int(req.target_chunk_seconds * chars_per_second))
    hard_chars = max(req.chunk_min_chars, int(req.hard_chunk_seconds * chars_per_second))
    soft_limit = max(
        req.chunk_min_chars,
        min(req.chunk_soft_chars, req.max_chunk_chars, target_chars),
    )
    hard_limit = max(
        soft_limit,
        min(req.chunk_hard_chars, max(req.max_chunk_chars + 10, soft_limit), hard_chars),
    )
    return soft_limit, hard_limit


def derive_retry_limits(soft_limit: int, hard_limit: int, min_chars: int) -> tuple[int, int]:
    next_soft = max(min_chars, int(soft_limit * 0.7))
    next_hard = max(next_soft, int(hard_limit * 0.7))
    return next_soft, next_hard


def _is_truncation_suspected(
    *,
    predicted_seconds: float | None,
    actual_seconds: float,
    trim_tail: bool,
    normalized_text: str,
) -> bool:
    if predicted_seconds is None or not trim_tail:
        return False
    sentence_like_end = normalized_text.endswith(("\u3002", "\u300d", "\u3011", "\uff09"))
    shortfall = predicted_seconds - actual_seconds
    return sentence_like_end and shortfall > 0.18


def _load_reference_field(req: VoiceRequest) -> tuple[str | None, str | None, str | None]:
    ref_embed = req.ref_embed or req.profile_path or req.embed_path
    if ref_embed is None and req.speaker_name:
        speaker_name = req.speaker_name.strip()
        candidates = [
            DEFAULT_EMBED_PROFILE_DIR / f"{speaker_name}.speaker.safetensors",
            DEFAULT_EMBED_PROFILE_DIR / speaker_name,
        ]
        resolved = next((path for path in candidates if path.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"speaker_name could not be resolved in {DEFAULT_EMBED_PROFILE_DIR}: {speaker_name}"
            )
        ref_embed = str(resolved)
    return req.ref_wav, req.ref_latent, ref_embed


def _merge_audio_files(
    chunk_wav_paths: list[Path],
    final_wav_path: Path,
    *,
    chunk_gap_ms: int,
) -> tuple[float, float, int, float]:
    merged_audio: list[torch.Tensor] = []
    sample_rate: int | None = None
    chunk_audio_seconds_sum = 0.0
    for path in chunk_wav_paths:
        wav, sr = torchaudio.load(str(path))
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            raise RuntimeError(
                f"Mismatched sample rate while merging chunks: expected {sample_rate}, got {sr}"
            )
        merged_audio.append(wav)
        chunk_audio_seconds_sum += _measure_audio_seconds(wav, sr)
    if not merged_audio or sample_rate is None:
        raise RuntimeError("No chunk audio available to merge.")
    inserted_gap_count = max(0, len(merged_audio) - 1)
    inserted_silence_seconds = 0.0
    if chunk_gap_ms > 0 and inserted_gap_count > 0:
        gap_samples = max(1, int(sample_rate * (chunk_gap_ms / 1000.0)))
        gap_audio = torch.zeros((merged_audio[0].shape[0], gap_samples), dtype=merged_audio[0].dtype)
        merged_with_gaps: list[torch.Tensor] = []
        for index, wav in enumerate(merged_audio):
            merged_with_gaps.append(wav)
            if index < len(merged_audio) - 1:
                merged_with_gaps.append(gap_audio)
        final_audio = torch.cat(merged_with_gaps, dim=-1)
        inserted_silence_seconds = (gap_samples * inserted_gap_count) / float(sample_rate)
    else:
        final_audio = torch.cat(merged_audio, dim=-1)
    save_wav(final_wav_path, final_audio, sample_rate)
    final_audio_seconds = _measure_audio_seconds(final_audio, sample_rate)
    return final_audio_seconds, chunk_audio_seconds_sum, inserted_gap_count, inserted_silence_seconds


def synthesize_chunks(req: VoiceRequest) -> dict[str, Any]:
    runtime, created = RUNTIME_MANAGER.get_runtime(req.device)
    ref_wav, ref_latent, ref_embed = _load_reference_field(req)
    if runtime.model_cfg.use_speaker_condition_resolved and not (
        req.no_ref or ref_wav or ref_latent or ref_embed
    ):
        raise ValueError(
            "This checkpoint expects speaker conditioning. Provide ref_wav, ref_latent, ref_embed, or set no_ref=true."
        )
    if req.sanitize_symbols:
        final_input_text, text_sanitized = sanitize_text_for_tts(req.text)
    else:
        final_input_text = req.text.strip()
        text_sanitized = False
    LOGGER.info("Input text original=%r", req.text)
    LOGGER.info("Input text sanitized=%r", final_input_text)
    LOGGER.info("sanitize_symbols=%s", req.sanitize_symbols)
    final_wav_path, chunk_dir = resolve_output_paths(req.output_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    soft_limit, hard_limit = derive_chunk_limits(req)
    LOGGER.info(
        "Chunk planning soft_limit=%s hard_limit=%s max_seconds=%s target_chunk_seconds=%s hard_chunk_seconds=%s",
        soft_limit,
        hard_limit,
        req.max_seconds,
        req.target_chunk_seconds,
        req.hard_chunk_seconds,
    )
    chunks = split_text_into_chunks(
        final_input_text,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        min_chars=req.chunk_min_chars,
    )
    if not chunks:
        raise ValueError("No non-empty chunks were produced from the input text.")

    wall_start = time.perf_counter()
    chunk_wav_paths: list[Path] = []
    per_chunk: list[dict[str, Any]] = []
    skipped_chunks: list[dict[str, Any]] = []
    retried_chunks: list[dict[str, Any]] = []
    output_chunk_index = 0
    retry_split_count = 0
    save_file_counter = 0

    def process_chunk(chunk: ChunkPlan, *, retry_count: int, local_soft_limit: int, local_hard_limit: int) -> None:
        nonlocal output_chunk_index, retry_split_count, save_file_counter
        chunk_total_start = time.perf_counter()
        evaluation = evaluate_chunk(
            chunk_index=chunk.index,
            boundary_reason=chunk.reason,
            original_text=chunk.text,
            sanitize_symbols=req.sanitize_symbols,
        )
        LOGGER.info(
            "Chunk %s original=%r normalized=%r char_len=%s skipped_reason=%s",
            evaluation.chunk_index,
            evaluation.original_text,
            evaluation.normalized_text,
            evaluation.chunk_char_len,
            evaluation.skipped_reason,
        )
        if evaluation.skipped_reason is not None:
            skipped_chunks.append(
                {
                    "chunk_index": evaluation.chunk_index,
                    "boundary_reason": evaluation.boundary_reason,
                    "original_text": evaluation.original_text,
                    "sanitized_text": evaluation.sanitized_text,
                    "normalized_text": evaluation.normalized_text,
                    "chunk_char_len": evaluation.chunk_char_len,
                    "skipped_reason": evaluation.skipped_reason,
                    "sanitize_symbols": req.sanitize_symbols,
                }
            )
            return
        norm_start = time.perf_counter()
        normalized_preview = evaluation.normalized_text
        normalize_sec = time.perf_counter() - norm_start
        chunk_logs: list[str] = []
        chunk_logs.append(f"original_chunk_text={evaluation.original_text!r}")
        chunk_logs.append(f"sanitized_chunk_text={evaluation.sanitized_text!r}")
        chunk_logs.append(f"normalized_chunk_text={evaluation.normalized_text!r}")
        chunk_logs.append(f"chunk_char_len={evaluation.chunk_char_len}")
        chunk_logs.append(f"request_text={evaluation.sanitized_text!r}")
        chunk_logs.append("skipped_reason=None")
        chunk_logs.append(f"sanitize_symbols={req.sanitize_symbols}")
        sampling_request = SamplingRequest(
            text=evaluation.sanitized_text,
            caption=req.caption,
            ref_wav=ref_wav,
            ref_latent=ref_latent,
            ref_embed=ref_embed,
            no_ref=req.no_ref,
            ref_normalize_db=req.ref_normalize_db,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode=req.decode_mode,
            seconds=req.seconds,
            duration_scale=req.duration_scale,
            max_ref_seconds=req.max_ref_seconds,
            max_seconds=req.max_seconds,
            num_steps=req.num_steps,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_speaker=req.cfg_scale_speaker,
            cfg_guidance_mode=req.cfg_guidance_mode,
            context_kv_cache=req.context_kv_cache,
            trim_tail=req.trim_tail,
            seed=req.seed,
        )
        result = runtime.synthesize(
            sampling_request,
            log_fn=lambda message, bucket=chunk_logs: bucket.append(message),
        )
        chunk_logs.append(f"used_seed={result.used_seed}")
        LOGGER.info("Chunk %s used_seed=%s", chunk.index, result.used_seed)
        LOGGER.info("Chunk %s sanitize_symbols=%s", chunk.index, req.sanitize_symbols)
        audio_to_save = _append_silence(result.audio, result.sample_rate, req.append_silence_ms)
        save_start = time.perf_counter()
        save_file_counter += 1
        chunk_path = chunk_dir / f"chunk_{save_file_counter:03d}.wav"
        save_wav(chunk_path, audio_to_save, result.sample_rate)
        save_sec = time.perf_counter() - save_start
        total_sec = time.perf_counter() - chunk_total_start
        timings = {name: round(sec, 6) for name, sec in result.stage_timings}
        timings["normalize_text"] = round(normalize_sec, 6)
        timings["save_wav"] = round(save_sec, 6)
        timings["total"] = round(total_sec, 6)
        predicted_seconds = _extract_predicted_seconds(result.messages)
        audio_seconds = round(_measure_audio_seconds(audio_to_save, result.sample_rate), 6)
        hit_max_seconds = bool(
            (predicted_seconds is not None and predicted_seconds >= req.max_seconds - 0.05)
            or audio_seconds >= req.max_seconds - 0.05
        )
        was_truncated_suspected = hit_max_seconds or _is_truncation_suspected(
            predicted_seconds=predicted_seconds,
            actual_seconds=audio_seconds,
            trim_tail=req.trim_tail,
            normalized_text=normalized_preview,
        )
        truncation_reason = "hit_max_seconds" if hit_max_seconds else None
        if was_truncated_suspected:
            chunk_logs.append(f"truncation_reason={truncation_reason or 'tail_trim_suspected'}")
            LOGGER.warning(
                "Chunk %s suspected truncation retry_count=%s predicted=%s audio=%s reason=%s",
                chunk.index,
                retry_count,
                predicted_seconds,
                audio_seconds,
                truncation_reason or "tail_trim_suspected",
            )
        if (
            was_truncated_suspected
            and req.retry_split_on_truncation
            and retry_count < req.max_split_retries
            and len(evaluation.sanitized_text) > req.chunk_min_chars * 2
        ):
            next_soft_limit, next_hard_limit = derive_retry_limits(
                local_soft_limit,
                local_hard_limit,
                req.chunk_min_chars,
            )
            retry_chunks = split_text_into_chunks(
                evaluation.sanitized_text,
                soft_limit=next_soft_limit,
                hard_limit=next_hard_limit,
                min_chars=req.chunk_min_chars,
            )
            if len(retry_chunks) > 1:
                retry_split_count += 1
                retried_chunks.append(
                    {
                        "chunk_index": chunk.index,
                        "retry_reason": truncation_reason or "suspected_truncation",
                        "retry_count": retry_count + 1,
                        "retry_split_count": len(retry_chunks),
                        "original_text": evaluation.original_text,
                        "request_text": evaluation.sanitized_text,
                        "predicted_seconds": predicted_seconds,
                        "audio_seconds": audio_seconds,
                        "sanitize_symbols": req.sanitize_symbols,
                    }
                )
                for retry_chunk in retry_chunks:
                    process_chunk(
                        retry_chunk,
                        retry_count=retry_count + 1,
                        local_soft_limit=next_soft_limit,
                        local_hard_limit=next_hard_limit,
                    )
                return
        output_chunk_index += 1
        per_chunk.append(
            {
                "chunk_index": output_chunk_index,
                "source_chunk_index": chunk.index,
                "boundary_reason": chunk.reason,
                "text": evaluation.sanitized_text,
                "original_text": evaluation.original_text,
                "normalized_text": normalized_preview,
                "request_text": evaluation.sanitized_text,
                "skipped_reason": None,
                "chunk_char_len": evaluation.chunk_char_len,
                "text_char_len": len(evaluation.sanitized_text),
                "seed": result.used_seed,
                "used_seed": result.used_seed,
                "sanitize_symbols": req.sanitize_symbols,
                "audio_seconds": audio_seconds,
                "duration_scale": req.duration_scale,
                "trim_tail": req.trim_tail,
                "append_silence_ms": req.append_silence_ms,
                "max_seconds": req.max_seconds,
                "predicted_seconds": predicted_seconds,
                "was_truncated_suspected": was_truncated_suspected,
                "truncation_reason": truncation_reason,
                "retry_count": retry_count,
                "timings": timings,
                "stage_timings": timings,
                "runtime_messages": result.messages,
                "runtime_logs": chunk_logs,
            }
        )
        chunk_wav_paths.append(chunk_path)
    for chunk in chunks:
        process_chunk(
            chunk,
            retry_count=0,
            local_soft_limit=soft_limit,
            local_hard_limit=hard_limit,
        )
    if not chunk_wav_paths:
        wall_time = time.perf_counter() - wall_start
        return {
            "final_wav_path": None,
            "chunk_wav_paths": [],
            "total_audio_seconds": 0.0,
            "chunk_audio_seconds_sum": 0.0,
            "final_audio_seconds": 0.0,
            "wall_time": round(wall_time, 6),
            "chunk_count": 0,
            "skipped_chunks": skipped_chunks,
            "skipped_chunk_count": len(skipped_chunks),
            "text_sanitized": text_sanitized,
            "final_input_text": final_input_text,
            "sanitize_symbols": req.sanitize_symbols,
            "chunk_gap_ms": req.chunk_gap_ms,
            "inserted_gap_count": 0,
            "inserted_silence_seconds": 0.0,
            "per_chunk": [],
            "per_chunk_timing": [],
            "max_seconds": req.max_seconds,
            "target_chunk_seconds": req.target_chunk_seconds,
            "hard_chunk_seconds": req.hard_chunk_seconds,
            "has_truncated_suspected": False,
            "truncated_chunk_count": 0,
            "retried_chunks": retried_chunks,
            "retry_reason": None,
            "retry_split_count": retry_split_count,
            "used_ref_embed": ref_embed,
            "runtime_cache_hit": not created,
            "runtime_device": req.device,
        }
    final_audio_seconds, chunk_audio_seconds_sum, inserted_gap_count, inserted_silence_seconds = _merge_audio_files(
        chunk_wav_paths,
        final_wav_path,
        chunk_gap_ms=req.chunk_gap_ms,
    )
    wall_time = time.perf_counter() - wall_start
    truncated_chunk_count = sum(1 for chunk in per_chunk if chunk["was_truncated_suspected"])
    has_truncated_suspected = truncated_chunk_count > 0
    return {
        "final_wav_path": str(final_wav_path),
        "chunk_wav_paths": [str(path) for path in chunk_wav_paths],
        "total_audio_seconds": round(final_audio_seconds, 6),
        "chunk_audio_seconds_sum": round(chunk_audio_seconds_sum, 6),
        "final_audio_seconds": round(final_audio_seconds, 6),
        "wall_time": round(wall_time, 6),
        "chunk_count": len(chunk_wav_paths),
        "skipped_chunks": skipped_chunks,
        "skipped_chunk_count": len(skipped_chunks),
        "text_sanitized": text_sanitized,
        "final_input_text": final_input_text,
        "sanitize_symbols": req.sanitize_symbols,
        "chunk_gap_ms": req.chunk_gap_ms,
        "inserted_gap_count": inserted_gap_count,
        "inserted_silence_seconds": round(inserted_silence_seconds, 6),
        "max_seconds": req.max_seconds,
        "target_chunk_seconds": req.target_chunk_seconds,
        "hard_chunk_seconds": req.hard_chunk_seconds,
        "has_truncated_suspected": has_truncated_suspected,
        "truncated_chunk_count": truncated_chunk_count,
        "retried_chunks": retried_chunks,
        "retry_reason": "hit_max_seconds" if retried_chunks else None,
        "retry_split_count": retry_split_count,
        "used_ref_embed": ref_embed,
        "per_chunk": per_chunk,
        "per_chunk_timing": per_chunk,
        "runtime_cache_hit": not created,
        "runtime_device": req.device,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_MANAGER.load_default_runtime()
    yield


app = FastAPI(title="Irodori Perfect API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return build_health()


@app.post("/voice")
def voice(req: VoiceRequest) -> dict[str, Any]:
    try:
        return synthesize_chunks(req)
    except Exception as exc:
        LOGGER.exception("Voice synthesis failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        ) from exc
