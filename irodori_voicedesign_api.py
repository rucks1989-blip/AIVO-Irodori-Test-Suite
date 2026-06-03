from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
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

DEFAULT_CHECKPOINT_REPO = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_REF_NORMALIZE_DB = -16.0
DEFAULT_OUTPUT_DIR = APP_ROOT / "outputs" / "irodori_voicedesign_api"
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
LOGGER = logging.getLogger("irodori_voicedesign_api")
API_LOG_PATH = APP_ROOT / "irodori_voicedesign_api.log"
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
    runtime_loaded: bool
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    vram_total_mb: float | None
    vram_allocated_mb: float | None
    model_name: str
    device: str
    supports_caption: bool
    supports_ref_embed: bool
    supports_ref_wav: bool
    startup_error: str | None = None


class VoiceRequest(BaseModel):
    text: str = Field(..., min_length=1)
    caption: str | None = None
    speaker_name: str | None = None
    ref_embed: str | None = None
    ref_wav: str | None = None
    output_path: str | None = None
    seed: int | None = None
    sanitize_symbols: bool = True
    chunk_gap_ms: int = 0
    max_seconds: float = 30.0
    target_chunk_seconds: float = 22.0
    hard_chunk_seconds: float = 26.0
    max_chunk_chars: int = 110
    num_steps: int = 40
    duration_scale: float = 1.0
    context_kv_cache: bool = True
    ref_normalize_db: float | None = DEFAULT_REF_NORMALIZE_DB
    decode_mode: str = "sequential"
    trim_tail: bool = True
    cfg_scale_text: float = 3.0
    cfg_scale_caption: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_guidance_mode: str = "independent"
    max_ref_seconds: float | None = 30.0
    device: str = DEFAULT_DEVICE
    chunk_soft_chars: int = 140
    chunk_hard_chars: int = 220
    chunk_min_chars: int = 48
    auto_split_by_estimated_duration: bool = True

    @model_validator(mode="after")
    def validate_request(self) -> "VoiceRequest":
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0.")
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
        if self.chunk_min_chars <= 0:
            raise ValueError("chunk_min_chars must be > 0.")
        if self.chunk_soft_chars < self.chunk_min_chars:
            raise ValueError("chunk_soft_chars must be >= chunk_min_chars.")
        if self.chunk_hard_chars < self.chunk_soft_chars:
            raise ValueError("chunk_hard_chars must be >= chunk_soft_chars.")
        if self.chunk_gap_ms < 0:
            raise ValueError("chunk_gap_ms must be >= 0.")
        ref_count = sum(1 for value in [self.speaker_name, self.ref_embed, self.ref_wav] if value)
        if ref_count > 1:
            raise ValueError("speaker_name, ref_embed, and ref_wav are mutually exclusive.")
        return self


class RuntimeManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[RuntimeKey, InferenceRuntime] = {}
        self._startup_error: str | None = None
        self._checkpoint_path: str | None = None

    @property
    def startup_error(self) -> str | None:
        return self._startup_error

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    def _resolve_checkpoint_path(self) -> str:
        if self._checkpoint_path is None:
            self._checkpoint_path = hf_hub_download(
                repo_id=DEFAULT_CHECKPOINT_REPO,
                filename="model.safetensors",
            )
            LOGGER.info("Resolved VoiceDesign checkpoint: %s", self._checkpoint_path)
        return self._checkpoint_path

    def make_key(self, device: str) -> RuntimeKey:
        return RuntimeKey(
            checkpoint=self._resolve_checkpoint_path(),
            model_device=device,
            codec_repo=DEFAULT_CODEC_REPO,
            model_precision="bf16" if device.startswith("cuda") else "fp32",
            codec_device=device,
            codec_precision="bf16" if device.startswith("cuda") else "fp32",
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
                LOGGER.info("VoiceDesign runtime cache hit for device=%s", device)
                return runtime, False
            LOGGER.info("Loading VoiceDesign runtime for device=%s", device)
            runtime = InferenceRuntime.from_key(key)
            self._runtimes[key] = runtime
            return runtime, True

    def warmup_runtime(self, device: str) -> None:
        runtime, created = self.get_runtime(device)
        if not created:
            return
        warmup_req = SamplingRequest(
            text="こんにちは。",
            caption="落ち着いた女性ナレーター",
            no_ref=True,
            num_steps=4,
            max_seconds=5.0,
            duration_scale=1.0,
            decode_mode="sequential",
            cfg_scale_text=3.0,
            cfg_scale_caption=3.0,
            cfg_scale_speaker=5.0,
            cfg_guidance_mode="independent",
            context_kv_cache=True,
            trim_tail=True,
        )
        runtime.synthesize(warmup_req, log_fn=LOGGER.info)

    def load_default_runtime(self) -> None:
        try:
            self.warmup_runtime(DEFAULT_DEVICE)
        except Exception as exc:  # noqa: BLE001
            self._startup_error = f"{exc.__class__.__name__}: {exc}"
            LOGGER.exception("VoiceDesign runtime warmup failed")

    def default_runtime_status(self) -> tuple[RuntimeKey | None, InferenceRuntime | None]:
        with self._lock:
            if not self._runtimes:
                return None, None
            key, runtime = next(iter(self._runtimes.items()))
            return key, runtime


RUNTIME_MANAGER = RuntimeManager()


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


def _shiftable_prefix_length(text: str) -> int:
    length = 0
    for char in text:
        if char in SHIFTABLE_LEADING_CHARS or unicodedata.category(char).startswith("S"):
            length += 1
            continue
        if char in {"!", "?", "\u2026"}:
            length += 1
            continue
        break
    return length


def _strip_and_log_chunk(text: str) -> str:
    return text.strip()


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


def split_text_into_chunks(text: str, *, soft_limit: int, hard_limit: int, min_chars: int) -> list[ChunkPlan]:
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
        LOGGER.info("VoiceDesign chunk %s boundary=%s text=%r", chunk.index, chunk.reason, chunk.text)
    return chunks


def derive_chunk_limits(req: VoiceRequest) -> tuple[int, int]:
    if not req.auto_split_by_estimated_duration:
        return req.chunk_soft_chars, req.chunk_hard_chars
    chars_per_second = 5.0
    target_chars = max(req.chunk_min_chars, int(req.target_chunk_seconds * chars_per_second))
    hard_chars = max(req.chunk_min_chars, int(req.hard_chunk_seconds * chars_per_second))
    soft_limit = max(req.chunk_min_chars, min(req.chunk_soft_chars, req.max_chunk_chars, target_chars))
    hard_limit = max(soft_limit, min(req.chunk_hard_chars, max(req.max_chunk_chars + 10, soft_limit), hard_chars))
    return soft_limit, hard_limit


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


def _measure_audio_seconds(audio: torch.Tensor, sample_rate: int) -> float:
    return float(audio.shape[-1]) / float(sample_rate)


def _extract_predicted_seconds(messages: list[str]) -> float | None:
    for message in messages:
        match = re.search(r"\((\d+(?:\.\d+)?)s\)", message)
        if match:
            return float(match.group(1))
    return None


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
            raise RuntimeError(f"Mismatched sample rate while merging chunks: expected {sample_rate}, got {sr}")
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


def get_device_vram(device_name: str) -> tuple[float | None, float | None, str | None]:
    if not torch.cuda.is_available():
        return None, None, None
    device = torch.device(device_name)
    props = torch.cuda.get_device_properties(device)
    total_mb = props.total_memory / (1024 * 1024)
    allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    return total_mb, allocated_mb, props.name


def _resolve_reference_inputs(req: VoiceRequest) -> tuple[str | None, str | None, bool]:
    resolved_ref_embed = req.ref_embed
    if resolved_ref_embed is None and req.speaker_name:
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
        resolved_ref_embed = str(resolved)
    effective_no_ref = not any([resolved_ref_embed, req.ref_wav])
    return resolved_ref_embed, req.ref_wav, effective_no_ref


def build_health() -> HealthResponse:
    key, runtime = RUNTIME_MANAGER.default_runtime_status()
    total_mb, allocated_mb, cuda_name = get_device_vram(DEFAULT_DEVICE)
    supports_caption = bool(runtime.model_cfg.use_caption_condition) if runtime is not None else True
    supports_ref = bool(runtime.model_cfg.use_speaker_condition_resolved) if runtime is not None else True
    return HealthResponse(
        runtime_loaded=runtime is not None,
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device_name=cuda_name,
        vram_total_mb=round(total_mb, 2) if total_mb is not None else None,
        vram_allocated_mb=round(allocated_mb, 2) if allocated_mb is not None else None,
        model_name=DEFAULT_CHECKPOINT_REPO,
        device=DEFAULT_DEVICE,
        supports_caption=supports_caption,
        supports_ref_embed=supports_ref,
        supports_ref_wav=supports_ref,
        startup_error=RUNTIME_MANAGER.startup_error,
    )


def synthesize_chunks(req: VoiceRequest) -> dict[str, Any]:
    runtime, created = RUNTIME_MANAGER.get_runtime(req.device)
    ref_embed, ref_wav, effective_no_ref = _resolve_reference_inputs(req)
    if req.sanitize_symbols:
        final_input_text, text_sanitized = sanitize_text_for_tts(req.text)
    else:
        final_input_text = req.text.strip()
        text_sanitized = False
    LOGGER.info("VoiceDesign input text original=%r", req.text)
    LOGGER.info("VoiceDesign input text sanitized=%r", final_input_text)
    LOGGER.info("VoiceDesign caption=%r", req.caption)
    LOGGER.info("VoiceDesign resolved_ref_embed=%r ref_wav=%r no_ref=%s", ref_embed, ref_wav, effective_no_ref)
    final_wav_path, chunk_dir = resolve_output_paths(req.output_path)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    soft_limit, hard_limit = derive_chunk_limits(req)
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
    save_file_counter = 0

    for chunk in chunks:
        evaluation = evaluate_chunk(
            chunk_index=chunk.index,
            boundary_reason=chunk.reason,
            original_text=chunk.text,
            sanitize_symbols=req.sanitize_symbols,
        )
        if evaluation.skipped_reason is not None:
            skipped_chunks.append(
                {
                    "chunk_index": evaluation.chunk_index,
                    "boundary_reason": evaluation.boundary_reason,
                    "original_text": evaluation.original_text,
                    "normalized_text": evaluation.normalized_text,
                    "request_text": evaluation.sanitized_text,
                    "skipped_reason": evaluation.skipped_reason,
                    "sanitize_symbols": req.sanitize_symbols,
                }
            )
            continue
        chunk_logs: list[str] = []
        sampling_request = SamplingRequest(
            text=evaluation.sanitized_text,
            caption=req.caption,
            ref_wav=ref_wav,
            ref_embed=ref_embed,
            ref_latent=None,
            no_ref=effective_no_ref,
            ref_normalize_db=req.ref_normalize_db,
            ref_ensure_max=True,
            num_candidates=1,
            decode_mode=req.decode_mode,
            seconds=None,
            duration_scale=req.duration_scale,
            max_ref_seconds=req.max_ref_seconds,
            max_seconds=req.max_seconds,
            num_steps=req.num_steps,
            cfg_scale_text=req.cfg_scale_text,
            cfg_scale_caption=req.cfg_scale_caption,
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
        save_file_counter += 1
        chunk_path = chunk_dir / f"chunk_{save_file_counter:03d}.wav"
        save_wav(chunk_path, result.audio, result.sample_rate)
        timings = {name: round(sec, 6) for name, sec in result.stage_timings}
        audio_seconds = round(_measure_audio_seconds(result.audio, result.sample_rate), 6)
        predicted_seconds = _extract_predicted_seconds(result.messages)
        per_chunk.append(
            {
                "chunk_index": len(per_chunk) + 1,
                "source_chunk_index": chunk.index,
                "boundary_reason": chunk.reason,
                "original_text": evaluation.original_text,
                "normalized_text": evaluation.normalized_text,
                "request_text": evaluation.sanitized_text,
                "chunk_char_len": evaluation.chunk_char_len,
                "seed": result.used_seed,
                "used_seed": result.used_seed,
                "sanitize_symbols": req.sanitize_symbols,
                "audio_seconds": audio_seconds,
                "predicted_seconds": predicted_seconds,
                "stage_timings": timings,
                "runtime_messages": result.messages,
                "runtime_logs": chunk_logs,
            }
        )
        chunk_wav_paths.append(chunk_path)

    if not chunk_wav_paths:
        wall_time = time.perf_counter() - wall_start
        return {
            "caption": req.caption,
            "final_wav_path": None,
            "chunk_wav_paths": [],
            "chunk_count": 0,
            "seed": req.seed,
            "used_ref_embed": ref_embed,
            "text_sanitized": text_sanitized,
            "final_input_text": final_input_text,
            "sanitize_symbols": req.sanitize_symbols,
            "chunk_gap_ms": req.chunk_gap_ms,
            "inserted_gap_count": 0,
            "inserted_silence_seconds": 0.0,
            "chunk_audio_seconds_sum": 0.0,
            "final_audio_seconds": 0.0,
            "total_audio_seconds": 0.0,
            "per_chunk": [],
            "skipped_chunks": skipped_chunks,
            "runtime_cache_hit": not created,
            "wall_time": round(wall_time, 6),
        }

    final_audio_seconds, chunk_audio_seconds_sum, inserted_gap_count, inserted_silence_seconds = _merge_audio_files(
        chunk_wav_paths,
        final_wav_path,
        chunk_gap_ms=req.chunk_gap_ms,
    )
    wall_time = time.perf_counter() - wall_start
    return {
        "caption": req.caption,
        "final_wav_path": str(final_wav_path),
        "chunk_wav_paths": [str(path) for path in chunk_wav_paths],
        "chunk_count": len(chunk_wav_paths),
        "seed": per_chunk[0]["seed"] if per_chunk else req.seed,
        "used_ref_embed": ref_embed,
        "text_sanitized": text_sanitized,
        "final_input_text": final_input_text,
        "sanitize_symbols": req.sanitize_symbols,
        "chunk_gap_ms": req.chunk_gap_ms,
        "inserted_gap_count": inserted_gap_count,
        "inserted_silence_seconds": round(inserted_silence_seconds, 6),
        "chunk_audio_seconds_sum": round(chunk_audio_seconds_sum, 6),
        "final_audio_seconds": round(final_audio_seconds, 6),
        "total_audio_seconds": round(final_audio_seconds, 6),
        "per_chunk": per_chunk,
        "skipped_chunks": skipped_chunks,
        "runtime_cache_hit": not created,
        "wall_time": round(wall_time, 6),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_MANAGER.load_default_runtime()
    yield


app = FastAPI(title="Irodori VoiceDesign API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return build_health()


@app.post("/voice")
def voice(req: VoiceRequest) -> dict[str, Any]:
    try:
        return synthesize_chunks(req)
    except Exception as exc:
        LOGGER.exception("VoiceDesign synthesis failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            },
        ) from exc
