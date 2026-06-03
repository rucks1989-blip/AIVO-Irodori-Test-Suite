from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8010"
REQUEST_TIMEOUT_SECONDS = 600
MAX_SIGNED_63BIT_INT = (1 << 63) - 1
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMBED_PROFILE_DIR = PROJECT_ROOT / "irodori_embed_profiles"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def list_speaker_profiles(profile_dir: Path) -> list[dict[str, str]]:
    profiles: list[dict[str, str]] = []
    if not profile_dir.exists():
        return profiles
    for path in sorted(profile_dir.glob("*.speaker.safetensors")):
        display_name = path.name.removesuffix(".speaker.safetensors")
        profiles.append(
            {
                "name": display_name,
                "path": str(path),
            }
        )
    return profiles


def build_request_payload(form_values: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": form_values["text"],
        "max_seconds": float(form_values["max_seconds"]),
        "target_chunk_seconds": float(form_values["target_chunk_seconds"]),
        "hard_chunk_seconds": float(form_values["hard_chunk_seconds"]),
        "max_chunk_chars": int(form_values["max_chunk_chars"]),
        "append_silence_ms": int(form_values["append_silence_ms"]),
        "chunk_gap_ms": int(form_values["chunk_gap_ms"]),
        "sanitize_symbols": bool(form_values["sanitize_symbols"]),
    }

    if form_values["num_steps_enabled"]:
        payload["num_steps"] = int(form_values["num_steps"])
    if form_values["use_seed"]:
        payload["seed"] = int(form_values["seed"])

    speaker_mode = form_values["speaker_mode"]
    if speaker_mode == "speaker_name":
        speaker_name = form_values["speaker_name"].strip()
        if speaker_name:
            payload["speaker_name"] = speaker_name
    elif speaker_mode == "ref_embed":
        ref_embed = form_values["ref_embed"].strip()
        if ref_embed:
            payload["ref_embed"] = ref_embed

    return payload


def validate_seed_input(use_seed: bool, seed_text: str) -> str | None:
    if not use_seed:
        return None
    candidate = seed_text.strip()
    if candidate == "":
        return "use_seed=true のときは seed を入力してください。"
    try:
        seed_value = int(candidate)
    except ValueError:
        return "seed は整数で入力してください。"
    if seed_value < 0:
        return "seed に負数は使えません。"
    if seed_value > MAX_SIGNED_63BIT_INT:
        return f"seed は {MAX_SIGNED_63BIT_INT} 以下の 63bit 正整数にしてください。"
    return None


def payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def call_health(api_url: str) -> tuple[int | None, Any, str | None]:
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/health",
            timeout=30,
        )
        try:
            return response.status_code, response.json(), None
        except ValueError:
            return response.status_code, response.text, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


def call_voice(api_url: str, payload: dict[str, Any]) -> tuple[int | None, Any, str | None]:
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/voice",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        try:
            return response.status_code, response.json(), None
        except ValueError:
            return response.status_code, response.text, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"{type(exc).__name__}: {exc}"


def init_state() -> None:
    defaults = {
        "is_generating": False,
        "last_request_hash": None,
        "last_request_payload": None,
        "last_response_json": None,
        "last_error_text": None,
        "last_http_status": None,
        "last_request_started_at": None,
        "last_request_finished_at": None,
        "last_wall_time": None,
        "last_health_status": None,
        "last_health_payload": None,
        "last_health_error": None,
        "health_requested_at": None,
        "voice_request_count": 0,
        "health_request_count": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_chunk_details(response_json: dict[str, Any]) -> None:
    per_chunk = response_json.get("per_chunk") or []
    if not per_chunk:
        st.info("chunk 情報はありません。")
        return

    for index, chunk in enumerate(per_chunk, start=1):
        with st.expander(f"Chunk {index}", expanded=False):
            st.write(f"sanitize_symbols: {chunk.get('sanitize_symbols')}")
            st.write(f"seed: {chunk.get('seed', chunk.get('used_seed'))}")
            st.write(f"original_text: {chunk.get('original_text')}")
            st.write(f"normalized_text: {chunk.get('normalized_text')}")
            st.write(f"request_text: {chunk.get('request_text')}")
            st.write(f"predicted_seconds: {chunk.get('predicted_seconds')}")
            st.write(f"audio_seconds: {chunk.get('audio_seconds')}")
            st.write(f"was_truncated_suspected: {chunk.get('was_truncated_suspected')}")


def render_audio_player(final_wav_path: str | None) -> None:
    if not final_wav_path:
        return

    wav_path = Path(final_wav_path)
    if not wav_path.exists():
        st.warning("final_wav_path は返っていますが、ファイルが見つかりません。")
        return

    st.audio(wav_path.read_bytes(), format="audio/wav")


def main() -> None:
    st.set_page_config(page_title="Irodori Test UI", layout="wide")
    init_state()
    speaker_profiles = list_speaker_profiles(DEFAULT_EMBED_PROFILE_DIR)
    speaker_names = [profile["name"] for profile in speaker_profiles]

    st.title("Irodori Thin API Client")
    st.caption("この UI は既存 API に JSON を POST するだけの薄いクライアントです。")

    with st.sidebar:
        st.subheader("API")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL, key="api_url")
        health_clicked = st.button("Health Check", use_container_width=True)
        if health_clicked:
            started_at = now_iso()
            status_code, payload, error_text = call_health(api_url)
            st.session_state.health_requested_at = started_at
            st.session_state.last_health_status = status_code
            st.session_state.last_health_payload = payload
            st.session_state.last_health_error = error_text
            st.session_state.health_request_count += 1

        st.write(f"health_request_count: {st.session_state.health_request_count}")
        if st.session_state.health_requested_at:
            st.write(f"last_health_requested_at: {st.session_state.health_requested_at}")
        if st.session_state.last_health_status is not None:
            st.write(f"last_health_http_status: {st.session_state.last_health_status}")
        if st.session_state.last_health_error:
            st.error(st.session_state.last_health_error)
        elif st.session_state.last_health_payload is not None:
            st.json(st.session_state.last_health_payload)

    col_left, col_right = st.columns([1.2, 1.0])

    with col_left:
        use_seed = st.checkbox("use_seed", value=False, key="ui_use_seed")
        seed = st.text_input(
            "seed",
            value="",
            key="ui_seed_text",
            disabled=not use_seed,
            placeholder="例: 7186657473745802163",
        )
        speaker_toolbar_left, speaker_toolbar_right = st.columns([1.2, 1.0])
        with speaker_toolbar_left:
            st.write(f"speaker profiles: {len(speaker_profiles)}")
            if speaker_names:
                st.caption(", ".join(speaker_names))
            else:
                st.caption("speaker profile が見つかりません。")
        with speaker_toolbar_right:
            if st.button("speaker一覧更新", use_container_width=True):
                st.rerun()

        with st.form("voice_request_form", clear_on_submit=False):
            top_submit_clicked = st.form_submit_button(
                "Generate",
                key="generate_top",
                type="primary",
                disabled=st.session_state.is_generating,
                use_container_width=True,
            )
            text = st.text_area("Text", height=280, placeholder="読み上げたいテキストを入力")

            speaker_mode = st.radio(
                "Speaker Input Mode",
                options=["speaker_name", "ref_embed"],
                format_func=lambda value: "speaker_name を使う" if value == "speaker_name" else "ref_embed のフルパスを使う",
                horizontal=True,
            )
            if speaker_mode == "speaker_name":
                if speaker_names:
                    default_index = speaker_names.index("由比ヶ浜結衣") if "由比ヶ浜結衣" in speaker_names else 0
                    speaker_name = st.selectbox("speaker_name", options=speaker_names, index=default_index)
                else:
                    speaker_name = st.selectbox("speaker_name", options=[""], index=0)
                    st.warning(f"{DEFAULT_EMBED_PROFILE_DIR} に *.speaker.safetensors がありません。")
                ref_embed = ""
            else:
                speaker_name = ""
                ref_embed = st.text_input("ref_embed", value="")

            max_seconds = st.number_input("max_seconds", min_value=1.0, value=30.0, step=1.0)
            target_chunk_seconds = st.number_input(
                "target_chunk_seconds",
                min_value=1.0,
                value=22.0,
                step=1.0,
            )
            hard_chunk_seconds = st.number_input(
                "hard_chunk_seconds",
                min_value=1.0,
                value=26.0,
                step=1.0,
            )
            max_chunk_chars = st.number_input("max_chunk_chars", min_value=1, value=110, step=1)
            append_silence_ms = st.number_input("append_silence_ms", min_value=0, value=0, step=10)
            chunk_gap_ms = st.selectbox("chunk_gap_ms", options=[0, 50, 100, 150, 200], index=0)
            sanitize_symbols = st.checkbox("sanitize_symbols", value=True)

            num_steps_enabled = st.checkbox("num_steps を送る", value=False)
            num_steps = st.number_input("num_steps", min_value=1, value=32, step=1, disabled=not num_steps_enabled)

            force_regenerate = st.checkbox(
                "force_regenerate",
                value=False,
                help="直前と同じリクエストでも確認なしで再生成します。",
            )

            bottom_submit_clicked = st.form_submit_button(
                "Generate",
                key="generate_bottom",
                type="primary",
                disabled=st.session_state.is_generating,
                use_container_width=True,
            )
            submit_clicked = top_submit_clicked or bottom_submit_clicked

        form_values = {
            "text": text,
            "speaker_mode": speaker_mode,
            "speaker_name": speaker_name,
            "ref_embed": ref_embed,
            "max_seconds": max_seconds,
            "target_chunk_seconds": target_chunk_seconds,
            "hard_chunk_seconds": hard_chunk_seconds,
            "max_chunk_chars": max_chunk_chars,
            "append_silence_ms": append_silence_ms,
            "chunk_gap_ms": chunk_gap_ms,
            "sanitize_symbols": sanitize_symbols,
            "use_seed": use_seed,
            "seed": seed,
            "num_steps_enabled": num_steps_enabled,
            "num_steps": num_steps,
        }

        seed_error = validate_seed_input(use_seed, seed)
        payload = build_request_payload(form_values) if seed_error is None else build_request_payload(
            {**form_values, "use_seed": False}
        )
        current_hash = payload_hash(payload)
        same_as_previous = (
            st.session_state.last_request_hash is not None
            and current_hash == st.session_state.last_request_hash
        )

        if seed_error is not None:
            st.error(seed_error)
        if same_as_previous and not force_regenerate:
            st.warning("直前と同じリクエストです。再生成する場合は force_regenerate をオンにしてください。")

        can_submit = bool(payload.get("text", "").strip())
        if not can_submit:
            st.info("Text を入力すると Generate できます。")

        if submit_clicked:
            if st.session_state.is_generating:
                st.warning("生成中です。完了までお待ちください。")
            elif not can_submit:
                st.error("Text が空です。")
            elif seed_error is not None:
                st.error(seed_error)
            elif same_as_previous and not force_regenerate:
                st.warning("同一リクエストの再送は停止しました。force_regenerate をオンにすると再生成できます。")
            else:
                st.session_state.is_generating = True
                started_at = now_iso()
                st.session_state.last_request_started_at = started_at
                st.session_state.last_request_payload = payload
                st.session_state.last_request_hash = current_hash
                st.session_state.voice_request_count += 1
                status_code, response_json, error_text = call_voice(api_url, payload)
                finished_at = now_iso()
                st.session_state.last_request_finished_at = finished_at
                st.session_state.last_http_status = status_code
                st.session_state.last_response_json = response_json
                st.session_state.last_error_text = error_text
                if isinstance(response_json, dict) and response_json.get("wall_time") is not None:
                    st.session_state.last_wall_time = response_json.get("wall_time")
                else:
                    st.session_state.last_wall_time = None
                st.session_state.is_generating = False

        st.subheader("Request")
        st.write(f"voice_request_count: {st.session_state.voice_request_count}")
        st.write(f"request_hash: {current_hash}")
        st.json(payload)

    with col_right:
        st.subheader("Result")
        if st.session_state.last_request_started_at:
            st.write(f"request_started_at: {st.session_state.last_request_started_at}")
        if st.session_state.last_request_finished_at:
            st.write(f"request_finished_at: {st.session_state.last_request_finished_at}")
        if st.session_state.last_http_status is not None:
            st.write(f"http_status: {st.session_state.last_http_status}")
        if st.session_state.last_wall_time is not None:
            st.write(f"wall_time: {st.session_state.last_wall_time}")

        if st.session_state.last_error_text:
            st.error(st.session_state.last_error_text)

        response_json = st.session_state.last_response_json
        if isinstance(response_json, dict):
            final_wav_path = response_json.get("final_wav_path")
            st.write(f"final_wav_path: {final_wav_path}")
            chunk_wav_paths = response_json.get("chunk_wav_paths")
            st.write("chunk_wav_paths:")
            st.json(chunk_wav_paths)
            skipped_chunks = response_json.get("skipped_chunks")
            st.write("skipped_chunks:")
            st.json(skipped_chunks)
            render_audio_player(final_wav_path)
            render_chunk_details(response_json)
            st.subheader("Response JSON")
            st.json(response_json)
        elif response_json is not None:
            st.text(str(response_json))


if __name__ == "__main__":
    main()
