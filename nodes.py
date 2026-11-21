"""ComfyUI node for turning reference-video motion into a generation prompt."""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any, Dict, List

from .llm_client import PROVIDER_OLLAMA, PROVIDERS, VisionLLMClient, reset_stop_request, stop_request
from .video_processing import PreparedFrame, PreparedVideo, chunks, prepare_video

try:
    from aiohttp import web
    from server import PromptServer
except Exception:  # Allows source inspection outside a running ComfyUI process.
    web = None
    PromptServer = None


ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT_PATH = ROOT / "prompts" / "minimax_h3_default.txt"
MODEL_PLACEHOLDER = "Refresh Models"
MAX_IMAGES_PER_REQUEST = 12


def _default_system_prompt() -> str:
    try:
        return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Could not load the default system prompt: {DEFAULT_PROMPT_PATH}") from exc


def _json_response(payload: Dict[str, Any], status: int = 200):
    return web.json_response(payload, status=status) if web is not None else payload


def _send_progress(node_id: str, **payload: Any) -> None:
    if PromptServer is None:
        return
    try:
        PromptServer.instance.send_sync("videoprompt_llm_progress", {"node_id": str(node_id), **payload})
    except Exception:
        pass


async def _handle_models(request):
    try:
        payload = await request.json()
        client = VisionLLMClient(str(payload.get("provider") or PROVIDER_OLLAMA), "")
        models = await asyncio.to_thread(client.list_models)
        return _json_response({"models": models})
    except Exception as exc:
        return _json_response({"models": [], "error": str(exc)}, status=400)


async def _handle_stop(request):
    try:
        payload = await request.json()
        stopped = stop_request(str(payload.get("node_id") or ""))
        return _json_response({"ok": True, "stopped_active_request": stopped})
    except Exception as exc:
        return _json_response({"ok": False, "error": str(exc)}, status=400)


async def _handle_unload(request):
    try:
        payload = await request.json()
        client = VisionLLMClient(str(payload.get("provider") or PROVIDER_OLLAMA), "")
        model = str(payload.get("model") or "").strip()
        if not model or model == MODEL_PLACEHOLDER:
            raise ValueError("Select an installed model before unloading.")
        await asyncio.to_thread(client.unload, model)
        return _json_response({"ok": True, "message": f"Unloaded {model}."})
    except Exception as exc:
        return _json_response({"ok": False, "error": str(exc)}, status=400)


if PromptServer is not None:
    PromptServer.instance.routes.post("/videoprompt_llm/models")(_handle_models)
    PromptServer.instance.routes.post("/videoprompt_llm/stop")(_handle_stop)
    PromptServer.instance.routes.post("/videoprompt_llm/unload")(_handle_unload)


def _seed_for_request(seed: int, mode: str, index: int) -> int:
    base = max(0, min(0xFFFFFFFF, int(seed)))
    if mode == "Increment":
        return (base + index) & 0xFFFFFFFF
    if mode == "Decrement":
        return (base - index) & 0xFFFFFFFF
    if mode == "Randomize":
        return random.SystemRandom().randint(0, 0xFFFFFFFF)
    return base


def _keep_alive_value(model_after_run: str, keep_minutes: int, is_final: bool) -> Any:
    if model_after_run == "Unload" and is_final:
        return 0
    if model_after_run == "Keep for N Minutes":
        return f"{max(1, int(keep_minutes))}m"
    return -1


def _frame_manifest(frames: List[PreparedFrame]) -> str:
    return "\n".join(
        f"- Attached image {position}: source frame {frame.source_index}, timestamp {frame.timestamp:.3f}s"
        for position, frame in enumerate(frames, 1)
    )


def _analysis_request(video: PreparedVideo, frames: List[PreparedFrame], part: int, total: int) -> str:
    return f"""Analyze chronological reference-video frames {part}/{total} as visual evidence only.

Source metadata:
- Source: {video.source_kind}
- FPS used for timestamps: {video.fps:.6g}
- Source frame count: {video.source_frame_count}
- Estimated duration: {video.duration:.3f} seconds

Attached frames, in chronological order:
{_frame_manifest(frames)}

Record only observable evidence: initial and final state, head/gaze/face changes, arms/hands,
shoulders/torso/posture, locomotion/rotation, action transitions, framing, camera motion versus
subject motion, and visible cuts. Treat gaps between sampled frames as uncertainty. Do not invent
audio, dialogue, emotion, intent, cuts, camera moves, or unobserved in-between motion. This is an
intermediate evidence note for a later synthesis, so preserve timestamps and ordering clearly."""


def _final_request(video: PreparedVideo, observations: List[str], user_prompt: str) -> str:
    evidence = "\n\n".join(
        f"--- Chronological observation chunk {index} ---\n{text}"
        for index, text in enumerate(observations, 1)
    )
    user_section = user_prompt.strip() or "No additional user request was provided."
    return f"""Create the final output required by the system instruction from the evidence below.

Video timing metadata:
- FPS used for timestamps: {video.fps:.6g}
- Source frame count: {video.source_frame_count}
- Estimated duration: {video.duration:.3f} seconds
- Sampled visual frames: {len(video.frames)}

Additional user request (apply it only within the system instruction):
{user_section}

Chronological visual evidence:
{evidence}

Resolve overlaps between chunks without changing temporal order. Do not claim certainty about
motion hidden between sampled frames. No audio track was analyzed. Return only the final requested
prompt, without analysis notes, Markdown fences, or a preamble."""


def _direct_request(video: PreparedVideo, user_prompt: str) -> str:
    user_section = user_prompt.strip() or "No additional user request was provided."
    return f"""Analyze the attached chronological frames and return only the final output required by
the system instruction.

Video timing metadata:
- Source: {video.source_kind}
- FPS used for timestamps: {video.fps:.6g}
- Source frame count: {video.source_frame_count}
- Estimated duration: {video.duration:.3f} seconds

Attached frames, in chronological order:
{_frame_manifest(video.frames)}

Additional user request:
{user_section}

Use only visible evidence. Distinguish subject movement from camera movement. Treat sampling gaps as
uncertainty, create new Shots only for visible cuts, and do not invent audio, dialogue, music,
emotion, intent, or unobserved motion. Return only the final prompt without a preamble."""


def _is_minimax_base_prompt(text: str) -> bool:
    required = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
        "[Shot 1]",
    )
    positions = [text.find(item) for item in required[:3]]
    return (
        all(item in text for item in required)
        and positions == sorted(positions)
        and not any(label in text for label in ("subject_definitions:", "retention_analysis:", "<Video 1>"))
    )


def _maybe_unload_comfyui_models(policy: str) -> None:
    if policy != "Unload Before LLM":
        return
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception as exc:
        raise RuntimeError(f"Failed to unload ComfyUI models before the LLM request: {exc}") from exc


class MiniMaxH3VideoLLMLoader:
    DESCRIPTION = (
        "Analyze chronological frames from a ComfyUI VIDEO with a local vision LLM and output a "
        "MiniMax H3 base-format prompt. A Custom System Prompt replaces the bundled H3 instruction."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (list(PROVIDERS), {"default": PROVIDERS[0]}),
                "ollama_model": ([MODEL_PLACEHOLDER], {"default": MODEL_PLACEHOLDER}),
                "lm_studio_model": ([MODEL_PLACEHOLDER], {"default": MODEL_PLACEHOLDER}),
                "thinking": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "seed_mode": (["Fixed", "Increment", "Decrement", "Randomize"], {"default": "Fixed"}),
                "model_after_run": (
                    ["Keep Loaded", "Keep for N Minutes", "Unload"], {"default": "Unload"}
                ),
                "keep_minutes": ("INT", {"default": 5, "min": 1, "max": 240, "step": 1}),
                "unload_comfyui_models": (
                    ["Do Not Unload", "Unload Before LLM"], {"default": "Do Not Unload"}
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 240.0,
                        "step": 0.001,
                        "tooltip": "0 uses VIDEO metadata; IMAGE batches fall back to 24 FPS.",
                    },
                ),
                "sampling_mode": (
                    ["Uniform", "Every N Frames", "Every N Seconds"], {"default": "Uniform"}
                ),
                "sampling_interval": (
                    "FLOAT", {"default": 1.0, "min": 0.001, "max": 10000.0, "step": 0.1}
                ),
                "max_analysis_frames": (
                    "INT", {"default": 24, "min": 1, "max": 128, "step": 1}
                ),
                "analysis_resolution": (
                    "INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}
                ),
                "custom_system_prompt": (
                    "STRING", {"default": "", "multiline": True, "dynamicPrompts": False}
                ),
                "prompt": (
                    "STRING", {"default": "", "multiline": True, "dynamicPrompts": False}
                ),
                "video": ("VIDEO", {"forceInput": True}),
            },
            "optional": {
                "image_frames": (
                    "IMAGE",
                    {
                        "forceInput": True,
                        "tooltip": "Chronological IMAGE batch fallback; not a subject reference input.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "analyze"
    CATEGORY = "DJwanho / LLM"

    @classmethod
    def VALIDATE_INPUTS(cls, provider=None, ollama_model=None, lm_studio_model=None, **kwargs):
        active_provider = str(provider or PROVIDERS[0])
        active_model = lm_studio_model if active_provider == "LM Studio" else ollama_model
        value = str(active_model or "").strip()
        if not value or value == MODEL_PLACEHOLDER:
            return "Press Refresh Models and select a vision-capable model."
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def analyze(
        self,
        provider: str,
        ollama_model: str,
        lm_studio_model: str,
        thinking: bool,
        seed: int,
        seed_mode: str,
        model_after_run: str,
        keep_minutes: int,
        unload_comfyui_models: str,
        fps: float,
        sampling_mode: str,
        sampling_interval: float,
        max_analysis_frames: int,
        analysis_resolution: int,
        custom_system_prompt: str,
        prompt: str,
        video=None,
        image_frames=None,
        unique_id=None,
    ):
        node_id = str(unique_id or "")
        reset_stop_request(node_id)
        _send_progress(node_id, status="preparing", thinking="Preparing chronological frames.", result="")
        prepared = prepare_video(
            video, image_frames, fps, sampling_mode, sampling_interval, max_analysis_frames, analysis_resolution
        )
        custom = str(custom_system_prompt or "").strip()
        system_prompt = custom if custom else _default_system_prompt()
        using_default = not bool(custom)
        model = str(lm_studio_model if provider == "LM Studio" else ollama_model).strip()
        client = VisionLLMClient(provider, "")
        _maybe_unload_comfyui_models(unload_comfyui_models)
        frame_chunks = list(chunks(prepared.frames, MAX_IMAGES_PER_REQUEST))
        responses: List[str] = []
        thinking_parts: List[str] = []
        request_index = 0

        def run_request(user_text: str, frames: List[PreparedFrame], final: bool):
            nonlocal request_index
            current_index = request_index
            request_index += 1
            _send_progress(
                node_id, status="running", thinking=f"Running LLM request {current_index + 1}.", result=""
            )
            response = client.chat(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_text,
                images_base64=[frame.png_base64 for frame in frames],
                thinking=bool(thinking),
                seed=_seed_for_request(seed, seed_mode, current_index),
                keep_alive=_keep_alive_value(model_after_run, keep_minutes, final),
                request_id=node_id,
            )
            if response.thinking:
                thinking_parts.append(response.thinking)
            _send_progress(
                node_id,
                status="running",
                thinking="\n\n".join(thinking_parts) if thinking else "",
                result=response.result,
            )
            if not response.result:
                raise RuntimeError("The selected model returned no final result.")
            return response.result

        try:
            if len(frame_chunks) == 1:
                final_result = run_request(_direct_request(prepared, prompt), prepared.frames, True)
            else:
                for part, frame_chunk in enumerate(frame_chunks, 1):
                    responses.append(
                        run_request(
                            _analysis_request(prepared, frame_chunk, part, len(frame_chunks)), frame_chunk, False
                        )
                    )
                final_result = run_request(_final_request(prepared, responses, prompt), [], True)

            if using_default and not _is_minimax_base_prompt(final_result):
                repair = f"""Rewrite the draft below so it strictly contains the three required fields in order,
starts its visual timeline with [Shot 1], and contains no preamble or Markdown fence. Preserve only
evidence already present. Do not invent audio.

Draft:
{final_result}"""
                final_result = run_request(repair, [], True)
                if not _is_minimax_base_prompt(final_result):
                    raise RuntimeError(
                        "The model did not return the required MiniMax H3 base prompt fields after one repair attempt."
                    )
        except Exception as exc:
            _send_progress(node_id, status="error", thinking=str(exc), result="")
            raise
        finally:
            if model_after_run == "Unload":
                try:
                    client.unload(model)
                except Exception as exc:
                    thinking_parts.append(f"Post-run unload warning: {exc}")

        final_thinking = "\n\n".join(thinking_parts) if (thinking or thinking_parts) else ""
        _send_progress(
            node_id,
            status="done",
            thinking=final_thinking,
            result=final_result,
            metadata={
                "fps": prepared.fps,
                "duration": prepared.duration,
                "sampled_frames": len(prepared.frames),
            },
        )
        return {"ui": {"thinking": [final_thinking], "text": [final_result]}, "result": (final_result,)}


NODE_CLASS_MAPPINGS = {"MiniMaxH3VideoLLMLoader": MiniMaxH3VideoLLMLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3VideoLLMLoader": "MiniMax H3 Video LLM Loader"}
