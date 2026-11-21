"""Video frame extraction and timestamp-preserving sampling for ComfyUI VIDEO."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class PreparedFrame:
    source_index: int
    timestamp: float
    png_base64: str
    width: int
    height: int


@dataclass(frozen=True)
class PreparedVideo:
    frames: List[PreparedFrame]
    source_frame_count: int
    fps: float
    duration: float
    source_kind: str


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, Fraction):
            return float(value.numerator) / float(value.denominator)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def _cap_indices(indices: Sequence[int], maximum: int) -> List[int]:
    values = sorted(set(int(v) for v in indices if int(v) >= 0))
    if len(values) <= maximum:
        return values
    if maximum <= 1:
        return [values[0]]
    positions = [round(i * (len(values) - 1) / (maximum - 1)) for i in range(maximum)]
    return [values[p] for p in positions]


def choose_frame_indices(
    frame_count: int,
    fps: float,
    sampling_mode: str,
    sampling_interval: float,
    max_frames: int,
) -> List[int]:
    if frame_count <= 0:
        raise ValueError("The input contains no video frames.")
    maximum = max(1, int(max_frames))
    mode = str(sampling_mode or "Uniform").strip().lower()
    interval = max(0.001, float(sampling_interval))

    if mode == "every n frames":
        step = max(1, int(round(interval)))
        indices = list(range(0, frame_count, step))
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
    elif mode == "every n seconds":
        if fps <= 0:
            raise ValueError("A positive FPS is required for Every N Seconds sampling.")
        step = max(1, int(round(interval * fps)))
        indices = list(range(0, frame_count, step))
        if indices[-1] != frame_count - 1:
            indices.append(frame_count - 1)
    else:
        count = min(maximum, frame_count)
        if count == 1:
            indices = [0]
        else:
            indices = [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]
    return _cap_indices(indices, maximum)


def _normalise_batch(images: Any) -> torch.Tensor:
    if images is None:
        raise ValueError("No IMAGE frame batch was provided.")
    tensor = images if isinstance(images, torch.Tensor) else torch.as_tensor(images)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D image batch, received shape {tuple(tensor.shape)}.")
    if tensor.shape[-1] in (1, 3, 4):
        batch = tensor[..., :3]
    elif tensor.shape[1] in (1, 3, 4):
        batch = tensor[:, :3].permute(0, 2, 3, 1)
    else:
        raise ValueError("Video frames must use BHWC or BCHW layout with 1, 3, or 4 channels.")
    batch = batch.detach().to(device="cpu", dtype=torch.float32)
    if batch.shape[-1] == 1:
        batch = batch.repeat(1, 1, 1, 3)
    if batch.numel() and float(batch.max()) > 1.5:
        batch = batch / 255.0
    return batch.clamp(0.0, 1.0).contiguous()


def _resize_frame(frame_hwc: torch.Tensor, max_edge: int) -> torch.Tensor:
    height, width = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
    limit = max(128, int(max_edge))
    scale = min(1.0, limit / float(max(height, width)))
    if scale >= 1.0:
        return frame_hwc
    out_h = max(1, int(round(height * scale)))
    out_w = max(1, int(round(width * scale)))
    chw = frame_hwc.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(chw, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def _encode_png(frame_hwc: torch.Tensor) -> tuple[str, int, int]:
    frame = (frame_hwc.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
    image = Image.fromarray(frame, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), image.width, image.height


def _component_value(components: Any, name: str) -> Any:
    if isinstance(components, dict):
        return components.get(name)
    return getattr(components, name, None)


def _video_metadata(video: Any) -> tuple[int, float, float]:
    count = 0
    fps = 0.0
    duration = 0.0
    for method_name, target in (
        ("get_frame_count", "count"),
        ("get_frame_rate", "fps"),
        ("get_duration", "duration"),
    ):
        method = getattr(video, method_name, None)
        if not callable(method):
            continue
        try:
            value = method()
        except Exception:
            continue
        if target == "count":
            count = max(0, int(value))
        elif target == "fps":
            fps = max(0.0, _as_float(value))
        else:
            duration = max(0.0, _as_float(value))
    return count, fps, duration


def _read_selected_from_stream(video: Any, wanted: Sequence[int]) -> Optional[List[torch.Tensor]]:
    source_method = getattr(video, "get_stream_source", None)
    if not callable(source_method):
        return None
    try:
        import av

        source = source_method()
        wanted_set = set(wanted)
        selected = {}
        with av.open(source, mode="r") as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted_set:
                    array = frame.to_ndarray(format="rgb24")
                    selected[index] = torch.from_numpy(array).to(torch.float32) / 255.0
                    if len(selected) == len(wanted_set):
                        break
        if len(selected) != len(wanted_set):
            return None
        return [selected[index] for index in wanted]
    except Exception:
        return None


def prepare_video(
    video: Any,
    image_frames: Any,
    fps_override: float,
    sampling_mode: str,
    sampling_interval: float,
    max_frames: int,
    analysis_resolution: int,
) -> PreparedVideo:
    source_kind = "VIDEO"
    metadata_count = 0
    metadata_fps = 0.0
    metadata_duration = 0.0

    if video is not None:
        metadata_count, metadata_fps, metadata_duration = _video_metadata(video)
        effective_fps = float(fps_override) if float(fps_override) > 0 else metadata_fps
        if effective_fps <= 0:
            effective_fps = 24.0
        if metadata_count > 0:
            wanted = choose_frame_indices(metadata_count, effective_fps, sampling_mode, sampling_interval, max_frames)
            streamed = _read_selected_from_stream(video, wanted)
            if streamed is not None:
                selected_frames = streamed
            else:
                components_method = getattr(video, "get_components", None)
                if not callable(components_method):
                    raise ValueError("The connected VIDEO object does not expose get_components().")
                components = components_method()
                batch = _normalise_batch(_component_value(components, "images"))
                component_fps = _as_float(_component_value(components, "frame_rate"))
                if float(fps_override) <= 0 and component_fps > 0:
                    effective_fps = component_fps
                metadata_count = int(batch.shape[0])
                wanted = choose_frame_indices(metadata_count, effective_fps, sampling_mode, sampling_interval, max_frames)
                selected_frames = [batch[index] for index in wanted]
        else:
            components_method = getattr(video, "get_components", None)
            if not callable(components_method):
                raise ValueError("The connected VIDEO object does not expose readable frames.")
            components = components_method()
            batch = _normalise_batch(_component_value(components, "images"))
            component_fps = _as_float(_component_value(components, "frame_rate"))
            if float(fps_override) <= 0 and component_fps > 0:
                effective_fps = component_fps
            metadata_count = int(batch.shape[0])
            wanted = choose_frame_indices(metadata_count, effective_fps, sampling_mode, sampling_interval, max_frames)
            selected_frames = [batch[index] for index in wanted]
    elif image_frames is not None:
        source_kind = "IMAGE batch"
        batch = _normalise_batch(image_frames)
        metadata_count = int(batch.shape[0])
        effective_fps = float(fps_override) if float(fps_override) > 0 else 24.0
        wanted = choose_frame_indices(metadata_count, effective_fps, sampling_mode, sampling_interval, max_frames)
        selected_frames = [batch[index] for index in wanted]
    else:
        raise ValueError("Connect a VIDEO input or an IMAGE batch of chronological video frames.")

    prepared: List[PreparedFrame] = []
    for source_index, frame in zip(wanted, selected_frames):
        resized = _resize_frame(frame, analysis_resolution)
        encoded, width, height = _encode_png(resized)
        prepared.append(PreparedFrame(int(source_index), float(source_index) / effective_fps, encoded, width, height))
    calculated_duration = metadata_count / effective_fps if effective_fps > 0 else 0.0
    duration = metadata_duration if metadata_duration > 0 else calculated_duration
    return PreparedVideo(prepared, metadata_count, effective_fps, duration, source_kind)


def chunks(values: Sequence[PreparedFrame], size: int) -> Iterable[List[PreparedFrame]]:
    width = max(1, int(size))
    for start in range(0, len(values), width):
        yield list(values[start : start + width])
