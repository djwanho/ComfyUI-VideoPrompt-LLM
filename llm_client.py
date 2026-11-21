"""Local Vision LLM clients used by the ComfyUI node."""

from __future__ import annotations

import ipaddress
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests


PROVIDER_OLLAMA = "Ollama"
PROVIDER_LM_STUDIO = "LM Studio"
PROVIDERS = (PROVIDER_OLLAMA, PROVIDER_LM_STUDIO)

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_RESPONSES: Dict[str, requests.Response] = {}
_STOP_EVENTS: Dict[str, threading.Event] = {}


@dataclass(frozen=True)
class LLMResponse:
    result: str
    thinking: str


def validate_local_server_url(value: str, provider: str) -> str:
    default = "http://127.0.0.1:11434" if provider == PROVIDER_OLLAMA else "http://127.0.0.1:1234"
    url = str(value or default).strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Server URL must be an HTTP(S) URL.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Server URL must not contain credentials, a query, or a fragment.")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("Server URL may contain only the optional /v1 suffix.")
    host = parsed.hostname.lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError(
                "For safety, use localhost or a literal private/LAN IP address for the local LLM server."
            ) from exc
        if not (address.is_private or address.is_loopback or address.is_link_local):
            raise ValueError("Public LLM server addresses are not allowed by this node.")
    return parsed._replace(path="", params="", query="", fragment="").geturl().rstrip("/")


def _register_response(request_id: str, response: requests.Response) -> threading.Event:
    with _ACTIVE_LOCK:
        event = _STOP_EVENTS.setdefault(request_id, threading.Event())
        _ACTIVE_RESPONSES[request_id] = response
        return event


def _clear_response(request_id: str, response: Optional[requests.Response] = None) -> None:
    with _ACTIVE_LOCK:
        current = _ACTIVE_RESPONSES.get(request_id)
        if response is None or current is response:
            _ACTIVE_RESPONSES.pop(request_id, None)


def stop_request(request_id: str) -> bool:
    with _ACTIVE_LOCK:
        event = _STOP_EVENTS.setdefault(str(request_id), threading.Event())
        event.set()
        response = _ACTIVE_RESPONSES.get(str(request_id))
    if response is not None:
        response.close()
        return True
    return False


def reset_stop_request(request_id: str) -> None:
    with _ACTIVE_LOCK:
        _STOP_EVENTS[str(request_id)] = threading.Event()


def _error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return str(payload.get("error") or payload.get("message") or payload)
    except Exception:
        pass
    return response.text[:2000] or response.reason


class VisionLLMClient:
    def __init__(self, provider: str, server_url: str, timeout_seconds: int = 600):
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        self.base_url = validate_local_server_url(server_url, provider)
        self.timeout = max(10, int(timeout_seconds))

    def list_models(self) -> List[str]:
        if self.provider == PROVIDER_OLLAMA:
            response = requests.get(f"{self.base_url}/api/tags", timeout=15, allow_redirects=False)
            if not response.ok:
                raise RuntimeError(_error_detail(response))
            items = response.json().get("models", [])
            names = {str(item.get("model") or item.get("name") or "").strip() for item in items if item}
            return sorted(name for name in names if name)
        response = requests.get(f"{self.base_url}/v1/models", timeout=15, allow_redirects=False)
        if not response.ok:
            raise RuntimeError(_error_detail(response))
        items = response.json().get("data", [])
        return sorted({str(item.get("id")) for item in items if item.get("id")})

    def unload(self, model: str) -> None:
        if self.provider == PROVIDER_OLLAMA:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=60,
                allow_redirects=False,
            )
        else:
            response = requests.post(
                f"{self.base_url}/api/v1/models/unload",
                json={"model": model},
                timeout=60,
                allow_redirects=False,
            )
        if not response.ok:
            raise RuntimeError(_error_detail(response))

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images_base64: Iterable[str],
        thinking: bool,
        seed: int,
        keep_alive: Any,
        request_id: str,
    ) -> LLMResponse:
        images = list(images_base64)
        if self.provider == PROVIDER_OLLAMA:
            return self._chat_ollama(
                model, system_prompt, user_prompt, images, thinking, seed, keep_alive, request_id
            )
        return self._chat_lm_studio(model, system_prompt, user_prompt, images, thinking, seed, request_id)

    def _stream(self, url: str, payload: Dict[str, Any], request_id: str) -> Iterable[Dict[str, Any]]:
        response = requests.post(
            url, json=payload, timeout=self.timeout, stream=True, allow_redirects=False
        )
        event = _register_response(request_id, response)
        try:
            if not response.ok:
                raise RuntimeError(f"{self.provider} API error {response.status_code}: {_error_detail(response)}")
            for raw_line in response.iter_lines(decode_unicode=True):
                if event.is_set():
                    raise RuntimeError("LLM generation was stopped by the user.")
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("error"):
                    raise RuntimeError(str(item["error"]))
                yield item
        finally:
            _clear_response(request_id, response)
            response.close()

    def _chat_ollama(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: List[str],
        thinking: bool,
        seed: int,
        keep_alive: Any,
        request_id: str,
    ) -> LLMResponse:
        message: Dict[str, Any] = {"role": "user", "content": user_prompt}
        if images:
            message["images"] = images
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, message],
            "stream": True,
            "think": bool(thinking),
            "options": {"seed": int(seed)},
            "keep_alive": keep_alive,
        }
        answer_parts: List[str] = []
        thinking_parts: List[str] = []
        for item in self._stream(f"{self.base_url}/api/chat", payload, request_id):
            message_data = item.get("message") or {}
            answer_parts.append(str(message_data.get("content") or ""))
            thinking_parts.append(str(message_data.get("thinking") or item.get("thinking") or ""))
        return LLMResponse("".join(answer_parts).strip(), "".join(thinking_parts).strip())

    def _chat_lm_studio(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: List[str],
        thinking: bool,
        seed: int,
        request_id: str,
    ) -> LLMResponse:
        content: Any = user_prompt
        if images:
            content = [{"type": "text", "text": user_prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
                for encoded in images
            )
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "stream": True,
            "seed": int(seed),
            "reasoning": "on" if thinking else "off",
        }
        try:
            return self._consume_lm_stream(payload, request_id)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "reasoning" not in message and "unknown" not in message and "unrecognized" not in message:
                raise
            payload.pop("reasoning", None)
            return self._consume_lm_stream(payload, request_id)

    def _consume_lm_stream(self, payload: Dict[str, Any], request_id: str) -> LLMResponse:
        answer_parts: List[str] = []
        thinking_parts: List[str] = []
        for item in self._stream(f"{self.base_url}/v1/chat/completions", payload, request_id):
            choices = item.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            answer_parts.append(str(delta.get("content") or ""))
            thinking_parts.append(str(delta.get("reasoning_content") or delta.get("reasoning") or ""))
        return LLMResponse("".join(answer_parts).strip(), "".join(thinking_parts).strip())
