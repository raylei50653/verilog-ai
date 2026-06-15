import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.cancellation import CancellationToken, PipelineCancelled


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


def build_local_file_tools() -> list[dict[str, Any]]:
    """Full tool set (read + grep + edit_file) — used by general agents."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a local UTF-8 text file by absolute path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_search",
                "description": "Search for a text pattern inside a file or directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "path": {"type": "string", "description": "Absolute file or directory path"},
                    },
                    "required": ["pattern", "path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Replace a local UTF-8 text file with the provided full content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                        "content": {"type": "string", "description": "Full replacement file content"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
    ]


def build_fix_tools() -> list[dict[str, Any]]:
    """Minimal tool set for targeted RTL fixes: read then replace a snippet."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the current RTL file to see its exact content before editing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "replace_in_file",
                "description": (
                    "Replace an exact substring in a file. "
                    "old_string must match the file content character-for-character "
                    "(including indentation and newlines). "
                    "Prefer small, targeted replacements over full-file rewrites."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Absolute file path"},
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to find (must exist verbatim in the file)",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement text",
                        },
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
    ]


def _extract_content_from_choice(message: dict[str, Any], on_token: Callable[[str], None] | None = None) -> str:
    content = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", "") or ""

    if not content and reasoning:
        matches = list(re.finditer(r"module\s+\w+", reasoning))
        if matches:
            content = reasoning[matches[-1].start():]
        else:
            content = reasoning

    if on_token and content:
        on_token(content)
    return content


def _normalize_stream_response(
    response: requests.Response,
    on_token: Callable[[str], None],
    cancel_token: CancellationToken | None = None,
) -> LLMResponse:
    content = ""
    reasoning = ""
    thinking_started = False
    thinking_ended = False

    try:
        for line in response.iter_lines():
            if cancel_token is not None and cancel_token.is_cancelled:
                response.close()
                raise PipelineCancelled("cancelled by user")
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            if not line_str.startswith("data: "):
                continue
            data_payload = line_str[6:]
            if data_payload == "[DONE]":
                break
            try:
                chunk = json.loads(data_payload)
                delta = chunk["choices"][0].get("delta", {})
                delta_content = delta.get("content", "") or ""
                delta_reasoning = delta.get("reasoning_content", "") or ""

                if delta_reasoning:
                    reasoning += delta_reasoning
                    if not thinking_started:
                        thinking_started = True
                        on_token("/* [Thinking]\n")
                    on_token(delta_reasoning)

                if delta_content:
                    if thinking_started and not thinking_ended:
                        thinking_ended = True
                        on_token("\n*/\n\n")
                    content += delta_content
                    on_token(delta_content)
            except Exception:
                pass
    except requests.exceptions.RequestException:
        response.close()
        if cancel_token is not None and cancel_token.is_cancelled:
            raise PipelineCancelled("cancelled by user")
        raise

    if thinking_started and not thinking_ended:
        on_token("\n*/\n\n")

    if not content and reasoning:
        matches = list(re.finditer(r"module\s+\w+", reasoning))
        if matches:
            content = reasoning[matches[-1].start():]
        else:
            content = reasoning
        on_token(content)

    return LLMResponse(
        text=content,
        model="",
        usage={"prompt_tokens": 0, "completion_tokens": 0},
    )


import threading

_local_storage = threading.local()


def set_allowed_paths(paths: list[str] | None) -> None:
    _local_storage.allowed_paths = paths


def get_allowed_paths() -> list[str] | None:
    return getattr(_local_storage, "allowed_paths", None)


def is_path_allowed(path_to_check: str | Path, allowed_dirs: list[str]) -> bool:
    try:
        check_path = Path(path_to_check).expanduser().absolute()
        for allowed in allowed_dirs:
            allowed_path = Path(allowed).expanduser().absolute()
            if check_path == allowed_path:
                return True
            try:
                check_path.relative_to(allowed_path)
                return True
            except ValueError:
                pass
    except Exception:
        pass
    return False


def _execute_local_tool(name: str, arguments: dict[str, Any]) -> str:
    allowed_paths = get_allowed_paths()
    try:
        if name == "read_file":
            path = Path(str(arguments["path"])).expanduser()
            if allowed_paths is not None and not is_path_allowed(path, allowed_paths):
                return f"Permission denied: Path {path} is outside the allowed directory scope."
            print(f"[tool used] read_file {path}")
            return path.read_text(encoding="utf-8")

        if name == "grep_search":
            pattern = str(arguments["pattern"])
            path = str(arguments["path"])
            if allowed_paths is not None and not is_path_allowed(path, allowed_paths):
                return f"Permission denied: Path {path} is outside the allowed directory scope."
            print(f"[tool used] grep_search pattern={pattern!r} path={path}")
            result = subprocess.run(
                ["rg", "-n", pattern, path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode in (0, 1):
                return result.stdout
            return result.stderr

        if name == "edit_file":
            path = Path(str(arguments["path"])).expanduser()
            if allowed_paths is not None and not is_path_allowed(path, allowed_paths):
                return f"Permission denied: Path {path} is outside the allowed directory scope."
            content = str(arguments["content"])
            print(f"[tool used] edit_file {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            _local_storage.last_edit_content = content
            return f"Wrote {len(content)} bytes to {path}"

        if name == "replace_in_file":
            path = Path(str(arguments["path"])).expanduser()
            if allowed_paths is not None and not is_path_allowed(path, allowed_paths):
                return f"Permission denied: Path {path} is outside the allowed directory scope."
            old_string = str(arguments["old_string"])
            new_string = str(arguments["new_string"])
            current = path.read_text(encoding="utf-8")
            if old_string not in current:
                return (
                    f"replace_in_file error: old_string not found verbatim in {path}. "
                    "Use read_file first to get the exact text."
                )
            new_content = current.replace(old_string, new_string, 1)
            path.write_text(new_content, encoding="utf-8")
            _local_storage.last_edit_content = new_content
            print(f"[tool used] replace_in_file {path} ({len(old_string)}→{len(new_string)} chars)")
            return f"Replaced successfully in {path}"

    except Exception as exc:
        return f"Tool execution error: {exc}"

    return f"Unsupported tool: {name}"


def _append_tool_messages(
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
) -> list[dict[str, Any]]:
    next_messages = list(messages)
    next_messages.append(assistant_message)
    for tool_call in assistant_message.get("tool_calls", []):
        function = tool_call.get("function", {})
        arguments_raw = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else (arguments_raw or {})
        except Exception:
            arguments = {}
        name = function.get("name", "")
        result = _execute_local_tool(name, arguments)
        # last_edit_content is set inside _execute_local_tool for edit_file / replace_in_file
        next_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result,
            }
        )
    return next_messages


class LLMBackend(ABC):
    def __init__(self) -> None:
        self._aborted = False

    def abort(self) -> None:
        self._aborted = True
        if hasattr(self, "_session"):
            try:
                self._session.close()
            except Exception:
                pass
            self._session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            self._session.mount("http://", HTTPAdapter(max_retries=retries))
            self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        return self._generate(
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            enable_thinking=False,
            on_token=on_token,
            tools=tools,
            cancel_token=cancel_token,
        )

    def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        return self._generate(
            system_prompt,
            user_prompt,
            temperature,
            max_tokens,
            enable_thinking=True,
            on_token=on_token,
            tools=tools,
            cancel_token=cancel_token,
        )

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        # Clear stale edit content from previous conversations on this thread
        _local_storage.last_edit_content = ""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self._generate_from_messages(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            on_token=on_token,
            tools=tools,
            cancel_token=cancel_token,
        )

    @abstractmethod
    def _generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_call_depth: int = 0,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class LlamaServerBackend(LLMBackend):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        super().__init__()
        self.base_url = (base_url or os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1")).rstrip("/")
        self.model = model or os.getenv("DEFAULT_MODEL", "deepseek-coder-v2")
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def _post(
        self,
        payload: dict[str, Any],
        *,
        stream: bool,
        cancel_token: CancellationToken | None = None,
    ) -> requests.Response:
        for attempt in range(3):
            if self._aborted:
                raise PipelineCancelled("backend aborted by user")
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(30, 300),
                    stream=stream,
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                if cancel_token is not None and cancel_token.is_cancelled:
                    raise PipelineCancelled("cancelled by user") from exc
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def _generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_call_depth: int = 0,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        can_stream = on_token is not None and not tools
        if can_stream:
            payload["stream"] = True
            response = self._post(payload, stream=True, cancel_token=cancel_token)
            normalized = _normalize_stream_response(response, on_token, cancel_token)
            normalized.model = self.model
            return normalized

        response = self._post(payload, stream=False, cancel_token=cancel_token)
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        if tools and message.get("tool_calls"):
            if tool_call_depth >= 5:
                last_edit = getattr(_local_storage, "last_edit_content", "")
                if last_edit:
                    print(f"[CRITICAL] Exceeded max tool call depth (5). Returning last edit_file content.")
                    return LLMResponse(
                        text=last_edit,
                        model=data.get("model", self.model),
                        usage={"prompt_tokens": 0, "completion_tokens": 0},
                    )
                content = message.get("content", "") or ""
                print(f"[CRITICAL] Exceeded max tool call depth (5). Aborting tool execution.")
                return LLMResponse(
                    text=content,
                    model=data.get("model", self.model),
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                )
            next_messages = _append_tool_messages(messages, message)
            return self._generate_from_messages(
                messages=next_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                on_token=on_token,
                tools=tools,
                tool_call_depth=tool_call_depth + 1,
                cancel_token=cancel_token,
            )

        content = _extract_content_from_choice(message, on_token)
        usage = data.get("usage", {})
        return LLMResponse(
            text=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, base_url: str, api_key: str, model: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def _post(
        self,
        payload: dict[str, Any],
        *,
        stream: bool,
        cancel_token: CancellationToken | None = None,
    ) -> requests.Response:
        if self._aborted:
            raise PipelineCancelled("backend aborted by user")
        if cancel_token is not None:
            cancel_token.raise_if_cancelled()
        try:
            response = self._session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=300,
                stream=stream,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            if cancel_token is not None and cancel_token.is_cancelled:
                raise PipelineCancelled("cancelled by user") from exc
            raise

    def _generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Callable[[str], None] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_call_depth: int = 0,
        cancel_token: CancellationToken | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        can_stream = on_token is not None and not tools
        if can_stream:
            payload["stream"] = True
            response = self._post(payload, stream=True, cancel_token=cancel_token)
            normalized = _normalize_stream_response(response, on_token, cancel_token)
            normalized.model = self.model
            return normalized

        response = self._post(payload, stream=False, cancel_token=cancel_token)
        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]

        if tools and message.get("tool_calls"):
            if tool_call_depth >= 5:
                last_edit = getattr(_local_storage, "last_edit_content", "")
                if last_edit:
                    print(f"[CRITICAL] Exceeded max tool call depth (5). Returning last edit_file content.")
                    return LLMResponse(
                        text=last_edit,
                        model=data.get("model", self.model),
                        usage={"prompt_tokens": 0, "completion_tokens": 0},
                    )
                content = message.get("content", "") or ""
                print(f"[CRITICAL] Exceeded max tool call depth (5). Aborting tool execution.")
                return LLMResponse(
                    text=content,
                    model=data.get("model", self.model),
                    usage={"prompt_tokens": 0, "completion_tokens": 0},
                )
            next_messages = _append_tool_messages(messages, message)
            return self._generate_from_messages(
                messages=next_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                enable_thinking=enable_thinking,
                on_token=on_token,
                tools=tools,
                tool_call_depth=tool_call_depth + 1,
                cancel_token=cancel_token,
            )

        content = _extract_content_from_choice(message, on_token)
        usage = data.get("usage", {})
        return LLMResponse(
            text=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )


def create_backend(model: str | None = None) -> LLMBackend:
    backend_type = os.getenv("LLM_BACKEND", "llama_server")
    default_model = model or os.getenv("DEFAULT_MODEL", "deepseek-coder-v2")
    if backend_type == "llama_server":
        return LlamaServerBackend(model=default_model)
    if backend_type == "openai":
        return OpenAICompatibleBackend(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=default_model,
        )
    raise ValueError(f"Unknown backend type: {backend_type}")
