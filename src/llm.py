import os
import time
from abc import ABC, abstractmethod
from typing import Callable, Any
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class LLMBackend(ABC):
    def abort(self) -> None:
        if hasattr(self, "_session"):
            try:
                self._session.close()
            except Exception:
                pass
            self._session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            self._session.mount("http://", HTTPAdapter(max_retries=retries))
            self._session.mount("https://", HTTPAdapter(max_retries=retries))

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        return self._generate(system_prompt, user_prompt, temperature, max_tokens, enable_thinking=False, on_token=on_token)

    def generate_with_thinking(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 16384,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        return self._generate(system_prompt, user_prompt, temperature, max_tokens, enable_thinking=True, on_token=on_token)

    def _generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        enable_thinking: bool,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if not enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        if on_token:
            payload["stream"] = True

        for attempt in range(3):
            try:
                r = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=(30, 300),
                    stream=bool(on_token),
                )
                r.raise_for_status()
                break
            except requests.exceptions.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        if on_token:
            content = ""
            reasoning = ""
            thinking_started = False
            thinking_ended = False
            import json
            for line in r.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
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

            if thinking_started and not thinking_ended:
                on_token("\n*/\n\n")

            if not content and reasoning:
                import re
                matches = list(re.finditer(r"module\s+\w+", reasoning))
                if matches:
                    content = reasoning[matches[-1].start():]
                else:
                    content = reasoning
                on_token(content)

            return LLMResponse(
                text=content,
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

        data = r.json()

        choice = data["choices"][0]
        msg = choice["message"]

        content = msg.get("content", "") or ""
        reasoning = msg.get("reasoning_content", "") or ""

        if not content and reasoning:
            import re
            matches = list(re.finditer(r"module\s+\w+", reasoning))
            if matches:
                content = reasoning[matches[-1].start():]
            else:
                content = reasoning

        usage = data.get("usage", {})
        return LLMResponse(
            text=content,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

class LlamaServerBackend(LLMBackend):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("LLAMA_SERVER_BASE_URL", "http://localhost:8080/v1")).rstrip("/")
        self.model = model or os.getenv("DEFAULT_MODEL", "deepseek-coder-v2")
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
    ) -> LLMResponse:
        return self._generate(system_prompt, user_prompt, temperature, max_tokens, enable_thinking=False, on_token=on_token)


class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self._session.mount("http://", HTTPAdapter(max_retries=retries))
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if on_token:
            payload["stream"] = True

        r = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300,
            stream=bool(on_token),
        )
        r.raise_for_status()

        if on_token:
            content = ""
            reasoning = ""
            thinking_started = False
            thinking_ended = False
            import json
            for line in r.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
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

            if thinking_started and not thinking_ended:
                on_token("\n*/\n\n")

            return LLMResponse(
                text=content,
                model=self.model,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
            )

        data = r.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "") or ""

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
    elif backend_type == "openai":
        return OpenAICompatibleBackend(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=default_model,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
