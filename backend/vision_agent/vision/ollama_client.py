"""
Resilient client for interacting with the local Ollama Vision LLM instance.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
import requests

from vision_agent.config.settings import settings
from vision_agent.utils.logging_utils import get_logger

logger = get_logger("ollama_client")


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model_name = model_name or settings.VISION_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug(f"Ollama connection check failed: {exc}")
            return False

    def list_models(self) -> List[str]:
        """Fetch list of installed models from Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception as exc:
            logger.warning(f"Failed to list Ollama models: {exc}")
        return []

    def verify_model_available(self, model: Optional[str] = None) -> tuple[bool, str]:
        """
        Verify if the configured model is installed.
        Returns (is_available, error_or_instruction_message).
        """
        target_model = model or self.model_name
        if not self.check_connection():
            return (
                False,
                f"Ollama is not reachable at {self.base_url}. Please ensure Ollama is running (`ollama serve`).",
            )

        installed_models = self.list_models()
        # Check direct or base name match (e.g. 'moondream:latest' vs 'moondream')
        found = any(
            target_model == m or target_model == m.split(":")[0] or m.startswith(target_model)
            for m in installed_models
        )

        if not found:
            return (
                False,
                f"Model '{target_model}' is not installed in Ollama.\n"
                f"Run the following command to download it:\n\n"
                f"  ollama pull {target_model}\n",
            )
        return True, ""

    def generate(
        self,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a completion for text and optional image(s) using Ollama.
        """
        if model:
            target_model = model
        elif images_base64:
            target_model = self.model_name
        else:
            target_model = settings.TEXT_MODEL or self.model_name

        url = f"{self.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        if images_base64:
            payload["images"] = images_base64

        try:
            logger.debug(f"Sending prompt to Ollama ({target_model}), images count: {len(images_base64 or [])}")
            resp = requests.post(url, json=payload, timeout=self.timeout)

            if resp.status_code != 200:
                error_msg = resp.text
                try:
                    error_json = resp.json()
                    error_msg = error_json.get("error", resp.text)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Ollama returned error (status {resp.status_code}): {error_msg}"
                )

            result = resp.json()
            return result.get("response", "").strip()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. Verify that the Ollama service is active."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s while processing model {target_model}."
            )
