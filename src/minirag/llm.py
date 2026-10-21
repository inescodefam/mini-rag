"""LLM clients.

Every LLM implements ``complete(system, user) -> str``.

* ``StubLLM``: returns the prompt it was given (no network, no AWS credentials).
  Used for tests and offline runs so the exact prompt can be inspected.
* ``BedrockLLM``: Claude through the Bedrock Converse API. Uses the default AWS
  credential chain; no keys in code.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional, Protocol, Tuple

# Newer Claude models on Bedrock are invoked through a cross-region inference
# profile id (the "us." prefix). Override with MINIRAG_LLM_MODEL_ID.
DEFAULT_BEDROCK_LLM_MODEL_ID = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_BEDROCK_REGION = "eu-north-1"


class LLM(Protocol):
    """Anything that turns a system prompt and a user prompt into text."""

    def complete(self, system: str, user: str) -> str:
        ...


class StubLLM:
    """Returns the prompt instead of calling a model.

    The reply is ``"<system>\\n\\n<user>"``. Every call is also recorded in
    ``calls`` as ``(system, user)`` so tests can assert whether the LLM was
    called at all (it must not be when the pipeline refuses).
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "%s\n\n%s" % (system, user)


class BedrockLLM:
    """Claude via the Bedrock ``Converse`` API.

    Configuration (constructor args win over env vars):

    * model id: ``MINIRAG_LLM_MODEL_ID``
    * region: ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (default ``us-east-1``) 

    The boto3 client is created lazily on the first call, so constructing this
    class needs no credentials. Pass ``client`` to inject a client in tests.
    Temperature defaults to 0 to keep grounded answers as repeatable as
    possible.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        self.model_id = (
            model_id
            or os.environ.get("MINIRAG_LLM_MODEL_ID")
            or DEFAULT_BEDROCK_LLM_MODEL_ID
        )
        self.region_name = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_BEDROCK_REGION
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            import boto3  # imported lazily so tests/offline use don't need it

            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def complete(self, system: str, user: str) -> str:
        response = self._get_client().converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        blocks = response["output"]["message"]["content"]
        return "".join(block["text"] for block in blocks if "text" in block).strip()
