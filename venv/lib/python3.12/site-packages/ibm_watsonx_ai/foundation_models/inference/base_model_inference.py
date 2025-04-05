#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import json
import copy
from contextlib import contextmanager, asynccontextmanager
from warnings import warn

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Generator,
    Any,
    Literal,
    AsyncGenerator,
    Callable,
    Type,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields

import httpx
import requests as _requests

from ibm_watsonx_ai.foundation_models.utils.utils import (
    HAPDetectionWarning,
    PIIDetectionWarning,
)
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
    BaseSchema,
)
import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_client_error import WMLClientError, UnsupportedOperation

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient

__all__ = ["BaseModelInference"]

_RETRY_STATUS_CODES = [429, 503, 504, 520]


class BaseModelInference(WMLResource, ABC):
    """Base interface class for the model interface."""

    DEFAULT_CONCURRENCY_LIMIT = 8

    def __init__(
        self,
        name: str,
        client: APIClient,
        persistent_connection: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ):
        self._persistent_connection = persistent_connection

        self._transport_params = requests._httpx_transport_params(client)

        WMLResource.__init__(self, name, client)
        if self._persistent_connection:
            self._http_client = client.httpx_client
        else:
            self._http_client = requests  # type: ignore[assignment]
            persistent_connection_warn = (
                "`persistent_connection` is deprecated and will be removed in future. "
            )
            warn(persistent_connection_warn, category=DeprecationWarning)

        self._async_http_client = client.async_httpx_client
        # Set initially 8 requests per second as it is default for prod instances
        # if header "x-requests-limit-rate" is different capacity will be updated
        self.rate_limiter = requests.TokenBucket(rate=8, capacity=8)

        self.retry_status_codes = retry_status_codes
        self.max_retries = max_retries
        self.delay_time = delay_time

    @abstractmethod
    def get_details(self) -> dict:
        """Get model interface's details

        :return: details of model or deployment
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat response.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> Generator:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat as generator.
        """
        raise NotImplementedError

    @abstractmethod
    async def achat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    async def achat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> AsyncGenerator:
        """
        Given a messages as input, and parameters the selected inference
        will generate a chat as a async generator.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        raise NotImplementedError

    @abstractmethod
    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> dict:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        raise NotImplementedError

    @abstractmethod
    async def agenerate_stream(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> AsyncGenerator:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_text_stream(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        raise NotImplementedError

    def _prepare_chat_payload(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_beta_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _send_inference_payload_raw(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        _http_client: httpx.Client | None = None,
    ) -> httpx.Response | _requests.Response:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        http_client = _http_client or self._http_client

        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._post(http_client=http_client, **post_params)

    def _send_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        _http_client: requests.HTTPXClient | None = None,
    ) -> dict:

        response_scoring = self._send_inference_payload_raw(
            prompt=prompt,
            params=params,
            generate_url=generate_url,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            _http_client=_http_client,
        )

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="generated_text",
        )

    def _send_chat_payload(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        generate_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> dict:
        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        response_scoring = self._post(self._http_client, **post_params)

        return self._handle_response(
            200,
            "chat",
            response_scoring,
            _field_to_hide="choices",
        )

    def _send_deployment_chat_payload(
        self,
        deployment_chat_url: str,
        messages: list[dict],
        context: str | None = None,
    ) -> dict:
        payload: dict = {"messages": messages}
        if context:
            payload.update({"context": context})

        post_params: dict[str, Any] = dict(
            url=deployment_chat_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        response_scoring = self._post(self._http_client, **post_params)

        if response_scoring.status_code == 404:
            raise UnsupportedOperation(
                Messages.get_message(message_id="chat_deployment_not_supported")
            )

        return self._handle_response(
            200,
            "chat",
            response_scoring,
            _field_to_hide="choices",
        )

    async def _asend_deployment_chat_payload(
        self,
        deployment_chat_url: str,
        messages: list[dict],
        context: str | None = None,
    ) -> dict:
        payload: dict = {"messages": messages}
        if context:
            payload.update({"context": context})

        post_params: dict[str, Any] = dict(
            url=deployment_chat_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        response = await self._apost(self._async_http_client, **post_params)

        if response.status_code == 404:
            raise UnsupportedOperation(
                Messages.get_message(message_id="chat_deployment_not_supported")
            )

        return self._handle_response(
            200,
            "achat",
            response,
            _field_to_hide="choices",
        )

    async def _asend_inference_payload(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        response = await self._apost(
            self._async_http_client,
            url=generate_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        return self._handle_response(200, "agenerate", response)

    async def _agenerate_stream_with_url(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> AsyncGenerator:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        if isinstance(self._async_http_client, requests.HTTPXAsyncClient):
            stream_function = self._async_http_client.post_stream
        else:
            stream_function = self._async_http_client.stream  # type: ignore[assignment]

        kw_args: dict = dict(
            method="POST",
            url=generate_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
                    if chunk.rstrip() == "event: error":
                        chunk = await anext(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during generating stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and "generated_text" in chunk:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")

                        yield parsed_response

            elif resp.status_code != 200:
                await resp.aread()
                raise WMLClientError(
                    f"Request failed with: ({resp.text} {resp.status_code})"
                )

    def _generate_chat_stream_with_url(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        chat_stream_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> Generator:

        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        kw_args: dict = dict(
            url=chat_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        if isinstance(self._http_client, requests.HTTPXClient):
            stream_function = self._http_client.post_stream
            kw_args |= {"method": "POST"}
        elif isinstance(self._http_client, httpx.Client):
            stream_function = self._http_client.stream  # type: ignore[assignment]
            kw_args |= {"method": "POST"}
        else:
            stream_function = requests.Session().post  # type: ignore[assignment]
            kw_args |= {
                "stream": True,
            }
        with self._stream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)  # type: ignore[call-arg]
                )
                for chunk in resp_iter:
                    if isinstance(resp, _requests.Response):
                        chunk = chunk.decode("utf-8")  # type: ignore[union-attr]
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    @requests._with_retry_stream()
    @contextmanager
    def _stream(
        self,
        stream_function: Callable,
        **kw_args: Any,
    ) -> Generator[httpx.Response, None, None]:
        """Handles streaming with retry."""
        with stream_function(**kw_args) as resp:
            yield resp

    @requests._with_async_retry_stream()
    @asynccontextmanager
    async def _astream(
        self,
        stream_function: Callable,
        **kw_args: Any,
    ) -> AsyncGenerator[httpx.Response, None]:
        async with stream_function(**kw_args) as resp:
            yield resp

    def _generate_deployment_chat_stream_with_url(
        self,
        deployment_chat_stream_url: str,
        messages: list[dict],
        context: str | None = None,
    ) -> Generator:

        payload: dict = {"messages": messages}
        if context:
            payload.update({"context": context})

        kw_args: dict = dict(
            url=deployment_chat_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        if isinstance(self._http_client, requests.HTTPXClient):
            stream_function = self._http_client.post_stream
            kw_args |= {"method": "POST"}
        elif isinstance(self._http_client, httpx.Client):
            stream_function = self._http_client.stream  # type: ignore[assignment]
            kw_args |= {"method": "POST"}
        else:
            stream_function = requests.Session().post  # type: ignore[assignment]
            kw_args |= {
                "stream": True,
            }

        with self._stream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)  # type: ignore[call-arg]
                )
                for chunk in resp_iter:
                    if isinstance(resp, _requests.Response):
                        chunk = chunk.decode("utf-8")  # type: ignore[union-attr]
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            elif resp.status_code == 404:
                raise UnsupportedOperation(
                    Messages.get_message(message_id="chat_deployment_not_supported")
                )

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    async def _agenerate_deployment_chat_stream_with_url(
        self,
        deployment_chat_stream_url: str,
        messages: list[dict],
        context: str | None = None,
    ) -> AsyncGenerator:

        payload: dict = {"messages": messages}
        if context:
            payload.update({"context": context})

        if isinstance(self._async_http_client, requests.HTTPXAsyncClient):
            stream_function = self._async_http_client.post_stream
        else:
            stream_function = self._async_http_client.stream  # type: ignore[assignment]

        kw_args: dict = dict(
            method="POST",
            url=deployment_chat_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )
        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            elif resp.status_code != 200:
                await resp.aread()
                raise WMLClientError(
                    f"Request failed with: ({resp.text} {resp.status_code})"
                )

    async def _agenerate_chat_stream_with_url(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None,
        chat_stream_url: str,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> AsyncGenerator:

        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        if isinstance(self._async_http_client, requests.HTTPXAsyncClient):
            stream_function = self._async_http_client.post_stream
        else:
            stream_function = self._async_http_client.stream  # type: ignore[assignment]

        kw_args: dict = dict(
            method="POST",
            url=chat_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        async with self._astream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = resp.aiter_lines()

                async for chunk in resp_iter:
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and response:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        yield parsed_response

            elif resp.status_code != 200:
                await resp.aread()
                raise WMLClientError(
                    f"Request failed with: ({resp.text} {resp.status_code})"
                )

    def _generate_with_url(
        self,
        prompt: list | str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> list | dict:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        if isinstance(prompt, list):
            # For batch of prompts use keep-alive connection even if persistent_connection=False
            http_client: requests.HTTPXClient | httpx.Client = (
                self._client.httpx_client
                if not self._persistent_connection
                else self._http_client
            )

            def make_request(_prompt: str) -> dict:
                """Rate-limited request with dynamic token adjustment and retry logic."""
                self.rate_limiter.acquire()

                inference_response = self._send_inference_payload_raw(
                    prompt=_prompt,
                    params=params,
                    generate_url=generate_url,
                    guardrails=guardrails,
                    guardrails_hap_params=guardrails_hap_params,
                    guardrails_pii_params=guardrails_pii_params,
                    _http_client=http_client,
                )
                rate_limit = int(
                    inference_response.headers.get("x-requests-limit-rate", 8)
                )
                if rate_limit and rate_limit != self.rate_limiter.capacity:
                    self.rate_limiter.capacity = rate_limit

                rate_limit_remaining = int(
                    inference_response.headers.get(
                        "x-requests-limit-remaining", self.rate_limiter.capacity
                    )
                )
                self.rate_limiter.adjust_tokens(rate_limit_remaining)

                return self._handle_response(
                    200,
                    "generate",
                    inference_response,
                    _field_to_hide="generated_text",
                )

            if (prompt_length := len(prompt)) <= concurrency_limit:
                with ThreadPoolExecutor(max_workers=prompt_length) as executor:
                    generated_responses = list(executor.map(make_request, prompt))
            else:
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    generated_responses = list(executor.map(make_request, prompt))
            return generated_responses

        else:
            response = self._send_inference_payload(
                prompt,
                params,
                generate_url,
                guardrails,
                guardrails_hap_params,
                guardrails_pii_params,
            )
        return response

    def _generate_with_url_async(
        self,
        prompt: list | str | None,
        params: dict | TextGenParameters | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> Generator:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        async_params = params or {}
        async_params = copy.deepcopy(async_params)

        if isinstance(async_params, BaseSchema):
            async_params = async_params.to_dict()

        async_params["return_options"] = {"input_text": True}
        if isinstance(prompt, list):
            # For batch of prompts use keep-alive connection even if persistent_connection=False
            http_client: requests.HTTPXClient | httpx.Client = (
                self._client.httpx_client
                if not self._persistent_connection
                else self._http_client
            )

            def make_request(_prompt: str) -> dict:
                """Rate-limited request with dynamic token adjustment and retry logic."""
                self.rate_limiter.acquire()

                inference_response = self._send_inference_payload_raw(
                    prompt=_prompt,
                    params=async_params,
                    generate_url=generate_url,
                    guardrails=guardrails,
                    guardrails_hap_params=guardrails_hap_params,
                    guardrails_pii_params=guardrails_pii_params,
                    _http_client=http_client,
                )
                rate_limit = int(
                    inference_response.headers.get("x-requests-limit-rate", 8)
                )
                if rate_limit and rate_limit != self.rate_limiter.capacity:
                    self.rate_limiter.capacity = rate_limit

                rate_limit_remaining = int(
                    inference_response.headers.get(
                        "x-requests-limit-remaining", self.rate_limiter.capacity
                    )
                )
                self.rate_limiter.adjust_tokens(rate_limit_remaining)

                return self._handle_response(
                    200,
                    "generate",
                    inference_response,
                    _field_to_hide="generated_text",
                )

            if (prompt_length := len(prompt)) <= concurrency_limit:
                with ThreadPoolExecutor(max_workers=prompt_length) as executor:
                    generate_futures = [
                        executor.submit(make_request, single_prompt)
                        for single_prompt in prompt
                    ]
                    try:
                        for future in as_completed(generate_futures):
                            yield future.result()
                    except:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise

            else:
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    generate_futures = [
                        executor.submit(make_request, single_prompt)
                        for single_prompt in prompt
                    ]
                    try:
                        for future in as_completed(generate_futures):
                            yield future.result()
                    except:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise
        else:
            response = self._send_inference_payload(
                prompt,
                async_params,
                generate_url,
                guardrails,
                guardrails_hap_params,
                guardrails_pii_params,
            )
            yield response

    def _generate_stream_with_url(
        self,
        prompt: str | None,
        params: dict | TextGenParameters | None,
        generate_stream_url: str,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> Generator:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        kw_args: dict = dict(
            url=generate_stream_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        if isinstance(self._http_client, requests.HTTPXClient):
            stream_function = self._http_client.post_stream
            kw_args |= {"method": "POST"}
        elif isinstance(self._http_client, httpx.Client):
            stream_function = self._http_client.stream  # type: ignore[assignment]
            kw_args |= {"method": "POST"}
        else:
            stream_function = requests.Session().post  # type: ignore[assignment]
            kw_args |= {
                "stream": True,
            }
        with self._stream(stream_function, **kw_args) as resp:
            if resp.status_code == 200:
                resp_iter = (
                    resp.iter_lines()
                    if isinstance(resp, httpx.Response)
                    else resp.iter_lines(decode_unicode=False)  # type: ignore[call-arg]
                )
                for chunk in resp_iter:
                    if isinstance(resp, _requests.Response):
                        chunk = chunk.decode("utf-8")  # type: ignore[union-attr]
                    if chunk.rstrip() == "event: error":
                        chunk = next(resp_iter)
                        field_name, _, response = chunk.partition(":")
                        raise WMLClientError(
                            error_msg="Error event occurred during generating stream.",
                            reason=response,
                        )

                    field_name, _, response = chunk.partition(":")
                    if field_name == "data" and "generated_text" in chunk:
                        try:
                            parsed_response = json.loads(response)
                        except json.JSONDecodeError:
                            raise Exception(f"Could not parse {response} as json")
                        if raw_response:
                            yield parsed_response
                            continue
                        yield self._return_guardrails_stats(parsed_response)[
                            "generated_text"
                        ]

            else:
                if isinstance(resp, httpx.Response):
                    resp.read()
                raise WMLClientError(
                    f"Request failed with: {resp.text} ({resp.status_code})"
                )

    def _tokenize_with_url(
        self,
        prompt: str,
        tokenize_url: str,
        return_tokens: bool,
    ) -> dict:
        payload = self._prepare_inference_payload(prompt)

        parameters = payload.get("parameters", {})
        parameters.update({"return_tokens": return_tokens})
        payload["parameters"] = parameters

        post_params: dict[str, Any] = dict(
            url=tokenize_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        if not isinstance(self._http_client, httpx.Client):
            post_params["_retry_status_codes"] = _RETRY_STATUS_CODES

        response_scoring = self._post(self._http_client, **post_params)

        if response_scoring.status_code == 404:
            raise WMLClientError("Tokenize is not supported for this release")
        return self._handle_response(200, "tokenize", response_scoring)

    def _return_guardrails_stats(self, single_response: dict) -> dict:
        results = single_response["results"][0]
        hap_details = (
            results.get("moderations", {}).get("hap")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("hap")
        )  # Remove 'else' on CPD 5.0 release
        if hap_details:
            if hap_details[0].get("input"):
                unsuitable_input_warning = next(
                    warning.get("message")
                    for warning in single_response.get("system", {}).get("warnings")
                    if warning.get("id") == "UNSUITABLE_INPUT"
                )
                warn(unsuitable_input_warning, category=HAPDetectionWarning)
            else:
                harmful_text_warning = (
                    f"Potentially harmful text detected: {hap_details}"
                )
                warn(harmful_text_warning, category=HAPDetectionWarning)
        pii_details = (
            results.get("moderations", {}).get("pii")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("pii")
        )  # Remove 'else' on CPD 5.0 release
        if pii_details:
            if pii_details[0].get("input"):
                unsuitable_input_warning = next(
                    warning.get("message")
                    for warning in single_response.get("system", {}).get("warnings")
                    if warning.get("id") == "UNSUITABLE_INPUT"
                )
                warn(unsuitable_input_warning, category=PIIDetectionWarning)
            else:
                identifiable_information_warning = (
                    f"Personally identifiable information detected: {pii_details}"
                )
                warn(identifiable_information_warning, category=PIIDetectionWarning)
        return results

    @staticmethod
    def _update_moderations_params(additional_params: dict) -> dict:
        default_params = {"input": {"enabled": True}, "output": {"enabled": True}}
        if additional_params:
            for key, value in default_params.items():
                if key in additional_params:
                    if additional_params[key]:
                        if "threshold" in additional_params:
                            default_params[key]["threshold"] = additional_params[
                                "threshold"
                            ]
                    else:
                        default_params[key]["enabled"] = False
                else:
                    if "threshold" in additional_params:
                        default_params[key]["threshold"] = additional_params[
                            "threshold"
                        ]
            if "mask" in additional_params:
                default_params.update({"mask": additional_params["mask"]})
        return default_params

    @staticmethod
    def _validate_and_overwrite_params(
        params: dict[str, Any], valid_param: TextChatParameters | TextGenParameters
    ) -> dict[str, Any]:
        """Validate and fix parameters"""
        chat_valid_params = {field.name.lower() for field in fields(valid_param)}
        valid_params = {}
        invalid_params = {}

        for param, value in params.items():
            if param.lower() in chat_valid_params:
                valid_params[param] = value
            else:
                invalid_params[param] = value

        if invalid_params:
            invalid_params_warning = f"Parameters [{', '.join(invalid_params)}] is/are not recognized and will be ignored."
            warn(invalid_params_warning)

        return valid_params

    @requests._with_retry()
    def _post(
        self, http_client: Any, *args: Any, **kwargs: Any
    ) -> httpx.Response | _requests.Response:
        return http_client.post(*args, **kwargs)

    @requests._with_async_retry()
    async def _apost(
        self, async_http_client: Any, *args: Any, **kwargs: Any
    ) -> httpx.Response:
        return await async_http_client.post(*args, **kwargs)
