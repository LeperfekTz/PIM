#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from enum import Enum
from copy import deepcopy
from typing import (
    Generator,
    cast,
    TYPE_CHECKING,
    overload,
    Literal,
    AsyncGenerator,
)
from warnings import catch_warnings, simplefilter

__all__ = ["FMModelInference"]


from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    BaseSchema,
    TextGenParameters,
)
from ibm_watsonx_ai.foundation_models.utils.utils import (
    _check_model_state,
    get_model_specs,
)
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.messages.messages import Messages
from .base_model_inference import BaseModelInference

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class FMModelInference(BaseModelInference):
    """Base abstract class for the model interface."""

    def __init__(
        self,
        *,
        model_id: str,
        api_client: APIClient,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        validate: bool = True,
        persistent_connection: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ):
        self.model_id = model_id
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value

        self.params = params
        FMModelInference._validate_type(
            params, "params", [dict, TextChatParameters, TextGenParameters], False, True
        )

        self._client = api_client
        self._tech_preview = False
        if validate:
            model_specs: dict = dict()

            if self._client._use_fm_ga_api:
                model_specs = self._client.foundation_models.get_model_specs()  # type: ignore[assignment]
            else:
                with catch_warnings():
                    simplefilter("ignore", category=DeprecationWarning)
                    model_specs = get_model_specs(self._client.credentials.url)  # type: ignore[arg-type]

            supported_models = [
                spec["model_id"] for spec in model_specs.get("resources", [])
            ]

            if self.model_id not in supported_models:
                if self._client._use_fm_ga_api:
                    model_specs = self._client.foundation_models.get_model_specs(  # type: ignore[assignment]
                        tech_preview=True
                    )
                    supported_models.clear()
                    for spec in model_specs.get("resources", []):
                        supported_models.append(spec["model_id"])
                        if self.model_id == spec["model_id"]:
                            if "tech_preview" in spec:  # check if tech_preview model
                                self._tech_preview = True
                            break

                if not self._tech_preview:
                    raise WMLClientError(
                        error_msg=f"Model '{self.model_id}' is not supported for this environment. "
                        f"Supported models: {supported_models}"
                    )

            # check if model is in constricted mode
            _check_model_state(
                self._client,
                self.model_id,
                tech_preview=self._tech_preview,
                model_specs=model_specs,
            )

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        BaseModelInference.__init__(
            self,
            __name__,
            self._client,
            persistent_connection,
            max_retries,
            delay_time,
            retry_status_codes,
        )

    def get_details(self) -> dict:
        """Get model's details

        :return: details of model or deployment
        :rtype: dict
        """
        if self._client._use_fm_ga_api:
            return self._client.foundation_models.get_model_specs(self.model_id, tech_preview=self._tech_preview)  # type: ignore[return-value]
        else:
            with catch_warnings():
                simplefilter("ignore", category=DeprecationWarning)
                return get_model_specs(self._client.credentials.url, self.model_id)  # type: ignore[arg-type]

    def chat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> dict:

        text_chat_url = (
            self._client.service_instance._href_definitions.get_fm_chat_href("chat")
        )

        return self._send_chat_payload(
            messages=messages,
            params=params,
            generate_url=text_chat_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    def chat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> Generator:

        text_chat_stream_url = (
            self._client.service_instance._href_definitions.get_fm_chat_href(
                "chat_stream"
            )
        )

        return self._generate_chat_stream_with_url(
            messages=messages,
            params=params,
            chat_stream_url=text_chat_stream_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    async def achat(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> dict:

        text_chat_url = (
            self._client.service_instance._href_definitions.get_fm_chat_href("chat")
        )

        payload = self._prepare_chat_payload(
            messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

        response = await self._apost(
            self._async_http_client,
            url=text_chat_url,
            json=payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
        )

        return self._handle_response(200, "achat", response, _field_to_hide="choices")

    async def achat_stream(
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: Literal["none", "auto"] | None = None,
        context: str | None = None,
    ) -> AsyncGenerator:

        text_chat_stream_url = (
            self._client.service_instance._href_definitions.get_fm_chat_href(
                "chat_stream"
            )
        )

        return self._agenerate_chat_stream_with_url(
            messages=messages,
            params=params,
            chat_stream_url=text_chat_stream_url,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )

    @overload
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | TextGenParameters | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: Literal[False] = ...,
        validate_prompt_variables: bool = ...,
    ) -> dict | list[dict]: ...

    @overload
    def generate(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        async_mode: Literal[True],
        validate_prompt_variables: bool,
    ) -> Generator: ...

    @overload
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | TextGenParameters | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: bool = ...,
        validate_prompt_variables: bool = ...,
    ) -> dict | list[dict] | Generator: ...

    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = BaseModelInference.DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
    ) -> dict | list[dict] | Generator:
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )
        self._validate_type(
            prompt, "prompt", [str, list], True, raise_error_for_list=True
        )
        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )

        generate_text_url = (
            self._client.service_instance._href_definitions.get_fm_generation_href(
                "text"
            )
        )
        prompt = cast(str | list, prompt)
        if async_mode:
            self.params = cast(dict | TextGenParameters, self.params)

            return self._generate_with_url_async(
                prompt=prompt,
                params=params or self.params,
                generate_url=generate_text_url,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
                concurrency_limit=concurrency_limit,
            )

        else:
            return self._generate_with_url(
                prompt=prompt,
                params=params,
                generate_url=generate_text_url,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
                concurrency_limit=concurrency_limit,
            )

    async def _agenerate_single(
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> dict:

        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )

        self._validate_type(prompt, "prompt", str, True)
        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )
        generate_text_url = (
            self._client.service_instance._href_definitions.get_fm_generation_href(
                "text"
            )
        )

        async_params = params or self.params or {}

        if isinstance(async_params, BaseSchema):
            async_params = async_params.to_dict()

        return await self._asend_inference_payload(
            prompt,
            async_params,
            generate_text_url,
            guardrails,
            guardrails_hap_params,
            guardrails_pii_params,
        )

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
        will generate a completion text as async generator response.
        """
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change its value for other scenarios."
            )

        self._validate_type(
            guardrails_hap_params, "guardrails_hap_params", dict, mandatory=False
        )
        self._validate_type(
            guardrails_pii_params, "guardrails_pii_params", dict, mandatory=False
        )

        self._validate_type(prompt, "prompt", str, True)

        generate_text_url = (
            self._client.service_instance._href_definitions.get_fm_generation_stream_href()
        )

        async_params = params or self.params or {}

        if isinstance(async_params, BaseSchema):
            async_params = async_params.to_dict()

        return self._agenerate_stream_with_url(
            prompt,
            async_params,
            generate_text_url,
            guardrails,
            guardrails_hap_params,
            guardrails_pii_params,
        )

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
        # if user change default value for `validate_prompt_variables` params raise an error
        if not validate_prompt_variables:
            raise ValueError(
                "`validate_prompt_variables` is only applicable for Prompt Template Asset deployment. Do not change it value for others scenarios."
            )
        self._validate_type(prompt, "prompt", str, True)
        if self._client._use_fm_ga_api:
            generate_text_stream_url = (
                self._client.service_instance._href_definitions.get_fm_generation_stream_href()
            )
        else:
            generate_text_stream_url = (
                self._client.service_instance._href_definitions.get_fm_generation_href(
                    f"text_stream"
                )
            )  # Remove on CPD 5.0 release
        prompt = cast(str, prompt)
        return self._generate_stream_with_url(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            generate_stream_url=generate_text_stream_url,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
        )

    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        """
        Given a text prompt as input, and return_tokens parameter will return tokenized input text.
        """
        self._validate_type(prompt, "prompt", str, True)
        generate_tokenize_url = (
            self._client.service_instance._href_definitions.get_fm_tokenize_href()
        )

        return self._tokenize_with_url(
            prompt=prompt,
            tokenize_url=generate_tokenize_url,
            return_tokens=return_tokens,
        )

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return {
            "model_id": self.model_id,
            "params": (
                self.params.to_dict()
                if isinstance(self.params, BaseSchema)
                else self.params
            ),
            "project_id": self._client.default_project_id,
            "space_id": self._client.default_space_id,
        }

    def _prepare_inference_payload(  # type: ignore[override]
        self,
        prompt: str,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        payload: dict = {
            "model_id": self.model_id,
            "input": prompt,
        }
        if guardrails:
            if guardrails_hap_params is None:
                guardrails_hap_params = dict(
                    input=True, output=True
                )  # HAP enabled if guardrails = True

            for guardrail_type, guardrails_params in zip(
                ("hap", "pii"), (guardrails_hap_params, guardrails_pii_params)
            ):
                if guardrails_params is not None:
                    if "moderations" not in payload:
                        payload["moderations"] = {}
                    payload["moderations"].update(
                        {
                            guardrail_type: self._update_moderations_params(
                                guardrails_params
                            )
                        }
                    )

        if params is not None:
            parameters = params

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        elif self.params is not None:

            self.params = cast(dict | TextGenParameters, self.params)
            parameters = deepcopy(self.params)

            if isinstance(parameters, dict):
                parameters = self._validate_and_overwrite_params(
                    parameters, TextGenParameters()
                )

            elif isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        else:
            parameters = None

        if parameters:
            payload["parameters"] = parameters

        if (
            "parameters" in payload
            and GenTextParamsMetaNames.DECODING_METHOD in payload["parameters"]
        ):
            if isinstance(
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD],
                DecodingMethods,
            ):
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD] = payload[
                    "parameters"
                ][GenTextParamsMetaNames.DECODING_METHOD].value

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        if "parameters" in payload and "return_options" in payload["parameters"]:
            if not (
                payload["parameters"]["return_options"].get("input_text", False)
                or payload["parameters"]["return_options"].get("input_tokens", False)
            ):
                raise WMLClientError(
                    Messages.get_message(
                        message_id="fm_required_parameters_not_provided"
                    )
                )

        return payload

    def _prepare_chat_payload(  # type: ignore[override]
        self,
        messages: list[dict],
        params: dict | TextChatParameters | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
        tool_choice_option: str | None = None,
    ) -> dict:

        payload: dict = {
            "model_id": self.model_id,
            "messages": messages,
        }

        if params is not None:
            parameters = params

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        elif self.params is not None:

            self.params = cast(dict | TextChatParameters, self.params)
            parameters = deepcopy(self.params)

            if isinstance(parameters, dict):
                parameters = self._validate_and_overwrite_params(
                    parameters, TextChatParameters()
                )

            elif isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        else:
            parameters = None

        if parameters:
            payload.update(parameters)

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if tool_choice_option:
            payload["tool_choice_option"] = tool_choice_option

        return payload

    def _prepare_beta_inference_payload(  # type: ignore[override]
        self,  # Remove on CPD 5.0 release
        prompt: str,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        payload: dict = {
            "model_id": self.model_id,
            "input": prompt,
        }
        if guardrails:
            default_moderations_params = {"input": True, "output": True}
            payload.update(
                {
                    "moderations": {
                        "hap": (
                            default_moderations_params | (guardrails_hap_params or {})
                        )
                    }
                }
            )

            if guardrails_pii_params is not None:
                payload["moderations"].update({"pii": guardrails_pii_params})

        if params is not None:
            parameters = params

            if isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        elif self.params is not None:

            self.params = cast(dict | TextGenParameters, self.params)
            parameters = deepcopy(self.params)

            if isinstance(parameters, dict):
                parameters = self._validate_and_overwrite_params(
                    parameters, TextGenParameters()
                )

            elif isinstance(parameters, BaseSchema):
                parameters = parameters.to_dict()

        else:
            parameters = None

        if parameters:
            payload["parameters"] = parameters

        if (
            "parameters" in payload
            and GenTextParamsMetaNames.DECODING_METHOD in payload["parameters"]
        ):
            if isinstance(
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD],
                DecodingMethods,
            ):
                payload["parameters"][GenTextParamsMetaNames.DECODING_METHOD] = payload[
                    "parameters"
                ][GenTextParamsMetaNames.DECODING_METHOD].value

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        if "parameters" in payload and "return_options" in payload["parameters"]:
            if not (
                payload["parameters"]["return_options"].get("input_text", False)
                or payload["parameters"]["return_options"].get("input_tokens", False)
            ):
                raise WMLClientError(
                    Messages.get_message(
                        message_id="fm_required_parameters_not_provided"
                    )
                )

        return payload
