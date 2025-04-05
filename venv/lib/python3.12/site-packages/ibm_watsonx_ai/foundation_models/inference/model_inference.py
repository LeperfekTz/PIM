#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, cast, overload, Literal, AsyncGenerator
from enum import Enum
from warnings import warn

import httpx

from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ParamOutOfRange,
    InvalidMultipleArguments,
    MissingExtension,
)
from ibm_watsonx_ai._wrappers.requests import (
    _get_httpx_client,
    _get_async_client,
)
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
)
import ibm_watsonx_ai._wrappers.requests as requests
from .base_model_inference import BaseModelInference
from .fm_model_inference import FMModelInference
from .deployment_model_inference import DeploymentModelInference

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials
    from langchain_ibm import WatsonxLLM


class ModelInference(BaseModelInference):
    """Instantiate the model interface.

    .. hint::
        To use the ModelInference class with LangChain, use the :func:`WatsonxLLM <langchain_ibm.WatsonxLLM>` wrapper.

    :param model_id: type of model to use
    :type model_id: str, optional

    :param deployment_id: ID of tuned model's deployment
    :type deployment_id: str, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict, optional

    :param params: parameters to use during request generation
    :type params: dict, TextGenParameters, TextChatParameters, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param validate: Model ID validation, defaults to True
    :type validate: bool, optional

    :param persistent_connection: Whether to keep persistent connection when evaluating `generate`, `generate_text` or `tokenize` methods.
                                  This parameter is only applicable for the mentioned methods when the prompt is a str type.
                                  To close the connection, run `model.close_persistent_connection()`, defaults to True. Added in 1.1.2.
    :type persistent_connection: bool, optional

    :param max_retries: number of retries performed when request was not successful and status code is in retry_status_codes, defaults to 10
    :type max_retries: int, optional

    :param delay_time: delay time to retry request, factor in exponential backoff formula: wx_delay_time * pow(2.0, attempt), defaults to 0.5s
    :type delay_time: float, optional

    :param retry_status_codes: list of status codes which will be considered for retry mechanism, defaults to [429, 503, 504, 520]
    :type retry_status_codes: list[int], optional

    .. note::
        * You must provide one of these parameters: [``model_id``, ``deployment_id``]
        * When the ``credentials`` parameter is passed, you must provide one of these parameters: [``project_id``, ``space_id``].

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

        # To display example params enter
        GenParams().get_example_values()

        generate_params = {
            GenParams.MAX_NEW_TOKENS: 25
        }

        model_inference = ModelInference(
            model_id=ModelTypes.FLAN_UL2,
            params=generate_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        deployment_inference = ModelInference(
            deployment_id="<ID of deployed model>",
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        deployment_id: str | None = None,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        credentials: dict | Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
        validate: bool = True,
        persistent_connection: bool = True,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ) -> None:
        self.model_id = model_id
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value

        self.deployment_id = deployment_id

        if self.model_id and self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="Both arguments were provided.",
            )
        elif not self.model_id and not self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="None of the arguments were provided.",
            )

        self.params = params
        ModelInference._validate_type(
            params, "params", [dict, TextChatParameters, TextGenParameters], False, True
        )

        if credentials:
            from ibm_watsonx_ai import APIClient

            self.set_api_client(APIClient(credentials, verify=verify))
        elif api_client:
            self.set_api_client(api_client)
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if validate or model_id:
            # 1. When `model_id` is provided, regardless of `validate` value: space, project or api_client is required.
            # 2. When `deployment_id` is provided and `validate ` is set to True:
            # space, project or api_client with default project/space is required.
            if space_id:
                self._client.set.default_space(space_id)
            elif project_id:
                self._client.set.default_project(project_id)
            elif not (
                api_client
                and (
                    api_client.default_space_id
                    or api_client.default_project_id
                    or model_id
                )
            ):
                raise InvalidMultipleArguments(
                    params_names_list=["space_id", "project_id"],
                    reason="None of the arguments were provided.",
                )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        self._inference: BaseModelInference
        if self.model_id:
            self._inference = FMModelInference(
                model_id=self.model_id,
                api_client=self._client,
                params=self.params,
                validate=validate,
                persistent_connection=persistent_connection,
                max_retries=max_retries,
                delay_time=delay_time,
                retry_status_codes=retry_status_codes,
            )
        else:
            self.deployment_id = cast(str, self.deployment_id)
            self._inference = DeploymentModelInference(
                deployment_id=self.deployment_id,
                api_client=self._client,
                params=self.params,
                validate=validate,
                persistent_connection=persistent_connection,
                max_retries=max_retries,
                delay_time=delay_time,
                retry_status_codes=retry_status_codes,
            )

        self._transport_params = requests._httpx_transport_params(self._client)

    def get_details(self) -> dict:
        """Get the details of a model interface

        :return: details of the model or deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_inference.get_details()

        """
        return self._inference.get_details()

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
        Given a list of messages comprising a conversation, the model will return a response.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: dict

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
            generated_response = model.chat(messages=messages)

            # Print all response
            print(generated_response)

            # Print only content
            print(generated_response['choices'][0]['message']['content'])

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return self._inference.chat(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """
        Given a list of messages comprising a conversation, the model will return a response in stream.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: generator

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
            generated_response = model.chat_stream(messages=messages)

            for chunk in generated_response:
                if chunk['choices']:
                    print(chunk['choices'][0]['delta'].get('content', ''), end='', flush=True)

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return self._inference.chat_stream(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """
        Given a list of messages comprising a conversation with a chat model in an asynchronous manner.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: dict

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
            generated_response = await model.achat(messages=messages)

            # Print all response
            print(generated_response)

            # Print only content
            print(generated_response['choices'][0]['message']['content'])

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return await self._inference.achat(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
        )

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
        Given a list of messages comprising a conversation, the model will return a response in stream in an asynchronous manner.

        :param messages: The messages for this chat session.
        :type messages: list[dict]

        :param params: meta props for chat generation, use ``ibm_watsonx_ai.foundation_models.schema.TextChatParameters.show()``
        :type params: dict, TextChatParameters, optional

        :param tools: Tool functions that can be called with the response.
        :type tools: list

        :param tool_choice: Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
        :type tool_choice: dict, optional

        :param tool_choice_option: Tool choice option
        :type tool_choice_option: Literal["none", "auto"], optional

        :param context: context variable can be present in chat `system_prompt` or chat messages content fields and are
            identified by sentence '{{ context }}'. Supported only with `deployment_id`, defaults to None.
        :type context: str, optional

        :return: scoring result containing generated chat content.
        :rtype: AsyncGenerator

        **Example:**

        .. code-block:: python

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"}
            ]
            generated_response = await model.achat_stream(messages=messages)

            async for chunk in generated_response:
                if chunk['choices']:
                    print(chunk['choices'][0]['delta'].get('content', ''), end='', flush=True)

        """
        self._validate_type(messages, "messages", list, True)
        self._validate_type(params, "params", [dict, TextChatParameters], False, True)

        if context and self.model_id:
            raise WMLClientError(
                "The `context` parameter is only supported for inferring a chat prompt deployment."
            )

        return await self._inference.achat_stream(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
            context=context,
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
        """Generates a completion text as generated_text after getting a text prompt as input and parameters for the
        selected model (model_id) or deployment (deployment_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests to be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: prompt string or list of strings. If list of strings is passed, requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP)
                        is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param async_mode: If True, yields results asynchronously (using a generator). In this case, both prompt and
                           generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool

        :param validate_prompt_variables: If True and `ModelInference` instance has been initialized with `validate=True`, prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool, optional

        :return: scoring result the contains the generated content
        :rtype: dict

        **Example:**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_response = model_inference.generate(prompt=q)
            print(generated_response['results'][0]['generated_text'])

        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)
        self._validate_type(
            concurrency_limit,
            "concurrency_limit",
            [int, float],
            False,
            raise_error_for_list=True,
        )

        if isinstance(concurrency_limit, float):  # convert float (ex. 10.0) to int
            concurrency_limit = int(concurrency_limit)

        if concurrency_limit > 10 or concurrency_limit < 1:
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )
        if async_mode:
            warning_async_mode = (
                "In this mode, the results will be returned in the order in which the server returns the responses. "
                "Please notice that it does not support non-blocking requests scheduling. "
                "To use non-blocking native async inference method you may use `ModelInference.agenerate(...)`"
            )
            warn(warning_async_mode)
        return self._inference.generate(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            async_mode=async_mode,
            validate_prompt_variables=validate_prompt_variables,
        )

    async def _agenerate_single(  # type: ignore[override]
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return await self._inference._agenerate_single(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
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
        """Generates a stream as agenerate_stream after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: scoring result that contains the generated content
        :rtype: AsyncGenerator

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the sun"
            generated_response = await model_inference.agenerate_stream(prompt=q)

            async for chunk in generated_response:
                print(chunk, end='', flush=True)

        """

        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return await self._inference.agenerate_stream(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    @overload
    def generate_text(
        self,
        prompt: str | None = ...,
        params: dict | TextGenParameters | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        validate_prompt_variables: bool = ...,
    ) -> str: ...

    @overload
    def generate_text(
        self,
        prompt: list,
        params: dict | TextGenParameters | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        validate_prompt_variables: bool = ...,
    ) -> list[str]: ...

    @overload
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        raw_response: Literal[True],
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        validate_prompt_variables: bool,
    ) -> list[dict] | dict: ...

    @overload
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        raw_response: bool,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        validate_prompt_variables: bool,
    ) -> str | list | dict: ...

    def generate_text(
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = BaseModelInference.DEFAULT_CONCURRENCY_LIMIT,
        validate_prompt_variables: bool = True,
    ) -> str | list | dict:
        """Generates a completion text as generated_text after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests to be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: prompt string or list of strings. If list of strings is passed, requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param raw_response: returns the whole response object
        :type raw_response: bool, optional

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: generated content
        :rtype: str | list | dict

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)


        **Example:**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_text = model_inference.generate_text(prompt=q)
            print(generated_text)

        """
        metadata = ModelInference.generate(
            self,
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            validate_prompt_variables=validate_prompt_variables,
        )
        if raw_response:
            return metadata
        else:
            if isinstance(prompt, list):
                return [
                    self._return_guardrails_stats(single_response)["generated_text"]
                    for single_response in metadata
                ]
            else:
                return self._return_guardrails_stats(metadata)["generated_text"]  # type: ignore[arg-type]

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
        """Generates a streamed text as generate_text_stream after getting a text prompt as input and
        parameters for the selected model (model_id). For prompt template deployment, `prompt` should be None.

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: scoring result that contains the generated content
        :rtype: generator

        .. note::
            By default, only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the sun"
            generated_response = model_inference.generate_text_stream(prompt=q)

            for chunk in generated_response:
                print(chunk, end='', flush=True)

        """
        self._validate_type(params, "params", [dict, TextGenParameters], False, True)

        return self._inference.generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        """
        The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
        It splits text into words or sub-words, which then are converted to IDs through a look-up table (vocabulary).
        Tokenization allows the model to have a reasonable vocabulary size.

        .. note::
            The tokenization method is available only for base models and is not supported for deployments.

        :param prompt: prompt string, defaults to None
        :type prompt: str, optional

        :param return_tokens: parameter for text tokenization, defaults to False
        :type return_tokens: bool

        :return: result of tokenizing the input string
        :rtype: dict

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the moon"
            tokenized_response = model_inference.tokenize(prompt=q, return_tokens=True)
            print(tokenized_response["result"])

        """
        return self._inference.tokenize(prompt=prompt, return_tokens=return_tokens)

    def to_langchain(self) -> WatsonxLLM:
        """

        :return: WatsonxLLM wrapper for watsonx foundation models
        :rtype: WatsonxLLM

        **Example:**

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            flan_ul2_model = ModelInference(
                model_id=ModelTypes.FLAN_UL2,
                credentials=Credentials(
                                    api_key = IAM_API_KEY,
                                    url = "https://us-south.ml.cloud.ibm.com"),
                project_id="*****"
                )

            prompt_template = "What color is the {flower}?"

            llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
            llm_chain.invoke('sunflower')

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            deployed_model = ModelInference(
                deployment_id="<ID of deployed model>",
                credentials=Credentials(
                                    api_key = IAM_API_KEY,
                                    url = "https://us-south.ml.cloud.ibm.com"),
                space_id="*****"
                )

            prompt_template = "What color is the {car}?"

            llm_chain = LLMChain(llm=deployed_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
            llm_chain.invoke('sunflower')

        """
        try:
            from langchain_ibm import WatsonxLLM
        except ImportError:
            raise MissingExtension("langchain_ibm")
        return WatsonxLLM(watsonx_model=self)

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return self._inference.get_identifying_params()

    def close_persistent_connection(self) -> None:
        """
        Only applicable if persistent_connection was set to True in ModelInference initialization.
        Calling this method closes the current `httpx.Client` and recreates a new `httpx.Client` with default values:
        timeout: httpx.Timeout(timeout=30 * 60, connect=10)
        limit: httpx.Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY)
        """
        if self._inference._persistent_connection and isinstance(
            self._inference._http_client, httpx.Client
        ):
            self._inference._http_client.close()
            self._client.httpx_client = _get_httpx_client(
                transport_params=self._transport_params
            )
            self._inference._http_client = self._client.httpx_client

    def set_api_client(self, api_client: APIClient) -> None:
        """
        Set or refresh the APIClient object associated with ModelInference object.

        :param api_client: initialized APIClient object with a set project ID or space ID.
        :type api_client: APIClient, optional

        **Example:**

        .. code-block:: python

            api_client = APIClient(credentials=..., space_id=...)
            model_inference.set_api_client(api_client=api_client)

        """

        self._client = api_client
        if hasattr(self, "_inference"):
            self._inference._client = api_client

    async def agenerate(
        self,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> dict:
        """Generate a response in an asynchronous manner.

        :param prompt: prompt string, defaults to None
        :type prompt: str | None, optional

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: dict | None, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP) is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected, then the `HAPDetectionWarning` is issued
        :type guardrails: bool, optional

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict | None, optional

        :param validate_prompt_variables: If True, the prompt variables provided in `params` are validated with the ones in the Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool, optional

        :return: raw response that contains the generated content
        :rtype: dict
        """
        self._validate_type(params, "params", dict, False)
        return await self._inference._agenerate_single(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    async def aclose_persistent_connection(self) -> None:
        """Only applicable if persistent_connection was set to True in the ModelInference initialization."""
        if self._inference._persistent_connection and isinstance(
            self._inference._async_http_client, httpx.AsyncClient
        ):
            await self._inference._async_http_client.aclose()
            self._client.async_httpx_client = _get_async_client(
                transport_params=self._transport_params
            )
            self._inference._async_http_client = self._client.async_httpx_client
