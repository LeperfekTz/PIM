#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, overload, Literal, Any
from warnings import warn, catch_warnings, simplefilter

from ibm_watsonx_ai.foundation_models.inference import ModelInference
from ibm_watsonx_ai.wml_client_error import MissingExtension
from ibm_watsonx_ai.foundation_models.schema import (
    TextChatParameters,
    TextGenParameters,
)

if TYPE_CHECKING:
    from langchain_ibm import WatsonxLLM
    from ibm_watsonx_ai import Credentials


class Model(ModelInference):
    """Instantiate the model interface.

    .. deprecated:: 1.1.21
        Use :func:`ModelInference() <ibm_watsonx_ai.foundation_models.inference.ModelInference>` instead.

    .. hint::
        To use the Model class with LangChain, use the :func:`to_langchain() <ibm_watsonx_ai.foundation_models.Model.to_langchain>` function.

    :param model_id: type of model to use
    :type model_id: str

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict

    :param params: parameters to use during generate requests
    :type params: dict, TextGenParameters, TextChatParameters, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of following as verify:

        * the path to a CA_BUNDLE file
        * the path of a directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param validate: model ID validation, defaults to True
    :type validate: bool, optional

    .. note::
        One of these parameters is required: ['project_id ', 'space_id'].

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import Model
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

        # To display example params enter
        GenParams().get_example_values()

        generate_params = {
            GenParams.MAX_NEW_TOKENS: 25
        }

        model = Model(
            model_id=ModelTypes.FLAN_UL2,
            params=generate_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )
    """

    def __init__(
        self,
        model_id: str,
        credentials: Credentials | dict,
        params: dict | TextChatParameters | TextGenParameters | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: str | bool | None = None,
        validate: bool = True,
    ) -> None:
        with catch_warnings():
            simplefilter("default", category=DeprecationWarning)
            model_class_deprecated_warning = (
                "The `Model` class is deprecated and will be removed in a future release. "
                "Please use the `ModelInference` class instead. "
                "To update your imports, use: `from ibm_watsonx_ai.foundation_models import ModelInference`."
            )
            warn(model_class_deprecated_warning, category=DeprecationWarning)

        ModelInference.__init__(
            self,
            model_id=model_id,
            credentials=credentials,
            params=params,
            project_id=project_id,
            space_id=space_id,
            verify=verify,
            validate=validate,
        )

    def get_details(self) -> dict:
        """Get the model's details.

        :return: model's details
        :rtype: dict

        **Example:**

        .. code-block:: python

            model.get_details()

        """
        return super().get_details()

    def to_langchain(self) -> WatsonxLLM:
        """

        :return: WatsonxLLM wrapper for watsonx foundation models
        :rtype: WatsonxLLM

        **Example:**

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai.foundation_models import Model
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            flan_ul2_model = Model(
                model_id=ModelTypes.FLAN_UL2,
                credentials=Credentials(
                            api_key = IAM_API_KEY,
                            url = "https://us-south.ml.cloud.ibm.com"),
                project_id="*****"
                )

            prompt_template = "What color is the {flower}?"

            llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
            llm_chain.invoke('sunflower')

        """
        try:
            from langchain_ibm import WatsonxLLM
        except ImportError:
            raise MissingExtension("langchain_ibm")

        return WatsonxLLM(watsonx_model=self)

    @overload  # type: ignore[override]
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | TextGenParameters | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: Literal[False] = ...,
    ) -> dict | list[dict]: ...

    @overload  # type: ignore[override]
    def generate(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        async_mode: Literal[True],
    ) -> Generator: ...

    @overload  # type: ignore[override]
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | TextGenParameters | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: bool = ...,
    ) -> dict | list[dict] | Generator: ...

    def generate(  # type: ignore[override]
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = ModelInference.DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
    ) -> dict | list[dict] | Generator:
        """Generates a completion text as generated_text after getting
        a text prompt as input and parameters for the selected model (model_id).

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests that will be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: the prompt string or list of strings. If a list of strings is passed, requests will be managed in parallel with the rate of concurency_limit
        :type prompt: str, list

        :param guardrails: if True, the detection filter for potentially hateful, abusive, and/or profane language (HAP)
                            is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type params: dict

        :param async_mode: if True, then yields results asynchronously using a generator. In this case, both prompt and
                           generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool

        :return: scoring result that contains the generated content
        :rtype: dict

        **Example:**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_response = model.generate(prompt=q)
            print(generated_response['results'][0]['generated_text'])

        """
        return super().generate(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            async_mode=async_mode,
            validate_prompt_variables=True,  # keep default value, changing not permitted
        )

    @overload  # type: ignore[override]
    def generate_text(
        self,
        prompt: str | None = ...,
        params: dict | TextGenParameters | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
    ) -> str: ...

    @overload  # type: ignore[override]
    def generate_text(
        self,
        prompt: list,
        params: dict | TextGenParameters | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
    ) -> list[str]: ...

    @overload  # type: ignore[override]
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        raw_response: Literal[True],
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
    ) -> list[dict] | dict: ...

    @overload  # type: ignore[override]
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | TextGenParameters | None,
        raw_response: bool,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
    ) -> str | list | dict: ...

    def generate_text(  # type: ignore[override]
        self,
        prompt: str | list | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = ModelInference.DEFAULT_CONCURRENCY_LIMIT,
    ) -> str | list | dict:
        """Generates a completion text as generated_text after getting
        a text prompt as input and parameters for the selected model (model_id).

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param concurrency_limit: number of requests to be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: the prompt string or list of strings. If a list of strings is passed, requests will be managed in parallel with the rate of concurency_limit
        :type prompt: str, list

        :param raw_response: return the whole response object
        :type raw_response: bool, optional

        :param guardrails: if True, the detection filter for potentially hateful, abusive, and/or profane language (HAP)
                           is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type params: dict

        :return: generated content
        :rtype: str or dict

        **Example:**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_text = model.generate_text(prompt=q)
            print(generated_text)

        """
        return super().generate_text(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            validate_prompt_variables=True,  # keep default value, changing not permitted in this scenario
        )

    def generate_text_stream(  # type: ignore[override]
        self,
        prompt: str | None = None,
        params: dict | TextGenParameters | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> Generator:
        """Generates a streamed text as generate_text_stream after getting
        a text prompt as input and parameters for the selected model (model_id).

        :param params: MetaProps for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, TextGenParameters, optional

        :param prompt: the prompt string
        :type prompt: str,

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, the detection filter for potentially hateful, abusive, and/or profane language (HAP)
                           is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: MetaProps for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type params: dict

        :return: scoring result that contains the generated content
        :rtype: generator

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the sun"
            generated_response = model.generate_text_stream(prompt=q)

            for chunk in generated_response:
                print(chunk, end='', flush=True)

        """
        return super().generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=True,  # keep default value, changing not permitted in this scenario
        )

    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        """
        The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
        It splits text into words or sub-words, which are are converted to IDs through a look-up table (vocabulary).
        Tokenization allows the model to have a reasonable vocabulary size.

        :param prompt: prompt string
        :type prompt: str

        :param return_tokens: parameter for text tokenization, defaults to False
        :type return_tokens: bool

        :return: result of tokenizing the input string
        :rtype: dict

        **Example:**

        .. code-block:: python

            q = "Write an epigram about the moon"
            tokenized_response = model.tokenize(prompt=q, return_tokens=True)
            print(tokenized_response["result"])

        """
        return super().tokenize(prompt=prompt, return_tokens=return_tokens)

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
            print(response['choices'][0]['message']['content'])

        """
        return super().chat(
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
                print(chunk['choices'][0]['delta'].get('content', ''), end='', flush=True)

        """
        return super().chat_stream(
            messages=messages,
            params=params,
            tools=tools,
            tool_choice=tool_choice,
            tool_choice_option=tool_choice_option,
        )
