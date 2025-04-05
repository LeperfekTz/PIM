#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
from typing import Any, Callable
from warnings import simplefilter, warn
from pathlib import Path
import json

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore
from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
    LangChainChunker,
)

from ibm_watsonx_ai.foundation_models.extensions.rag.retriever import (
    BaseRetriever,
    Retriever,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.default_inference_function import (
    default_inference_function,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.default_indexing_function import (
    default_indexing_function,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.default_inference_service import (
    inference_service as default_inference_service,
)
from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.default_inference_service_chroma import (
    inference_service as default_inference_service_chroma,
)

from ibm_watsonx_ai.foundation_models.prompts import PromptTemplateManager
from ibm_watsonx_ai.foundation_models.utils.enums import PromptTemplateFormats
from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    InvalidValue,
    MissingValue,
    ValidationError,
    WMLClientError,
)
from ibm_watsonx_ai.wml_resource import WMLResource

from .pattern_assets import RAGPatternFunction, RAGPatternService, Context


class RAGPattern:
    """Class for defining, querying and deploying Retrieval-Augmented Generation (RAG) patterns."""

    QUESTION_PLACEHOLDER = "{question}"
    DOCUMENT_PLACEHOLDER = "{document}"
    REFERENCE_DOCUMENTS_PLACEHOLDER = "{reference_documents}"

    def __init__(
        self,
        *,
        space_id: str | None = None,
        project_id: str | None = None,
        api_client: APIClient | None = None,
        auto_store: bool | None = False,
        credentials: Credentials | dict | None = None,
        model: ModelInference | None = None,
        prompt_id: str | None = None,
        indexing_function: Callable | None = None,
        inference_function: Callable | None = None,
        inference_service: Callable | None = None,
        indexing_function_params: dict | None = None,
        inference_function_params: dict | None = None,
        store_params: dict | None = None,
        retriever: BaseRetriever | None = None,
        vector_store: VectorStore | None = None,
        chunker: LangChainChunker | None = None,
        word_to_token_ratio: float = 1.5,
        ### Scenario with Chroma
        input_data_references: list[DataConnection] | None = None,
        #######################
        **kwargs: Any,
    ) -> None:
        """Initialize the ``RAGPattern`` object.

        .. note:: If the pattern's components (``vector_store``, ``prompt_id``, ``model``) are specified, the pattern
        will use default function template for querying and deployment. If custom ``inference_function`` is
        specified, the pattern's components are not utilized.

        .. hint:: Both default function template and custom ``inference_function`` provided by user can by modified
        by changing :meth:`pretty_print`'s output.

        :param space_id: ID of the Watson Studio space
        :type space_id: str

        :param project_id: ID of the Watson Studio project
        :type project_id: str

        :param api_client: initialized APIClient object, defaults to None
        :type api_client: APIClient, optional

        :param auto_store: whether to store the ``inference_function`` in the repository upon initialization, defaults to False
        :type auto_store: bool, optional

        :param credentials: credentials to Watson Machine Learning instance, defaults to None
        :type credentials: Credentials or dict, optional

        :param model: initialized :class:`ModelInference <ibm_watsonx_ai.foundation_models.inference.model_inference.ModelInference>` object, defaults to None
        :type model: ModelInference, optional

        :param prompt_id: Initialized ID of :class:`PromptTemplate <ibm_watsonx_ai.foundation_models.prompts.prompt_template.PromptTemplate>` object stored in space.
            Required to have ``{question}`` and ``{reference_documents}`` input variables when used with default python function, defaults to None
        :type prompt_id: str, optional

        :param indexing_function: custom python function generator containing document indexing, defaults to None
        :type indexing_function: Callable, optional

        :param inference_function: custom python function generator containing RAG logic, defaults to None
        :type inference_function: Callable, optional

        :param inference_service: custom AI-Service containing RAG logic, defaults to None
        :type inference_service: Callable, optional

        :param indexing_function_params: optional parameters passed to the ``indexing_function``, defaults to None
        :type indexing_function_params: dict, optional

        :param inference_function_params: optional parameters passed to the ``inference_function``, defaults to None
        :type inference_function_params: dict, optional

        :param store_params: properties used for storing function in the repository, to see available meta names use: ``client.repository.FunctionMetaNames.show()``, defaults to None
        :type store_params: dict, optional

        :param retriever: initialized retriever of type :class:`BaseRetriever <ibm_watsonx_ai.foundation_models.extensions.rag.retriever.base_retriever.BaseRetriever>` object, defaults to None
        :type retriever: BaseRetriever, optional

        :param vector_store: initialized :class:`VectorStore <ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.vector_store.VectorStore>` object, defaults to None
        :type vector_store: VectorStore, optional

        :param chunker: initialized chunker of type :class:`LangChainChunker <ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker.LangChainChunker>` object, defaults to None
        :type chunker: LangChainChunker, optional

        :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for approximating the token count, defaults to 1.5
        :type word_to_token_ratio: float, optional

        :param input_data_references: a list of DataConnection instances from which the knowledge base will be recreated inside the deployed default AI service with `chroma` type vector store, defaults to None
        :type input_data_references: list[DataConnection] | None, optional

        .. note::
            For ``inference_function`` to be populated with parameters passed at initialization the function's signature must have a default parameter called ``params`` as its last parameter.

            .. code-block:: python

                def custom_inference_function(custom_arg='value', params=None):
                    def score(payload):
                        return payload
                    return score


        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai import Credentials, APIClient
            from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern

            def custom_inference_service(context):
                task_token = context.generate_token()

                def generate(context):
                    return {"body": context.get_json()}
                return generate


            api_client = APIClient(
                credentials=Credentials(
                    api_key=IAM_API_KEY, url="https://us-south.ml.cloud.ibm.com"
                )
            )
            pattern = RAGPattern(
                space_id="<ID of the space>",
                inference_service=custom_inference_service,
                api_client=api_client
            )

            pattern.inference_service.deploy(name="<deployment name>")

            # inference deployment
            api_client.deployments.run_ai_service(pattern.deployment_id, payload)

        .. code-block:: python

            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.helpers import DataConnection
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern, VectorStore
            from ibm_watsonx_ai.foundation_models.extensions.rag.chunker import LangChainChunker

            chroma_vector_store = VectorStore(..., datasource_type='chroma')
            model = ModelInference(...)
            chunker = LangChainChunker(...)

            pattern = RAGPattern(
                space_id="<ID of the space>",
                vector_store=chroma_vector_store,
                prompt_id="<ID of the prompt template>",
                model=model,
                credentials=Credentials(
                            api_key = IAM_API_KEY,
                            url = "https://us-south.ml.cloud.ibm.com"),
                chunker=chunker,
                input_data_references=[Dataconnection(data_asset_id="<id to data asset>")]
            )

            pattern.inference_service.pretty_print() # inspect autogenerated inference service body

        .. code-block:: python

            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern

            def custom_inference_function(custom_arg='value', params=None):
                def score(payload):
                    return payload
                return score

            pattern = RAGPattern(
                space_id="<ID of the space>",
                inference_function=custom_inference_function,
                credentials=Credentials(
                                api_key = IAM_API_KEY,
                                url = "https://us-south.ml.cloud.ibm.com")
            )

        .. code-block:: python

            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern, VectorStore

            vector_store = VectorStore(...)
            model = ModelInference(...)

            pattern = RAGPattern(
                space_id="<ID of the space>",
                vector_store=vector_store,
                prompt_id="<ID of the prompt template>",
                model=model,
                credentials=Credentials(
                            api_key = IAM_API_KEY,
                            url = "https://us-south.ml.cloud.ibm.com")
            )

        """
        WMLResource._validate_type(
            word_to_token_ratio, "word_to_token_ratio", [int, float], False, True
        )
        self.space_id = space_id
        self.project_id = project_id
        self.model = model
        self.prompt_id = prompt_id
        self.inference_function_params = inference_function_params or {}
        self.indexing_function_params = indexing_function_params or {}
        self.store_params = store_params
        self.retriever = retriever
        self.vector_store = vector_store
        self.chunker = chunker

        if word_to_token_ratio <= 0:
            raise ValueError(
                "The value of 'word_to_token_ratio' must be strictly positive, exceeding 0."
            )
        else:
            self.word_to_token_ratio = word_to_token_ratio
        self.kwargs = kwargs

        self.prompt_template_text = None
        self.context_template_text = None

        self.default_sw_spec = "runtime-24.1-py3.11"

        self._validate_kwargs()

        if api_client is not None:
            self._credentials = api_client.credentials
            self._client = api_client
        elif credentials is not None:
            if isinstance(credentials, dict):
                credentials = Credentials.from_dict(credentials)
            self._credentials = credentials
            self._client = APIClient(self._credentials)
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if self.space_id is not None:
            self.project_id = None
            if self.space_id != self._client.default_space_id:
                self._client.set.default_space(self.space_id)
        elif self.project_id:
            self.space_id = None
            if self.project_id != self._client.default_project_id:
                self._client.set.default_project(self.project_id)
        else:
            self.space_id = self._client.default_space_id
            self.project_id = self._client.default_project_id
            if self.space_id is None and self.project_id is None:
                raise InvalidMultipleArguments(
                    params_names_list=["space_id", "project_id"],
                    reason="None of the arguments were provided or set in api_client/credentials.",
                )

        if inference_function is None and inference_service is None:
            inference_custom_asset = None
            if not vector_store and not retriever:
                raise InvalidMultipleArguments(
                    params_names_list=["vector_store", "retriever"],
                    reason="None of the arguments were provided.",
                )

            if not prompt_id and not kwargs.get("prompt_template_text"):
                raise MissingValue(
                    value_name="prompt_id",
                    reason="Prompt ID must be provided when python function is not provided.",
                )

            if not model:
                raise MissingValue(
                    value_name="model",
                    reason="ModelInference object must be provided when python function is not provided.",
                )
        else:
            if inference_service is not None and inference_function is not None:
                inference_custom_asset = "function_service"
            elif inference_service is not None:
                inference_custom_asset = "service"
            else:
                inference_custom_asset = "function"

        if vector_store and not retriever:
            self.retriever = Retriever.from_vector_store(vector_store=vector_store)
        elif retriever:
            self.vector_store = retriever.vector_store  # type: ignore[assignment]

        if prompt_id:
            self.prompt_template_text = self._load_prompt_text(prompt_id)
        elif prompt_template_text := kwargs.get("prompt_template_text"):
            self._validate_template_text(
                prompt_template_text,
                [self.QUESTION_PLACEHOLDER, self.REFERENCE_DOCUMENTS_PLACEHOLDER],
            )
            self.prompt_template_text = prompt_template_text
            if context_template_text := kwargs.get("context_template_text"):
                self._validate_template_text(
                    context_template_text, [self.DOCUMENT_PLACEHOLDER]
                )
                self.context_template_text = context_template_text

        self._input_data_references = input_data_references

        if self._input_data_references is not None and (
            inference_service is not None
            or getattr(self.vector_store, "_datasource_type", None) != "chroma"
        ):
            raise WMLClientError(
                "Param `input_data_references` only supported for default generated inference service and Chroma type VectorStore"
            )
        if self._input_data_references and self.chunker is None:
            raise WMLClientError(
                "`chunker` param is required when `input_data_references` provided"
            )

        self.indexing_function: RAGPatternFunction | None
        self.inference_function: RAGPatternFunction | None
        self.inference_service: RAGPatternService | None

        self.deployment_function: RAGPatternFunction | None = None

        self._allow_store = self.vector_store is None or (
            (self.vector_store._datasource_type != "chroma")
            or self._input_data_references is not None
        )

        indexing_function = indexing_function or (
            default_indexing_function if self.vector_store and self.chunker else None
        )
        if indexing_function:
            self.indexing_function = RAGPatternFunction(
                api_client=self._client,
                function=indexing_function,
                default_params=self._default_indexing_function_params(),
                store_params=self.store_params,
                _allow_store=self._allow_store,
            )
            if auto_store:
                self.indexing_function._store_component()
        else:
            self.indexing_function = None

        inference_function = inference_function or default_inference_function
        if inference_custom_asset is None or "function" in inference_custom_asset:
            self.inference_function = RAGPatternFunction(
                api_client=self._client,
                function=inference_function,
                default_params=self._default_inference_function_params(),
                store_params=self.store_params,
                cached=False,
                _allow_store=self._allow_store,
            )
            if auto_store:
                self.inference_function._store_component()
        else:
            self.inference_function = None

        store_params_ai_service: dict | None

        if inference_service is None:
            inference_service = (
                default_inference_service_chroma
                if self._input_data_references is not None
                else default_inference_service
            )
            store_params_ai_service = self.store_params or {}
            current_dir = Path(__file__).parent
            if (
                self._client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION
                not in store_params_ai_service
            ):
                with (
                    current_dir / "_default_inference_service_schema" / "request.json"
                ).open("r", encoding="utf-8") as file:
                    request_schema = json.load(file)

                store_params_ai_service.update(
                    {
                        self._client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema
                    }
                )

            if (
                self._client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION
                not in store_params_ai_service
            ):
                with (
                    current_dir / "_default_inference_service_schema" / "response.json"
                ).open("r", encoding="utf-8") as file:
                    response_schema = json.load(file)

                store_params_ai_service.update(
                    {
                        self._client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema
                    }
                )

        else:
            store_params_ai_service = self.store_params

        if inference_custom_asset is None or "service" in inference_custom_asset:
            if (
                inference_service is default_inference_service
                or inference_service is default_inference_service_chroma
            ):
                default_params = self._default_inference_service_params()
            else:
                # No default params for custom inference service
                default_params = {}

            self.inference_service = RAGPatternService(
                api_client=self._client,
                ai_service=inference_service,
                default_params=default_params,
                store_params=store_params_ai_service,
                _allow_store=self._allow_store,
            )
            if auto_store:
                self.inference_service._store_component()
        else:
            self.inference_service = None

    def deploy(
        self,
        name: str,
        space_id: str | None = None,
        store_params: dict | None = None,
        deploy_params: dict | None = None,
    ) -> dict:
        """Store and deploy ``inference_function`` to the space.

        .. deprecated:: 1.2.0
               `RAGPattern.deploy(...)` method is deprecated, please use "RAGPattern.inference_function.deploy(...)" instead

        :param name: Name for the stored function object as well as the deployed function. Can be overwritten by ``store_params`` and ``deploy_params``.
        :type name: str

        :param space_id: ID of the space to deploy ``inference_function`` to. Must be provided if ``space_id`` was not set at initialization.
        :type space_id: str, optional

        :param store_params: properties used for storing function in the repository, to see available meta names use: ``client.repository.FunctionMetaNames.show()``, defaults to None
        :type store_params: dict, optional

        :param deploy_params: properties used for deploying function to the space, to see available meta names use: ``client.deployments.ConfigurationMetaNames.show()``, defaults to None
        :type deploy_params: dict, optional

        :raises InvalidValue: If `inference_function` is not specified when initializing RAGPattern with custom inference objects

        :return: details of the deployed python function
        :rtype: dict

        **Example:**

        .. code-block:: python

            pattern.deploy(name="Example deployment name")

        .. code-block:: python

            deployment_details = pattern.deploy(
                name="Example deployment name",
                store_params={"software_spec_id": "<ID of the custom sw spec>"},
                deploy_params={"description": "Optional deployed function description"}
            )

        """

        deploy_method_deprecated_warning = "`deploy` method is deprecated, please use `inference_function.deploy(...)` instead"
        warn(deploy_method_deprecated_warning, category=DeprecationWarning)
        simplefilter("ignore", category=DeprecationWarning)

        if not space_id and not self.space_id:
            raise MissingValue(
                value_name="space_id",
                reason="Deployment space ID must be provided to deploy RAGPattern's inference function.",
            )

        if self.inference_function is not None:
            self.deployment_function = RAGPatternFunction(
                api_client=self._client,
                function=self.inference_function.function,
                default_params=self._default_inference_function_params(),
                store_params=self.store_params,
                _allow_store=self._allow_store,
            )

            if space_id and space_id != self.space_id:
                self.space_id = space_id
                self.project_id = None
                self._client.set.default_space(space_id)

            if (
                not (
                    self.inference_function.function_id
                    and self.inference_function.context == Context.SPACE
                )
                or store_params != self.store_params
            ):
                self.deployment_function._store_component(store_params)

            if deploy_params:
                deploy_params.update(
                    {self._client.repository.FunctionMetaNames.NAME: name}
                )
            else:
                deploy_params = {self._client.repository.FunctionMetaNames.NAME: name}

            return self.deployment_function._deploy_asset(
                deploy_params,
            )
        else:
            raise InvalidValue(
                "inference_function",
                reason=(
                    "No inference function provided. "
                    "Please note that `RAGPattern.deploy` method can be used only when `inference_function` is provided."
                ),
            )

    def query(self, payload: dict) -> dict:
        """Query the python function locally, without deploying.

        .. deprecated:: 1.2.0
               `RAGPattern.query(...)` method is deprecated, please use "RAGPattern.inference_function(...)" instead

        :param payload: payload for the scoring function
        :type payload: dict

        :raises InvalidValue: If `inference_function` is not specified when initializing RAGPattern with custom inference objects

        :return: result of the scoring function
        :rtype: dict

        **Example:**

        .. code-block:: python

            payload = {
                client.deployments.ScoringMetaNames.INPUT_DATA: [
                    {
                        "values": ["question 1", "question 2"],
                    }
                ]
            }
            result = pattern.query(payload)

        """

        simplefilter("default", category=DeprecationWarning)
        query_method_deprecated_warning = (
            "`query` method is deprecated, please use `inference_function(...)` instead"
        )
        warn(query_method_deprecated_warning, category=DeprecationWarning)
        simplefilter("ignore", category=DeprecationWarning)

        input_data = payload[self._client.deployments.ScoringMetaNames.INPUT_DATA]
        if not "access_token" in input_data[0]:
            input_data[0]["access_token"] = self._client.token

        if self.inference_function is not None:
            return self.inference_function()(payload)
        else:
            raise InvalidValue(
                "inference_function",
                reason=(
                    "No inference function provided. "
                    "Please note that `RAGPattern.query` method can be used only when `inference_function` is provided"
                ),
            )

    def delete(self, delete_stored_function: bool = True) -> None:
        """Delete stored functions object and/or deployed function from space.

        .. deprecated:: 1.2.0
               `RAGPattern.delete(...)` method is deprecated, please use "RAGPattern.inference_function.delete(...)" instead

        :param delete_stored_function: whether to delete stored function object from the repository, defaults to True
        :type delete_stored_function: bool, optional
        """
        delete_method_deprecated_warning = (
            "`delete` method is deprecated. "
            "Instead, please use `api_client.deployments.delete(deployment_id)` and `api_client.repository.delete(asset_id)` "
            "to delete the deployment and asset in the repository, respectively."
        )
        warn(delete_method_deprecated_warning, category=DeprecationWarning)

        if self.deployment_function:
            self._delete_function(self.deployment_function, delete_stored_function)

        if self.indexing_function:
            self._delete_function(self.indexing_function, delete_stored_function)

        if self.inference_function:
            self._delete_function(self.inference_function, delete_stored_function)

    def _validate_kwargs(self) -> None:
        """Check if all passed keyword arguments are supported.

        :raises InvalidValue: if any keyword argument is not supported
        """
        SUPPORTED_KWARGS = [
            "prompt_template_text",
            "context_template_text",
            "default_max_sequence_length",
        ]

        for kwarg in self.kwargs.keys():
            if kwarg not in SUPPORTED_KWARGS:
                raise InvalidValue(
                    kwarg,
                    reason=f"{kwarg} is not supported as a keyword argument. Supported kwargs: {SUPPORTED_KWARGS}",
                )

    def _validate_template_text(
        self, template_text: str, required_input_variables: list[str]
    ) -> None:
        """Check if template text has required input variables."

        :param template_text: template as text with placeholders
        :type template_text: str

        :param required_input_variables: input variables' names to check for
        :type required_input_variables: list[str]

        :raises ValidationError: if any required input variable missing
        """
        for key in required_input_variables:
            if key not in template_text:
                raise ValidationError(key)

    def _load_prompt_text(self, prompt_id: str) -> str:
        """Load prompt as string and validate input variables.
        ``RAGPattern.QUESTION_PLACEHOLDER`` and ``RAGPattern.REFERENCE_DOCUMENTS_PLACEHOLDER`` are expected by the default inference function.

        :param prompt_id: ID of :class:`PromptTemplate <ibm_watsonx_ai.foundation_models.prompts.prompt_template.PromptTemplate>` stored in space
        :type prompt_id: str

        :return: prompt with placeholders as string
        :rtype: str
        """
        prompt_mgr = PromptTemplateManager(api_client=self._client)
        prompt_text = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.STRING)

        required_input_variables = [
            self.QUESTION_PLACEHOLDER,
            self.REFERENCE_DOCUMENTS_PLACEHOLDER,
        ]
        self._validate_template_text(prompt_text, required_input_variables)

        return prompt_text

    def _swap_apikey_for_token(self, credentials: dict) -> dict:
        """Remove api_key form credentials and add token.
        Used primarily to prevent api_key from displaying in stored function code preview.

        :param credentials: credentials to modify
        :type credentials: dict

        :return: credentials with api_key removed and token added
        :rtype: dict
        """
        result = credentials.copy()
        result.pop("api_key", None)
        result["token"] = self._client.token

        return result

    def _default_inference_function_params(self) -> dict:
        """Generates default param dictionary for the inference function.

        :return: dictionary containing nescessary parameters for inference function ``params`` parameter
        :rtype: dict
        """
        default_params = {
            "credentials": self._swap_apikey_for_token(self._credentials.to_dict()),
            "space_id": self.space_id,
            "project_id": self.project_id,
            "retriever": self.retriever.to_dict() if self.retriever else None,
            "vector_store": self.vector_store.to_dict() if self.vector_store else None,
            "prompt_template_text": self.prompt_template_text,
            "context_template_text": self.context_template_text,
            "model": self.model.get_identifying_params() if self.model else None,
            "inference_function_params": self.inference_function_params,
            "default_max_sequence_length": self.kwargs.get(
                "default_max_sequence_length"
            ),
        }

        from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.prompt_builder import (
            WORD_TO_TOKEN_RATIO,
        )

        if self.word_to_token_ratio != WORD_TO_TOKEN_RATIO:
            default_params |= {"word_to_token_ratio": self.word_to_token_ratio}

        return default_params

    def _default_inference_service_params(self) -> dict:
        """Generates default param dictionary for the inference function.

        :return: dictionary containing necessary parameters for inference function ``params`` parameter
        :rtype: dict
        """

        default_params = {
            "credentials": self._swap_apikey_for_token(self._credentials.to_dict()),
            "space_id": self.space_id,
            "project_id": self.project_id,
            "retriever": self.retriever.to_dict() if self.retriever else None,
            "vector_store": self.vector_store.to_dict() if self.vector_store else None,
            "prompt_template_text": self.prompt_template_text,
            "context_template_text": self.context_template_text,
            "model": self.model.get_identifying_params() if self.model else None,
            "default_max_sequence_length": self.kwargs.get(
                "default_max_sequence_length"
            ),
        }

        from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.prompt_builder import (
            WORD_TO_TOKEN_RATIO,
        )

        if self.word_to_token_ratio != WORD_TO_TOKEN_RATIO:
            default_params |= {"word_to_token_ratio": self.word_to_token_ratio}

        if self._input_data_references is not None and self.chunker is not None:
            default_params |= {
                "input_data_references": [
                    data_connection.to_dict()
                    for data_connection in self._input_data_references
                ]
            }
            default_params |= {"chunker": self.chunker.to_dict()}

        return default_params

    def _default_indexing_function_params(self) -> dict:
        """Generates default param dictionary for the indexing function.

        :return: dictionary containing necessary parameters for indexing function ``params`` parameter
        :rtype: dict
        """

        return {
            "credentials": self._swap_apikey_for_token(self._credentials.to_dict()),
            "space_id": self.space_id,
            "project_id": self.project_id,
            "vector_store": self.vector_store.to_dict() if self.vector_store else None,
            "chunker": self.chunker.to_dict() if self.chunker else None,
            "indexing_params": self.indexing_function_params,
        }

    @staticmethod
    def create_custom_software_spec(client: APIClient) -> dict:
        """Create a custom software specification for RAGPattern functions deployment.

        :return: details of the custom software specification
        :rtype: dict
        """
        BASE_SW_SPEC_NAME = "runtime-24.1-py3.11"
        SW_SPEC_NAME = "rag_24.1-py3.11"
        PKG_EXTN_NAME = "rag_pattern-py3.11"
        CONFIG_PATH = "config.yaml"
        CONFIG_TYPE = "conda_yml"
        CONFIG_CONTENT = f"""
        name: python311
        channels:
          - empty
        dependencies:
          - pip:
            - ibm-watsonx-ai[rag]
        prefix: /opt/anaconda3/envs/python311
        """

        try:
            sw_spec_id = client.software_specifications.get_id_by_name(SW_SPEC_NAME)
            return client.software_specifications.get_details(sw_spec_id)
        except:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                f.write(CONFIG_CONTENT)

            try:
                pkg_extn_meta_props = {
                    client.package_extensions.ConfigurationMetaNames.NAME: PKG_EXTN_NAME,
                    client.package_extensions.ConfigurationMetaNames.TYPE: CONFIG_TYPE,
                }

                pkg_extn_details = client.package_extensions.store(
                    meta_props=pkg_extn_meta_props, file_path=CONFIG_PATH
                )
                pkg_extn_uid = client.package_extensions.get_id(pkg_extn_details)

                sw_spec_meta_props = {
                    client.software_specifications.ConfigurationMetaNames.NAME: SW_SPEC_NAME,
                    client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {
                        "guid": client.software_specifications.get_id_by_name(
                            BASE_SW_SPEC_NAME
                        )
                    },
                }

                sw_spec_details = client.software_specifications.store(
                    meta_props=sw_spec_meta_props
                )
                sw_spec_id = client.software_specifications.get_id(sw_spec_details)

                client.software_specifications.add_package_extension(
                    sw_spec_id, pkg_extn_uid
                )
            finally:
                os.remove(CONFIG_PATH)

            return client.software_specifications.get_details(sw_spec_id)

    def _delete_function(
        self, pattern_function: RAGPatternFunction, delete_stored: bool = True
    ) -> None:
        """Delete stored function object and/or deployed function from space.

        :param pattern_function: function to delete
        :type pattern_function: RAGPatternFunction

        :param delete_stored: whether to delete stored function from repository as well, defaults to True
        :type delete_stored: bool, optional

        :raises WMLClientError: if deleting deployment or stored function fails
        """
        if pattern_function.deployment_id:
            try:
                self._client.set.default_space(pattern_function.context_id)
                self._client.deployments.delete(pattern_function.deployment_id)
                pattern_function.deployment_id = None
                pattern_function.context = None
                pattern_function.context_id = None
            except WMLClientError as e:
                raise WMLClientError(
                    f"Could not delete deployment with ID: '{pattern_function.deployment_id}'"
                ) from e

        if delete_stored and pattern_function.function_id:
            try:
                if pattern_function.context == Context.PROJECT:
                    self._client.set.default_project(pattern_function.context_id)  # type: ignore[arg-type]
                elif pattern_function.context == Context.SPACE:
                    self._client.set.default_space(pattern_function.context_id)  # type: ignore[arg-type]
                self._client.repository.delete(pattern_function.function_id)
                pattern_function.function_id = None
                pattern_function.context = None
                pattern_function.context_id = None
            except WMLClientError as e:
                raise WMLClientError(
                    f"Could not delete function with ID: '{pattern_function.function_id}' in {pattern_function.context}: '{pattern_function.context_id}'"
                ) from e
