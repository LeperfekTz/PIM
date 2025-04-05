#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import inspect
import re
import ast
import uuid
import gzip
from enum import Enum
from functools import cache
from typing import Any, Callable, cast
import tempfile
from abc import ABC, abstractmethod

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.ai_services import AIServices
from ibm_watsonx_ai.deployments import RuntimeContext
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.utils.utils import is_lib_installed

from ibm_watsonx_ai.foundation_models.extensions.rag.utils.utils import (
    FunctionTransformer,
    FunctionVisitor,
    _get_components_replace_data,
)

from ibm_watsonx_ai.foundation_models.utils.utils import _copy_function
from ibm_watsonx_ai.wml_client_error import (
    MissingValue,
    WMLClientError,
)


class Context(str, Enum):
    PROJECT = "project"
    SPACE = "space"

    def __str__(self) -> str:
        return self.value


class BaseRAGPatternService(ABC):
    """Abstract Base Class for RAGPattern function/AI service.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param function: python function generator used as RAG Pattern
    :type function: Callable

    :param default_params: default parameters that will be passed to the provided function
    :type default_params: dict

    :param store_params: parameters used when storing asset, defaults to None
    :type store_params: dict | None, optional

    :param cached: if True function calls will be cached, defaults to False
    :type cached: bool, optional

    :param _allow_store: determines whether to allow storing the asset in WML repository, defaults to True
    :type _allow_store: bool, optional
    """

    DEFAULT_SW_SPEC = "runtime-24.1-py3.11"

    def __init__(
        self,
        api_client: APIClient,
        function: Callable,
        default_params: dict,
        store_params: dict | None = None,
        cached: bool = False,
        _allow_store: bool = True,
    ) -> None:

        self.api_client = api_client
        self.store_params = store_params
        self._allow_store = _allow_store
        self.cached = cached
        self.context: Context | None = None
        self.context_id: str | None = None
        self._asset_id: str | None = None
        self.deployment_id: str | None = None

        self._function = self._populate_default_params(
            _copy_function(function), default_params
        )

        self.cached_function: Callable | None

    @abstractmethod
    def _populate_default_params(
        self, function: Callable, default_params: dict
    ) -> Callable:
        """Populate default params in function object"""
        raise NotImplementedError

    @abstractmethod
    def _store(self, meta_props: dict) -> dict:
        """Store function/AI service as WML asset"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.cached:
            # If self.cached is True then self.cached_function has type Callable
            return self.cached_function(*args, **kwargs)  # type: ignore[misc]
        else:
            return self._function(*args, **kwargs)

    def _deploy_asset(self, deploy_params: dict | None = None) -> dict:
        """Deploy RAGPattern inference/indexing objects to the current client WML space.

        :param deploy_params: properties used for deploying the function object, to see available meta names use: ``client.deployments.ConfigurationMetaNames.show()``, defaults to None
        :type deploy_params: dict | None, optional

        :raises ValueError: if function is not stored

        :return: details of the deployed function
        :rtype: dict
        """

        self._validate_before_deployment()

        if deploy_params and deploy_params.get(
            self.api_client.repository.FunctionMetaNames.NAME
        ):
            name = deploy_params.get(self.api_client.repository.FunctionMetaNames.NAME)
        else:
            name = self._function.__name__

        meta_props = {  # type: ignore[var-annotated]
            self.api_client.deployments.ConfigurationMetaNames.NAME: name,
            self.api_client.deployments.ConfigurationMetaNames.ONLINE: {},
        }

        if deploy_params:
            meta_props.update(deploy_params)

        deployment_details = self.api_client.deployments.create(
            artifact_id=self._asset_id, meta_props=meta_props
        )

        self.deployment_id = self.api_client.deployments.get_id(deployment_details)

        return deployment_details

    def deploy(
        self,
        name: str,
        space_id: str | None = None,
        store_params: dict | None = None,
        deploy_params: dict | None = None,
    ) -> dict:
        """Store and deploy RAGPattern asset to the space."""

        if space_id and space_id != self.api_client.default_space_id:
            self.api_client.set.default_space(space_id)

        if not (self._asset_id and self.context == Context.SPACE):
            self._store_component(store_params=store_params)

        if deploy_params:
            deploy_params.update(
                {self.api_client.repository.FunctionMetaNames.NAME: name}
            )
        else:
            deploy_params = {self.api_client.repository.FunctionMetaNames.NAME: name}

        return self._deploy_asset(
            deploy_params,
        )

    def _store_component(
        self,
        store_params: dict | None = None,
    ) -> dict:
        """Store the ``pattern_function`` contents in the repository.

        :param store_params: properties used for storing the function/service object, defaults to None
        :type store_params: dict | None, optional

        :return: details of the stored function/service
        :rtype: dict
        """

        if not self._allow_store:
            raise WMLClientError(
                "Inference function with Chroma vector store can't be used without indexing function."
            )

        # Both function and AI service API support fields NAME, SOFTWARE_SPEC_ID
        store_params = store_params or self.store_params
        if store_params and store_params.get(
            self.api_client.repository.FunctionMetaNames.NAME
        ):
            name = store_params.get(self.api_client.repository.FunctionMetaNames.NAME)
        else:
            name = self._function.__name__

        if (
            not store_params
            or self.api_client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID
            not in store_params
        ):
            software_spec_id = self.api_client.software_specifications.get_id_by_name(
                self.DEFAULT_SW_SPEC
            )
        else:
            software_spec_id = store_params[
                self.api_client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID
            ]

        meta_props = {
            self.api_client.repository.FunctionMetaNames.NAME: name,
            self.api_client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
        }

        if store_params:
            meta_props.update(store_params)

        function_details = self._store(meta_props=meta_props)

        if project_id := self.api_client.default_project_id:
            self.context = Context.PROJECT
            self.context_id = project_id
        elif space_id := self.api_client.default_space_id:
            self.context = Context.SPACE
            self.context_id = space_id

        return function_details

    def _validate_before_deployment(self) -> None:
        """Validate if creating deployment is possible"""
        pass


class RAGPatternService(BaseRAGPatternService):
    """Initialize ``RAGPatternService`` object.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param ai_service: AI service python function generator used as RAG Pattern
    :type ai_service: Callable

    :param default_params: default parameters that will be passed to the provided function
    :type default_params: dict

    :param store_params: parameters used when storing asset, defaults to None
    :type store_params: dict | None, optional

    :param cached: if True AI service function calls will be cached, defaults to False
    :type cached: bool, optional

    :param _allow_store: determines whether to allow storing the AI service asset in WML repository, defaults to True
    :type _allow_store: bool, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        ai_service: Callable,
        default_params: dict,
        store_params: dict | None = None,
        cached: bool = False,
        _allow_store: bool = True,
    ) -> None:

        self._code: str | None = None

        super().__init__(
            api_client=api_client,
            function=ai_service,
            default_params=default_params,
            store_params=store_params,
            cached=cached,
            _allow_store=_allow_store,
        )

        if self.cached:
            self.cached_function = cache(self.ai_service)
            self.cached_function(RuntimeContext(api_client=api_client))

    @property
    def ai_service(self) -> Callable:
        return self._function

    @property
    def ai_service_id(self) -> str | None:
        return self._asset_id

    @ai_service_id.setter
    def ai_service_id(self, value: str) -> None:
        self._asset_id = value

    def _populate_default_params(
        self, function: Callable[..., Any], default_params: dict
    ) -> Callable[..., Any]:
        """Populate AI service params by updating and overwriting.
        Method populates in template `inference_service` the placeholders that starts with `REPLACE_THIS_CODE_WITH_` using default_params.

        :param function: AI service function which placeholders should be populated
        :type function: Callable

        :return: function with params populated if signature matches
        :rtype: Callable
        """
        if default_params:
            args_spec = inspect.getfullargspec(function)
            defaults: tuple | list = args_spec.defaults or []
            args = args_spec.args or []

            if args and args[-1] == "vector_store_settings":
                vectorstore_params = default_params.get("vector_store") or {}
                vector_store_settings = {
                    "connection_id": vectorstore_params.get("connection_id"),
                    "index_name": vectorstore_params.get("index_name"),
                }
                if space_id := default_params.get("space_id"):
                    vector_store_settings["space_id"] = space_id
                else:
                    vector_store_settings["project_id"] = default_params.get(
                        "project_id"
                    )
                function.__defaults__ = (
                    *defaults[:-1],
                    vector_store_settings,
                )

            source = AIServices._populate_default_params(function)

            tree = ast.parse(source)
            visitor = FunctionVisitor()
            visitor.visit(tree)

            func_def = visitor.function

            # Credentials params
            credentials_params = _get_components_replace_data(
                default_params.get("credentials", {}),
                Credentials.__init__,
                suffix="credentials",
            )

            # APIClient params, for Chroma scenario
            api_client_params = _get_components_replace_data(
                {
                    "space_id": default_params.get("space_id"),
                    "project_id": default_params.get("project_id"),
                },
                APIClient.__init__,
                suffix="api_client",
            )

            # ModelInference Params
            model_init_params = default_params.get("model", {}) or {}
            new_model_init_params = _get_components_replace_data(
                model_init_params, ModelInference.__init__, "model"
            )

            from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
                LangChainChunker,
            )

            # LangChainChunker Params
            chunker_init_params = default_params.get("chunker", {}) or {}
            new_chunker_init_params = _get_components_replace_data(
                chunker_init_params, LangChainChunker.__init__, "langchain_chunker"
            )

            # VectorStore params
            vector_store_init_params: dict = default_params.get("vector_store", {})

            ## Remove credential, project/space id and verify fields from wx embeddings
            ## since they will be restored from APIClient instance
            if (
                vector_store_init_params is not None
                and "ibm_watsonx_ai.foundation_models.embeddings.embeddings"
                in (
                    embeddings_init_params := (
                        vector_store_init_params.get("embeddings", {}) or {}
                    )
                ).get("__module__", "")
            ):
                embeddings_init_params.pop("credentials", None)
                embeddings_init_params.pop("project_id", None)
                embeddings_init_params.pop("space_id", None)
                embeddings_init_params.pop("verify", None)

            replace_data = dict(
                # For AST Call node
                **credentials_params,
                **api_client_params,
                REPLACE_THIS_CODE_WITH_RETRIEVER={
                    "value": default_params.get("retriever"),
                    "replace": True,
                },
                **new_model_init_params,
                REPLACE_THIS_CODE_WITH_PROMPT_TEMPLATE_TEXT={
                    "value": default_params.get("prompt_template_text"),
                    "replace": True,
                },
                REPLACE_THIS_CODE_WITH_DEFAULT_MAX_SEQUENCE_LENGTH={
                    "value": default_params.get("default_max_sequence_length"),
                    "replace": default_params.get("default_max_sequence_length")
                    is not None,
                },
                REPLACE_THIS_CODE_WITH_CONTEXT_TEMPLATE_TEXT={
                    "value": default_params.get("context_template_text"),
                    "replace": True,
                },
                **new_chunker_init_params,
                # For AST Assign Node
                REPLACE_THIS_CODE_WITH_WORD_TO_TOKEN_RATIO=default_params.get(
                    "word_to_token_ratio"
                ),
                REPLACE_THIS_CODE_WITH_INPUT_DATA_REFERENCES=default_params.get(
                    "input_data_references"
                ),
                REPLACE_THIS_CODE_WITH_VECTOR_STORE_ASSIGN=vector_store_init_params,
                REPLACE_THIS_CODE_WITH_VECTOR_STORE_CALL={
                    "value": vector_store_init_params,
                    "replace": True,
                },
            )

            replacer = FunctionTransformer(
                cast(ast.FunctionDef, func_def), **replace_data
            )
            new_tree = replacer.visit(tree)
            ast.fix_missing_locations(new_tree)

            self._code = ast.unparse(new_tree)

            if is_lib_installed("black"):
                import black

                # If default values in inference service make the line too long
                # use shorter signature
                code_lines = self._code.split("\n")
                tmp_signature = "def default_service(context):"
                formatted_code = black.format_str(
                    tmp_signature + "\n" + "\n".join(code_lines[1:]),
                    mode=black.FileMode(),
                ).rstrip()

                self._code = (
                    code_lines[0] + "\n" + "\n".join(formatted_code.split("\n")[1:])
                )

            with tempfile.NamedTemporaryFile(
                suffix="inference_service.py", delete=True
            ) as tmp_file:
                tmp_file.write(self._code.encode())

                compiled_code = compile(new_tree, filename=tmp_file.name, mode="exec")
                namespace: dict = {}
                exec(compiled_code, namespace)
                function = namespace[function.__name__]
        return function

    def _store(self, meta_props: dict) -> dict:
        """Store the AI service function content in the repository.

        :param store_params: properties used for storing the service object, defaults to None
        :type store_params: dict | None, optional

        :return: details of the stored AI service
        :rtype: dict
        """
        if self._code is not None:

            tmp_uid = "tmp_ai_service_python_function_code_{}.py.gz".format(
                str(uuid.uuid4()).replace("-", "_")
            )
            with tempfile.NamedTemporaryFile(
                suffix=tmp_uid, delete=True
            ) as archive_name:
                with gzip.GzipFile(mode="wb", fileobj=archive_name) as gzip_file:
                    gzip_file.write(self._code.encode())
                archive_name.seek(0)
                function_details = self.api_client.repository.store_ai_service(
                    ai_service=archive_name.name, meta_props=meta_props
                )

        else:
            function_details = self.api_client.repository.store_ai_service(
                ai_service=self.ai_service, meta_props=meta_props
            )

        self.ai_service_id = self.api_client.repository.get_ai_service_id(
            function_details
        )

        return function_details

    def pretty_print(self, insert_to_cell: bool = False) -> None:
        """Print the AI service's source code to inspect or modify.

        :param insert_to_cell: whether to insert python service's source code to a new notebook cell, defaults to False
        :type insert_to_cell: bool, optional
        """

        if self._code is not None:
            code = self._code
        else:
            code = AIServices._populate_default_params(self.ai_service)

        if insert_to_cell:
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            comment = "# generated by RAGPatternService.pretty_print\n"
            ipython.set_next_input(comment + code, replace=False)
        else:
            print(code)

    def _validate_before_deployment(self) -> None:
        """Validation before creating deployment"""
        if not self._allow_store:
            raise WMLClientError(
                "Inference AI service with Chroma vector store can't be used without indexing function."
            )

        if self._asset_id is None:
            raise ValueError(
                "AI service was not stored. Either store the AI service or provide `ai_service_id`."
            )
        return super()._validate_before_deployment()

    def deploy(
        self,
        name: str,
        space_id: str | None = None,
        store_params: dict | None = None,
        deploy_params: dict | None = None,
    ) -> dict:
        """Store and deploy RAGPattern AI service to the space.

        :param name: Name for the stored AI service object as well as the deployed AI service. Can be overwritten by ``store_params`` and ``deploy_params``.
        :type name: str

        :param space_id: ID of the space to deploy AI service to. Must be provided if ``space_id`` was not set at initialization.
        :type space_id: str, optional

        :param store_params: properties used for storing AI service in the repository, to see available meta names use: ``client.repository.AIServiceMetaNames.show()``, defaults to None
        :type store_params: dict, optional

        :param deploy_params: properties used for deploying AI service to the space, to see available meta names use: ``client.deployments.ConfigurationMetaNames.show()``, defaults to None
        :type deploy_params: dict, optional

        :return: details of the deployed asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            pattern.inference_service.deploy(name="Example deployment name")

        .. code-block:: python

            deployment_details = pattern.inference_service.deploy(
                name="Example deployment name",
                store_params={"software_spec_id": "<ID of the custom sw spec>"},
                deploy_params={"description": "Optional deployed AI service description"}
            )

        To override vector store `connection_id`, `index_name` or set specific scope id (project_id/space_id) use following:
        .. code-block:: python

            deployment_details = pattern.inference_service.deploy(
                name="Example deployment name",
                store_params={"software_spec_id": "<ID of the custom sw spec>"},
                deploy_params={
                                "online": {
                                    "parameters": {
                                        "vector_store_settings": {
                                            "connection_id": "<connection_to_vector_store>",
                                            "index_name": "<index_name>",
                                            "project_id": "<project_id>",
                                        }
                                    }
                                }
                            }
            )

        """

        if not space_id and not self.api_client.default_space_id:

            raise MissingValue(
                value_name="space_id",
                reason=f"Deployment space ID must be provided to deploy RAGPattern's inference AI service.",
            )
        return super().deploy(
            name=name,
            space_id=space_id,
            store_params=store_params,
            deploy_params=deploy_params,
        )


class RAGPatternFunction(BaseRAGPatternService):
    """Initialize ``RAGPatternSFunction`` object.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param ai_service: python AI service generator used as RAG Pattern
    :type ai_service: Callable

    :param default_params: default parameters that will be passed to the provided function
    :type default_params: dict

    :param store_params: parameters used when storing asset, defaults to None
    :type store_params: dict | None, optional

    :param cached: if True function calls will be cached, defaults to False
    :type cached: bool, optional

    :param _allow_store: determines whether to allow storing the function asset in WML repository, defaults to True
    :type _allow_store: bool, optional
    """

    def __init__(
        self,
        function: Callable,
        default_params: dict,
        cached: bool = False,
        api_client: APIClient | None = None,
        store_params: dict | None = None,
        _allow_store: bool = True,
    ) -> None:

        # api_client is mandatory to store and deploy function asset
        WMLResource._validate_type(api_client, "api_client", APIClient, mandatory=True)
        api_client = cast(APIClient, api_client)

        super().__init__(
            api_client=api_client,
            function=function,
            default_params=default_params,
            store_params=store_params,
            cached=cached,
            _allow_store=_allow_store,
        )

        if cached:
            self.cached_function = cache(self.function)
            self.cached_function()

    @property
    def function(self) -> Callable:
        return self._function

    @property
    def function_id(self) -> str | None:
        return self._asset_id

    @function_id.setter
    def function_id(self, value: str) -> None:
        self._asset_id = value

    def _populate_default_params(
        self, function: Callable, default_params: dict
    ) -> Callable:
        """Populate function's default params by updating and overwriting.
        Default parameter named ``params`` is used to pass information that is used inside deployed function.
        Can be used both with default function template and custom function (if signature matches).

        :param function: function which default params should be populated
        :type function: Callable

        :return: function with params populated if signature matches
        :rtype: Callable
        """
        args_spec = inspect.getfullargspec(function)
        defaults: tuple | list = args_spec.defaults or []
        args = args_spec.args or []

        if len(args) > 0 and args[-1] == "params":
            if provided_deployable_params := defaults[-1]:
                default_params.update(provided_deployable_params)
            function.__defaults__ = (*defaults[:-1], default_params)

        return function

    def _store(self, meta_props: dict) -> dict:
        """Store the ``pattern_function`` contents in the repository.

        :param store_params: properties used for storing the function object, defaults to None
        :type store_params: dict | None, optional

        :return: details of the stored function
        :rtype: dict
        """
        function_details = self.api_client.repository.store_function(
            function=self.function, meta_props=meta_props
        )

        self.function_id = self.api_client.repository.get_function_id(function_details)

        return function_details

    def pretty_print(self, insert_to_cell: bool = False) -> None:
        """Print the python function's source code to inspect or modify.

        :param insert_to_cell: whether to insert python function's source code to a new notebook cell, defaults to False
        :type insert_to_cell: bool, optional
        """

        def hide_credentials(defaults: dict) -> dict:
            return {
                key: (hide_credentials(val) if isinstance(val, dict) else val)
                for key, val in defaults.items()
                if "credentials" not in key
            }

        code = inspect.getsource(self._function)
        args_spec = inspect.getfullargspec(self._function)

        defaults: tuple | list = args_spec.defaults or []
        args = args_spec.args or []

        args_pattern = ",".join([rf"\s*{arg}\s*=\s*(.+)\s*" for arg in args])

        pattern = rf"^def {self._function.__name__}\s*\({args_pattern}\)\s*:"

        res = re.match(pattern, code)

        for i in range(len(defaults) - 1, -1, -1):
            default = defaults[i]
            if isinstance(default, dict):
                default = hide_credentials(default)
            code = (
                code[: res.start(i + 1)] + default.__repr__() + code[res.end(i + 1) :]  # type: ignore[union-attr]
            )

        if insert_to_cell:
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            comment = "# generated by RAGPatternFunction.pretty_print\n# credentials have been redacted\n\n"
            ipython.set_next_input(comment + code, replace=False)
        else:
            print(code)

    def _validate_before_deployment(self) -> None:
        """Validation before creating deployment"""
        if not self._allow_store:
            raise WMLClientError(
                "Inference function with Chroma vector store can't be used without indexing function."
            )

        if self._asset_id is None:
            raise ValueError(
                "Function was not stored. Either store the function or provide `function_id`."
            )
        return super()._validate_before_deployment()

    def deploy(
        self,
        name: str,
        space_id: str | None = None,
        store_params: dict | None = None,
        deploy_params: dict | None = None,
    ) -> dict:
        """Store and deploy RAGPattern function to the space.

        :param name: Name for the stored function object as well as the deployed function. Can be overwritten by ``store_params`` and ``deploy_params``.
        :type name: str

        :param space_id: ID of the space to deploy function to. Must be provided if ``space_id`` was not set at initialization.
        :type space_id: str, optional

        :param store_params: properties used for storing function in the repository, to see available meta names use: ``client.repository.FunctionMetaNames.show()``, defaults to None
        :type store_params: dict, optional

        :param deploy_params: properties used for deploying function to the space, to see available meta names use: ``client.deployments.ConfigurationMetaNames.show()``, defaults to None
        :type deploy_params: dict, optional

        :return: details of the deployed function
        :rtype: dict

        **Example:**

        .. code-block:: python

            pattern.inference_function.deploy(name="Example deployment name")

        .. code-block:: python

            deployment_details = pattern.inference_function.deploy(
                name="Example deployment name",
                store_params={"software_spec_id": "<ID of the custom sw spec>"},
                deploy_params={"description": "Optional deployed function description"}
            )

        """

        if not space_id and not self.api_client.default_space_id:

            raise MissingValue(
                value_name="space_id",
                reason=f"Deployment space ID must be provided to deploy RAGPattern's inference {'service' if isinstance(self, RAGPatternService) else 'service'}.",
            )
        return super().deploy(
            name=name,
            space_id=space_id,
            store_params=store_params,
            deploy_params=deploy_params,
        )
