#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import time
from pandas import DataFrame
from typing import TYPE_CHECKING, Any, Literal

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.utils import (
    print_text_header_h1,
    StatusLogger,
    DisableWarningsLogger,
)
from ibm_watsonx_ai.utils.autoai.utils import run_id_required, is_ipython
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    UnsupportedOperation,
    MissingValue,
)
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.helpers.connections import DataConnection
from ibm_watsonx_ai.metanames import RAGOptimizerConfigurationMetaNames

if TYPE_CHECKING:
    from ibm_watsonx_ai.workspace import WorkSpace
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern import RAGPattern

__all__ = ["RAGEngine"]


class RAGEngine(WMLResource):
    """RAG Engine provides unified API to work with AutoAI RAG Pattern on WML.

    :param workspace: WorkSpace object with wml client and all needed parameters
    :type workspace: WorkSpace
    """

    def __init__(self, workspace: "WorkSpace"):
        self.workspace = workspace
        self._client = workspace.api_client
        WMLResource.__init__(self, __name__, self._client)

        self._current_run_id = None
        self._cached_details: dict | None = None
        self._training_metadata: dict = {}
        self._params: dict = {}
        self.ConfigurationMetaNames = RAGOptimizerConfigurationMetaNames()

        if not self._is_autoai_rag_endpoint_available():
            raise UnsupportedOperation(
                Messages.get_message(message_id="rag_optimizer_not_supported")
            )

    def run(
        self,
        *,
        input_data_references: list[DataConnection],
        test_data_references: list[DataConnection],
        results_reference: DataConnection,
        vector_store_references: list[DataConnection] | None = None,
        background_mode: bool = True,
    ) -> dict:
        """Engine for create an AutoAI RAG job that will find the best RAG pattern.

        :param input_data_references: Data storage connection details to inform where training data is stored
        :type input_data_references: list[DataConnection]

        :param test_data_references: A set of test data references
        :type test_data_references: list[DataConnection]

        :param results_reference: The training results
        :type results_reference: DataConnection

        :param vector_store_references: A set of vector store references
        :type vector_store_references: list[DataConnection], optional

        :param background_mode: Indicator if run() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: run details
        :rtype: dict

        """
        self._cached_details = (
            None  # attribute set to None for correct catching _cached_details
        )
        for input_conn in input_data_references:
            if self.workspace.api_client.project_type == "local_git_storage":
                input_conn.location.userfs = "true"  # type: ignore[union-attr]
            input_conn.set_client(self.workspace.api_client)

        for test_conn in test_data_references:
            if self.workspace.api_client.project_type == "local_git_storage":
                input_conn.location.userfs = "true"  # type: ignore[union-attr]
            test_conn.set_client(self.workspace.api_client)

        self._initialize_training_metadata(
            input_data_references=input_data_references,
            test_data_references=test_data_references,
            results_reference=results_reference,
            vector_store_references=vector_store_references,
        )

        url = self._client.service_instance._href_definitions.get_autoai_rag_href()

        response_train_post = requests.post(
            url=url,
            json=self._training_metadata,
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
        )

        run_details = self._handle_response(201, "training", response_train_post)

        self._current_run_id = run_details["metadata"]["id"]

        if background_mode:
            return self.get_run_details()
        else:
            print_text_header_h1("Running '{}'".format(self._current_run_id))

            def get_status(details: dict) -> str:
                try:
                    status = details["entity"]["status"]["state"]
                except KeyError:
                    # Valid case for CPD 5.1
                    status = details.get("entity", {}).get("state", "error")
                return status

            details = self.get_run_details()
            status = get_status(details)

            with StatusLogger(status) as status_logger:
                while status not in ["error", "completed", "canceled", "failed"]:
                    time.sleep(5)
                    details = self.get_run_details()
                    status = get_status(details)
                    status_logger.log_state(status)

            if "completed" in status:
                print(
                    "\nTraining of '{}' finished successfully.".format(
                        str(self._current_run_id)
                    )
                )
            else:
                error_msg = (
                    details.get("entity", {}).get("message", {}).get("text", None)
                )
                print(
                    "\nTraining of '{}' failed with status: '{}'. Error message: '{}'".format(
                        self._current_run_id, str(status), error_msg
                    )
                )

            self._logger.debug("Response({}): {}".format(status, run_details))
            return self.get_run_details()

    @run_id_required
    def cancel_run(self, hard_delete: bool = False) -> str:
        """Engine for cancels a RAG Optimizer run.

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancellation is successful
        :rtype: str

        """
        self._cached_details = (
            None  # attribute set to None to not use cached details after deleting
        )

        self._client._check_if_either_is_set()

        url = self._client.service_instance._href_definitions.get_autoai_rag_id_href(
            self._current_run_id
        )
        params = self._client._params()

        if hard_delete is True:
            params.update({"hard_delete": "true"})

        response_delete = requests.delete(
            url=url,
            params=params,
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204, "rag optimizer deletion", response_delete, False
        )

    def get_details(
        self,
        run_id: str | None = None,
        limit: int | None = None,
        get_all: bool = False,
    ) -> dict:
        """Engine for get details.

        :param run_id: ID of the fit/run
        :type run_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks, defaults to `False`
        :type get_all: bool, optional

        :return: RAGOptimizer details
        :rtype: dict

        """

        self._client._check_if_either_is_set()

        url = self._client.service_instance._href_definitions.get_autoai_rag_href()

        return self._get_artifact_details(
            base_url=url,
            id=run_id,
            limit=limit,
            resource_name="optimized patterns",
            _all=get_all,
        )

    @run_id_required
    def get_run_status(self) -> str:
        """Engine for check status/state of initialized RAGOptimizer run if ran in background mode.

        :return: run status details
        :rtype: str

        """

        details = self.get_run_details()

        if details is not None:
            try:
                with DisableWarningsLogger():
                    status = WMLResource._get_required_element_from_dict(
                        details, "details", ["entity", "status", "state"]
                    )
            except MissingValue:
                # Valid case for CPD 5.1
                status = WMLResource._get_required_element_from_dict(
                    details, "details", ["entity", "state"]
                )
            return status
        else:
            raise WMLClientError(
                "Getting trained model status failed. Unable to get model details for training_id: '{}'.".format(
                    self._current_run_id
                )
            )

    @run_id_required
    def get_run_details(self) -> dict:
        """Engine for get run details.

        :return: RAGOptimizer run details
        :rtype: dict

        """

        if self._cached_details is not None:
            return self._cached_details
        else:
            details = self.get_details(self._current_run_id)

            try:
                with DisableWarningsLogger():
                    status = WMLResource._get_required_element_from_dict(
                        details, "details", ["entity", "status", "state"]
                    )
            except MissingValue:
                # Valid case for CPD 5.1
                status = WMLResource._get_required_element_from_dict(
                    details, "details", ["entity", "state"]
                )

            if status in ["completed", "failed"]:
                self._cached_details = details
            else:
                self._cached_details = None

        return details

    @staticmethod
    def sort_results_by_metric(details: dict, metric_name: str) -> list:
        """
        Method for sort results by metric

        :param details: pattern details
        :type details: dict

        :param metric_name: metrics name to be sorted by
        :type metric_name: dict

        :return: clist of sorted metrics
        :rtype: list

        """

        results = details.get("entity", {}).get("results")

        def get_mean_value(test_data: list, inner_metric_name: str) -> float:
            for metric in test_data:
                if metric["metric_name"] == inner_metric_name:
                    return metric["mean"]
            return float("-inf")

        sorted_results = sorted(
            results,
            key=lambda x: get_mean_value(x["metrics"]["test_data"], metric_name),
            reverse=True,
        )
        return sorted_results

    def summary(self, scoring: str | list[str] | None = None) -> "DataFrame":
        """Engine for return RAGOptimizer summary details.

        :param scoring: scoring metric which user wants to use to sort patterns by,
            when not provided use optimized one
        :type scoring: str | list, optional

        :return: computed patterns and metrics
        :rtype: pandas.DataFrame

        """

        details = self.get_run_details()

        self._check_if_metrics_available(details)

        if isinstance(scoring, str):
            optimization_metrics = [scoring]
        elif isinstance(scoring, list):
            optimization_metrics = scoring
        else:
            optimization_metrics = self._params.get("optimization_metrics", None)
            if optimization_metrics is None:
                optimization_metrics = [
                    details.get("entity", {})
                    .get("results", [])[0]
                    .get("metrics", {})
                    .get("test_data", [])[0]
                    .get("metric_name")
                ]

        results = self.sort_results_by_metric(details, optimization_metrics[0])

        rag_pattern_names = [name["context"]["rag_pattern"]["name"] for name in results]

        ordered_columns = [
            "mean_answer_correctness",
            "mean_faithfulness",
            "mean_context_correctness",
        ]
        data_sorted_by_value: dict = {col: [] for col in ordered_columns}

        setting_keys = {
            "chunking": ["method", "chunk_size", "chunk_overlap"],
            "embeddings": ["model_id"],
            "vector_store": ["distance_metric"],
            "retrieval": ["method", "number_of_chunks"],
            "generation": ["model_id"],
        }
        data_sorted_by_value.update(
            {
                f"{key}.{sub_key}": []
                for key, sub_keys in setting_keys.items()
                for sub_key in sub_keys
            }
        )

        def add_to_data(data_dict: dict, dict_key: str, dict_value: Any) -> None:
            """Helper function to add values to the dictionary."""
            data_dict.setdefault(dict_key, []).append(dict_value)

        for result in results:
            for metric_item in result["metrics"]["test_data"]:
                metric_name = metric_item["metric_name"]
                if "mean" in metric_item:
                    add_to_data(
                        data_sorted_by_value, f"mean_{metric_name}", metric_item["mean"]
                    )

            settings = result["context"]["rag_pattern"]["settings"]

            for key, sub_keys in setting_keys.items():
                if key in settings:
                    for sub_key in sub_keys:
                        if sub_key in settings[key]:
                            add_to_data(
                                data_sorted_by_value,
                                f"{key}.{sub_key}",
                                settings[key][sub_key],
                            )

        data_sorted_by_optimization_metrics: dict = {"Pattern_Name": rag_pattern_names}

        for optimization_metric in optimization_metrics:
            for sorted_metric in data_sorted_by_value:
                if optimization_metric in sorted_metric:
                    data_sorted_by_optimization_metrics[sorted_metric] = (
                        data_sorted_by_value[sorted_metric]
                    )

        data_sorted_by_optimization_metrics.update(data_sorted_by_value)

        data_frame = DataFrame(data=data_sorted_by_optimization_metrics)
        data_frame.set_index("Pattern_Name", inplace=True)

        return data_frame

    def get_pattern(self, pattern_name: str | None = None) -> "RAGPattern":
        """Engine for return RAGPattern from RAGOptimizer training.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :return: RAGPattern class for defining, querying and deploying Retrieval-Augmented Generation (RAG) patterns.
        :rtype: RAGPattern

        """

        details = self.get_run_details()

        pattern_details = self.get_best_pattern_details(
            details=details, pattern_name=pattern_name
        )

        embeddings_settings = pattern_details["context"]["rag_pattern"]["settings"][
            "embeddings"
        ]
        embeddings = Embeddings(
            model_id=embeddings_settings["model_id"],
            params={
                EmbedParams.TRUNCATE_INPUT_TOKENS: embeddings_settings.get(
                    "truncate_input_tokens"
                )
            },
            api_client=self._client,
        )

        vector_store_settings = pattern_details["context"]["rag_pattern"]["settings"][
            "vector_store"
        ]
        datasource_type = None
        connection_id = None

        if vector_store_references := details["entity"].get("vector_store_references"):
            connection_id = vector_store_references[0]["connection"]["id"]
        else:
            datasource_type = vector_store_settings["datasource_type"]

        from ibm_watsonx_ai.foundation_models.extensions.rag import VectorStore

        vector_store = VectorStore(
            connection_id=connection_id,
            datasource_type=datasource_type,
            index_name=vector_store_settings.get("index_name"),
            distance_metric=vector_store_settings.get("distance_metric"),
            embeddings=embeddings,
            api_client=self._client,
        )

        retrieval_settings = pattern_details["context"]["rag_pattern"]["settings"][
            "retrieval"
        ]

        from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever

        retriever = Retriever(
            vector_store=vector_store,
            method=retrieval_settings.get("method"),
            number_of_chunks=retrieval_settings.get("number_of_chunks"),
            window_size=retrieval_settings.get("window_size"),
        )

        generation_settings = pattern_details["context"]["rag_pattern"]["settings"][
            "generation"
        ]
        model = ModelInference(
            model_id=generation_settings["model_id"],
            params=generation_settings["parameters"],
            project_id=details["metadata"].get("project_id"),
            space_id=details["metadata"].get("space_id"),
            api_client=self._client,
        )

        from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
            LangChainChunker,
        )

        chunking_settings = pattern_details["context"]["rag_pattern"]["settings"][
            "chunking"
        ]
        chunker = LangChainChunker(**chunking_settings)

        from ibm_watsonx_ai.foundation_models.extensions.rag.pattern import RAGPattern

        data_connections = None
        if vector_store._datasource_type == "chroma" and (
            input_data_references := details["entity"]["input_data_references"]
        ):
            data_connections = [
                DataConnection.from_dict(data_reference)
                for data_reference in input_data_references
            ]

        pattern = RAGPattern(
            retriever=retriever,
            model=model,
            prompt_template_text=generation_settings["prompt_template_text"],
            context_template_text=generation_settings["context_template_text"],
            space_id=details["metadata"].get("space_id"),
            project_id=details["metadata"].get("project_id"),
            api_client=self._client,
            chunker=chunker,
            input_data_references=data_connections,
        )

        return pattern

    def get_pattern_details(self, pattern_name: str | None = None) -> dict:
        """Engine for fetch specific pattern details, e.g. steps etc.

        :param pattern_name: pattern name e.g. Pattern1, if not specified, best pattern parameters will be fetched
        :type pattern_name: str, optional

        :return: pattern parameters
        :rtype: dict

        """

        details = self.get_run_details()

        pattern_details = self.get_best_pattern_details(
            details=details, pattern_name=pattern_name
        )

        return pattern_details.get("context", {}).get("rag_pattern")

    def get_inference_notebook(
        self,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Engine for download specified inference notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved inference notebook
        :rtype: str

        """

        return self._get_specific_notebook(
            type_of_notebook="inference_notebook",
            pattern_name=pattern_name,
            local_path=local_path,
            filename=filename,
        )

    def get_indexing_notebook(
        self,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Engine for download specified indexing notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved indexing notebook
        :rtype: str

        """

        return self._get_specific_notebook(
            type_of_notebook="indexing_notebook",
            pattern_name=pattern_name,
            local_path=local_path,
            filename=filename,
        )

    def _get_specific_notebook(
        self,
        type_of_notebook: Literal["indexing_notebook", "inference_notebook"],
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """
        Abstract class for get specific notebook

        :param type_of_notebook: type of notebook
        :type type_of_notebook: str

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved notebook
        :rtype: str

        """
        details = self.get_run_details()

        pattern_details = self.get_best_pattern_details(
            details=details, pattern_name=pattern_name
        )

        pattern_name = pattern_details["context"]["rag_pattern"]["name"]

        notebook_location = pattern_details["context"]["rag_pattern"]["location"][
            type_of_notebook
        ]

        results_ref = details["entity"]["results_reference"]

        data_connection = DataConnection._from_dict(results_ref)

        if data_connection.location is not None:
            attr_name = (
                "file_name"
                if hasattr(data_connection.location, "file_name")
                else "path"
            )
            setattr(data_connection.location, attr_name, notebook_location)

        data_connection.set_client(self._client)

        if not filename:
            filename = f"{pattern_name}_{type_of_notebook}.ipynb"

        if not filename.endswith(".ipynb"):
            filename += ".ipynb"

        filename = f"{local_path}/{filename}"

        data_connection.download(filename=filename)

        if is_ipython():
            from IPython.display import display
            from ibm_watsonx_ai.utils import create_download_link

            display(create_download_link(filename))

        return filename

    def get_best_pattern_details(
        self, details: dict, pattern_name: str | None = None
    ) -> dict:
        """
        Return best pattern details

        :param details: pattern details
        :type details: dict

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :return: pattern parameters
        :rtype: dict

        """

        self._check_if_metrics_available(details)

        results = details.get("entity", {}).get("results")

        if pattern_name:
            pattern_details = next(
                (
                    pattern
                    for pattern in results
                    if pattern["context"]["rag_pattern"]["name"] == pattern_name
                ),
                None,
            )
            if pattern_details is None:
                raise WMLClientError(
                    f"Invalid pattern name. Available pattern name: {[name['context']['rag_pattern']['name'] for name in results]}"
                )

        else:
            if self._params.get("optimization_metrics", None):
                optimization_metric = self._params.get("optimization_metrics", None)
            else:
                optimization_metric = [
                    details.get("entity", {})
                    .get("results", [])[0]
                    .get("metrics", {})
                    .get("test_data", [])[0]
                    .get("metric_name")
                ]

            pattern_details = None
            metrics_best_score = 0
            for pattern in results:
                for test_data in pattern["metrics"]["test_data"]:
                    if (
                        test_data["metric_name"] == optimization_metric[0]
                        and test_data["mean"] > metrics_best_score
                    ):
                        pattern_details = pattern
                        metrics_best_score = test_data["mean"]

        return pattern_details

    def get_logs(self) -> str:
        """
        Get logs of an AutoAI RAG job

        return: path to saved logs
        :rtype: str

        """

        details = self.get_run_details()

        results_ref = details["entity"]["results_reference"]

        logs_location = details["entity"]["results_reference"]["location"][
            "training_log"
        ]

        data_connection = DataConnection._from_dict(results_ref)

        if data_connection.location is not None:
            attr_name = (
                "file_name"
                if hasattr(data_connection.location, "file_name")
                else "path"
            )
            setattr(data_connection.location, attr_name, logs_location)
        data_connection.set_client(self._client)

        filename = logs_location.split("/")[-1]
        data_connection.download(filename=filename)

        if is_ipython():
            from IPython.display import display
            from ibm_watsonx_ai.utils import create_download_link

            display(create_download_link(filename))

        return filename

    def get_evaluation_results(self, pattern_name: str | None = None) -> str:
        """
        Get evaluation results of an AutoAI RAG job

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        return: path to saved evaluation results
        :rtype: str

        """
        details = self.get_run_details()
        results_ref = details["entity"]["results_reference"]

        pattern_details = self.get_best_pattern_details(
            details=details, pattern_name=pattern_name
        )
        pattern_name = pattern_details["context"]["rag_pattern"]["name"]
        evaluation_results_location = pattern_details["context"]["rag_pattern"][
            "location"
        ]["evaluation_results"]

        data_connection = DataConnection._from_dict(results_ref)

        if data_connection.location is not None:
            attr_name = (
                "file_name"
                if hasattr(data_connection.location, "file_name")
                else "path"
            )
            setattr(data_connection.location, attr_name, evaluation_results_location)
        data_connection.set_client(self._client)

        filename = f'{pattern_name}_{evaluation_results_location.split("/")[-1]}'

        data_connection.download(filename=filename)

        if is_ipython():
            from IPython.display import display
            from ibm_watsonx_ai.utils import create_download_link

            display(create_download_link(filename))

        return filename

    def initiate_optimizer_metadata(self, params: dict, **kwargs: Any) -> None:
        """Method for initiate optimizer metadata"""
        self._training_metadata = {
            self.ConfigurationMetaNames.NAME: params[self.ConfigurationMetaNames.NAME]
        }
        if self.ConfigurationMetaNames.DESCRIPTION in params:
            self._training_metadata[self.ConfigurationMetaNames.DESCRIPTION] = params[
                self.ConfigurationMetaNames.DESCRIPTION
            ]

        constraints = {
            "embedding_models": params.get("embedding_models"),
            "retrieval_methods": params.get("retrieval_methods"),
            "foundation_models": params.get("foundation_models"),
            "generation": params.get("generation"),
            "max_number_of_rag_patterns": params.get("max_number_of_rag_patterns"),
            "retrieval": params.get("retrieval"),
        }

        # note: For CPD 5.1, only chunking_methods = ["recursive"] (default) was supported.
        if not self._client.CPD_version == 5.1:
            constraints["chunking"] = params.get("chunking")

        constraints = {k: v for k, v in constraints.items() if v is not None}

        optimization = (
            {"metrics": params.get("optimization_metrics")}
            if "optimization_metrics" in params
            else None
        )

        parameters: dict = {}

        if constraints:
            parameters["constraints"] = constraints
        if optimization:
            parameters["optimization"] = optimization

        parameters.update({"output_logs": True})

        if parameters:
            self._training_metadata["parameters"] = parameters

        if kwargs.get("custom") is not None:
            self._training_metadata["custom"] = kwargs["custom"]

        if (
            self._client.default_space_id is None
            and self._client.default_project_id is None
        ):
            raise WMLClientError(
                Messages.get_message(
                    message_id="it_is_mandatory_to_set_the_space_project_id"
                )
            )
        else:
            if self._client.default_space_id is not None:
                self._training_metadata["space_id"] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                self._training_metadata["project_id"] = self._client.default_project_id

    def _initialize_training_metadata(
        self,
        input_data_references: list[DataConnection],
        test_data_references: list[DataConnection],
        results_reference: DataConnection,
        vector_store_references: list[DataConnection] | None = None,
    ) -> None:
        """Initialization of training metadata.

        :param input_data_references: Data storage connection details to inform where training data is stored
        :type input_data_references: list[DataConnection]

        :param test_data_references: A set of test data references
        :type test_data_references: list[DataConnection]

        :param results_reference: The training results
        :type results_reference: DataConnection

        :param vector_store_references: A set of vector store references
        :type vector_store_references: list[DataConnection], optional

        """

        self._training_metadata[self.ConfigurationMetaNames.INPUT_DATA_REFERENCES] = [
            connection._to_dict() for connection in input_data_references
        ]

        self._training_metadata[self.ConfigurationMetaNames.TEST_DATA_REFERENCES] = [
            connection._to_dict() for connection in test_data_references
        ]

        self._training_metadata[self.ConfigurationMetaNames.RESULTS_REFERENCE] = (
            results_reference._to_dict()
        )

        hardware_specifications_name = "L"  # Added as a default
        hardware_specifications_id = (
            self._client.hardware_specifications.get_id_by_name(
                hardware_specifications_name
            )
        )

        self._training_metadata[self.ConfigurationMetaNames.HARDWARE_SPEC] = {
            "id": hardware_specifications_id,
            "name": hardware_specifications_name,
        }

        if vector_store_references:
            self._training_metadata[
                self.ConfigurationMetaNames.VECTOR_STORE_REFERENCES
            ] = [connection._to_dict() for connection in vector_store_references]

    @staticmethod
    def _check_if_metrics_available(details: dict) -> None:
        """Method for checking if metrics available"""
        try:
            if not details.get("entity", {}).get("results", [])[0].get("context", {}):
                raise WMLClientError(
                    Messages.get_message(message_id="rag_optimizer_no_metrics")
                )
        except IndexError:
            raise WMLClientError(
                Messages.get_message(message_id="rag_optimizer_no_metrics")
            )

    def _is_autoai_rag_endpoint_available(self) -> bool:
        try:
            url = self._client.service_instance._href_definitions.get_autoai_rag_href()

            response_autoai_rag_api = self._client._session.get(
                url=f"{url}?limit=1",
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            return response_autoai_rag_api.status_code != 404
        except:
            return False
