#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from pandas import DataFrame
from typing import TYPE_CHECKING, Any, cast
from warnings import warn
from ibm_watsonx_ai.metanames import RAGOptimizerConfigurationMetaNames
from ibm_watsonx_ai.helpers.connections import (
    S3Location,
    FSLocation,
    ContainerLocation,
    DataConnection,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai.experiment.autoai.engines import RAGEngine
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern import RAGPattern

__all__ = ["RAGOptimizer"]


class RAGOptimizer:
    """RAGOptimizer class for RAG pattern operation.

    :param name: name for the RAGOptimizer
    :type name: str

    :param engine: engine for remote work on Service instance
    :type engine: RAGEngine

    :param description: description for the RAGOptimizer
    :type description: str, optional

    :param embedding_models: The embedding models to try.
    :type embedding_models: list[str], optional

    :param retrieval_methods: Retrieval methods to be used.
    :type retrieval_methods: list[str], optional

    :param foundation_models: The foundation models to try.
    :type foundation_models: list[str], optional

    :param max_number_of_rag_patterns: The maximum number of RAG patterns to create.
    :type max_number_of_rag_patterns: int, optional

    :param optimization_metrics: The metric name(s) to be used for optimization.
    :type optimization_metrics: list[str], optional

    :param generation: Properties describing the generation step.
    :type generation: dict[str, Any], optional

    :param retrieval: Retrieval settings to be used.
    :type retrieval: list[dict[str, Any]], optional

    """

    def __init__(
        self,
        name: str,
        engine: "RAGEngine",
        description: str | None = None,
        chunking_methods: list[str] | None = None,
        embedding_models: list[str] | None = None,
        retrieval_methods: list[str] | None = None,
        foundation_models: list[str] | None = None,
        max_number_of_rag_patterns: int | None = None,
        optimization_metrics: list[str] | None = None,
        chunking: list[dict] | None = None,
        generation: dict[str, Any] | None = None,
        retrieval: list[dict[str, Any]] | None = None,
        **kwargs: dict[str, Any],
    ):
        self._engine = engine

        if chunking_methods is not None:
            chunking_methods_deprecated_warning = "The parameter chunking_methods is deprecated, please use `chunking` instead"
            warn(chunking_methods_deprecated_warning, category=DeprecationWarning)

        self._params = {
            "name": name,
            "description": description,
            "chunking": chunking,
            "embedding_models": embedding_models,
            "retrieval_methods": retrieval_methods,
            "retrieval": retrieval,
            "foundation_models": foundation_models,
            "generation": generation,
            "max_number_of_rag_patterns": max_number_of_rag_patterns,
            "optimization_metrics": optimization_metrics,
        }
        self._engine.initiate_optimizer_metadata(self._params, **kwargs)
        self._engine._params = self._params

        self.ConfigurationMetaNames = RAGOptimizerConfigurationMetaNames()

    def _get_engine(self) -> RAGEngine:
        """Return Engine for development purposes."""
        return self._engine

    def get_params(self) -> dict:
        """Get configuration parameters of RAGOptimizer.

        :return: RAGOptimizer parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_params()

            # Result:
            # {
            #     'name': 'RAG AutoAi ',
            #     'description': 'Sample description',
            #     'max_number_of_rag_patterns': 5,
            #     'optimization_metrics': ['answer_correctness']
            # }

        """
        params_without_none = {k: v for k, v in self._params.items() if v is not None}
        return params_without_none

    def run(
        self,
        input_data_references: list[DataConnection],
        test_data_references: list[DataConnection],
        results_reference: DataConnection | None = None,
        vector_store_references: list[DataConnection] | None = None,
        background_mode: bool = True,
    ) -> dict:
        """Create an AutoAI RAG job that will find the best RAG pattern.

        :param input_data_references: Data storage connection details to inform where training data is stored
        :type input_data_references: list[DataConnection]

        :param test_data_references: A set of test data references
        :type test_data_references: list[DataConnection]

        :param results_reference: The training results
        :type results_reference: DataConnection, optional

        :param vector_store_references: A set of vector store references
        :type vector_store_references: list[DataConnection], optional

        :param background_mode: Indicator if run() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :return: run details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            from ibm_watsonx_ai.utils.autoai.enums import TShirtSize
            from ibm_watsonx_ai.helpers import DataConnection, ContainerLocation

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.run(
                input_data_references=[DataConnection(
                    data_asset_id=training_data_asset_id
                )],
                test_data_references=[DataConnection(
                    data_asset_id=test_data_asset_id
                )],
                vector_store_references=[DataConnection(
                    connection_asset_id=milvus_connection_id
                )],
                results_reference=[DataConnection(
                    location=ContainerLocation(
                        path="."
                    )
                )],
                background_mode=False
            )

        """
        results_reference = self._determine_result_reference(
            results_reference, "default_autoai_rag_out"
        )

        results_reference = cast(DataConnection, results_reference)

        return self._engine.run(
            input_data_references=input_data_references,
            test_data_references=test_data_references,
            results_reference=results_reference,
            vector_store_references=vector_store_references,
            background_mode=background_mode,
        )

    def _determine_result_reference(
        self,
        results_reference: DataConnection | None,
        result_path: str,
    ) -> DataConnection:
        if results_reference is None:
            if self._engine._client.CLOUD_PLATFORM_SPACES:
                results_reference = DataConnection(
                    location=ContainerLocation(path=result_path)
                )
            else:
                location = FSLocation()
                client = self._engine._client
                if self._engine._client.default_project_id is None:

                    location.path = location.path.format(
                        option="spaces", id=client.default_space_id
                    )

                else:
                    location.path = location.path.format(
                        option="projects", id=client.default_project_id
                    )
                results_reference = DataConnection(connection=None, location=location)

        elif getattr(results_reference, "type", False) == "fs":
            client = self._engine._client
            results_reference.location = cast(FSLocation, results_reference.location)
            if self._engine._client.default_project_id is None:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="spaces", id=client.default_space_id
                    )
                )
            else:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="projects", id=client.default_project_id
                    )
                )

        if not isinstance(
            results_reference.location,
            (S3Location, FSLocation, ContainerLocation),
        ):
            raise TypeError(
                "Unsupported results location type. Results reference can be stored"
                " only on S3Location or FSLocation or ContainerLocation."
            )

        return results_reference

    def cancel_run(self, hard_delete: bool = False) -> str:
        """Cancels a RAG Optimizer run.

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancellation is successful
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)
            rag_optimizer.run(...)

            rag_optimizer.cancel_run()
            # or
            rag_optimizer.cancel_run(hard_delete=True)

        """

        return self._engine.cancel_run(hard_delete=hard_delete)

    def get_run_status(self) -> str:
        """Check status/state of initialized RAGOptimizer run if ran in background mode.

        :return: run status details
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_run_status()

            # Result:
            # 'completed'

        """
        return self._engine.get_run_status()

    def get_run_details(self) -> dict:
        """Get run details.

        :return: RAGOptimizer run details
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_run_details()

        """
        return self._engine.get_run_details()

    def summary(self, scoring: str | list[str] | None = None) -> "DataFrame":
        """Return RAGOptimizer summary details.

        :param scoring: scoring metric which user wants to use to sort patterns by,
            when not provided use optimized one
        :type scoring: str | list, optional

        :return: computed patterns and metrics
        :rtype: pandas.DataFrame

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.summary()
            rag_optimizer.summary(scoring='answer_correctness')
            rag_optimizer.summary(scoring=['answer_correctness', 'context_correctness'])

            # Result:
            #                  mean_answer_correctness  ...  ci_high_faithfulness
            # Pattern_Name	                            ...
            # Pattern5                        0.79165   ...                0.5102
            # Pattern1                        0.72915   ...                0.4839
            # Pattern2                        0.64585   ...                0.8333
            # Pattern4                        0.64585   ...                0.5312

        """
        return self._engine.summary(scoring=scoring)

    def get_pattern(self, pattern_name: str | None = None) -> "RAGPattern":
        """Return RAGPattern from RAGOptimizer training.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :return: RAGPattern class for defining, querying and deploying Retrieval-Augmented Generation (RAG) patterns.
        :rtype: RAGPattern

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            pattern_1 = rag_optimizer.get_pattern()
            pattern_2 = rag_optimizer.get_pattern(pattern_name='Pattern2')

        """
        return self._engine.get_pattern(pattern_name=pattern_name)

    def get_pattern_details(self, pattern_name: str | None = None) -> dict:
        """Fetch specific pattern details, e.g. steps etc.

        :param pattern_name: pattern name e.g. Pattern1, if not specified, best pattern parameters will be fetched
        :type pattern_name: str, optional

        :return: pattern parameters
        :rtype: dict

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_pattern_details()
            rag_optimizer.get_pattern_details(pattern_name='Pattern1')

        """
        return self._engine.get_pattern_details(pattern_name=pattern_name)

    def get_inference_notebook(
        self,
        *,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Download specified inference notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved inference notebook
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            inference_notebook_path_1 = rag_optimizer.get_inference_notebook()
            inference_notebook_path_2 = rag_optimizer.get_inference_notebook(
                pattern_name='Pattern1',
                filename='inference_notebook'
            )

        """
        return self._engine.get_inference_notebook(
            pattern_name=pattern_name, local_path=local_path, filename=filename
        )

    def get_indexing_notebook(
        self,
        *,
        pattern_name: str | None = None,
        local_path: str = ".",
        filename: str | None = None,
    ) -> str:
        """Download specified indexing notebook from Service.

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        :param local_path: local filesystem path, if not specified, current directory is used
        :type local_path: str, optional

        :param filename: filename under which the pattern notebook will be saved
        :type filename: str, optional

        :return: path to saved indexing notebook
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            indexing_notebook_path_1 = rag_optimizer.get_indexing_notebook()
            indexing_notebook_path_2 = rag_optimizer.get_indexing_notebook(
                pattern_name='Pattern1',
                filename='indexing_notebook'
            )

        """
        return self._engine.get_indexing_notebook(
            pattern_name=pattern_name, local_path=local_path, filename=filename
        )

    def get_logs(self) -> str:
        """
        Get logs of an AutoAI RAG job

        return: path to saved logs
        :rtype: str

        **Example**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_logs()

        """
        return self._engine.get_logs()

    def get_evaluation_results(self, pattern_name: str | None = None) -> str:
        """
        Get evaluation results of an AutoAI RAG job

        :param pattern_name: pattern name, if you want to see the patterns names, please use summary() method,
            if this parameter is None, the best pattern will be fetched
        :type pattern_name: str, optional

        return: path to saved evaluation results
        :rtype: str

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI

            experiment = AutoAI(credentials, ...)
            rag_optimizer = experiment.rag_optimizer(...)

            rag_optimizer.get_evaluation_results()
            # or
            rag_optimizer.get_evaluation_results(pattern_name='Pattern1')

        """
        return self._engine.get_evaluation_results(pattern_name=pattern_name)
