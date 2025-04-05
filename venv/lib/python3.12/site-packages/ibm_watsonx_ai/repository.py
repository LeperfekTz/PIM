#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any, Literal, TypeAlias, cast

from dataclasses import dataclass

from requests import Response

from ibm_watsonx_ai._wrappers import requests

from ibm_watsonx_ai.experiments import Experiments
from ibm_watsonx_ai.functions import Functions
from ibm_watsonx_ai.ai_services import AIServices
from ibm_watsonx_ai.libs.repo.mlrepositoryclient import MLRepositoryClient
from ibm_watsonx_ai.lifecycle import SpecStates
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import (
    ExperimentMetaNames,
    FunctionMetaNames,
    PipelineMetanames,
    SpacesMetaNames,
    ModelMetaNames,
    RepositoryMemberMetaNames,
    AIServiceMetaNames,
)
from ibm_watsonx_ai.models import Models
from ibm_watsonx_ai.pipelines import Pipelines
from ibm_watsonx_ai.utils import inherited_docstring, get_url, get_user_agent_header
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource


if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    import numpy
    import pandas
    import pyspark

    LabelColumnNamesType: TypeAlias = (
        numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str]
    )
    TrainingDataType: TypeAlias = (
        pandas.DataFrame | numpy.ndarray | pyspark.sql.Dataframe | list
    )
    TrainingTargetType: TypeAlias = (
        pandas.DataFrame | pandas.Series | numpy.ndarray | list
    )
    FeatureNamesArrayType: TypeAlias = numpy.ndarray | list


class Repository(WMLResource):
    """Store and manage models, functions, spaces, pipelines, and experiments
    using the Watson Machine Learning Repository.

    To view ModelMetaNames, use:

    .. code-block:: python

        client.repository.ModelMetaNames.show()

    To view ExperimentMetaNames, use:

    .. code-block:: python

        client.repository.ExperimentMetaNames.show()

    To view FunctionMetaNames, use:

    .. code-block:: python

        client.repository.FunctionMetaNames.show()

    To view PipelineMetaNames, use:

    .. code-block:: python

        client.repository.PipelineMetaNames.show()

    To view AIServiceMetaNames, use:

    .. code-block:: python

        client.repository.AIServiceMetaNames.show()

    """

    @dataclass
    class ModelAssetTypes:
        """Data class with supported model asset types."""

        DO_DOCPLEX_20_1: str = "do-docplex_20.1"
        DO_OPL_20_1: str = "do-opl_20.1"
        DO_CPLEX_20_1: str = "do-cplex_20.1"
        DO_CPO_20_1: str = "do-cpo_20.1"
        DO_DOCPLEX_22_1: str = "do-docplex_22.1"
        DO_OPL_22_1: str = "do-opl_22.1"
        DO_CPLEX_22_1: str = "do-cplex_22.1"
        DO_CPO_22_1: str = "do-cpo_22.1"
        WML_HYBRID_0_1: str = "wml-hybrid_0.1"
        PMML_4_2_1: str = "pmml_4.2.1"
        PYTORCH_ONNX_1_12: str = "pytorch-onnx_1.12"
        PYTORCH_ONNX_RT22_2: str = "pytorch-onnx_rt22.2"
        PYTORCH_ONNX_2_0: str = "pytorch-onnx_2.0"
        PYTORCH_ONNX_RT23_1: str = "pytorch-onnx_rt23.1"
        SCIKIT_LEARN_1_1: str = "scikit-learn_1.1"
        MLLIB_3_3: str = "mllib_3.3"
        SPSS_MODELER_17_1: str = "spss-modeler_17.1"
        SPSS_MODELER_18_1: str = "spss-modeler_18.1"
        SPSS_MODELER_18_2: str = "spss-modeler_18.2"
        TENSORFLOW_2_9: str = "tensorflow_2.9"
        TENSORFLOW_RT22_2: str = "tensorflow_rt22.2"
        TENSORFLOW_2_12: str = "tensorflow_2.12"
        TENSORFLOW_RT23_1: str = "tensorflow_rt23.1"
        XGBOOST_1_6: str = "xgboost_1.6"
        PROMPT_TUNE_1_0: str = "prompt_tune_1.0"
        CUSTOM_FOUNDATION_MODEL_1_0: str = "custom_foundation_model_1.0"
        CURATED_FOUNDATION_MODEL_1_0: str = "curated_foundation_model_1.0"

    cloud_platform_spaces = False
    icp_platform_spaces = False

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)
        self._ml_repository_client: MLRepositoryClient | None = None

        self.ExperimentMetaNames = ExperimentMetaNames()
        self.FunctionMetaNames = FunctionMetaNames()
        self.PipelineMetaNames = PipelineMetanames()
        self.SpacesMetaNames = SpacesMetaNames()
        self.ModelMetaNames = ModelMetaNames()
        self.MemberMetaNames = RepositoryMemberMetaNames()
        self.AIServiceMetaNames = AIServiceMetaNames()

        # make sure that old repo client is aware of token changes
        self._client._auth_method._on_token_set = self._refresh_repo_client
        self._client._auth_method._on_token_creation = self._refresh_repo_client
        self._client._auth_method._on_token_refresh = self._refresh_repo_client

        self._refresh_repo_client()

    def _refresh_repo_client(self) -> None:
        self._ml_repository_client = MLRepositoryClient(self._credentials.url)
        self._ml_repository_client.authorize_with_token(self._client.token)
        self._ml_repository_client._add_header("User-Agent", get_user_agent_header())

    @inherited_docstring(
        Experiments.store, {"experiments.get_href": "repository.get_experiment_href"}
    )
    def store_experiment(self, meta_props: dict) -> dict:
        return self._client.experiments.store(meta_props)

    @inherited_docstring(Pipelines.store)
    def store_pipeline(self, meta_props: dict) -> dict:
        return self._client.pipelines.store(meta_props)

    @inherited_docstring(Models.store, {"store()": "store_model()"})
    def store_model(
        self,
        model: str | object | None = None,
        meta_props: dict | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: object | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict | None = None,
        training_id: str | None = None,
    ) -> dict:
        return self._client._models.store(
            model=model,
            meta_props=meta_props,
            training_data=training_data,
            training_target=training_target,
            pipeline=pipeline,
            feature_names=feature_names,
            label_column_names=label_column_names,
            subtrainingId=subtrainingId,
            round_number=round_number,
            experiment_metadata=experiment_metadata,
            training_id=training_id,
        )

    def clone(
        self,
        artifact_id: str,
        space_id: str | None = None,
        action: str = "copy",
        rev_id: str | None = None,
    ) -> dict:
        raise WMLClientError(Messages.get_message(message_id="cloning_not_supported"))

    @inherited_docstring(Functions.store)
    def store_function(
        self, function: str | Callable, meta_props: str | dict[str, Any]
    ) -> dict:
        return self._client._functions.store(function, meta_props)

    @inherited_docstring(Models.create_revision)
    def create_model_revision(self, model_id: str | None = None, **kwargs: Any) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.create_revision(model_id=model_id)

    @inherited_docstring(Pipelines.create_revision)
    def create_pipeline_revision(
        self, pipeline_id: str | None = None, **kwargs: Any
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.create_revision(pipeline_id=pipeline_id)

    @inherited_docstring(Functions.create_revision)
    def create_function_revision(
        self, function_id: str | None = None, **kwargs: Any
    ) -> dict:
        return self._client._functions.create_revision(
            function_id=function_id, **kwargs
        )

    @inherited_docstring(Experiments.create_revision)
    def create_experiment_revision(self, experiment_id: str) -> dict:
        return self._client.experiments.create_revision(experiment_id=experiment_id)

    @inherited_docstring(Models.update, {"meta_props": "updated_meta_props"})
    def update_model(
        self,
        model_id: str | None = None,
        updated_meta_props: dict | None = None,
        update_model: Any | None = None,
        **kwargs: Any,
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.update(model_id, updated_meta_props, update_model)

    @inherited_docstring(Experiments.update)
    def update_experiment(
        self,
        experiment_id: str | None = None,
        changes: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client.experiments.update(experiment_id, changes, **kwargs)

    @inherited_docstring(Functions.update)
    def update_function(
        self,
        function_id: str | None,
        changes: dict | None = None,
        update_function: str | Callable | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client._functions.update(
            function_id, changes, update_function, **kwargs
        )

    @inherited_docstring(Pipelines.update)
    def update_pipeline(
        self,
        pipeline_id: str | None = None,
        changes: dict | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.update(pipeline_id, changes, rev_id, **kwargs)

    def load(self, artifact_id: str | None = None, **kwargs: Any) -> object:
        """Load a model from the repository to object in a local environment.

        .. note::
            The use of the load() method is restricted and not permitted for AutoAI models.

        :param artifact_id: ID of the stored model
        :type artifact_id: str

        :return: trained model
        :rtype: object

        **Example**

        .. code-block:: python

            model = client.repository.load(model_id)

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        return self._client._models.load(artifact_id)

    def download(
        self,
        artifact_id: str | None = None,
        filename: str = "downloaded_artifact.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download the configuration file for an artifact with the specified ID.

        :param artifact_id: unique ID of the model or function
        :type artifact_id: str
        :param filename: name of the file to which the artifact content will be downloaded
        :type filename: str, optional
        :param rev_id: revision ID
        :type rev_id: str, optional
        :param format: format of the content, applicable for models
        :type format: str, optional

        :return: path to the downloaded artifact content
        :rtype: str

        **Examples**

        .. code-block:: python

            client.repository.download(model_id, 'my_model.tar.gz')
            client.repository.download(model_id, 'my_model.json') # if original model was saved as json, works only for xgboost 1.3

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)

        self._validate_type(artifact_id, "artifact_id", str, True)
        self._validate_type(filename, "filename", str, True)

        res = self._check_artifact_type(str(artifact_id))

        if res["model"] is True:
            return self._client._models.download(artifact_id, filename, rev_id, format)
        elif res["function"]:
            return self._client._functions.download(artifact_id, filename, rev_id)
        elif res["ai_service"]:
            return self._client._ai_services.download(artifact_id, filename, rev_id)
        else:
            raise WMLClientError(
                "Unexpected type of artifact to download or Artifact with artifact_id: '{}' does not exist.".format(
                    artifact_id
                )
            )

    def delete(
        self, artifact_id: str | None = None, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Delete a model, experiment, pipeline, function, or AI service from the repository.

        :param artifact_id: unique ID of the stored model, experiment, function, pipeline, or AI service
        :type artifact_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.repository.delete(artifact_id)

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")

        Repository._validate_type(artifact_id, "artifact_id", str, True)
        if self._if_deployment_exist_for_asset(artifact_id):
            raise WMLClientError(
                "Cannot delete artifact that has existing deployments. Please delete all associated deployments and try again"
            )
        params = self._client._params()
        params.update({"purge_on_delete": "true"})

        response = requests.delete(
            self._client.service_instance._href_definitions.get_asset_href(artifact_id),
            params=params,
            headers=self._client._get_headers(),
        )

        if response.status_code == 200 or response.status_code == 204:
            if response.status_code == 200:
                response = self._handle_response(200, "delete assets", response)
                return response
            else:
                response = self._handle_response(204, "delete assets", response)
                return response
        else:
            if response.status_code == 404:
                raise WMLClientError(
                    "Artifact with artifact_id: '{}' does not exist.".format(
                        artifact_id
                    )
                )
            else:
                raise WMLClientError(
                    "Deletion error for the given id : ", response.text
                )

    def get_details(
        self,
        artifact_id: str | None = None,
        spec_state: SpecStates | None = None,
        artifact_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of stored artifacts. If `artifact_id` and `artifact_name` are not specified,
        the metadata of all models, experiments, functions, pipelines, and ai services is returned.
        If only `artifact_name` is specified, metadata of all artifacts with the name is returned.

        :param artifact_id: unique ID of the stored model, experiment, function, or pipeline
        :type artifact_id: str, optional

        :param spec_state: software specification state, can be used only when `artifact_id` is None
        :type spec_state: SpecStates, optional

        :param artifact_name: name of the stored model, experiment, function, pipeline, or ai service
            can be used only when `artifact_id` is None
        :type artifact_name: str, optional

        :return: metadata of the stored artifact(s)
        :rtype:
            - dict (if artifact_id is not None)
            - {"models": dict, "experiments": dict, "pipeline": dict, "functions": dict, "ai_service": dict} (if artifact_id is None)

        **Examples**

        .. code-block:: python

            details = client.repository.get_details(artifact_id)
            details = client.repository.get_details(artifact_name='Sample_model')
            details = client.repository.get_details()


        Example of getting all repository assets with deprecated software specifications:

        .. code-block:: python

            from ibm_watsonx_ai.lifecycle import SpecStates

            details = client.repository.get_details(spec_state=SpecStates.DEPRECATED)

        """
        artifact_id = _get_id_from_deprecated_uid(
            kwargs, artifact_id, "artifact", can_be_none=True
        )
        Repository._validate_type(artifact_id, "artifact_id", str, False)
        Repository._validate_type(artifact_name, "artifact_name", str, False)

        if artifact_id is None:
            model_details = self._client._models.get_details(
                spec_state=spec_state, model_name=artifact_name
            )
            experiment_details = (
                self.get_experiment_details(experiment_name=artifact_name)
                if not spec_state
                else {"resources": []}
            )
            pipeline_details = (
                self.get_pipeline_details(pipeline_name=artifact_name)
                if not spec_state
                else {"resources": []}
            )
            function_details = self._client._functions.get_details(
                spec_state=spec_state, function_name=artifact_name
            )
            try:
                ai_service_details = self._client._ai_services.get_details(
                    spec_state=spec_state, ai_service_name=artifact_name
                )
            except WMLClientError:
                ai_service_details = None

            details = {
                "models": model_details,
                "experiments": experiment_details,
                "pipeline": pipeline_details,
                "functions": function_details,
            }

            if ai_service_details is not None:
                details["ai_service"] = ai_service_details

        else:
            artifact_type = self._check_artifact_type(str(artifact_id))

            if artifact_type["model"] is True:
                details = self.get_model_details(artifact_id)
            elif artifact_type["experiment"] is True:
                details = self.get_experiment_details(artifact_id)
            elif artifact_type["pipeline"] is True:
                details = self.get_pipeline_details(artifact_id)
            elif artifact_type["function"] is True:
                details = self.get_function_details(artifact_id)
            elif artifact_type["ai_service"] is True:
                details = self.get_ai_service_details(artifact_id)
            else:
                raise WMLClientError(
                    "Getting artifact details failed. Artifact id: '{}' not found.".format(
                        artifact_id
                    )
                )

        return details

    def get_id_by_name(self, artifact_name: str) -> str:
        """Get the ID of a stored artifact by name.

        :param artifact_name: name of the stored artifact
        :type artifact_name: str

        :return: ID of the stored artifact if exactly one with the 'artifact_name' exists. Otherwise, raise an error.
        :rtype: str

        **Example:**

        .. code-block:: python

            artifact_id = client.repository.get_id_by_name(artifact_name)

        """

        details = self.get_details(artifact_name=artifact_name)

        # Check whether 0, 1, or more artifacts were found in 'details' results
        details_by_name = {}
        for artifact_type, artifact_details in details.items():
            if len(artifact_details["resources"]) == 1 and not details_by_name:
                # Found first artifact
                details_by_name = artifact_details["resources"][0]
            elif len(artifact_details["resources"]) > 0 and details_by_name:
                # Found another artifact of different type
                raise WMLClientError(
                    Messages.get_message(
                        artifact_name,
                        message_id="multiple_artifacts_found_by_name",
                    )
                )
            elif len(artifact_details["resources"]) > 1:
                # Found more than 1 artifact of a specific type
                raise WMLClientError(
                    Messages.get_message(
                        artifact_name,
                        message_id="multiple_artifacts_found_by_name",
                    )
                )

        if not details_by_name:
            raise WMLClientError(
                f"Artifact with artifact_name: '{artifact_name}' does not exist."
            )

        return details_by_name["metadata"]["id"]

    @inherited_docstring(Models.get_details)
    def get_model_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(
            kwargs, model_id, "model", can_be_none=True
        )
        return self._client._models.get_details(
            model_id=model_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            model_name=model_name,
        )

    @inherited_docstring(Models.get_revision_details)
    def get_model_revision_details(
        self, model_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")
        return self._client._models.get_revision_details(model_id, rev_id)

    @inherited_docstring(Experiments.get_details)
    def get_experiment_details(
        self,
        experiment_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        experiment_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client.experiments.get_details(
            experiment_id=experiment_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            experiment_name=experiment_name,
            **kwargs,
        )

    @inherited_docstring(Experiments.get_revision_details)
    def get_experiment_revision_details(
        self, experiment_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client.experiments.get_revision_details(
            experiment_id, rev_id, **kwargs
        )

    @inherited_docstring(Functions.get_details)
    def get_function_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client._functions.get_details(
            function_id=function_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            function_name=function_name,
            **kwargs,
        )

    @inherited_docstring(Functions.get_revision_details)
    def get_function_revision_details(
        self, function_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client._functions.get_revision_details(
            function_id, rev_id, **kwargs
        )

    @inherited_docstring(Pipelines.get_details)
    def get_pipeline_details(
        self,
        pipeline_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        pipeline_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(
            kwargs, pipeline_id, "pipeline", can_be_none=True
        )
        Repository._validate_type(pipeline_id, "pipeline_id", str, False)
        Repository._validate_type(limit, "limit", int, False)
        Repository._validate_type(asynchronous, "asynchronous", bool, False)
        Repository._validate_type(get_all, "get_all", bool, False)
        return self._client.pipelines.get_details(
            pipeline_id=pipeline_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            pipeline_name=pipeline_name,
            **kwargs,
        )

    @inherited_docstring(Pipelines.get_revision_details)
    def get_pipeline_revision_details(
        self, pipeline_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.get_revision_details(
            pipeline_id, rev_id, **kwargs
        )

    @staticmethod
    @inherited_docstring(Models.get_href)
    def get_model_href(model_details: dict) -> str:
        return Models.get_href(model_details)

    @staticmethod
    @inherited_docstring(Models.get_id)
    def get_model_id(model_details: dict) -> str:
        return Models.get_id(model_details)

    @staticmethod
    @inherited_docstring(
        Experiments.get_id,
        {"experiments.get_details": "repository.get_experiment_details"},
    )
    def get_experiment_id(experiment_details: dict) -> str:
        return Experiments.get_id(experiment_details)

    @staticmethod
    @inherited_docstring(
        Experiments.get_href,
        {"experiments.get_details": "repository.get_experiment_details"},
    )
    def get_experiment_href(experiment_details: dict) -> str:
        return Experiments.get_href(experiment_details)

    @staticmethod
    @inherited_docstring(Functions.get_id)
    def get_function_id(function_details: dict) -> str:
        return Functions.get_id(function_details)

    @staticmethod
    @inherited_docstring(Functions.get_href)
    def get_function_href(function_details: dict) -> str:
        return Functions.get_href(function_details)

    @staticmethod
    @inherited_docstring(
        Pipelines.get_href, {"pipelines.get_details": "repository.get_pipeline_details"}
    )
    def get_pipeline_href(pipeline_details: dict) -> str:
        return Pipelines.get_href(pipeline_details)

    @staticmethod
    @inherited_docstring(Pipelines.get_id)
    def get_pipeline_id(pipeline_details: dict) -> str:
        return Pipelines.get_id(pipeline_details)

    def list(self, framework_filter: str | None = None) -> pandas.DataFrame:
        """Get and list stored models, pipelines, functions, experiments, and AI services in a table/DataFrame format.

        :param framework_filter: get only the frameworks with the desired names
        :type framework_filter: str, optional

        :return: DataFrame with listed names and IDs of stored models
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.repository.list()
            client.repository.list(framework_filter='prompt_tune')

        """

        params = self._client._params()

        isIcp = self._client.ICP_PLATFORM_SPACES

        endpoints = {
            "model": self._client.service_instance._href_definitions.get_published_models_href(),
            "experiment": self._client.service_instance._href_definitions.get_experiments_href(),
            "pipeline": self._client.service_instance._href_definitions.get_pipelines_href(),
            "function": self._client.service_instance._href_definitions.get_functions_href(),
            "ai_service": self._client.service_instance._href_definitions.get_ai_services_href(),
        }

        artifact_get = {}
        for artifact in endpoints:
            params = self._client._params()
            artifact_get[artifact] = get_url(
                endpoints[artifact], self._client._get_headers(), params, isIcp
            )

        resources: dict[str, list] = {artifact: [] for artifact in endpoints}

        for artifact in endpoints:
            try:
                response = artifact_get[artifact]
                response_text = self._handle_response(
                    200, "getting all {}s".format(artifact), response
                )
                resources[artifact] = response_text["resources"]
            except Exception as e:
                self._logger.error(e)

        sw_spec_info = {
            s["id"]: s
            for s in self._client.software_specifications.get_details(state_info=True)[
                "resources"
            ]
        }

        def get_spec_info(spec_id: str, prop: str) -> str:
            if spec_id and spec_id in sw_spec_info:
                return sw_spec_info[spec_id].get(prop, "")
            else:
                return ""

        values = []
        for t in endpoints.keys():
            values += [
                (
                    m["metadata"]["id"],
                    m["metadata"]["name"],
                    m["metadata"]["created_at"],
                    m["entity"]["type"] if t == "model" else "-",
                    (
                        t
                        if t != "function" or t != "ai_service"
                        else m["entity"]["type"] + " function"
                    ),
                    get_spec_info(
                        m["entity"].get("software_spec", {}).get("id"), "state"
                    ),
                    get_spec_info(
                        m["entity"].get("software_spec", {}).get("id"),
                        "replacement",
                    ),
                )
                for m in resources[t]
            ]

        columns = [
            "ID",
            "NAME",
            "CREATED",
            "FRAMEWORK",
            "TYPE",
            "SPEC_STATE",
            "SPEC_REPLACEMENT",
        ]
        from pandas import DataFrame

        table = DataFrame(data=values, columns=columns)

        table = table.sort_values(by=["CREATED"], ascending=False).reset_index(
            drop=True
        )

        if framework_filter:
            table = table[table["FRAMEWORK"].str.contains(framework_filter)]

        return table

    @inherited_docstring(Models.list)
    def list_models(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> pandas.DataFrame:
        return self._client._models.list(
            limit=limit, asynchronous=asynchronous, get_all=get_all
        )

    @inherited_docstring(Experiments.list)
    def list_experiments(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client.experiments.list(limit=limit)

    @inherited_docstring(Functions.list)
    def list_functions(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client._functions.list(limit=limit)

    @inherited_docstring(Pipelines.list)
    def list_pipelines(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client.pipelines.list(limit=limit)

    def _check_artifact_type(self, artifact_id: str) -> dict[str, bool]:
        Repository._validate_type(artifact_id, "artifact_id", str, True)

        def _artifact_exists(response: Response | None) -> bool:
            return (
                (response is not None)
                and ("status_code" in dir(response))
                and (response.status_code == 200)
            )

        isIcp = self._client.ICP_PLATFORM_SPACES

        endpoints = {
            "model": self._client.service_instance._href_definitions.get_model_last_version_href(
                artifact_id
            ),
            "pipeline": self._client.service_instance._href_definitions.get_pipeline_href(
                artifact_id
            ),
            "experiment": self._client.service_instance._href_definitions.get_experiment_href(
                artifact_id
            ),
            "function": self._client.service_instance._href_definitions.get_function_href(
                artifact_id
            ),
            "ai_service": self._client.service_instance._href_definitions.get_ai_service_href(
                artifact_id
            ),
        }

        artifact_get = {}
        for artifact in endpoints:
            params = self._client._params()
            artifact_get[artifact] = get_url(
                endpoints[artifact], self._client._get_headers(), params, isIcp
            )

        response_get: dict[str, Response | None] = {
            artifact: None for artifact in endpoints
        }

        for artifact in endpoints:
            try:
                response_get[artifact] = artifact_get[artifact]
                artifact_res = cast(Response, response_get[artifact])

                self._logger.debug(
                    "Response({})[{}]: {}".format(
                        endpoints[artifact],
                        artifact_res.status_code,
                        artifact_res.text,
                    )
                )

            except Exception as e:
                self._logger.debug("Error during checking artifact type: " + str(e))

        artifact_type = {
            artifact: _artifact_exists(response_get[artifact])
            for artifact in response_get
        }
        return artifact_type

    def create_revision(self, artifact_id: str | None = None, **kwargs: Any) -> dict:
        """Create a revision for passed `artifact_id`.

        :param artifact_id: unique ID of a stored model, experiment, function, or pipelines
        :type artifact_id: str

        :return: artifact new revision metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            details = client.repository.create_revision(artifact_id)

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")

        Repository._validate_type(artifact_id, "artifact_id", str, True)

        artifact_type = self._check_artifact_type(str(artifact_id))
        if artifact_type["experiment"] is True:
            return self._client.experiments.create_revision(artifact_id)
        elif artifact_type["pipeline"] is True:
            return self._client.pipelines.create_revision(artifact_id)
        elif artifact_type["ai_service"] is True:
            return self._client._ai_services.create_revision(artifact_id)
        else:
            raise WMLClientError(
                "Getting artifact details failed. Artifact id: '{}' not found.".format(
                    artifact_id
                )
            )

    def _get_revision_details(
        self, artifact_id: str | None = None, **kwargs: Any
    ) -> dict:
        """Get metadata of the stored artifacts revisions.

        :param artifact_id: unique ID of a stored model, experiment, function, pipelines
        :type artifact_id: str

        :return: stored artifacts metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            details = client.repository.get_revision_details(artifact_id)

        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")

        Repository._validate_type(artifact_id, "artifact_id", str, True)

        artifact_type = self._check_artifact_type(str(artifact_id))

        if artifact_type["experiment"] is True:
            details = self._client.experiments.get_revision_details(artifact_id)
        elif artifact_type["pipeline"] is True:
            details = self._client.pipelines.get_revision_details(artifact_id)
        else:
            raise WMLClientError(
                "Getting artifact details failed. Artifact id: '{}' not found.".format(
                    artifact_id
                )
            )
        return details

    @inherited_docstring(Models.list_revisions)
    def list_models_revisions(
        self, model_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        return self._client._models.list_revisions(model_id, limit=limit, **kwargs)

    @inherited_docstring(Pipelines.list_revisions)
    def list_pipelines_revisions(
        self, pipeline_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        return self._client.pipelines.list_revisions(pipeline_id, limit=limit)

    @inherited_docstring(Functions.list_revisions)
    def list_functions_revisions(
        self, function_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        return self._client._functions.list_revisions(
            function_id, limit=limit, **kwargs
        )

    @inherited_docstring(Experiments.list_revisions)
    def list_experiments_revisions(
        self, experiment_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        return self._client.experiments.list_revisions(
            experiment_id, limit=limit, **kwargs
        )

    @inherited_docstring(Models.promote)
    def promote_model(
        self, model_id: str, source_project_id: str, target_space_id: str
    ) -> str:  # deprecated
        return self._client._models.promote(
            model_id, source_project_id, target_space_id
        )

    @inherited_docstring(AIServices.store)
    def store_ai_service(
        self, ai_service: str | Callable, meta_props: dict[str, Any]
    ) -> dict:
        return self._client._ai_services.store(ai_service, meta_props)

    @inherited_docstring(AIServices.get_details)
    def get_ai_service_details(
        self,
        ai_service_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        ai_service_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        return self._client._ai_services.get_details(
            ai_service_id=ai_service_id,
            limit=limit,
            asynchronous=asynchronous,
            get_all=get_all,
            spec_state=spec_state,
            ai_service_name=ai_service_name,
        )

    @inherited_docstring(AIServices.update)
    def update_ai_service(
        self,
        ai_service_id: str,
        changes: dict,
        update_ai_service: str | Callable | None = None,
    ) -> dict:
        return self._client._ai_services.update(
            ai_service_id, changes, update_ai_service
        )

    @staticmethod
    @inherited_docstring(AIServices.get_id)
    def get_ai_service_id(ai_service_details: dict) -> str:
        return AIServices.get_id(ai_service_details)

    @inherited_docstring(AIServices.list)
    def list_ai_services(self, limit: int | None = None) -> pandas.DataFrame:
        return self._client._ai_services.list(limit=limit)

    @inherited_docstring(AIServices.create_revision)
    def create_ai_service_revision(self, ai_service_id: str, **kwargs: Any) -> dict:
        return self._client._ai_services.create_revision(
            ai_service_id=ai_service_id, **kwargs
        )

    @inherited_docstring(AIServices.get_revision_details)
    def get_ai_service_revision_details(
        self, ai_service_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        return self._client._ai_services.get_revision_details(
            ai_service_id, rev_id, **kwargs
        )

    @inherited_docstring(AIServices.list_revisions)
    def list_ai_service_revisions(
        self, ai_service_id: str, limit: int | None = None
    ) -> pandas.DataFrame:
        return self._client._ai_services.list_revisions(ai_service_id, limit=limit)
