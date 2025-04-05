#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Any, cast, Callable
from warnings import warn
from contextlib import redirect_stdout

from pandas import DataFrame
from ..credentials import Credentials

from ..experiment import AutoAI
from ..utils import is_lale_pipeline
from ..utils.autoai.utils import (
    prepare_auto_ai_model_to_publish,
    prepare_auto_ai_model_to_publish_normal_scenario,
    remove_file,
    prepare_auto_ai_model_to_publish_notebook_normal_scenario,
    check_if_ts_pipeline_is_winner,
    convert_dataframe_to_fields_values_payload,
)
from ..utils.deployment.errors import (
    WrongDeploymnetType,
    ModelTypeNotSupported,
    NotAutoAIExperiment,
    MissingSpace,
    ServingNameNotAvailable,
)
from ..helpers import DataConnection
from ..workspace import WorkSpace

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

__all__ = ["BaseDeployment"]


class BaseDeployment(ABC):
    """Base abstract class for Deployment."""

    def __init__(
        self,
        deployment_type: str,
        source_instance_credentials: Credentials | WorkSpace | None = None,
        source_project_id: str | None = None,
        source_space_id: str | None = None,
        target_instance_credentials: Credentials | WorkSpace | None = None,
        target_project_id: str | None = None,
        target_space_id: str | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        **kwargs: Any,
    ):

        if space_id is None and source_space_id is None and target_space_id is None:
            raise MissingSpace(
                reason="Any of the [space_id, source_space_id, target_space_id] is not specified."
            )

        # note: backward compatibility
        if (source_wml_credentials := kwargs.get("source_wml_credentials")) is not None:
            if not source_instance_credentials:
                source_instance_credentials = source_wml_credentials

            source_wml_credentials_deprecated_warning = (
                "`source_wml_credentials` is deprecated and will be removed in future. "
                "Instead, please use `source_instance_credentials`."
            )
            warn(source_wml_credentials_deprecated_warning, category=DeprecationWarning)

        if (target_wml_credentials := kwargs.get("target_wml_credentials")) is not None:
            if not target_instance_credentials:
                target_instance_credentials = target_wml_credentials

            target_wml_credentials_deprecated_warning = (
                "`target_wml_credentials` is deprecated and will be removed in future. "
                "Instead, please use `target_instance_credentials`."
            )
            warn(target_wml_credentials_deprecated_warning, category=DeprecationWarning)

        if project_id is not None:
            source_project_id = project_id
            project_id_parameter_deprecated_warning = (
                "`project_id` parameter is deprecated, please use `source_project_id`"
            )
            warn(project_id_parameter_deprecated_warning, category=DeprecationWarning)

        if space_id is not None:
            source_space_id = space_id
            space_id_parameter_deprecated_warning = (
                "`space_id` parameter is deprecated, please use `source_space_id`"
            )
            warn(space_id_parameter_deprecated_warning)
        # --- end note

        # note: as workspace is not clear enough to understand, there is a possibility to use pure
        # credentials with project and space IDs, but in addition we
        # leave a possibility to use a previous WorkSpace implementation, it could be passed as a first argument
        self._source_workspace: WorkSpace | None
        target_project_id = cast(str, target_project_id)
        target_space_id = cast(str, target_space_id)
        source_project_id = cast(str, source_project_id)
        source_space_id = cast(str, source_space_id)

        if isinstance(source_instance_credentials, dict):  # backward compatibility
            source_instance_credentials = Credentials.from_dict(
                source_instance_credentials
            )

        if isinstance(source_instance_credentials, WorkSpace):
            self._source_workspace = source_instance_credentials

        elif isinstance(source_instance_credentials, Credentials):
            self._source_workspace = WorkSpace(
                credentials=copy(source_instance_credentials),
                project_id=source_project_id,
                space_id=source_space_id,
            )
        else:
            self._source_workspace = None

        if isinstance(target_instance_credentials, dict):  # backward compatibility
            target_instance_credentials = Credentials.from_dict(
                target_instance_credentials
            )

        if target_instance_credentials is None:
            self._target_workspace: WorkSpace | None
            if isinstance(source_instance_credentials, WorkSpace):
                self._target_workspace = WorkSpace(
                    credentials=source_instance_credentials.credentials,
                    space_id=(
                        target_space_id
                        if target_space_id is not None
                        else source_instance_credentials.space_id
                    ),
                )

            elif isinstance(source_instance_credentials, Credentials):
                self._target_workspace = WorkSpace(
                    credentials=copy(source_instance_credentials),
                    space_id=(
                        target_space_id
                        if target_space_id is not None
                        else source_space_id
                    ),
                )
            else:
                self._target_workspace = None

            self._target_workspace = cast(WorkSpace, self._target_workspace)
            if target_space_id or target_project_id:
                self._target_workspace.project_id = target_project_id
                self._target_workspace.space_id = target_space_id

        else:
            if isinstance(target_instance_credentials, WorkSpace):
                self._target_workspace = target_instance_credentials

            elif isinstance(target_instance_credentials, Credentials):
                self._target_workspace = WorkSpace(
                    credentials=copy(target_instance_credentials),
                    project_id=target_project_id,
                    space_id=target_space_id,
                )

                # note: only if user provides target instance information
                if self._source_workspace is None:
                    self._source_workspace = copy(self._target_workspace)

            else:
                self._target_workspace = None
        # --- end note

        if self._source_workspace is None and self._target_workspace is None:
            raise TypeError(
                f"{self.__class__.__name__} is missing one of the parameters: "
                f"['source_instance_credentials', 'target_instance_credentials']"
            )

        self.name: str | None = None
        self.id: str | None = None
        if deployment_type == "online":
            self.scoring_url: str | None = None

    def __repr__(self) -> str:
        return f"name: {self.name}, id: {self.id}"

    def __str__(self) -> str:
        return f"name: {self.name}, id: {self.id}"

    @abstractmethod
    def create(self, **kwargs: Any) -> None:
        """Create a deployment from a model.

        :param model: AutoAI model name
        :type model: str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param training_data: training data for the model
        :type training_data: pandas.DataFrame or numpy.ndarray, optional

        :param training_target: target/label data for the model
        :type training_target: pandas.DataFrame or numpy.ndarray, optional

        :param metadata: model meta properties
        :type metadata: dict, optional

        :param experiment_run_id: ID of a training/experiment (only applicable for AutoAI deployments)
        :type experiment_run_id: str, optional

        :param hardware_spec: hardware specification for deployment
        :type hardware_spec: str, optional
        """
        model_props: dict
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        self._source_workspace = cast(WorkSpace, self._source_workspace)

        if kwargs.get("serving_name"):
            (
                status_code,
                response,
            ) = self._target_workspace.api_client.deployments._get_serving_name_info(
                str(kwargs.get("serving_name"))
            )

            if status_code == 409:
                raise ServingNameNotAvailable(response["errors"][0])

        # note: This section is only for deployments with specified experiment_id
        if kwargs["experiment_run_id"] is not None:
            run_params = self._source_workspace.api_client.training.get_details(
                training_id=kwargs["experiment_run_id"], _internal=True
            )
            pipeline_details = self._source_workspace.api_client.pipelines.get_details(
                run_params["entity"]["pipeline"]["id"]
            )

            if not (
                "autoai" in str(pipeline_details) or "auto_ai" in str(pipeline_details)
            ):
                raise NotAutoAIExperiment(
                    kwargs["experiment_run_id"],
                    reason="Currently WebService class supports only AutoAI models.",
                )

            print("Preparing an AutoAI Deployment...")
            # TODO: remove part with model object depployment
            if not isinstance(kwargs["model"], str):
                passing_object_deprecation_warning = (
                    "Depreciation Warning: Passing an object will no longer be supported. "
                    "Please specify the AutoAI model name to deploy."
                )
                warn(passing_object_deprecation_warning, category=DeprecationWarning)

            if is_lale_pipeline(kwargs["model"]):
                model = kwargs["model"].export_to_sklearn_pipeline()
            else:
                model = kwargs["model"]

            # note: check if model is of lale type, if yes, convert it back to scikit
            if not isinstance(kwargs["model"], str):
                model_props = {
                    self._target_workspace.api_client.repository.ModelMetaNames.NAME: f"{kwargs['deployment_name']} Model",
                    self._target_workspace.api_client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                    self._target_workspace.api_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: self._target_workspace.api_client.software_specifications.get_id_by_name(
                        "hybrid_0.1"
                    ),
                }

                schema, artifact_name = prepare_auto_ai_model_to_publish(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=kwargs["experiment_run_id"],
                    api_client=self._source_workspace.api_client,
                )

                model_props[
                    self._target_workspace.api_client.repository.ModelMetaNames.INPUT_DATA_SCHEMA
                ] = [schema]

            else:
                # raise an error when TS pipeline is discarded one
                check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

                # Note: We need to fetch credentials when 'container' is the type
                if run_params["entity"]["results_reference"]["type"] == "container":
                    data_connection = DataConnection._from_dict(
                        _dict=run_params["entity"]["results_reference"]
                    )
                    data_connection._api_client = self._source_workspace.api_client
                else:
                    data_connection = None
                # --- end note

                try:
                    auto_pipelines_parameters = (
                        pipeline_details.get("entity", {})
                        .get("document", {})
                        .get("pipelines", [])[0]
                        .get("nodes", [])[0]
                        .get("parameters")
                    )
                except:
                    auto_pipelines_parameters = None

                data_connection = cast(DataConnection, data_connection)

                (
                    artifact_name,
                    model_props,
                ) = prepare_auto_ai_model_to_publish_normal_scenario(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=kwargs["experiment_run_id"],
                    api_client=self._source_workspace.api_client,
                    space_id=self._target_workspace.space_id,
                    result_reference=data_connection,
                    auto_pipelines_parameters=auto_pipelines_parameters,
                )

            deployment_details = self._deploy(
                pipeline_model=artifact_name,
                deployment_name=kwargs["deployment_name"],
                serving_name=kwargs.get("serving_name"),
                meta_props=model_props,
                hardware_spec=kwargs.get("hardware_spec"),
            )

            remove_file(filename=artifact_name)

            self.name = kwargs["deployment_name"]
            self.id = deployment_details["metadata"].get("id")
            if kwargs["deployment_type"] == "online":
                deployment_details = cast(dict, deployment_details)
                self.scoring_url = (
                    self._target_workspace.api_client.deployments.get_scoring_href(
                        deployment_details
                    )
                )
        # --- end note

        # note: This section is for deployments from auto-gen notebook with COS connection
        else:
            # note: only if we have COS connections from the notebook
            if kwargs.get("metadata") is not None:
                print("Preparing an AutoAI Deployment...")
                # note: CP4D
                if (
                    self._source_workspace is not None
                    and self._source_workspace.api_client.ICP_PLATFORM_SPACES
                ):
                    optimizer = AutoAI(self._source_workspace).runs.get_optimizer(metadata=kwargs["metadata"])  # type: ignore[attr-defined]
                # note: CLOUD
                else:
                    optimizer = AutoAI().runs.get_optimizer(  # type: ignore[attr-defined]
                        metadata=kwargs["metadata"],
                        api_client=self._source_workspace.api_client,
                    )

                # note: only when user did not pass WMLS credentials during Service initialization
                if self._source_workspace is None:
                    self._source_workspace = copy(optimizer._workspace)

                if self._target_workspace is None:
                    self._target_workspace = copy(optimizer._workspace)
                # --- end note

                # TODO: remove part with model object depployment
                if not isinstance(kwargs["model"], str):
                    passing_object_deprecation_warning = (
                        "Depreciation Warning: Passing an object will no longer be supported. "
                        "Please specify the AutoAI model name to deploy."
                    )
                    warn(passing_object_deprecation_warning, category=DeprecationWarning)  # fmt: skip

                if is_lale_pipeline(kwargs["model"]):
                    model = kwargs["model"].export_to_sklearn_pipeline()
                else:
                    model = kwargs["model"]

                if not self._target_workspace.api_client.ICP_PLATFORM_SPACES:

                    optimizer_2 = AutoAI(self._source_workspace).runs.get_optimizer(metadata=kwargs["metadata"])  # type: ignore[attr-defined]
                    run_details = (
                        optimizer_2._workspace.api_client.training.get_details(
                            optimizer_2.get_params()["run_id"], _internal=True
                        )
                    )
                    self._target_workspace.space_id = cast(
                        str, self._target_workspace.space_id
                    )
                    (
                        artifact_name,
                        model_props,
                    ) = prepare_auto_ai_model_to_publish_notebook_normal_scenario(
                        pipeline_model=model,
                        result_connection=optimizer._result_client[0],
                        cos_client=optimizer._result_client[1],
                        run_params=run_details,
                        space_id=self._target_workspace.space_id,
                        auto_pipelines_parameters=optimizer.get_params(),
                    )

                    deployment_details = self._deploy(
                        pipeline_model=artifact_name,
                        deployment_name=kwargs["deployment_name"],
                        meta_props=model_props,
                        serving_name=kwargs.get("serving_name"),
                        result_client=optimizer._result_client,
                        hardware_spec=kwargs.get("hardware_spec"),
                    )

                # note: CP4D part
                else:
                    training_result_reference = kwargs["metadata"].get(
                        "training_result_reference"
                    )
                    run_id = training_result_reference.location.path.split("/")[-3]
                    run_params = self._source_workspace.api_client.training.get_details(
                        training_id=run_id, _internal=True
                    )

                    (
                        artifact_name,
                        model_props,
                    ) = prepare_auto_ai_model_to_publish_normal_scenario(
                        pipeline_model=model,
                        run_params=run_params,
                        run_id=run_id,
                        api_client=self._source_workspace.api_client,
                        space_id=self._target_workspace.space_id,
                        auto_pipelines_parameters=optimizer.get_params(),
                    )

                    deployment_details = self._deploy(
                        pipeline_model=artifact_name,
                        deployment_name=kwargs["deployment_name"],
                        serving_name=kwargs.get("serving_name"),
                        meta_props=model_props,
                        hardware_spec=kwargs.get("hardware_spec"),
                    )
                # --- end note
                remove_file(filename=artifact_name)

                self.name = kwargs["deployment_name"]
                self.id = deployment_details["metadata"].get("id")
                if kwargs["deployment_type"] == "online":
                    self.scoring_url = (
                        self._target_workspace.api_client.deployments.get_scoring_href(
                            deployment_details
                        )
                    )
            # --- end note

            else:
                raise ModelTypeNotSupported(
                    str(type(kwargs["model"])),
                    reason="Currently WebService class supports only AutoAI models.",
                )

    @abstractmethod
    def get_params(self) -> dict:
        """Get deployment parameters."""
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        return self._target_workspace.api_client.deployments.get_details(self.id)

    @abstractmethod
    def score(self, **kwargs: Any) -> dict:
        """Scoring on Service. Payload is passed to the scoring endpoint where the model has been deployed.

        :param payload: data to test the model
        :type payload: pandas.DataFrame
        """
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        if isinstance(kwargs["payload"], DataFrame):
            payload = convert_dataframe_to_fields_values_payload(kwargs["payload"])
            input_data = [payload]

        elif isinstance(kwargs["payload"], dict):
            observations_df = kwargs["payload"].get("observations", DataFrame())
            supporting_features_df = kwargs["payload"].get("supporting_features")

            if isinstance(observations_df, DataFrame) and (
                isinstance(supporting_features_df, DataFrame)
                or supporting_features_df is None
            ):
                observations_payload = convert_dataframe_to_fields_values_payload(
                    observations_df
                )
                observations_payload["id"] = "observations"
                input_data = [observations_payload]

                if supporting_features_df is not None:
                    supporting_features_payload = (
                        convert_dataframe_to_fields_values_payload(
                            supporting_features_df
                        )
                    )
                    supporting_features_payload["id"] = "supporting_features"
                    input_data.append(supporting_features_payload)
            else:
                raise TypeError(
                    "Missing data observations in payload or observations"
                    "or supporting_features are not pandas DataFrames."
                )

        else:
            raise TypeError(
                f"Scoring payload has invalid type: {type(kwargs['payload'])}. "
                f"Supported types are: pandas.DataFrame and dictionary."
            )

        transaction_id = kwargs.get("transaction_id")

        scoring_payload = {
            self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA: input_data
        }

        if kwargs.get("forecast_window") is not None:
            scoring_payload.update(
                {
                    self._target_workspace.api_client.deployments.ScoringMetaNames.SCORING_PARAMETERS: {
                        "forecast_window": kwargs.get("forecast_window")
                    }
                }
            )

        self.id = cast(str, self.id)
        score = self._target_workspace.api_client.deployments.score(
            self.id, scoring_payload, transaction_id=transaction_id
        )

        return score

    @abstractmethod
    def delete(self, **kwargs: Any) -> None:
        """Delete a deployment.

        :param deployment_id: ID of the deployment to delete, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        if kwargs["deployment_id"] is None:
            self._target_workspace.api_client.deployments.delete(self.id)
            self.name = None
            self.scoring_url = None
            self.id = None

        else:
            deployment_details = (
                self._target_workspace.api_client.deployments.get_details(
                    deployment_uid=kwargs["deployment_id"]
                )
            )
            if (
                deployment_details.get("entity", {}).get(kwargs["deployment_type"])
                is not None
            ):
                self._target_workspace.api_client.deployments.delete(
                    kwargs["deployment_id"]
                )

            else:
                raise WrongDeploymnetType(
                    f"{kwargs['deployment_type']}",
                    reason=f"Deployment with ID: {kwargs['deployment_id']} is not of \"{kwargs['deployment_type']}\" type!",
                )

    @abstractmethod
    def list(self, **kwargs: Any) -> DataFrame:
        """List deployments.

        :param limit: set the limit of how many deployments to list,
            default is `None` (all deployments should be fetched)
        :type limit: int, optional

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        deployments = self._target_workspace.api_client.deployments.get_details(
            limit=kwargs["limit"]
        )
        columns = ["created_at", "modified_at", "id", "name", "status"]

        data = [
            [
                deployment.get("metadata")["created_at"],
                deployment.get("metadata")["modified_at"],
                deployment.get("metadata")["id"],
                deployment.get("metadata")["name"],
                deployment.get("entity")["status"]["state"],
            ]
            for deployment in deployments.get("resources", [])
            if isinstance(
                deployment.get("entity", {}).get(kwargs["deployment_type"]), dict
            )
        ]

        deployments_df = DataFrame(data=data, columns=columns).sort_values(
            by=["created_at"], ascending=False
        )
        return deployments_df.head(n=kwargs["limit"])

    @abstractmethod
    def get(self, **kwargs: Any) -> None:
        """Get a deployment.

        :param deployment_id: ID of the deployment to work with
        :type deployment_id: str

        :param deployment_type: type of the deployment: [online, batch]
        :type deployment_type: str
        """
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        deployment_details = self._target_workspace.api_client.deployments.get_details(
            deployment_uid=kwargs["deployment_id"]
        )
        if (
            deployment_details.get("entity", {}).get(kwargs["deployment_type"])
            is not None
        ):
            self.name = deployment_details["metadata"].get("name")
            self.id = deployment_details["metadata"].get("id")
            if kwargs["deployment_type"] == "online":
                self.scoring_url = (
                    self._target_workspace.api_client.deployments.get_scoring_href(
                        deployment_details
                    )
                )

        else:
            raise WrongDeploymnetType(
                f"{kwargs['deployment_type']}",
                reason=f"Deployment with ID: {kwargs['deployment_id']} is not of \"{kwargs['deployment_type']}\" type!",
            )

    @abstractmethod
    def _deploy(self, **kwargs: Any) -> dict:
        """Protected method to create a deployment."""
        pass

    def _publish_model(self, pipeline_model: Pipeline | str, meta_props: dict) -> str:
        """Publish model into Service.

        :param pipeline_model: model of the pipeline to publish
        :type pipeline_model: Pipeline or str

        :param meta_props: model meta properties
        :type meta_props: dict

        :return: asset id
        :rtype: str
        """

        # Note: publish model to project and then promote to space
        self._source_workspace = cast(WorkSpace, self._source_workspace)
        self._target_workspace = cast(WorkSpace, self._target_workspace)
        if (
            self._source_workspace.project_id
            and self._source_workspace.project_id in str(meta_props)
        ):
            published_model_details = (
                self._source_workspace.api_client.repository.store_model(
                    model=pipeline_model, meta_props=meta_props
                )
            )

            project_asset_id = (
                self._source_workspace.api_client.repository.get_model_id(
                    published_model_details
                )
            )

            self._target_workspace.space_id = cast(str, self._target_workspace.space_id)
            asset_id = self._source_workspace.api_client.spaces.promote(
                asset_id=project_asset_id,
                source_project_id=self._source_workspace.project_id,
                target_space_id=self._target_workspace.space_id,
            )

        else:
            published_model_details = (
                self._target_workspace.api_client.repository.store_model(
                    model=pipeline_model, meta_props=meta_props
                )
            )

            asset_id = self._target_workspace.api_client.repository.get_model_id(
                published_model_details
            )

        print(f"Published model uid: {asset_id}")
        return asset_id

    @staticmethod
    def _project_to_space_to_project(method: Callable) -> Callable:
        @wraps(method)
        def _method(
            self: BaseDeployment, *method_args: Any, **method_kwargs: Any
        ) -> Callable:
            self._target_workspace = cast(WorkSpace, self._target_workspace)
            with redirect_stdout(open(os.devnull, "w")):
                if self._target_workspace.space_id is not None:
                    (
                        self._target_workspace.api_client.set.default_space(
                            self._target_workspace.space_id
                        )
                        if self._target_workspace.api_client.ICP_PLATFORM_SPACES
                        else None
                    )

            try:
                return method(self, *method_args, **method_kwargs)

            finally:
                if self._target_workspace.project_id:
                    with redirect_stdout(open(os.devnull, "w")):
                        if self._target_workspace.project_id is not None:
                            (
                                self._target_workspace.api_client.set.default_project(
                                    self._target_workspace.project_id
                                )
                                if self._target_workspace.api_client.ICP_PLATFORM_SPACES
                                else None
                            )

        return _method
