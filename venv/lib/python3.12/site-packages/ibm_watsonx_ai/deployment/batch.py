#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

import io
import os
import time
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from pandas import DataFrame

from .base_deployment import BaseDeployment
from ..helpers import DataConnection, AssetLocation
from ..utils import StatusLogger, print_text_header_h1
from ..utils.autoai.connection import (
    validate_source_data_connections,
    validate_deployment_output_connection,
)
from ..utils.autoai.utils import convert_dataframe_to_fields_values_payload
from ..utils.autoai.errors import NoneDataConnection
from ..utils.deployment.errors import BatchJobFailed, MissingScoringResults
from ..wml_client_error import WMLClientError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from pandas import DataFrame
    from numpy import ndarray
    from ..workspace import WorkSpace
    from ..credentials import Credentials

__all__ = ["Batch"]


class Batch(BaseDeployment):
    """The Batch Deployment class.
    With this class object, you can manage any batch deployment.

    :param source_instance_credentials: credentials to the instance where the training was performed
    :type source_instance_credentials: dict

    :param source_project_id: ID of the Watson Studio project where the training was performed
    :type source_project_id: str, optional

    :param source_space_id: ID of the Watson Studio Space where the training was performed
    :type source_space_id: str, optional

    :param target_instance_credentials: credentials to the instance where you want to deploy
    :type target_instance_credentials: dict

    :param target_project_id: ID of the Watson Studio project where you want to deploy
    :type target_project_id: str, optional

    :param target_space_id: ID of the Watson Studio Space where you want to deploy
    :type target_space_id: str, optional

    """

    def __init__(
        self,
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

        super().__init__(
            deployment_type="batch",
            source_wml_credentials=kwargs.get("source_wml_credentials"),
            source_project_id=source_project_id,
            source_space_id=source_space_id,
            target_wml_credentials=kwargs.get("target_wml_credentials"),
            target_project_id=target_project_id,
            target_space_id=target_space_id,
            project_id=project_id,
            space_id=space_id,
            source_instance_credentials=source_instance_credentials,
            target_instance_credentials=target_instance_credentials,
        )

        self.name = None
        self.id = None
        self.asset_id: str | None = None

    def __repr__(self) -> str:
        return f"name: {self.name}, id: {self.id}, asset_id: {self.asset_id}"

    def __str__(self) -> str:
        return f"name: {self.name}, id: {self.id}, asset_id: {self.asset_id}"

    def score(self, **kwargs: Any) -> dict:
        raise NotImplementedError("Batch deployment supports only job runs.")

    def create(  # type: ignore[override]
        self,
        model: str,
        deployment_name: str,
        metadata: dict | None = None,
        training_data: DataFrame | ndarray | None = None,
        training_target: DataFrame | ndarray | None = None,
        experiment_run_id: str | None = None,
        hardware_spec: str | None = None,
    ) -> None:
        """Create a deployment from a model.

        :param model: name of the AutoAI model
        :type model: str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param training_data: training data for the model
        :type training_data: pandas.DataFrame or numpy.ndarray, optional

        :param training_target: target/label data for the model
        :type training_target: pandas.DataFrame or numpy.ndarray, optional

        :param metadata: meta properties of the model
        :type metadata: dict, optional

        :param experiment_run_id: ID of a training/experiment (only applicable for AutoAI deployments)
        :type experiment_run_id: str, optional

        :param hardware_spec: hardware specification name of the deployment
        :type hardware_spec: str, optional

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.deployment import Batch

            deployment = Batch(
                    source_instance_credentials=Credentials(...),
                    source_project_id="...",
                    target_space_id="...")

            deployment.create(
                   experiment_run_id="...",
                   model=model,
                   deployment_name='My new deployment'
                   hardware_spec='L'
               )
        """
        return super().create(
            model=model,
            deployment_name=deployment_name,
            metadata=metadata,
            training_data=training_data,
            training_target=training_target,
            experiment_run_id=experiment_run_id,
            deployment_type="batch",
            hardware_spec=hardware_spec,
        )

    @BaseDeployment._project_to_space_to_project
    def get_params(self) -> dict:
        """Get deployment parameters."""
        return super().get_params()

    @BaseDeployment._project_to_space_to_project
    def run_job(
        self,
        payload: (
            DataFrame
            | list[DataConnection]
            | dict[str, DataFrame]
            | dict[str, DataConnection]
        ) = pd.DataFrame(),
        output_data_reference: DataConnection | None = None,
        transaction_id: str | None = None,
        background_mode: bool = True,
        hardware_spec: str | None = None,
    ) -> dict | DataConnection:
        """Batch scoring job. Payload or Payload data reference is required.
        Passed to the Service where the model has been deployed.

        :param payload: DataFrame that contains data to test the model or data storage connection details
            that inform the model where the payload data is stored
        :type payload: pandas.DataFrame or List[DataConnection] or Dict

        :param output_data_reference: DataConnection to the output COS for storing predictions,
             required only when DataConnections are used as a payload
        :type output_data_reference: DataConnection, optional

        :param transaction_id: ID under which the records should be saved in the payload table
            in IBM OpenScale
        :type transaction_id: str, optional

        :param background_mode: indicator whether the score() method will run in the background (async) or (sync)
        :type background_mode: bool, optional

        :param hardware_spec: hardware specification name for the scoring job
        :type hardware_spec: str, optional

        :return: details of the scoring job
        :rtype: dict

        **Examples**

        .. code-block:: python

            score_details = batch_service.run_job(payload=test_data)
            print(score_details['entity']['scoring'])

            # Result:
            # {'input_data': [{'fields': ['sepal_length',
            #               'sepal_width',
            #               'petal_length',
            #               'petal_width'],
            #              'values': [[4.9, 3.0, 1.4, 0.2]]}],
            # 'predictions': [{'fields': ['prediction', 'probability'],
            #               'values': [['setosa',
            #                 [0.9999320742502246,
            #                  5.1519823540224506e-05,
            #                  1.6405926235405522e-05]]]}]

            payload_reference = DataConnection(location=DSLocation(asset_id=asset_id))
            score_details = batch_service.run_job(payload=payload_reference, output_data_filename = "scoring_output.csv")
            score_details = batch_service.run_job(payload={'observations': payload_reference})
            score_details = batch_service.run_job(payload=[payload_reference])
            score_details = batch_service.run_job(payload={'observations': payload_reference, 'supporting_features': supporting_features_reference})  # supporting features time series forecasting sceanrio
            score_details = batch_service.run_job(payload=test_df, hardware_spec='S')
            score_details = batch_service.run_job(payload=test_df, hardware_spec=TShirtSize.L)
        """
        self._target_workspace: WorkSpace
        input_data: list[DataConnection] | list[dict]
        scoring_payload: dict
        if isinstance(payload, dict):
            observations = payload.get("observations", pd.DataFrame())
            supporting_features = payload.get("supporting_features")

            if isinstance(observations, DataFrame) and (
                isinstance(supporting_features, DataFrame)
                or supporting_features is None
            ):
                observations_payload = convert_dataframe_to_fields_values_payload(
                    observations, return_values_only=True
                )
                observations_payload["id"] = "observations"
                input_data = [observations_payload]

                if supporting_features is not None:
                    supporting_features_payload = (
                        convert_dataframe_to_fields_values_payload(
                            supporting_features, return_values_only=True
                        )
                    )
                    supporting_features_payload["id"] = "supporting_features"
                    input_data.append(supporting_features_payload)

                scoring_payload = {
                    self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA: input_data
                }

            elif isinstance(observations, DataConnection) and (
                isinstance(supporting_features, DataConnection)
                or supporting_features is None
            ):

                observations.id = "observations"
                input_data = [observations]
                if supporting_features is not None:
                    supporting_features.id = "supporting_features"
                    input_data.append(supporting_features)

                for data_conn in input_data:
                    if hasattr(data_conn, "location") and isinstance(
                        data_conn.location, AssetLocation
                    ):
                        data_conn.location.api_client = (
                            self._target_workspace.api_client
                        )

                input_data = validate_source_data_connections(
                    source_data_connections=input_data,
                    workspace=self._target_workspace,
                    deployment=True,
                )
                input_data = [
                    data_connection._to_dict() for data_connection in input_data
                ]

                if output_data_reference is None:
                    raise ValueError('"output_data_reference" should be provided.')

                if isinstance(output_data_reference, DataConnection):

                    # api_client sets correct href for Data Assets
                    if hasattr(output_data_reference, "location") and isinstance(
                        output_data_reference.location, AssetLocation
                    ):
                        output_data_reference.location.api_client = (
                            self._target_workspace.api_client
                        )

                    input_data = cast(list[DataConnection], input_data)
                    output_data_reference = validate_deployment_output_connection(
                        results_data_connection=output_data_reference,
                        workspace=self._target_workspace,
                        source_data_connections=input_data,
                    )
                    output_data_reference = output_data_reference._to_dict()  # type: ignore[assignment]

                input_data = cast(list[dict], input_data)
                scoring_payload = {
                    self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: input_data,
                    self._target_workspace.api_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE: output_data_reference,
                }

            else:
                raise TypeError(
                    "Missing data observations in payload or observations "
                    "or supporting_features are not pandas.DataFrames."
                )
        # note: support for DataFrame payload
        elif isinstance(payload, DataFrame):
            scoring_payload = {
                self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA: [
                    {"values": payload}
                ]
            }
        # note: support for DataConnections and dictionaries payload
        elif isinstance(payload, list):
            if isinstance(payload[0], DataConnection):
                if None in payload:
                    raise NoneDataConnection("payload")

                # api_client sets correct href for Data Assets
                for data_conn in payload:
                    if hasattr(data_conn, "location") and isinstance(
                        data_conn.location, AssetLocation
                    ):
                        data_conn.location.api_client = (
                            self._target_workspace.api_client
                        )

                payload = [
                    new_conn
                    for conn in payload
                    for new_conn in conn._subdivide_connection()
                ]
                payload = validate_source_data_connections(
                    source_data_connections=payload,
                    workspace=self._target_workspace,
                    deployment=True,
                )
                payload = [data_connection._to_dict() for data_connection in payload]  # type: ignore[assignment]
            elif isinstance(payload[0], dict):
                pass
            else:
                raise ValueError(
                    f"Current payload type: list of {type(payload[0])} is not supported."
                )

            if output_data_reference is None:
                raise ValueError('"output_data_reference" should be provided.')

            if isinstance(output_data_reference, DataConnection):

                # api_client sets correct href for Data Assets
                if hasattr(output_data_reference, "location") and isinstance(
                    output_data_reference.location, AssetLocation
                ):
                    output_data_reference.location.api_client = (
                        self._target_workspace.api_client
                    )

                output_data_reference = validate_deployment_output_connection(
                    results_data_connection=output_data_reference,
                    workspace=self._target_workspace,
                    source_data_connections=payload,
                )
                output_data_reference = output_data_reference._to_dict()  # type: ignore[assignment]

            scoring_payload = {
                self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES: payload,
                self._target_workspace.api_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE: output_data_reference,
            }

        else:
            raise ValueError(
                f"Incorrect payload type. Required: DataFrame or List[DataConnection], Passed: {type(payload)}"
            )

        if hardware_spec:
            hw_spec = [
                {
                    "node_runtime_id": "auto_ai.kb",
                    "hardware_spec": {"name": hardware_spec.upper()},
                }
            ]
        else:  # default
            details = self._target_workspace.api_client.deployments.get_details(self.id)
            hw_spec = details.get("entity", {}).get("hybrid_pipeline_hardware_specs")

        scoring_payload["hybrid_pipeline_hardware_specs"] = hw_spec

        self.id = cast(str, self.id)
        job_details = self._target_workspace.api_client.deployments.create_job(
            self.id, scoring_payload, _asset_id=self.asset_id
        )
        job_details = cast(dict, job_details)
        if background_mode:
            return job_details

        else:
            # note: monitor scoring job

            job_id = self._target_workspace.api_client.deployments.get_job_id(
                job_details
            )
            print_text_header_h1(
                "Synchronous scoring for id: '{}' started".format(job_id)
            )

            status = self.get_job_status(job_id)["state"]

            with StatusLogger(status) as status_logger:
                while status not in ["failed", "error", "completed", "canceled"]:
                    time.sleep(10)
                    status = self.get_job_status(job_id)["state"]
                    status_logger.log_state(status)
            # --- end note

            if "completed" in status:
                print("\nScoring job  '{}' finished successfully.".format(job_id))
            else:
                raise BatchJobFailed(
                    job_id,
                    f"Scoring job failed with status: {self.get_job_status(job_id)}",
                )

            return self.get_job_params(job_id)

    @BaseDeployment._project_to_space_to_project
    def rerun_job(
        self, scoring_job_id: str, background_mode: bool = True
    ) -> dict | DataFrame | DataConnection:
        """Rerun scoring job with the same parameters as job described by `scoring_job_id`.

        :param scoring_job_id: ID of the described scoring job
        :type scoring_job_id: str

        :param background_mode: indicator whether the score_rerun() method will run in the background (async) or (sync)
        :type background_mode: bool, optional

        :return: details of the scoring job
        :rtype: dict

        **Example:**

        .. code-block:: python

            scoring_details = deployment.score_rerun(scoring_job_id)
        """
        scoring_params = self.get_job_params(scoring_job_id)["entity"]["scoring"]
        input_data_references = (
            self._target_workspace.api_client.deployments.ScoringMetaNames.INPUT_DATA_REFERENCES
        )
        output_data_reference = (
            self._target_workspace.api_client.deployments.ScoringMetaNames.OUTPUT_DATA_REFERENCE
        )

        if input_data_references in scoring_params:
            payload_ref = [
                input_ref for input_ref in scoring_params[input_data_references]
            ]

            if "href" in scoring_params[output_data_reference]["location"]:
                del scoring_params[output_data_reference]["location"]["href"]

            return self.run_job(
                payload=payload_ref,
                output_data_reference=scoring_params["output_data_reference"],
                background_mode=background_mode,
            )
        else:
            raise NotImplementedError(
                "'rerun_job' method supports only jobs with "
                "payload passed as a list of DataConnections. If you want to rerun job "
                "with payload passed directly, please use 'run_job' one more time."
            )

    @BaseDeployment._project_to_space_to_project
    def delete(self, deployment_id: str | None = None) -> None:
        """Delete a deployment.

        :param deployment_id: ID of the deployment to be deleted, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        **Example:**

        .. code-block:: python

            deployment = Batch(workspace=...)
            # Delete current deployment
            deployment.delete()
            # Or delete a specific deployment
            deployment.delete(deployment_id='...')
        """
        super().delete(deployment_id=deployment_id, deployment_type="batch")

    @BaseDeployment._project_to_space_to_project
    def list(self, limit: int | None = None) -> DataFrame:
        """List deployments.

        :param limit: set the limit for number of listed deployments,
            default is `None` (all deployments should be fetched)
        :type limit: int, optional

        :return: Pandas DataFrame with information about deployments
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            deployment = Batch(workspace=...)
            deployments_list = deployment.list()
            print(deployments_list)

            # Result:
            #                  created_at  ...  status
            # 0  2020-03-06T10:50:49.401Z  ...   ready
            # 1  2020-03-06T13:16:09.789Z  ...   ready
            # 4  2020-03-11T14:46:36.035Z  ...  failed
            # 3  2020-03-11T14:49:55.052Z  ...  failed
            # 2  2020-03-11T15:13:53.708Z  ...   ready
        """
        return super().list(limit=limit, deployment_type="batch")

    @BaseDeployment._project_to_space_to_project
    def get(self, deployment_id: str) -> None:
        """Get a deployment.

        :param deployment_id: ID of the deployment
        :type deployment_id: str

        **Example:**

        .. code-block:: python

            deployment = Batch(workspace=...)
            deployment.get(deployment_id="...")
        """
        super().get(deployment_id=deployment_id, deployment_type="batch")

    @BaseDeployment._project_to_space_to_project
    def get_job_params(self, scoring_job_id: str | None = None) -> dict:
        """Get batch deployment job parameters.

        :param scoring_job_id: ID of the scoring job
        :type scoring_job_id: str

        :return: parameters of the scoring job
        :rtype: dict
        """
        return self._target_workspace.api_client.deployments.get_job_details(
            scoring_job_id
        )

    @BaseDeployment._project_to_space_to_project
    def get_job_status(self, scoring_job_id: str) -> dict:
        """Get the status of a scoring job.

        :param scoring_job_id: ID of the scoring job
        :type scoring_job_id: str

        :return: dictionary with state of the scoring job (one of: [completed, failed, starting, queued])
            and additional details if they exist
        :rtype: dict
        """
        return self._target_workspace.api_client.deployments.get_job_status(
            scoring_job_id
        )

    @BaseDeployment._project_to_space_to_project
    def get_job_result(self, scoring_job_id: str) -> DataFrame:
        """Get batch deployment results of a scoring job.

        :param scoring_job_id: ID of the scoring job
        :type scoring_job_id: str

        :return: batch deployment results of the scoring job
        :rtype: pandas.DataFrame

        :raises MissingScoringResults: in case of incompleted or failed job
            `MissingScoringResults` scoring exception is raised
        """
        scoring_params = self.get_job_params(scoring_job_id)["entity"]["scoring"]
        if scoring_params["status"]["state"] == "completed":
            if "predictions" in scoring_params:
                data = DataFrame(
                    scoring_params["predictions"][0]["values"],
                    columns=scoring_params["predictions"][0]["fields"],
                )
                return data
            else:
                conn = DataConnection._from_dict(
                    scoring_params["output_data_reference"]
                )
                conn._api_client = self._target_workspace.api_client

                return conn.read(
                    raw=True
                )  # if in future output may be excel file or with custom separator, here it should be recognized
        else:
            raise MissingScoringResults(
                scoring_job_id, reason="Scoring is not completed."
            )

    @BaseDeployment._project_to_space_to_project
    def get_job_id(self, batch_scoring_details: dict) -> str:
        """Get the ID from batch scoring details."""
        return self._target_workspace.api_client.deployments.get_job_id(
            batch_scoring_details
        )

    @BaseDeployment._project_to_space_to_project
    def list_jobs(self) -> DataFrame:
        """Returns pandas DataFrame with a list of deployment jobs"""

        resources = self._target_workspace.api_client.deployments.get_job_details()[
            "resources"
        ]
        columns = ["job id", "state", "creted", "deployment id"]
        values = []
        for scoring_details in resources:
            if "scoring" in scoring_details["entity"]:
                state = scoring_details["entity"]["scoring"]["status"]["state"]
                score_values = (
                    scoring_details["metadata"]["id"],
                    state,
                    scoring_details["metadata"]["created_at"],
                    scoring_details["entity"]["deployment"]["id"],
                )
                if self.id:
                    if self.id == scoring_details["entity"]["deployment"]["id"]:
                        values.append(score_values)
                else:
                    values.append(score_values)

        return DataFrame(values, columns=columns)

    @BaseDeployment._project_to_space_to_project
    def _deploy(
        self,
        pipeline_model: Pipeline,
        deployment_name: str,
        meta_props: dict,
        serving_name: (
            str | None
        ) = None,  # Not used, but added to match unified parameters for _deploy
        result_client: str | None = None,
        hardware_spec: str | None = None,
    ) -> dict:  # Not used, but added to match unified parameters for _deploy
        """Deploy a model into the Service.

        :param pipeline_model: model of the pipeline
        :type pipeline_model: Pipeline or str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param meta_props: meta properties of the model
        :type meta_props: dict

        :param result_client: tuple with a Result DataConnection object and an initialized COS client
        :type result_client: tuple[DataConnection, resource]

        :param hardware_spec: hardware specification for deployment
        :type hardware_spec: str, optional
        """
        deployment_details: dict | None = {}
        deployment_props: dict[str, Any]
        asset_uid = self._publish_model(
            pipeline_model=pipeline_model, meta_props=meta_props
        )

        self.asset_id = asset_uid

        deployment_props = {
            self._target_workspace.api_client.deployments.ConfigurationMetaNames.NAME: deployment_name,
            self._target_workspace.api_client.deployments.ConfigurationMetaNames.BATCH: {},
        }

        deployment_props[
            self._target_workspace.api_client.deployments.ConfigurationMetaNames.ASSET
        ] = {"id": asset_uid}

        deployment_props[
            self._target_workspace.api_client.deployments.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS
        ] = [
            {
                "node_runtime_id": "auto_ai.kb",
                "hardware_spec": {
                    "name": hardware_spec.upper() if hardware_spec else "M"
                },
            }
        ]

        print("Deploying model {} using V4 client.".format(asset_uid))
        try:
            deployment_details = self._target_workspace.api_client.deployments.create(
                artifact_uid=asset_uid,  # type: ignore[arg-type]
                meta_props=deployment_props,
            )

            deployment_details = cast(dict, deployment_details)
            self.deployment_id = self._target_workspace.api_client.deployments.get_id(
                deployment_details
            )

        except WMLClientError as e:
            raise e

        return deployment_details
