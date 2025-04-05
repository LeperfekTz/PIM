#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pandas as pd
from pandas import DataFrame

from .base_deployment import BaseDeployment
from ..wml_client_error import WMLClientError

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline
    from numpy import ndarray
    from ..workspace import WorkSpace
    from ..credentials import Credentials
    from ibm_watsonx_ai.helpers.connections import DataConnection
    from ibm_boto3 import resource

__all__ = ["WebService"]


class WebService(BaseDeployment):
    """WebService is an Online Deployment class.
    With this class object, you can manage any online (WebService) deployment.

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
            deployment_type="online",
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

        self.name: str | None = None
        self.scoring_url: str | None = None
        self.id: str | None = None
        self.asset_id: str | None = None

    def __repr__(self) -> str:
        return f"name: {self.name}, id: {self.id}, scoring_url: {self.scoring_url}, asset_id: {self.asset_id}"

    def __str__(self) -> str:
        return f"name: {self.name}, id: {self.id}, scoring_url: {self.scoring_url}, asset_id: {self.asset_id}"

    def create(  # type: ignore[override]
        self,
        model: str,
        deployment_name: str,
        serving_name: str | None = None,
        metadata: dict | None = None,
        training_data: DataFrame | ndarray | None = None,
        training_target: DataFrame | ndarray | None = None,
        experiment_run_id: str | None = None,
        hardware_spec: dict | None = None,
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

        :param serving_name: serving name of the deployment
        :type serving_name: str, optional

        :param metadata: meta properties of the model
        :type metadata: dict, optional

        :param experiment_run_id: ID of a training/experiment (only applicable for AutoAI deployments)
        :type experiment_run_id: str, optional

        :param hardware_spec: hardware specification for the deployment
        :type hardware_spec: dict, optional

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.deployment import WebService
            from ibm_watsonx_ai import Credentials

            deployment = WebService(
                    source_instance_credentials=Credentials(...),
                    source_project_id="...",
                    target_space_id="...")

            deployment.create(
                   experiment_run_id="...",
                   model=model,
                   deployment_name='My new deployment',
                   serving_name='my_new_deployment'
               )
        """
        return super().create(
            model=model,
            deployment_name=deployment_name,
            metadata=metadata,
            serving_name=serving_name,
            training_data=training_data,
            training_target=training_target,
            experiment_run_id=experiment_run_id,
            deployment_type="online",
            hardware_spec=hardware_spec,
        )

    @BaseDeployment._project_to_space_to_project
    def get_params(self) -> dict:
        """Get deployment parameters."""
        return super().get_params()

    @BaseDeployment._project_to_space_to_project
    def score(
        self,
        payload: dict | DataFrame = pd.DataFrame(),
        *,
        forecast_window: int | None = None,
        transaction_id: str | None = None,
    ) -> dict:
        """Online scoring. Payload is passed to the Service scoring endpoint where the model has been deployed.

        :param payload: DataFrame with data to test the model or dictionary with keys `observations`
            and `supporting_features`, and DataFrames with data for `observations` and `supporting_features`
            to score forecasting models
        :type payload: pandas.DataFrame or dict

        :param forecast_window: size of forecast window, supported only for forcasting, supported for CPD 5.0 and later
        :type forecast_window: int, optional

        :param transaction_id: ID under which the records should be saved in the payload table
            in IBM OpenScale
        :type transaction_id: str, optional

        :return: dictionary with list of model output/predicted targets
        :rtype: dict

        **Examples**

        .. code-block:: python

            predictions = web_service.score(payload=test_data)
            print(predictions)

            # Result:
            # {'predictions':
            #     [{
            #         'fields': ['prediction', 'probability'],
            #         'values': [['no', [0.9221385608558003, 0.07786143914419975]],
            #                   ['no', [0.9798324002736079, 0.020167599726392187]]
            #     }]}

            predictions = web_service.score(payload={'observations': new_observations_df})
            predictions = web_service.score(payload={'observations': new_observations_df, 'supporting_features': supporting_features_df}) # supporting features time series forecasting scenario
            predictions = web_service.score(payload={'observations': new_observations_df}
                                            forecast_window=1000) # forecast_window time series forecasting scenario
        """
        return super().score(
            payload=payload,
            forecast_window=forecast_window,
            transaction_id=transaction_id,
        )

    @BaseDeployment._project_to_space_to_project
    def delete(self, deployment_id: str | None = None) -> None:
        """Delete a deployment.

        :param deployment_id: ID of the deployment to be deleted, if empty, current deployment will be deleted
        :type deployment_id: str, optional

        **Example:**

        .. code-block:: python

            deployment = WebService(workspace=...)
            # Delete current deployment
            deployment.delete()
            # Or delete a specific deployment
            deployment.delete(deployment_id='...')
        """
        super().delete(deployment_id=deployment_id, deployment_type="online")

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

            deployment = WebService(workspace=...)
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
        return super().list(limit=limit, deployment_type="online")

    @BaseDeployment._project_to_space_to_project
    def get(self, deployment_id: str) -> None:
        """Get a deployment.

        :param deployment_id: ID of the deployment
        :type deployment_id: str

        **Example:**

        .. code-block:: python

            deployment = WebService(workspace=...)
            deployment.get(deployment_id="...")
        """
        super().get(deployment_id=deployment_id, deployment_type="online")

    @BaseDeployment._project_to_space_to_project
    def _deploy(
        self,
        pipeline_model: Pipeline,
        deployment_name: str,
        meta_props: dict,
        serving_name: str | None = None,
        result_client: tuple[DataConnection, resource] | None = None,
        hardware_spec: str | None = None,
    ) -> dict:
        """Deploy model into Service.

        :param pipeline_model: model of the pipeline to deploy
        :type pipeline_model: Pipeline or str

        :param deployment_name: name of the deployment
        :type deployment_name: str

        :param meta_props: meta properties of the model
        :type meta_props: dict

        :param serving_name: serving name of the deployment
        :type serving_name: str, optional

        :param result_client: tuple with a Result DataConnection object and an initialized COS client
        :rtype: tuple[DataConnection, resource]

        :return: details of the deployment
        :rtype: dict
        """
        from ..workspace import WorkSpace

        self._target_workspace = cast(WorkSpace, self._target_workspace)
        deployment_details: dict | None
        deployment_props: dict[str, Any]
        asset_uid = self._publish_model(
            pipeline_model=pipeline_model, meta_props=meta_props
        )

        self.asset_id = asset_uid

        conf_names = (
            self._target_workspace.api_client.deployments.ConfigurationMetaNames
        )

        deployment_props = {conf_names.NAME: deployment_name, conf_names.ONLINE: {}}

        if hardware_spec:
            deployment_props[conf_names.HARDWARE_SPEC] = hardware_spec

        if serving_name:
            deployment_props[conf_names.ONLINE]["parameters"] = {
                conf_names.SERVING_NAME: serving_name
            }

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
