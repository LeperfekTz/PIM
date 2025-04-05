#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

__all__ = ["LocalAutoPipelinesRuns"]

from typing import List, Dict, Union, TYPE_CHECKING, Optional

from pandas import DataFrame
from warnings import warn

from ibm_watsonx_ai.experiment.autoai.optimizers.local_auto_pipelines import (
    LocalAutoPipelines,
)
from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.utils.autoai.utils import prepare_cos_client
from .base_auto_pipelines_runs import BaseAutoPipelinesRuns


if TYPE_CHECKING:
    from ..optimizers import RemoteAutoPipelines


class LocalAutoPipelinesRuns(BaseAutoPipelinesRuns):
    """LocalAutoPipelinesRuns class is used to work with historical Optimizer runs (local optimizer and
    with data from COS, without API interaction).

    :param filter: filter, user can choose which runs to fetch specifying experiment name (option not yet available)
    :type filter: str, optional
    """

    def __init__(self, filter: str = None) -> None:
        self.experiment_name = filter
        self.training_data_reference = None
        self.training_result_reference = None

    def __call__(self, *, filter: str) -> "LocalAutoPipelinesRuns":
        raise NotImplementedError("Not yet implemented in the local scenario.")

    def list(self) -> "DataFrame":
        raise NotImplementedError("Not yet implemented in the local scenario.")

    def get_params(self, run_id: str = None) -> dict:
        raise NotImplementedError("Not yet implemented in the local scenario.")

    def get_run_details(self, run_id: str = None) -> dict:
        raise NotImplementedError("Not yet implemented in the local scenario.")

    def get_optimizer(
        self,
        run_id: Optional[str] = None,
        metadata: Dict[
            str, Union[List["DataConnection"], "DataConnection", str, int]
        ] = None,
        api_client: "APIClient" = None,
        **kwargs,
    ) -> Union["LocalAutoPipelines", "RemoteAutoPipelines"]:
        """Get historical optimizer from historical experiment.

        :param run_id: ID of the local historical experiment run (option not yet available)
        :type run_id: str, optional

        :param metadata: option to pass information about COS data reference
        :type metadata: dict, optional

        :param api_client: only for Container type to work properly
        :type api_client: ApiClient, optional

        **Example:**

        .. code-block:: python

            metadata = dict(
                   prediction_type ='classification',
                   prediction_column='species',
                   holdout_size=0.2,
                   scoring='roc_auc',
                   max_number_of_estimators=1,
                   training_data_reference = [DataConnection(
                       connection_asset_id=connection_id,
                       location=S3Location(
                           bucket='autoai-bucket',
                           path='iris_dataset.csv',
                       )
                   )],
                   training_result_reference = DataConnection(
                       connection_asset_id=connection_id,
                       location=S3Location(
                           bucket='autoai-bucket',
                           path='.',
                           model_location="0a8266be-0f3e-4ef9-af89-856022b7c1c9/data/automl/global_output/",
                           training_status="./75eec2e0-2600-4b7e-bcf2-ea54f2471400/9236e3ab-25e2-4daa-86a8-fd009d4e1e7d/training-status.json",
                       )
                   )
               )
            optimizer = AutoAI().runs.get_optimizer(metadata)
        """
        # note: backward compatibility
        if (wml_client := kwargs.get("wml_client")) is not None:
            if api_client is None:
                api_client = wml_client

            wml_client_deprecated_warning = (
                "`wml_client` is deprecated and will be removed in future. "
                "Instead, please use `api_client`."
            )
            warn(wml_client_deprecated_warning, category=DeprecationWarning)

        # --- end note
        if run_id is not None:
            raise NotImplementedError(
                "run_id option is not yet implemented in the local scenario."
            )

        else:
            training_result_reference = metadata.get("training_result_reference")

            # note: cloud auto-gen notebook scenario
            # note: save training connection to be able to further provide this data via get_data_connections
            self.training_data_reference: List["DataConnection"] = metadata.get(
                "training_data_references", metadata.get("training_data_reference")
            )
            self.training_result_reference: "DataConnection" = training_result_reference

            # note: fill experiment parameters to be able to recreate holdout split
            for data in self.training_data_reference:
                data._fill_experiment_parameters(
                    prediction_type=metadata.get("prediction_type"),
                    prediction_column=metadata.get("prediction_column"),
                    holdout_size=metadata.get("holdout_size"),
                    csv_separator=metadata.get("csv_separator", ","),
                    excel_sheet=metadata.get("excel_sheet", 0),
                    encoding=metadata.get("encoding", "utf-8"),
                )

            self.training_result_reference._fill_experiment_parameters(
                prediction_type=metadata.get("prediction_type"),
                prediction_column=metadata.get("prediction_column"),
                holdout_size=metadata.get("holdout_size"),
                csv_separator=metadata.get("csv_separator", ","),
                excel_sheet=metadata.get("excel_sheet", 0),
                encoding=metadata.get("encoding", "utf-8"),
            )
            # --- end note

            # Note: We need to fetch credentials when 'container' is the type
            if (
                hasattr(self.training_result_reference, "type")
                and (
                    self.training_result_reference.type == "container"
                    or self.training_result_reference.type == "data_asset"
                    or self.training_result_reference.type == "connection_asset"
                )
                and api_client is not None
            ):
                self.training_result_reference.set_client(api_client)

            for data_ref in self.training_data_reference:
                if (
                    hasattr(data_ref, "type")
                    and (
                        data_ref.type == "container"
                        or data_ref.type == "connection_asset"
                        or data_ref.type == "data_asset"
                    )
                    and api_client is not None
                ):
                    data_ref.set_client(api_client)
            # --- end note

            data_clients, result_client = prepare_cos_client(
                training_data_references=self.training_data_reference,
                training_result_reference=self.training_result_reference,
            )

            optimizer = LocalAutoPipelines(
                name="Auto-gen notebook from COS",
                prediction_type=metadata.get("prediction_type"),
                prediction_column=metadata.get("prediction_column"),
                scoring=metadata.get("scoring"),
                holdout_size=metadata.get("holdout_size"),
                max_num_daub_ensembles=metadata.get("max_number_of_estimators"),
                _data_clients=data_clients,
                _result_client=result_client,
            )
            optimizer._training_data_reference = self.training_data_reference
            optimizer._training_result_reference = self.training_result_reference

            return optimizer
            # --- end note

    def get_data_connections(self, run_id: str) -> List["DataConnection"]:
        raise NotImplementedError("Not yet implemented in the local scenario.")
