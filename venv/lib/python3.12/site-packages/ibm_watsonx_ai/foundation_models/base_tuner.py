#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from abc import ABC
from typing import Literal, TYPE_CHECKING, cast

from ibm_watsonx_ai.helpers import (
    DataConnection,
    ContainerLocation,
    S3Connection,
    AssetLocation,
    FSLocation,
    S3Location,
)
from ibm_watsonx_ai.utils.autoai.errors import ContainerTypeNotSupported
import numpy as np

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class BaseTuner(ABC):
    tuning_type: Literal["prompt", "fine", "ilab"]
    from ibm_watsonx_ai import APIClient

    _client: APIClient = None  # type: ignore[assignment]

    def __init__(self, tuning_type: Literal["prompt", "fine", "ilab"]):
        self.tuning_type = tuning_type

    def _validate_source_data_connections(
        self, source_data_connections: list[DataConnection]
    ) -> list[DataConnection]:
        for data_connection in source_data_connections:
            if isinstance(data_connection.location, ContainerLocation):
                if self._client.ICP_PLATFORM_SPACES:
                    raise ContainerTypeNotSupported()  # block Container type on CPD
                elif isinstance(data_connection.connection, S3Connection):
                    # note: remove S3 inline credential from data asset before training
                    data_connection.connection = None
                    if hasattr(data_connection.location, "bucket"):
                        delattr(data_connection.location, "bucket")
                    # --- end note
            if isinstance(data_connection.connection, S3Connection) and isinstance(
                data_connection.location, AssetLocation
            ):
                # note: remove S3 inline credential from data asset before training
                data_connection.connection = None

                for s3_attr in ["bucket", "path"]:
                    if hasattr(data_connection.location, s3_attr):
                        delattr(data_connection.location, s3_attr)
                # --- end note

        return source_data_connections

    def _determine_result_reference(
        self,
        results_reference: DataConnection | None,
        data_references: list[DataConnection],
        result_path: str = "default_tuning_output",
    ) -> DataConnection:
        # note: if user did not provide results storage information, use default ones
        if results_reference is None:
            if self._client.ICP_PLATFORM_SPACES:
                location = FSLocation(
                    path=f"/{{option}}/{{id}}/assets/wx_{self.tuning_type}_tune"
                )
                if self._client.default_project_id is None:
                    location.path = location.path.format(
                        option="spaces", id=self._client.default_space_id
                    )

                else:
                    location.path = location.path.format(
                        option="projects", id=self._client.default_project_id
                    )
                results_reference = DataConnection(connection=None, location=location)

            else:
                if isinstance(data_references[0].location, S3Location):
                    results_reference = DataConnection(
                        connection=data_references[0].connection,
                        location=S3Location(
                            bucket=data_references[0].location.bucket, path="."
                        ),
                    )

                elif isinstance(data_references[0].location, AssetLocation):
                    connection_id = data_references[0].location._get_connection_id(
                        self._client
                    )

                    if connection_id is not None:
                        results_reference = DataConnection(
                            connection_asset_id=connection_id,
                            location=S3Location(
                                bucket=data_references[0].location._get_bucket(
                                    self._client
                                ),
                                path=result_path,
                            ),
                        )

                    else:  # set container output location when default DAta Asset is as a train ref
                        results_reference = DataConnection(
                            location=ContainerLocation(path=result_path)
                        )

                else:
                    results_reference = DataConnection(
                        location=ContainerLocation(path=result_path)
                    )
        # -- end note

        # note: validate location types:
        if self._client.ICP_PLATFORM_SPACES:
            if not isinstance(results_reference.location, FSLocation):
                raise TypeError(
                    "Unsupported results location type. Results reference can be stored on FSLocation."
                )
        else:
            if not isinstance(
                results_reference.location, (S3Location, ContainerLocation)
            ):
                raise TypeError(
                    "Unsupported results location type. Results reference can be stored"
                    " only on S3Location or ContainerLocation."
                )
        # -- end note
        return results_reference

    @staticmethod
    def _get_average_loss_score_for_each_epoch(
        tuning_details: dict, epoch: int
    ) -> list:
        scores = []
        temp_score = []
        if "data" in tuning_details["entity"]["status"]["metrics"][0]:
            for ind, metric in enumerate(tuning_details["entity"]["status"]["metrics"]):
                if int(metric["data"]["epoch"]) == epoch:
                    temp_score.append(metric["data"]["value"])
                else:
                    epoch += 1
                    scores.append(np.average(temp_score))
                    temp_score = [metric["data"]["value"]]
            scores.append(np.average(temp_score))
        else:
            for ind, metric in enumerate(tuning_details["entity"]["status"]["metrics"]):
                if int(metric["ml_metrics"]["epoch"]) == epoch:
                    temp_score.append(metric["ml_metrics"]["loss"])
                else:
                    epoch += 1
                    scores.append(np.average(temp_score))
                    temp_score = [metric["ml_metrics"]["loss"]]
            scores.append(np.average(temp_score))
        return scores

    @staticmethod
    def _get_last_iteration_metrics_for_each_epoch(tuning_details: dict) -> list:
        last_iteration_metrics_for_each_epoch = []
        for ind in range(len(tuning_details["entity"]["status"]["metrics"])):
            if ind == 0:
                last_iteration_metrics_for_each_epoch.append(
                    tuning_details["entity"]["status"]["metrics"][0]
                )
            else:
                if (
                    tuning_details["entity"]["status"]["metrics"][ind]["ml_metrics"][
                        "epoch"
                    ]
                    == tuning_details["entity"]["status"]["metrics"][ind - 1][
                        "ml_metrics"
                    ]["epoch"]
                ):
                    last_iteration_metrics_for_each_epoch.pop()
                    last_iteration_metrics_for_each_epoch.append(
                        tuning_details["entity"]["status"]["metrics"][ind]
                    )
                else:
                    last_iteration_metrics_for_each_epoch.append(
                        tuning_details["entity"]["status"]["metrics"][ind]
                    )
        return last_iteration_metrics_for_each_epoch

    @staticmethod
    def _get_first_and_last_iteration_metrics_for_each_epoch(
        tuning_details: dict,
    ) -> list:
        first_and_last_iteration_metrics_for_each_epoch = []
        first_iteration = True

        tuning_metrics = tuning_details["entity"]["status"]["metrics"]
        for ind in range(len(tuning_metrics)):
            if ind == 0:
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_iteration = False
            elif first_iteration:
                first_and_last_iteration_metrics_for_each_epoch.append(
                    tuning_metrics[ind]
                )
                first_iteration = False
            else:
                if (
                    tuning_metrics[ind].get(
                        "data", tuning_metrics[ind].get("ml_metrics")
                    )["epoch"]
                    == tuning_metrics[ind - 1].get(
                        "data", tuning_metrics[ind - 1].get("ml_metrics")
                    )["epoch"]
                ):
                    first_and_last_iteration_metrics_for_each_epoch.pop()
                    first_and_last_iteration_metrics_for_each_epoch.append(
                        tuning_metrics[ind]
                    )
                else:
                    first_and_last_iteration_metrics_for_each_epoch.append(
                        tuning_metrics[ind]
                    )
                    first_iteration = True
        return first_and_last_iteration_metrics_for_each_epoch

    def _get_metrics_data_from_property_or_file(self, details: dict) -> dict:
        path = details["entity"]["status"]["metrics"][0]["context"][
            f"{self.tuning_type}_tuning"
        ]["metrics_location"]
        results_reference = details["entity"]["results_reference"]
        conn = DataConnection._from_dict(results_reference)
        conn._api_client = self._client
        tuning_type = cast(
            Literal["prompt_tuning", "fine_tuning", "ilab_tuning"],
            f"{self.tuning_type}_tuning",
        )
        metrics_data = conn._download_json_file(path, tuning_type=tuning_type)

        return metrics_data
