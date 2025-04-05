#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

__all__ = ["TabularIterableDataset"]

import os
import logging
from typing import TYPE_CHECKING, Iterator, Any, cast, Sequence
from warnings import warn

import pandas as pd

from ibm_watsonx_ai.helpers.connections.local import LocalBatchReader
from ibm_watsonx_ai.utils.autoai.enums import SamplingTypes, DocumentsSamplingTypes
from ibm_watsonx_ai.utils.autoai.errors import InvalidSizeLimit

if TYPE_CHECKING:
    from ibm_watsonx_ai.helpers.connections.flight_service import FlightConnection
    from ibm_watsonx_ai.helpers.connections import DataConnection
    from ibm_watsonx_ai import APIClient

# Note: try to import torch lib if available, this fallback is done based on
# torch dependency removal request
try:
    from torch.utils.data import IterableDataset

except ImportError:
    IterableDataset: type = object  # type: ignore[no-redef]
# --- end note

DEFAULT_SAMPLE_SIZE_LIMIT = (
    1073741824  # 1GB in Bytes is verified later by _set_sample_size_limit
)
DEFAULT_SAMPLING_TYPE = SamplingTypes.FIRST_VALUES
DEFAULT_DOCUMENTS_SAMPLING_TYPE = DocumentsSamplingTypes.RANDOM

logger = logging.getLogger(__name__)


# This dataset is intended to be an Iterable stream from Flight Service.
# It should iterate over flight logical batches and manages by Connection class
# how batches are downloaded and created. It should take into consideration only 2 batches at a time.
# If we have 2 batches already downloaded, it should block further download
# and wait for first batch to be consumed.
class TabularIterableDataset(IterableDataset):
    """
    Iterable class downloading data in batches.

    :param connection: connection to the dataset
    :type connection: DataConnection

    :param experiment_metadata: metadata retrieved from the experiment that created the model
    :type experiment_metadata: dict, optional

    :param enable_sampling: if set to `True`, will enable sampling, default: True
    :type enable_sampling: bool, optional

    :param sample_size_limit: upper limit for the overall data to be downloaded in bytes, default: 1 GB
    :type sample_size_limit: int, optional

    :param sampling_type: a sampling strategy on how to read the data,
        check `SamplingTypes` enum class for more options
    :type sampling_type: str, optional

    :param binary_data: if set to `True`, the downloaded data will be treated as binary data
    :type binary_data: bool, optional

    :param number_of_batch_rows: number of rows to read in each batch when reading from the flight connection
    :type number_of_batch_rows: int, optional

    :param stop_after_first_batch: if set to `True`, the loading will stop after downloading the first batch
    :type stop_after_first_batch: bool, optional

    :param total_size_limit: upper limit for overall data to be downloaded in Bytes, default: 1 GB,
        if more than one of: `total_size_limit`, `total_nrows_limit`, `total_percentage_limit` are set,
        then data are limited to the lower threshold, if None, then all data are downloaded in batches
        in the `iterable_read` method
    :type total_size_limit: int, optional

    :param total_nrows_limit: upper limit for overall data to be downloaded in a number of rows,
        if more than one of: `total_size_limit`, `total_nrows_limit`, `total_percentage_limit` are set,
        then data are limited to the lower threshold
    :type total_nrows_limit: int, optional

    :param total_percentage_limit: upper limit for overall data to be downloaded in percent of all dataset,
        must be a float number between 0 and 1, if more than one of: `total_size_limit`, `total_nrows_limit`,
        `total_percentage_limit` are set, then data are limited to the lower threshold
    :type total_percentage_limit: float, optional

    :param apply_literal_eval: when True then ast.literal_eval will be applied to all string columns.
    :type apply_literal_eval: bool, optional

    **Example:**

        .. code-block:: python

            experiment_metadata = {
                    "prediction_column": 'species',
                    "prediction_type": "classification",
                    "project_id": os.environ.get('PROJECT_ID'),
                    'credentials': credentials
            }

            connection = DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')


    **Example: default sampling - read first 1 GB of data**

        .. code-block:: python

            iterable_dataset = TabularIterableDataset(connection=connection,
                                                      enable_sampling=True,
                                                      sampling_type='first_n_records',
                                                      sample_size_limit = 1GB,
                                                      experiment_metadata=experiment_metadata)

    **Example: read all data records in batches/no subsampling**

        .. code-block:: python

            iterable_dataset = TabularIterableDataset(connection=connection,
                                                      enable_sampling=False,
                                                      experiment_metadata=experiment_metadata)

    **Example: stratified/random sampling**

        .. code-block:: python

            iterable_dataset = TabularIterableDataset(connection=connection,
                                                      enable_sampling=True,
                                                      sampling_type='stratified',
                                                      sample_size_limit = 1GB,
                                                      experiment_metadata=experiment_metadata)

    """

    def __init__(
        self,
        connection: DataConnection | dict,
        experiment_metadata: dict | None = None,
        enable_sampling: bool = True,
        sample_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        sampling_type: str = DEFAULT_SAMPLING_TYPE,
        binary_data: bool = False,
        number_of_batch_rows: int | None = None,
        stop_after_first_batch: bool = False,
        total_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        total_nrows_limit: int | None = None,
        total_percentage_limit: float = 1.0,
        apply_literal_eval: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.enable_sampling = enable_sampling
        self.sample_size_limit = sample_size_limit
        self.experiment_metadata = (
            experiment_metadata if experiment_metadata is not None else {}
        )
        self._api_client = getattr(connection, "_api_client", None)
        if self._api_client is None:
            self._api_client = kwargs.get(
                "api_client", kwargs.get("_api_client", kwargs.get("_wml_client"))
            )
        self.binary_data = binary_data
        self.sampling_type = sampling_type
        self.read_to_file = kwargs.get("read_to_file")
        self.authorized = self._check_authorization()
        self._set_size_limit(total_size_limit)
        self.total_nrows_limit = total_nrows_limit
        self.total_percentage_limit = total_percentage_limit
        self.apply_literal_eval = apply_literal_eval

        # Note: convert to dictionary if we have object from API client
        if not isinstance(connection, dict):
            dict_connection = connection._to_dict()

        else:
            dict_connection = connection
        # --- end note

        self.experiment_metadata = cast(dict[str, Any], self.experiment_metadata)
        # Note: backward compatibility after sampling refactoring #27255
        if kwargs.get("with_sampling") or kwargs.get("normal_read"):
            parameters_deprecated_warning = (
                "The parameters with_sampling and normal_read in TabularIterableDataset are deprecated. "
                "Use enable_sampling and sampling_type instead."
            )
            warn(parameters_deprecated_warning, category=DeprecationWarning)

            if kwargs.get("normal_read"):
                self.enable_sampling = False
            if kwargs.get("with_sampling"):
                from ibm_watsonx_ai.utils.autoai.enums import PredictionType

                self.enable_sampling = True

                if self.experiment_metadata.get("prediction_type") in [
                    PredictionType.REGRESSION
                ]:
                    self.sampling_type = SamplingTypes.RANDOM
                elif self.experiment_metadata.get("prediction_type") in [
                    PredictionType.CLASSIFICATION,
                    PredictionType.BINARY,
                    PredictionType.MULTICLASS,
                ]:
                    self.sampling_type = SamplingTypes.STRATIFIED
        # --- end note

        # if number_of_batch_rows is provided, batch_size does not matter anymore
        if self.authorized:
            is_cos_asset = bool(
                kwargs.get("flight_parameters", {})
                .get("datasource_type", {})
                .get("entity", {})
                .get("name", "")
                == "bluemixcloudobjectstorage"
            )
            # first used headers from experiment metadata if they were set.
            headers_: dict | None = None
            if self.experiment_metadata.get("headers"):
                headers_ = self.experiment_metadata.get("headers")
            elif self._api_client is not None:
                headers_ = self._api_client._get_headers()

            from ibm_watsonx_ai.helpers.connections.flight_service import (
                FlightConnection,
            )

            flight_parameters = self._update_params_with_connection_properties(
                connection=dict_connection,
                flight_parameters=kwargs.get("flight_parameters", {}),
                api_client=self._api_client,
            )

            headers_ = cast(dict, headers_)
            number_of_batch_rows = cast(int, number_of_batch_rows)
            self.connection: FlightConnection = FlightConnection(
                headers=headers_,
                sampling_type=self.sampling_type,
                label=self.experiment_metadata.get("prediction_column"),
                learning_type=self.experiment_metadata.get("prediction_type"),
                params=self.experiment_metadata,
                project_id=self.experiment_metadata.get(
                    "project_id", getattr(self._api_client, "default_project_id", None)
                ),
                space_id=self.experiment_metadata.get(
                    "space_id", getattr(self._api_client, "default_space_id", None)
                ),
                asset_id=(
                    None
                    if is_cos_asset
                    else dict_connection.get("location", {}).get("id")
                ),  # do not pass asset id for data assets located on COS
                connection_id=dict_connection.get("connection", {}).get("id"),
                data_location=dict_connection,
                data_batch_size_limit=self.sample_size_limit,
                flight_parameters=flight_parameters,
                extra_interaction_properties=kwargs.get(
                    "extra_interaction_properties", {}
                ),
                fallback_to_one_connection=kwargs.get(
                    "fallback_to_one_connection", True
                ),
                number_of_batch_rows=number_of_batch_rows,
                stop_after_first_batch=stop_after_first_batch,
                _api_client=self._api_client,
                return_subsampling_stats=kwargs.get("_return_subsampling_stats", False),
                total_size_limit=self.total_size_limit,
                total_nrows_limit=self.total_nrows_limit,
                total_percentage_limit=self.total_percentage_limit,
                apply_literal_eval=self.apply_literal_eval,
            )

        else:

            if (
                dict_connection.get("type") == "fs"
                and "location" in dict_connection
                and "path" in dict_connection["location"]
            ):
                self.connection: LocalBatchReader = LocalBatchReader(
                    file_path=dict_connection["location"]["path"],
                    batch_size=sample_size_limit,
                )

            else:
                raise NotImplementedError(
                    "For local data read please use 'fs' (file system) connection type. "
                    + "For remote data read enrich DataConnection with authorization data using "
                    + "`connection.set_client(api_client)` function or providing 'experiment_metadata'."
                )

    @property
    def _wml_client(self) -> APIClient:
        # note: backward compatibility
        wml_client_deprecated_warning = (
            "`_wml_client` is deprecated and will be removed in future. "
            "Instead, please use `_api_client`."
        )
        warn(wml_client_deprecated_warning, category=DeprecationWarning)
        # --- end note
        return self._api_client  # type: ignore[return-value]

    @_wml_client.setter
    def _wml_client(self, var: APIClient) -> None:
        # note: backward compatibility
        wml_client_deprecated_warning = (
            "`_wml_client` is deprecated and will be removed in future. "
            "Instead, please use `_api_client`."
        )
        warn(wml_client_deprecated_warning, category=DeprecationWarning)
        # --- end note
        self._api_client = var

    def _check_authorization(self) -> bool:
        """
        Check if you can authorize with Service.
        If the connection has api_client initialized, use it as an attribute.
        Otherwise, provide your credentials in the experiment_metadata dictionary.
        If the client is properly initialized, True will be returned.
        """
        if self._api_client is not None:
            return True

        if self.experiment_metadata is None:
            return False
        credentials = (
            creds
            if (creds := self.experiment_metadata.get("credentials")) is not None
            else self.experiment_metadata.get("wml_credentials")
        )
        if credentials is not None:
            from ibm_watsonx_ai import APIClient

            self._api_client = APIClient(credentials=credentials)
            return True

        elif self.experiment_metadata.get("headers") is not None:
            return True

        else:
            return False

    def _set_size_limit(self, size_limit: int) -> None:
        """If non-default value of total_size_limit was not passed,
        set Sample Size Limit based on T-Shirt size if code is run on training pod:
        For memory < 16 (T-Shirts: XS,S) default is 10MB,
        For memory < 32 & >= 16 (T-Shirts: M) default is 100MB,
        For memory = 32 (T-Shirt L) default is 0.7GB,
        For memory > 32 (T-Shirt XL) or runs outside pod default is 1GB.
        """
        self.total_size_limit: int | None
        from ibm_watsonx_ai.utils.autoai.connection import get_max_sample_size_limit

        max_tshirt_size_limit = (
            get_max_sample_size_limit() if os.getenv("MEM", False) else None
        )  # limit manual setting of sample size limit on autoai clusters #31527

        if self.enable_sampling:
            if max_tshirt_size_limit:
                if (
                    size_limit > max_tshirt_size_limit
                    and size_limit != DEFAULT_SAMPLE_SIZE_LIMIT
                ):
                    raise InvalidSizeLimit(size_limit, max_tshirt_size_limit)
                else:
                    self.total_size_limit = min(size_limit, max_tshirt_size_limit)
            else:
                self.total_size_limit = size_limit
        else:
            if size_limit == DEFAULT_SAMPLE_SIZE_LIMIT:
                self.total_size_limit = None  # do not limit reading if sampling is disabled, we want read all data
            else:
                self.total_size_limit = size_limit

    @staticmethod
    def _update_params_with_connection_properties(
        connection: dict,
        flight_parameters: dict,
        api_client: APIClient | None = None,
    ) -> dict:

        if (
            not flight_parameters.get("connection_properties")
            and connection.get("type") == "container"
        ):
            from ibm_watsonx_ai.helpers.connections import DataConnection

            data_connection = DataConnection._from_dict(connection)
            data_connection.set_client(api_client)

            flight_parameters = (
                data_connection._update_flight_parameters_with_connection_details(
                    flight_parameters
                )
            )
        return flight_parameters

    def write(
        self, data: pd.DataFrame | None = None, file_path: str | None = None
    ) -> None:
        """
        Writes data into the data source connection.

        :param data: structured data to be saved in data source connection, 'data' or 'file_path' must be provided
        :type data: DataFrame, optional

        :param file_path: path to the local file to be saved in a source data connection (binary transfer).
            'data' or 'file_path' need to be provided
        :type file_path: str, optional
        """
        if (data is None and file_path is None) or (
            data is not None and file_path is not None
        ):
            raise ValueError("Either 'data' or 'file_path' need to be provided.")

        if data is not None and not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"'data' need to be a pandas DataFrame, you provided: '{type(data)}'."
            )

        if file_path is not None and not isinstance(file_path, str):
            raise TypeError(
                f"'file_path' need to be a string, you provided: '{type(file_path)}'."
            )

        if data is not None:
            self.connection.write_data(data)

        else:
            file_path = cast(str, file_path)
            self.connection.write_binary_data(file_path)

    def __iter__(self) -> Iterator:
        """Iterate over Flight Dataset."""
        if self.authorized:
            if self.enable_sampling:
                if self.sampling_type == SamplingTypes.FIRST_VALUES:
                    return self.connection.iterable_read()
                else:
                    self.connection.enable_subsampling = True
                    return self.connection.iterable_read()
            else:
                if self.binary_data:
                    return self.connection.read_binary_data(read_to_file=self.read_to_file)  # type: ignore[return-value]
                else:
                    self.total_size_limit = None
                    return self.connection.iterable_read()
        else:
            self.connection = cast(LocalBatchReader, self.connection)
            return (batch for batch in self.connection)
