#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from warnings import warn

from ibm_watsonx_ai.foundation_models.base_tuner import BaseTuner
from ibm_watsonx_ai.foundation_models.ilab.documents import DocumentExtractions
from ibm_watsonx_ai.foundation_models.ilab.synthetic_data import SyntheticData
from ibm_watsonx_ai.foundation_models.ilab.taxonomies import Taxonomies
from ibm_watsonx_ai.helpers import (
    S3Location,
    ContainerLocation,
)
from ibm_watsonx_ai.helpers.connections.connections import GithubLocation
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.helpers.connections import (
    DataConnection,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class ILabTuner(BaseTuner):
    """Class of InstructLab fine tuner."""

    id: str | None = None
    _client: APIClient = None  # type: ignore[assignment]
    _training_metadata: dict | None = None

    def __init__(self, name: str, api_client: APIClient):
        BaseTuner.__init__(self, "ilab")
        self.name = name
        self._client = api_client

        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("InstructLab fine tuning is supported on Cloud only.")

        beta_release_warning = (
            "Warning: This beta release is a preview IBM Cloud service and is not meant for production use. "
            "Usage limitations that are explained in the Service Description apply."
        )
        warn(beta_release_warning)

        WMLResource._validate_type(self.name, "name", str, mandatory=True)

        self.documents = DocumentExtractions(self.name, self._client)
        self.taxonomies = Taxonomies(self.name, self._client)
        self.synthetic_data = SyntheticData(self.name, self._client)

    def run(
        self,
        training_data_references: list[DataConnection],
        training_results_reference: DataConnection | None = None,
        background_mode: bool = False,
    ) -> dict:
        """Run an ilab tuning process of a foundation model on top of the training data referenced by DataConnection.

        :param training_data_references: data storage connection details to inform where the training data is stored
        :type training_data_references: list[DataConnection]

        :param training_results_reference: data storage connection details to store pipeline training results
        :type training_results_reference: DataConnection, optional

        :param background_mode: indicator if the fit() method will run in the background, async or sync
        :type background_mode: bool, optional

        :return: run details
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            from ibm_watsonx_ai.helpers import DataConnection, GithubLocation

            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)

            taxonomy_import = ilab_tuner.taxonomies.run_import(
                name="my_taxonomy",
                data_reference=DataConnection(
                    location=GithubLocation(
                        secret_manager_url="...",
                        secret_id="...",
                        path="."
                    )
                ),
                results_reference=DataConnection(
                    location=ContainerLocation(path="."))
            )

            taxonomy = taxonomy_import.get_taxonomy()

            sdg = ilab_tuner.synthetic_data.generate(
                name="my_sdg",
                taxonomy=taxonomy
            )

            ilab_tuner.run(
                training_data_references=[sdg.get_results_reference()],
                training_results_reference=DataConnection(
                    location=ContainerLocation(
                        path="fine_tuning_result"
                    )
                )
            )
        """
        WMLResource._validate_type(
            training_data_references, "training_data_references", list, mandatory=True
        )
        WMLResource._validate_type(
            training_results_reference,
            "training_results_reference",
            object,
            mandatory=False,
        )

        for source_data_connection in [training_data_references]:
            if source_data_connection:
                self._validate_source_data_connections(source_data_connection)
        training_results_reference = self._determine_result_reference(
            results_reference=training_results_reference,
            data_references=training_data_references,
        )

        self._initialize_training_metadata(
            training_data_references,
            training_results_reference=training_results_reference,
        )

        self._training_metadata = cast(dict, self._training_metadata)
        tuning_details = self._client.training.run(
            meta_props=self._training_metadata,
            asynchronous=background_mode,
            _is_fine_tuning=True,
        )
        self.id = self._client.training.get_id(tuning_details)

        return self._client.training.get_details(self.id, _is_fine_tuning=True)

    def _initialize_training_metadata(
        self,
        training_data_references: list[DataConnection],
        training_results_reference: DataConnection | None = None,
    ) -> None:
        self._training_metadata = {
            self._client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
                connection._to_dict() for connection in training_data_references
            ],
            self._client.training.ConfigurationMetaNames.NAME: f"{self.name[:100]}",
            self._client.training.ConfigurationMetaNames.DESCRIPTION: self.name,
            "type": "ilab",
        }
        if training_results_reference:
            self._training_metadata["results_reference"] = (
                training_results_reference._to_dict()
            )
        if self._client.default_project_id:
            self._training_metadata["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            self._training_metadata["space_id"] = self._client.default_space_id

    def get_params(self) -> dict:
        """Get configuration parameters of ILabTuner.

        :return: ILabTuner parameters
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)

            ilab_tuner.get_params()

            # Result:
            #
            # {'name': 'ILab tuning'}
        """

        params = {"name": self.name}
        return params

    #####################
    #   Run operations  #
    #####################

    def get_run_status(self) -> str:
        """Check the status/state of an initialized ilab tuning run if it was run in background mode.

        :return: status of the ILab Tuning run
        :rtype: str

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)
            ilab_tuner.run(...)

            ilab_tuner.get_run_details()

            # Result:
            # 'completed'
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_ilab_tuning_not_scheduled")
            )

        return self._client.training.get_status(training_id=self.id, _is_fine_tuning=True).get("state")  # type: ignore[return-value]

    def get_run_details(self, include_metrics: bool = False) -> dict:
        """Get details of an ilab tuning run.

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: details of the ilab tuning
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)
            ilab_tuner.run(...)

            ilab_tuner.get_run_details()
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_ilab_tuning_not_scheduled")
            )

        details = self._client.training.get_details(
            training_id=self.id, _is_fine_tuning=True
        )

        if not include_metrics:
            if details["entity"]["status"].get("metrics", False):
                del details["entity"]["status"]["metrics"]

        return details

    def cancel_run(self) -> None:
        """Cancel a ILab Tuning run."""
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_ilab_tuning_not_scheduled")
            )

        self._client.training.cancel(training_id=self.id, _is_fine_tuning=True)

    def delete_run(self) -> None:
        """Delete a ILab Tuning run."""

        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_ilab_tuning_not_scheduled")
            )

        self._client.training.cancel(
            training_id=self.id, hard_delete=True, _is_fine_tuning=True
        )

    def get_data_connections(self) -> list[DataConnection]:
        """Create DataConnection objects for further usage
            (eg. to handle data storage connection).

        :return: list of DataConnections
        :rtype: list['DataConnection']

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            experiment = TuneExperiment(credentials, ...)
            ilab_tuner = experiment.ilab_tuner(...)
            ilab_tuner.run(...)

            data_connections = ilab_tuner.get_data_connections()
        """

        training_data_references = self.get_run_details()["entity"][
            "training_data_references"
        ]

        data_connections = [
            DataConnection._from_dict(_dict=data_connection)
            for data_connection in training_data_references
        ]

        for data_connection in data_connections:
            data_connection.set_client(self._client)
            data_connection._run_id = self.id

        return data_connections
