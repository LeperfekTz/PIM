#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, TYPE_CHECKING, TypeAlias, Iterator, Literal, Callable

import json
import logging
import time
from warnings import warn

from ibm_boto3.exceptions import Boto3Error
from lomond import WebSocket

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import TrainingConfigurationMetaNames
from ibm_watsonx_ai.utils import (
    print_text_header_h1,
    print_text_header_h2,
    TRAINING_RUN_DETAILS_TYPE,
    StatusLogger,
)
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError, ApiRequestFailure
from ibm_watsonx_ai.wml_resource import WMLResource

logging.getLogger("lomond").setLevel(logging.CRITICAL)
ListType: TypeAlias = list

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class Training(WMLResource):
    """Train new models."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)
        self.ConfigurationMetaNames = TrainingConfigurationMetaNames()

    def get_status(self, training_id: str | None = None, **kwargs: Any) -> dict:
        """Get the status of a created training.

        :param training_id: ID of the training
        :type training_id: str

        :return: training_status
        :rtype: dict

        **Example:**

        .. code-block:: python

            training_status = client.training.get_status(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        Training._validate_type(training_id, "training_id", str, True)

        details = self.get_details(
            training_id, _internal=True, _is_fine_tuning=_is_fine_tuning
        )

        if details is not None:
            return WMLResource._get_required_element_from_dict(
                details, "details", ["entity", "status"]
            )
        else:
            raise WMLClientError(
                "Getting trained model status failed. Unable to get model details for training_id: '{}'.".format(
                    training_id
                )
            )

    def get_details(
        self,
        training_id: str | None = None,
        limit: int | None = None,
        asynchronous: Literal[True, False] = False,
        get_all: Literal[True, False] = False,
        training_type: str | None = None,
        state: str | None = None,
        tag_value: str | None = None,
        training_definition_id: str | None = None,
        _internal: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of training(s). If training_id is not specified, the metadata of all model spaces are returned.

        :param training_id: unique ID of the training
        :type training_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param training_type: filter the fetched list of trainings based on the training type ["pipeline" or "experiment"]
        :type training_type: str, optional

        :param state: filter the fetched list of training based on their state:
            [`queued`, `running`, `completed`, `failed`]
        :type state: str, optional

        :param tag_value: filter the fetched list of training based on their tag value
        :type tag_value: str, optional

        :param training_definition_id: filter the fetched trainings that are using the given training definition
        :type training_definition_id: str, optional

        :return: metadata of training(s)
        :rtype:
          - **dict** - if training_id is not None
          - **{"resources": [dict]}** - if training_id is None

        **Examples**

        .. code-block:: python

            training_run_details = client.training.get_details(training_id)
            training_runs_details = client.training.get_details()
            training_runs_details = client.training.get_details(limit=100)
            training_runs_details = client.training.get_details(limit=100, get_all=True)
            training_runs_details = []
            for entry in client.training.get_details(limit=100, asynchronous=True, get_all=True):
                training_runs_details.extend(entry)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=True
        )
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        # For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, False)

        if _is_fine_tuning:
            url = (
                self._client.service_instance._href_definitions.get_fine_tunings_href()
            )
        else:
            url = self._client.service_instance._href_definitions.get_trainings_href()

        if training_id is None:
            query_params: dict | None = {
                param_name: param_value
                for param_name, param_value in (
                    ("type", training_type),
                    ("state", state),
                    ("tag.value", tag_value),
                    ("training_definition_id", training_definition_id),
                )
                if param_value is not None
            }
            # note: If query params is an empty dict convert it back to None value
            query_params = query_params if query_params != {} else None

            return self._get_artifact_details(
                base_url=url,
                id=training_id,
                limit=limit,
                resource_name="trained models",
                _async=asynchronous,
                _all=get_all,
                query_params=query_params,
            )
        else:
            return self._get_artifact_details(url, training_id, limit, "trained models")

    @staticmethod
    def get_href(training_details: dict) -> str:
        """Get the training href from the training details.

        :param training_details: metadata of the created training
        :type training_details: dict

        :return: training href
        :rtype: str

        **Example:**

        .. code-block:: python

            training_details = client.training.get_details(training_id)
            run_url = client.training.get_href(training_details)
        """

        Training._validate_type(training_details, "training_details", object, True)
        if "id" in training_details.get("metadata", {}):
            training_id = WMLResource._get_required_element_from_dict(
                training_details, "training_details", ["metadata", "id"]
            )
            return "/ml/v4/trainings/" + training_id
        else:
            Training._validate_type_of_details(
                training_details, TRAINING_RUN_DETAILS_TYPE
            )
            return WMLResource._get_required_element_from_dict(
                training_details, "training_details", ["metadata", "href"]
            )

    @staticmethod
    def get_id(training_details: dict) -> str:
        """Get the training ID from the training details.

        :param training_details: metadata of the created training
        :type training_details: dict

        :return: unique ID of the training
        :rtype: str

        **Example:**

        .. code-block:: python

            training_details = client.training.get_details(training_id)
            training_id = client.training.get_id(training_details)

        """
        Training._validate_type(training_details, "training_details", object, True)
        return WMLResource._get_required_element_from_dict(
            training_details, "training_details", ["metadata", "id"]
        )

    def run(self, meta_props: dict, asynchronous: bool = True, **kwargs: Any) -> dict:
        """Create a new Machine Learning training.

        :param meta_props: metadata of the training configuration. To see available meta names, use:

            .. code-block:: python

                client.training.ConfigurationMetaNames.show()

        :type meta_props: dict
        :param asynchronous:
            * `True` - training job is submitted and progress can be checked later
            * `False` - method will wait till job completion and print training stats
        :type asynchronous: bool, optional

        :return: metadata of the training created
        :rtype: dict

        .. note::

            You can provide one of the following values for training:
             * client.training.ConfigurationMetaNames.EXPERIMENT
             * client.training.ConfigurationMetaNames.PIPELINE
             * client.training.ConfigurationMetaNames.MODEL_DEFINITION

        **Examples**

        Example of meta_props for creating a training run in IBM Cloud Pak® for Data version 3.0.1 or above:

        .. code-block:: python

            metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Hand-written Digit Recognition',
                client.training.ConfigurationMetaNames.DESCRIPTION: 'Hand-written Digit Recognition Training',
                client.training.ConfigurationMetaNames.PIPELINE: {
                    "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                    "rev": "12",
                    "model_type": "string",
                    "data_bindings": [
                        {
                            "data_reference_name": "string",
                            "node_id": "string"
                        }
                    ],
                    "nodes_parameters": [
                        {
                            "node_id": "string",
                            "parameters": {}
                        }
                    ],
                    "hardware_spec": {
                        "id": "4cedab6d-e8e4-4214-b81a-2ddb122db2ab",
                        "rev": "12",
                        "name": "string",
                        "num_nodes": "2"
                    }
                },
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [{
                    'type': 's3',
                    'connection': {},
                    'location': {'href': 'v2/assets/asset1233456'},
                    'schema': { 'id': 't1', 'name': 'Tasks', 'fields': [ { 'name': 'duration', 'type': 'number' } ]}
                }],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'id' : 'string',
                    'connection': {
                        'endpoint_url': 'https://s3-api.us-geo.objectstorage.service.networklayer.com',
                        'access_key_id': '***',
                        'secret_access_key': '***'
                    },
                    'location': {
                        'bucket': 'wml-dev-results',
                        'path' : "path"
                    }
                    'type': 's3'
                }
            }

        Example of a Federated Learning training job:

        .. code-block:: python

            aggregator_metadata = {
                client.training.ConfigurationMetaNames.NAME: 'Federated_Learning_Tensorflow_MNIST',
                client.training.ConfigurationMetaNames.DESCRIPTION: 'MNIST digit recognition with Federated Learning using Tensorflow',
                client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [],
                client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
                    'type': results_type,
                    'name': 'outputData',
                    'connection': {},
                    'location': { 'path': '/projects/' + PROJECT_ID + '/assets/trainings/'}
                },
                client.training.ConfigurationMetaNames.FEDERATED_LEARNING: {
                    'model': {
                        'type': 'tensorflow',
                        'spec': {
                        'id': untrained_model_id
                    },
                    'model_file': untrained_model_name
                },
                'fusion_type': 'iter_avg',
                'metrics': 'accuracy',
                'epochs': 3,
                'rounds': 10,
                'remote_training' : {
                    'quorum': 1.0,
                    'max_timeout': 3600,
                    'remote_training_systems': [ { 'id': prime_rts_id }, { 'id': nonprime_rts_id} ]
                },
                'hardware_spec': {
                    'name': 'S'
                },
                'software_spec': {
                    'name': 'runtime-22.1-py3.9'
                }
            }

            aggregator = client.training.run(aggregator_metadata, asynchronous=True)
            aggregator_id = client.training.get_id(aggregator)
        """
        # For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(meta_props, "meta_props", object, True)
        Training._validate_type(asynchronous, "asynchronous", bool, True)
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        self.ConfigurationMetaNames._validate(meta_props)
        training_configuration_metadata = {
            "training_data_references": meta_props[
                self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
            ],
            "results_reference": meta_props[
                self.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE
            ],
        }

        if self.ConfigurationMetaNames.TEST_DATA_REFERENCES in meta_props:
            training_configuration_metadata["test_data_references"] = meta_props[
                self.ConfigurationMetaNames.TEST_DATA_REFERENCES
            ]

        if self.ConfigurationMetaNames.TEST_OUTPUT_DATA in meta_props:
            training_configuration_metadata["test_output_data"] = meta_props[
                self.ConfigurationMetaNames.TEST_OUTPUT_DATA
            ]

        if self.ConfigurationMetaNames.TAGS in meta_props:
            training_configuration_metadata["tags"] = meta_props[
                self.ConfigurationMetaNames.TAGS
            ]

        if self.ConfigurationMetaNames.PROMPT_TUNING in meta_props:
            training_configuration_metadata["prompt_tuning"] = meta_props[
                self.ConfigurationMetaNames.PROMPT_TUNING
            ]

        if self.ConfigurationMetaNames.FINE_TUNING in meta_props:
            training_configuration_metadata["parameters"] = meta_props[
                self.ConfigurationMetaNames.FINE_TUNING
            ]

        if self.ConfigurationMetaNames.AUTO_UPDATE_MODEL in meta_props:
            training_configuration_metadata["auto_update_model"] = meta_props[
                self.ConfigurationMetaNames.AUTO_UPDATE_MODEL
            ]

        # TODO remove when training service starts copying such data on their own

        training_configuration_metadata["name"] = meta_props[
            self.ConfigurationMetaNames.NAME
        ]
        training_configuration_metadata["description"] = meta_props[
            self.ConfigurationMetaNames.DESCRIPTION
        ]

        if self.ConfigurationMetaNames.PIPELINE in meta_props:
            training_configuration_metadata["pipeline"] = meta_props[
                self.ConfigurationMetaNames.PIPELINE
            ]
        if self.ConfigurationMetaNames.EXPERIMENT in meta_props:
            training_configuration_metadata["experiment"] = meta_props[
                self.ConfigurationMetaNames.EXPERIMENT
            ]
        if self.ConfigurationMetaNames.MODEL_DEFINITION in meta_props:
            training_configuration_metadata["model_definition"] = meta_props[
                self.ConfigurationMetaNames.MODEL_DEFINITION
            ]
        if self.ConfigurationMetaNames.SPACE_UID in meta_props:
            training_configuration_metadata["space_id"] = meta_props[
                self.ConfigurationMetaNames.SPACE_UID
            ]
        if "type" in meta_props:
            training_configuration_metadata["type"] = meta_props["type"]

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
                training_configuration_metadata["space_id"] = (
                    self._client.default_space_id
                )
            elif self._client.default_project_id is not None:
                training_configuration_metadata["project_id"] = (
                    self._client.default_project_id
                )

        if self.ConfigurationMetaNames.FEDERATED_LEARNING in meta_props:
            fl_deprecated_warning = "Federated Learning is deprecated and will be removed in IBM Cloud Pak for Data 5.2."
            warn(fl_deprecated_warning, category=DeprecationWarning)

            training_configuration_metadata["federated_learning"] = meta_props[
                self.ConfigurationMetaNames.FEDERATED_LEARNING
            ]
        if _is_fine_tuning:
            train_endpoint = (
                self._client.service_instance._href_definitions.get_fine_tunings_href()
            )
        else:
            train_endpoint = (
                self._client.service_instance._href_definitions.get_trainings_href()
            )

        params = self._client._params()
        if "space_id" in params.keys():
            params.pop("space_id")
        if "project_id" in params.keys():
            params.pop("project_id")

        if self._client.ICP_PLATFORM_SPACES:
            if "userfs" in params.keys():
                params.pop("userfs")

        response_train_post = requests.post(
            train_endpoint,
            json=training_configuration_metadata,
            params=params,
            headers=self._client._get_headers(),
        )

        run_details = self._handle_response(201, "training", response_train_post)

        trained_model_id = self.get_id(run_details)

        if asynchronous is True:
            return run_details
        else:
            print_text_header_h1("Running '{}'".format(trained_model_id))

            status = self.get_status(trained_model_id, _is_fine_tuning=_is_fine_tuning)
            state = status["state"]

            with StatusLogger(state) as status_logger:
                while state not in ["error", "completed", "canceled", "failed"]:
                    time.sleep(5)
                    status = self.get_status(
                        trained_model_id, _is_fine_tuning=_is_fine_tuning
                    )
                    state = status["state"]
                    status_logger.log_state(state)

            if "completed" in state:
                print(
                    "\nTraining of '{}' finished successfully.".format(
                        str(trained_model_id)
                    )
                )
            else:
                print(
                    "\nTraining of '{}' failed with status: '{}'.".format(
                        trained_model_id, str(status)
                    )
                )

            self._logger.debug("Response({}): {}".format(state, run_details))
            return self.get_details(
                trained_model_id, _internal=True, _is_fine_tuning=_is_fine_tuning
            )

    def list(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> DataFrame | Iterator | ListType:
        """List stored trainings in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: pandas.DataFrame with listed experiments
        :rtype: pandas.DataFrame

        **Examples**

        .. code-block:: python

            client.training.list()
            training_runs_df = client.training.list(limit=100)
            training_runs_df = client.training.list(limit=100, get_all=True)
            training_runs_df = []
            for entry in client.training.list(limit=100, asynchronous=True, get_all=True):
                training_runs_df.extend(entry)
        """
        # For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        def preprocess_details(details: dict) -> DataFrame | ListType:
            resources = details["resources"]
            values = [
                (
                    m["metadata"].get("id", m["metadata"].get("guid")),
                    m["entity"]["status"]["state"],
                    m["metadata"]["created_at"],
                )
                for m in resources
            ]

            return self._list(
                values,
                ["ID (training)", "STATE", "CREATED"],
                limit=None,
                sort_by=None,
            )

        if asynchronous:
            return (
                preprocess_details(details)
                for details in self.get_details(
                    limit=limit,
                    asynchronous=asynchronous,
                    get_all=get_all,
                    _internal=True,
                )
            )
        else:
            details = self.get_details(limit=limit, get_all=get_all, _internal=True)
            table = preprocess_details(details)
            return table

    def list_intermediate_models(
        self, training_id: str | None = None, **kwargs: Any
    ) -> None:
        """Print the intermediate_models in a table format.

        :param training_id: ID of the training
        :type training_id: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example:**

        .. code-block:: python

            client.training.list_intermediate_models()

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        # For CP4D, check if either spce or project ID is set
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                "This method is not supported for IBM Cloud Pak® for Data. "
            )

        self._client._check_if_either_is_set()
        details = self.get_details(training_id, _internal=True)
        # if status is completed then only lists global_output else display message saying "state value"
        training_state = details["entity"]["status"]["state"]
        if training_state == "completed":
            if (
                "metrics" in details["entity"]["status"]
                and details["entity"]["status"].get("metrics") is not None
            ):
                metrics_list = details["entity"]["status"]["metrics"]
                new_list = []
                for ml in metrics_list:
                    if "context" in ml and "intermediate_model" in ml["context"]:
                        name = ml["context"]["intermediate_model"].get("name", "")
                        if "location" in ml["context"]["intermediate_model"]:
                            path = ml["context"]["intermediate_model"]["location"].get(
                                "model", ""
                            )
                        else:
                            path = ""
                    else:
                        name = ""
                        path = ""

                    accuracy = ml["ml_metrics"].get("training_accuracy", "")
                    F1Micro = round(ml["ml_metrics"].get("training_f1_micro", 0), 2)
                    F1Macro = round(ml["ml_metrics"].get("training_f1_macro", 0), 2)
                    F1Weighted = round(
                        ml["ml_metrics"].get("training_f1_weighted", 0), 2
                    )
                    logLoss = round(ml["ml_metrics"].get("training_neg_log_loss", 0), 2)
                    PrecisionMicro = round(
                        ml["ml_metrics"].get("training_precision_micro", 0), 2
                    )
                    PrecisionWeighted = round(
                        ml["ml_metrics"].get("training_precision_weighted", 0), 2
                    )
                    PrecisionMacro = round(
                        ml["ml_metrics"].get("training_precision_macro", 0), 2
                    )
                    RecallMacro = round(
                        ml["ml_metrics"].get("training_recall_macro", 0), 2
                    )
                    RecallMicro = round(
                        ml["ml_metrics"].get("training_recall_micro", 0), 2
                    )
                    RecallWeighted = round(
                        ml["ml_metrics"].get("training_recall_weighted", 0), 2
                    )
                    createdAt = details["metadata"]["created_at"]
                    new_list.append(
                        [
                            name,
                            path,
                            accuracy,
                            F1Micro,
                            F1Macro,
                            F1Weighted,
                            logLoss,
                            PrecisionMicro,
                            PrecisionMacro,
                            PrecisionWeighted,
                            RecallMicro,
                            RecallMacro,
                            RecallWeighted,
                            createdAt,
                        ]
                    )
                    new_list.append([])

                from tabulate import tabulate

                header = [
                    "NAME",
                    "PATH",
                    "Accuracy",
                    "F1Micro",
                    "F1Macro",
                    "F1Weighted",
                    "LogLoss",
                    "PrecisionMicro",
                    "PrecisionMacro",
                    "PrecisionWeighted",
                    "RecallMicro",
                    "RecallMacro",
                    "RecallWeighted",
                    "CreatedAt",
                ]
                table = tabulate([header] + new_list)

                print(table)
            else:
                print(
                    " There is no intermediate model metrics are available for this training id. "
                )
        else:
            self._logger.debug("state is not completed")

    def cancel(
        self,
        training_id: str | None = None,
        hard_delete: bool = False,
        **kwargs: Any,
    ) -> Literal["SUCCESS"]:
        """Cancel a training that is currently running. This method can delete metadata
        details of a completed or canceled training run when `hard_delete` parameter is set to `True`.

        :param training_id: ID of the training
        :type training_id: str

        :param hard_delete: specify `True` or `False`:

            * `True` - to delete the completed or canceled training run
            * `False` - to cancel the currently running training run
        :type hard_delete: bool, optional

        :return: status "SUCCESS" if cancelation is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.training.cancel(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        _is_fine_tuning = kwargs.get("_is_fine_tuning", False)

        # For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Training._validate_type(training_id, "training_id", str, True)

        params = self._client._params()

        if hard_delete is True:
            params.update({"hard_delete": "true"})

        if _is_fine_tuning:

            train_endpoint = (
                self._client.service_instance._href_definitions.get_fine_tuning_href(
                    training_id
                )
            )
        else:
            train_endpoint = (
                self._client.service_instance._href_definitions.get_training_href(
                    training_id
                )
            )

        response_delete = requests.delete(
            train_endpoint,
            headers=self._client._get_headers(),
            params=params,
        )

        if (
            response_delete.status_code == 400
            and response_delete.text is not None
            and "Job already completed with state" in response_delete.text
        ):
            print(
                "Job is not running currently. Please use 'hard_delete=True' parameter to force delete"
                " completed or canceled training runs."
            )
            return "SUCCESS"
        else:
            return self._handle_response(
                204, "trained model deletion", response_delete, False
            )

    def _COS_logs(self, run_id: str, on_start: Callable = lambda: {}) -> None:
        on_start()
        run_details = self.get_details(run_id, _internal=True)
        if (
            "connection" in run_details["entity"]["results_reference"]
            and run_details["entity"]["results_reference"].get("connection") is not None
        ):
            endpoint_url = run_details["entity"]["results_reference"]["connection"][
                "endpoint_url"
            ]
            aws_access_key = run_details["entity"]["results_reference"]["connection"][
                "access_key_id"
            ]
            aws_secret = run_details["entity"]["results_reference"]["connection"][
                "secret_access_key"
            ]
            bucket = run_details["entity"]["results_reference"]["location"]["bucket"]

            if bucket == "":
                bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
            import ibm_boto3

            client_cos = ibm_boto3.client(
                service_name="s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret,
                endpoint_url=endpoint_url,
            )

            try:
                if self._client.CLOUD_PLATFORM_SPACES:
                    logs = (
                        run_details["entity"]
                        .get("results_reference")
                        .get("location")
                        .get("logs")
                    )
                    if logs is None:
                        print(
                            " There is no logs details for this Training run, hence no logs."
                        )
                        return

                    key = logs + "/learner-1/training-log.txt"

                else:
                    try:
                        key = (
                            "data/"
                            + run_details["metadata"].get(
                                "id", run_details["metadata"].get("guid")
                            )
                            + "/pipeline-model.json"
                        )

                        obj = client_cos.get_object(Bucket=bucket, Key=key)
                        pipeline_model = json.loads(
                            (obj["Body"].read().decode("utf-8"))
                        )

                    except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
                        if ex.response["Error"]["Code"] == "NoSuchKey":
                            print(
                                " Error - There is no training logs are found for the given training run id"
                            )
                            return
                        else:
                            print(ex)
                            return
                    if pipeline_model is not None:
                        key = (
                            pipeline_model["pipelines"][0]["nodes"][0]["parameters"][
                                "model_id"
                            ]
                            + "/learner-1/training-log.txt"
                        )
                    else:
                        print(
                            " Error - Cannot find the any logs for the given training run id"
                        )
                obj = client_cos.get_object(Bucket=bucket, Key=key)
                print(obj["Body"].read().decode("utf-8"))
            except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                if ex.response["Error"]["Code"] == "NoSuchKey":
                    print("ERROR - Cannot find training-log.txt in the bucket")
                else:
                    print(ex)
                    print("ERROR - Cannot get the training run log in the bucket")
        else:
            print(
                " There is no connection details for this Training run, hence no logs."
            )

    def _COS_metrics(self, run_id: str, on_start: Callable = lambda: {}) -> None:
        on_start()
        run_details = self.get_details(run_id, _internal=True)
        endpoint_url = run_details["entity"]["results_reference"]["connection"][
            "endpoint_url"
        ]
        aws_access_key = run_details["entity"]["results_reference"]["connection"][
            "access_key_id"
        ]
        aws_secret = run_details["entity"]["results_reference"]["connection"][
            "secret_access_key"
        ]
        bucket = run_details["entity"]["results_reference"]["location"]["bucket"]

        if bucket == "":
            bucket = run_details["entity"]["results_reference"]["target"]["bucket"]
        import ibm_boto3

        client_cos = ibm_boto3.client(
            service_name="s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret,
            endpoint_url=endpoint_url,
        )

        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                logs = (
                    run_details["entity"]
                    .get("results_reference")
                    .get("location")
                    .get("logs")
                )
                if logs is None:
                    print(
                        " Metric log location details for this Training run is not available."
                    )
                    return
                key = logs + "/learner-1/evaluation-metrics.txt"
            else:
                try:
                    key = (
                        run_details["metadata"].get(
                            "id", run_details["metadata"].get("guid")
                        )
                        + "/pipeline-model.json"
                    )

                    obj = client_cos.get_object(Bucket=bucket, Key=key)

                    pipeline_model = json.loads((obj["Body"].read().decode("utf-8")))
                except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:

                    if ex.response["Error"]["Code"] == "NoSuchKey":
                        print(
                            "ERROR - Cannot find pipeline_model.json in the bucket for training id "
                            + run_id
                        )
                        print(
                            "There is no training logs are found for the given training run id"
                        )
                        return
                    else:
                        print(ex)
                        return
                key = (
                    pipeline_model["pipelines"][0]["nodes"][0]["parameters"].get[
                        "model_id"
                    ]
                    + "/learner-1/evaluation-metrics.txt"
                )

            obj = client_cos.get_object(Bucket=bucket, Key=key)
            print(obj["Body"].read().decode("utf-8"))

        except ibm_boto3.exceptions.ibm_botocore.client.ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                print("ERROR - Cannot find evaluation-metrics.txt in the bucket")
            else:
                print(ex)
                print(
                    "ERROR - Cannot get the location of evaluation-metrics.txt details in the bucket"
                )

    def monitor_logs(self, training_id: str | None = None, **kwargs: Any) -> None:
        """Print the logs of a training created.

        :param training_id: training ID
        :type training_id: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example:**

        .. code-block:: python

            client.training.monitor_logs(training_id)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                "Metrics logs are not supported. This method is not supported for IBM Cloud Pak® for Data."
            )

        Training._validate_type(training_id, "training_id", str, True)

        self._simple_monitor_logs(
            training_id,  # type: ignore
            lambda: print_text_header_h1(
                "Log monitor started for training run: " + str(training_id)
            ),
        )

        print_text_header_h2("Log monitor done.")

    def _simple_monitor_logs(
        self, training_id: str, on_start: Callable = lambda: {}
    ) -> None:
        try:
            run_details = self.get_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print(
                    "Could not find the training run details for the given training run id."
                )
                return
            else:
                raise ex

        status = run_details["entity"]["status"]["state"]

        if (
            status == "completed"
            or status == "error"
            or status == "failed"
            or status == "canceled"
        ):
            self._COS_logs(
                training_id,
                lambda: print_text_header_h1(
                    "Log monitor started for training run: " + str(training_id)
                ),
            )
        else:
            if self._client.CLOUD_PLATFORM_SPACES:
                ws_param = self._client._params()
                if "project_id" in ws_param.keys():
                    proj_id = ws_param.get("project_id")
                    monitor_endpoint = (
                        self._credentials.url.replace("https", "wss")
                        + "/ml/v4/trainings/"
                        + training_id
                        + "?project_id="
                        + proj_id
                    )
                else:
                    space_id = ws_param.get("space_id")
                    monitor_endpoint = (
                        self._credentials.url.replace("https", "wss")
                        + "/ml/v4/trainings/"
                        + training_id
                        + "?space_id="
                        + space_id
                    )
            else:
                monitor_endpoint = (
                    self._credentials.url.replace("https", "wss")
                    + "/v4/trainings/"
                    + training_id
                )
            websocket = WebSocket(monitor_endpoint)

            try:
                websocket.add_header(
                    bytes("Authorization", "utf-8"),
                    bytes("Bearer " + self._client.token, "utf-8"),
                )
            except:
                websocket.add_header(
                    bytes("Authorization", "utf-8"),
                    bytes("bearer " + self._client.token),
                )

            on_start()

            for event in websocket:

                if event.name == "text":
                    text = json.loads(event.text)
                    entity = text["entity"]
                    if "status" in entity:
                        if "message" in entity["status"]:
                            message = entity["status"]["message"]
                            if len(message) > 0:
                                print(message)

            websocket.close()

    def monitor_metrics(self, training_id: str | None = None, **kwargs: Any) -> None:
        """Print the metrics of a created training.

        :param training_id: ID of the training
        :type training_id: str

        .. note::

            This method is not supported for IBM Cloud Pak® for Data.

        **Example:**

        .. code-block:: python

            client.training.monitor_metrics(training_id)
        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                "Metrics monitoring is not supported for IBM Cloud Pak® for Data"
            )

        Training._validate_type(training_id, "training_id", str, True)
        try:
            run_details = self.get_details(training_id, _internal=True)
        except ApiRequestFailure as ex:
            if "404" in str(ex.args[1]):
                print(
                    "Could not find the training run details for the given training run id. "
                )
                return
            else:
                raise ex
        status = run_details["entity"]["status"]["state"]

        if (
            status == "completed"
            or status == "error"
            or status == "failed"
            or status == "canceled"
        ):
            self._COS_metrics(
                training_id,  # type: ignore
                lambda: print_text_header_h1(
                    "Log monitor started for training run: " + str(training_id)
                ),
            )
        else:
            if self._client.CLOUD_PLATFORM_SPACES:
                ws_param = self._client._params()
                if "project_id" in ws_param.keys():
                    proj_id = ws_param.get("project_id")
                    monitor_endpoint = (
                        self._credentials.url.replace("https", "wss")
                        + "/ml/v4/trainings/"
                        + training_id
                        + "?project_id="
                        + proj_id
                    )
                else:
                    space_id = ws_param.get("space_id")
                    monitor_endpoint = (
                        self._credentials.url.replace("https", "wss")
                        + "/ml/v4/trainings/"
                        + training_id
                        + "?space_id="
                        + space_id
                    )
            else:
                monitor_endpoint = (
                    self._credentials.url.replace("https", "wss")
                    + "/v4/trainings/"
                    + training_id
                )
            websocket = WebSocket(monitor_endpoint)
            try:
                websocket.add_header(
                    bytes("Authorization", "utf-8"),
                    bytes("Bearer " + self._client.token, "utf-8"),
                )
            except:
                websocket.add_header(
                    bytes("Authorization", "utf-8"),
                    bytes("bearer " + self._client.token),
                )

            print_text_header_h1(
                "Metric monitor started for training run: " + str(training_id)
            )

            for event in websocket:
                if event.name == "text":
                    text = json.loads(event.text)
                    entity = text["entity"]
                    if "status" in entity:
                        status = entity["status"]
                        if "metrics" in status:
                            metrics = status["metrics"]
                            if len(metrics) > 0:
                                metric = metrics[0]
                                print(metric)

            websocket.close()

            print_text_header_h2("Metric monitor done.")

    def get_metrics(
        self, training_id: str | None = None, **kwargs: Any
    ) -> ListType[dict]:
        """Get metrics of a training run.

        :param training_id: ID of the training
        :type training_id: str

        :return: metrics of the training run
        :rtype: list of dict

        **Example:**

        .. code-block:: python

            training_status = client.training.get_metrics(training_id)

        """
        training_id = _get_id_from_deprecated_uid(
            kwargs, training_id, "training", can_be_none=False
        )

        Training._validate_type(training_id, "training_id", str, True)
        status = self.get_status(training_id)
        if "metrics" in status:
            return status["metrics"]
        else:
            details = self.get_details(training_id, _internal=True)
            if "metrics" in details:
                return details["metrics"]
            else:
                raise WMLClientError(
                    "No metrics details are available for the given training_id"
                )
