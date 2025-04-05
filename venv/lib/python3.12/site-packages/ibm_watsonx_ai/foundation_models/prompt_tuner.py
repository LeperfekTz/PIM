#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, cast

from ibm_watsonx_ai.foundation_models.base_tuner import BaseTuner
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.helpers.connections import (
    DataConnection,
)
from ibm_watsonx_ai.utils.autoai.utils import is_ipython
from ibm_watsonx_ai.foundation_models.utils import PromptTuningParams

import datetime


if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class PromptTuner(BaseTuner):
    id: str | None = None
    _client: APIClient = None  # type: ignore[assignment]
    _training_metadata: dict | None = None

    def __init__(
        self,
        name: str,
        task_id: str,
        *,
        description: str | None = None,
        base_model: str | None = None,
        accumulate_steps: int | None = None,
        batch_size: int | None = None,
        init_method: str | None = None,
        init_text: str | None = None,
        learning_rate: float | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        num_epochs: int | None = None,
        verbalizer: str | None = None,
        tuning_type: str | None = None,
        auto_update_model: bool = True,
        group_by_name: bool | None = None,
    ):
        BaseTuner.__init__(self, "prompt")
        self.name = name
        self.description = description if description else "Prompt tuning with SDK"
        self.auto_update_model = auto_update_model
        self.group_by_name = group_by_name

        base_model_red: dict = {"model_id": base_model}

        self.prompt_tuning_params = PromptTuningParams(
            base_model=base_model_red,
            accumulate_steps=accumulate_steps,
            batch_size=batch_size,
            init_method=init_method,
            init_text=init_text,
            learning_rate=learning_rate,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            num_epochs=num_epochs,
            task_id=task_id,
            tuning_type=tuning_type,
            verbalizer=verbalizer,
        )

        if not isinstance(self.name, str):
            raise WMLClientError(
                f"'name' param expected string, but got {type(self.name)}: {self.name}"
            )

        if self.description and (not isinstance(self.description, str)):
            raise WMLClientError(
                f"'description' param expected string, but got {type(self.description)}: "
                f"{self.description}"
            )

        if self.auto_update_model and (not isinstance(self.auto_update_model, bool)):
            raise WMLClientError(
                f"'auto_update_model' param expected bool, but got {type(self.auto_update_model)}: "
                f"{self.auto_update_model}"
            )

        if self.group_by_name and (not isinstance(self.group_by_name, bool)):
            raise WMLClientError(
                f"'group_by_name' param expected bool, but got {type(self.group_by_name)}: "
                f"{self.group_by_name}"
            )

    def run(
        self,
        training_data_references: list[DataConnection],
        training_results_reference: DataConnection | None = None,
        background_mode: bool = False,
    ) -> dict:
        """Run a prompt tuning process of a foundation model on top of the training data referenced by DataConnection.

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
            from ibm_watsonx_ai.helpers import DataConnection, S3Location

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)

            prompt_tuner.run(
                training_data_references=[DataConnection(
                    connection_asset_id=connection_id,
                    location=S3Location(
                        bucket='prompt_tuning_data',
                        path='pt_train_data.json')
                    )
                )]
                background_mode=False)
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
            test_data_references=None,
            training_results_reference=training_results_reference,
        )

        self._training_metadata = cast(dict, self._training_metadata)
        tuning_details = self._client.training.run(
            meta_props=self._training_metadata, asynchronous=background_mode
        )
        self.id = self._client.training.get_id(tuning_details)

        return self._client.training.get_details(
            self.id
        )  # TODO improve the background_mode = False option

    def _initialize_training_metadata(
        self,
        training_data_references: list[DataConnection],
        test_data_references: list[DataConnection] | None = None,
        training_results_reference: DataConnection | None = None,
    ) -> None:
        self._training_metadata = {
            self._client.training.ConfigurationMetaNames.TAGS: self._get_tags(),
            self._client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES: [
                connection._to_dict() for connection in training_data_references
            ],
            self._client.training.ConfigurationMetaNames.NAME: f"{self.name[:100]}",
            self._client.training.ConfigurationMetaNames.PROMPT_TUNING: self.prompt_tuning_params.to_dict(),
        }
        if test_data_references:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.TEST_DATA_REFERENCES
            ] = [connection._to_dict() for connection in test_data_references]
        if training_results_reference:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE
            ] = training_results_reference._to_dict()

        if self.description:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.DESCRIPTION
            ] = f"{self.description}"

        if self.auto_update_model is not None:
            self._training_metadata[
                self._client.training.ConfigurationMetaNames.AUTO_UPDATE_MODEL
            ] = self.auto_update_model

    def _get_tags(self) -> list:
        tags = ["prompt_tuning"]
        if self.group_by_name is not None and self.group_by_name:
            for training in self._client.training.get_details(
                tag_value="prompt_tuning"
            )["resources"]:
                if training["metadata"].get("name") == self.name:
                    # Find recent tags related to 'name'
                    tags = list(set(tags) | set(training["metadata"].get("tags")))
                    break

            if tags != ["prompt_tuning"]:
                self._client.generate_ux_tag = False

        return tags

    def get_params(self) -> dict:
        """Get configuration parameters of PromptTuner.

        :return: PromptTuner parameters
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)

            prompt_tuner.get_params()

            # Result:
            #
            # {'base_model': {'name': 'google/flan-t5-xl'},
            #  'task_id': 'summarization',
            #  'name': 'Prompt Tuning of Flan T5 model',
            #  'auto_update_model': False,
            #  'group_by_name': False}
        """

        params = self.prompt_tuning_params.to_dict()
        params["name"] = self.name
        params["description"] = self.description
        params["auto_update_model"] = self.auto_update_model
        params["group_by_name"] = self.group_by_name
        return params

    #####################
    #   Run operations  #
    #####################

    def get_run_status(self) -> str:
        """Check the status/state of an initialized prompt tuning run if it was run in background mode.

        :return: status of the Prompt Tuning run
        :rtype: str

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            prompt_tuner.get_run_details()

            # Result:
            # 'completed'
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_not_scheduled")
            )

        return self._client.training.get_status(training_id=self.id).get("state")  # type: ignore[return-value]

    def get_run_details(self, include_metrics: bool = False) -> dict:
        """Get details of a prompt tuning run.

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: details of the prompt tuning
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            prompt_tuner.get_run_details()
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_not_scheduled")
            )

        details = self._client.training.get_details(training_id=self.id)

        if include_metrics:
            try:
                details["entity"]["status"]["metrics"] = (
                    self._get_metrics_data_from_property_or_file(details)
                )
            except KeyError:
                pass
            finally:
                return details

        if details["entity"]["status"].get("metrics", False):
            del details["entity"]["status"]["metrics"]

        return details

    def plot_learning_curve(self) -> None:
        """Plot learning curves.

        .. note ::
            Available only for Jupyter notebooks.

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            prompt_tuner.plot_learning_curve()
        """
        if not is_ipython():
            raise WMLClientError(
                "Function `plot_learning_curve` is available only for Jupyter notebooks."
            )
        from ibm_watsonx_ai.utils.autoai.incremental import plot_learning_curve
        import matplotlib.pyplot as plt

        tuning_details = self.get_run_details(include_metrics=True)

        if "metrics" in tuning_details["entity"]["status"]:
            # average loss score for each epoch
            scores = self._get_average_loss_score_for_each_epoch(
                tuning_details=tuning_details, epoch=0
            )

            # date_time from the first and last iteration on each epoch
            if "data" in tuning_details["entity"]["status"]["metrics"][0]:
                date_times = [
                    datetime.datetime.strptime(
                        m_obj["data"]["timestamp"], "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    for m_obj in self._get_first_and_last_iteration_metrics_for_each_epoch(
                        tuning_details=tuning_details
                    )
                ]
            else:
                date_times = [
                    datetime.datetime.strptime(
                        m_obj["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
                    for m_obj in self._get_first_and_last_iteration_metrics_for_each_epoch(
                        tuning_details=tuning_details
                    )
                ]

            elapsed_time = []
            for i in range(1, len(date_times), 2):
                elapsed_time.append((date_times[i] - date_times[i - 1]).total_seconds())

            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            if scores:
                plot_learning_curve(
                    fig=fig,
                    axes=axes,
                    scores=scores,
                    fit_times=elapsed_time,
                    xlabels={"first_xlabel": "Epochs", "second_xlabel": "Epochs"},
                    titles={"first_plot": "Loss function"},
                )
        else:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_no_metrics")
            )

    def summary(self, scoring: str = "loss") -> DataFrame:
        """Print the details of PromptTuner models (prompt-tuned models).

        :param scoring: scoring metric for sorting pipelines,
            when not provided, uses loss one
        :type scoring: string, optional

        :return: computed models and metrics
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            prompt_tuner.summary()

            # Result:
            #                          Enhancements            Base model  ...         loss
            #       Model Name
            # Prompt_tuned_M_1      [prompt_tuning]     google/flan-t5-xl  ...     0.449197
        """

        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_not_scheduled")
            )

        from pandas import DataFrame

        details = self.get_run_details(include_metrics=True)

        metrics = details["entity"]["status"].get("metrics", [{}])[0]
        is_ml_metrics = "data" in metrics or "ml_metrics" in metrics

        if not is_ml_metrics:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_no_metrics")
            )

        columns = [
            "Model Name",
            "Enhancements",
            "Base model",
            "Auto store",
            "Epochs",
            scoring,
        ]
        values = []
        model_name = "model_" + self.id
        base_model_name = None
        epochs = None
        enhancements = []
        if scoring == "loss":
            model_metrics = [
                self._get_average_loss_score_for_each_epoch(
                    tuning_details=details, epoch=0
                )[-1]
            ]
        else:
            if "data" in details["entity"]["status"]["metrics"][0]:
                model_metrics = [
                    details["entity"]["status"]
                    .get("metrics", [{}])[-1]
                    .get("data", {})[scoring]
                ]
            else:
                model_metrics = [
                    details["entity"]["status"]
                    .get("metrics", [{}])[-1]
                    .get("ml_metrics", {})[scoring]
                ]

        if "prompt_tuning" in details["entity"]:
            enhancements = [details["entity"]["prompt_tuning"]["tuning_type"]]
            base_model_name = details["entity"]["prompt_tuning"]["base_model"][
                "model_id"
            ]
            epochs = details["entity"]["prompt_tuning"]["num_epochs"]

        values.append(
            (
                [model_name]
                + [enhancements]
                + [base_model_name]
                + [details["entity"]["auto_update_model"]]
                + [epochs]
                + model_metrics
            )
        )

        summary = DataFrame(data=values, columns=columns)
        summary.set_index("Model Name", inplace=True)

        return summary

    def get_model_id(self) -> str:
        """Get the model ID.

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            prompt_tuner.get_model_id()
        """

        run_details = self.get_run_details()
        if run_details["entity"]["auto_update_model"]:
            return run_details["entity"]["model_id"]
        else:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_no_model_id")
            )

    def cancel_run(self, hard_delete: bool = False) -> None:
        """Cancel or delete a Prompt Tuning run.

        :param hard_delete: if True, the completed or cancelled prompt tuning run is deleted,
                            if False, the current run is canceled. Default: False
        :type hard_delete: bool, optional
        """
        if self.id is None:
            raise WMLClientError(
                Messages.get_message(message_id="fm_prompt_tuning_not_scheduled")
            )

        self._client.training.cancel(training_id=self.id, hard_delete=hard_delete)

    def get_data_connections(self) -> list[DataConnection]:
        """Create DataConnection objects for further usage
            (eg. to handle data storage connection).

        :return: list of DataConnections
        :rtype: list['DataConnection']

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            experiment = TuneExperiment(credentials, ...)
            prompt_tuner = experiment.prompt_tuner(...)
            prompt_tuner.run(...)

            data_connections = prompt_tuner.get_data_connections()
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
