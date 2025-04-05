#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from datetime import datetime

from pandas import DataFrame

from ibm_watsonx_ai.foundation_models.ilab_tuner import ILabTuner
from ibm_watsonx_ai.foundation_models.prompt_tuner import PromptTuner
from ibm_watsonx_ai.foundation_models.fine_tuner import FineTuner
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
)
from ibm_watsonx_ai.foundation_models.utils.utils import (
    _is_fine_tuning_endpoint_available,
)
from ibm_watsonx_ai import APIClient

__all__ = ["TuneRuns"]


class TuneRuns:
    """The TuneRuns class is used to work with historical PromptTuner and FineTuner runs.

    :param client: APIClient to handle service operations
    :type client: APIClient

    :param filter: filter, choose which runs specifying the tuning name to fetch
    :type filter: str, optional

    :param limit: int number of records to be returned
    :type limit: int
    """

    def __init__(
        self, client: APIClient, filter: str | None = None, limit: int = 50
    ) -> None:

        self.client = client
        self.tuning_name = filter
        self.limit = limit

        self._is_fine_tuning_endpoint_available = _is_fine_tuning_endpoint_available(
            self.client
        )

    def __call__(self, *, filter: str | None = None, limit: int = 50) -> TuneRuns:
        self.tuning_name = filter
        self.limit = limit
        return self

    def list(self) -> DataFrame:
        """Lists historical runs with their status. If you have a lot of runs stored in the service,
        it might take a longer time to fetch all the information. If there is no limit set, it
        gets the last 50 records.

        :return: Pandas DataFrame with run IDs and status
        :rtype: pandas.DataFrame

        **Examples**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(...)
            df = experiment.runs.list()
        """

        columns = ["timestamp", "run_id", "state", "tuning name"]

        pt_runs_details = self.client.training.get_details(
            get_all=True if self.tuning_name else False,
            limit=None if self.tuning_name else self.limit,
            training_type="prompt_tuning",
            _internal=True,
        )

        records: list = []
        for run in pt_runs_details["resources"]:
            if len(records) >= self.limit:
                break

            if {"entity", "metadata"}.issubset(run.keys()):

                timestamp = run["metadata"].get("modified_at")
                run_id = run["metadata"].get("id", run["metadata"].get("guid"))
                state = run["entity"].get("status", {}).get("state")
                tuning_name = run["entity"].get("name", "Unknown")

                record = [timestamp, run_id, state, tuning_name]

                if self.tuning_name is None or (
                    self.tuning_name and self.tuning_name == tuning_name
                ):
                    records.append(record)

        if self._is_fine_tuning_endpoint_available:

            ft_runs_details = self.client.training.get_details(
                get_all=True if self.tuning_name else False,
                limit=None if self.tuning_name else self.limit,
                _internal=True,
                _is_fine_tuning=True,
            )["resources"]

            ilabt_runs_details = self.client.training.get_details(
                get_all=True if self.tuning_name else False,
                limit=None if self.tuning_name else self.limit,
                training_type="ilab",
                _internal=True,
                _is_fine_tuning=True,
            )["resources"]

            ft_runs_details.extend(ilabt_runs_details)

            for run in ft_runs_details:
                if len(records) >= self.limit:
                    break

                if {"entity", "metadata"}.issubset(run.keys()):

                    timestamp = run["metadata"].get("modified_at")
                    run_id = run["metadata"].get("id", run["metadata"].get("guid"))
                    state = run["entity"].get("status", {}).get("state")
                    tuning_name = run["metadata"].get("name", "Unknown")

                    record = [timestamp, run_id, state, tuning_name]

                    if self.tuning_name is None or (
                        self.tuning_name and self.tuning_name == tuning_name
                    ):
                        records.append(record)

        runs = DataFrame(data=records, columns=columns)
        return runs.sort_values(by=["timestamp"], ascending=False)

    def get_tuner(self, run_id: str) -> PromptTuner | FineTuner | ILabTuner:
        """Create an instance of PromptTuner or FineTuner or ILabTuner based on a tuning run with a specific run_id.

        :param run_id: ID of the run
        :type run_id: str

        :return: prompt tuner | fine | ilab tuner object
        :rtype: PromptTuner | FineTuner | ILabTuner class instance

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment

            experiment = TuneExperiment(credentials, ...)
            historical_tuner = experiment.runs.get_tuner(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')
        """
        # note: normal scenario

        if not isinstance(run_id, str):
            raise WMLClientError(
                f"Provided run_id type was {type(run_id)} (should be a string)"
            )
        if self._is_fine_tuning_endpoint_available:
            try:
                tuning_details = self.client.training.get_details(
                    run_id, _is_fine_tuning=True
                )
                entity = tuning_details.get("entity", {})
                tuning_type = (
                    "ilab_tuning" if entity.get("type") == "ilab" else "fine_tuning"
                )
            except ApiRequestFailure:
                tuning_details = self.client.training.get_details(run_id)
                entity = tuning_details.get("entity")
                tuning_type = "prompt_tuning"
        else:
            tuning_details = self.client.training.get_details(run_id)
            entity = tuning_details.get("entity")
            tuning_type = "prompt_tuning"

        if not entity:
            raise WMLClientError("Provided run_id was invalid")

        tuner: PromptTuner | FineTuner | ILabTuner

        match tuning_type:
            case "prompt_tuning":
                tuning_params = entity["prompt_tuning"]
                tuner = PromptTuner(
                    name=entity.get("name"),
                    task_id=tuning_params.get("task_id"),
                    description=entity.get("description"),
                    base_model=tuning_params.get("base_model", {}).get("name"),
                    accumulate_steps=tuning_params.get("accumulate_steps"),
                    batch_size=tuning_params.get("batch_size"),
                    init_method=tuning_params.get("init_method"),
                    init_text=tuning_params.get("init_text"),
                    learning_rate=tuning_params.get("learning_rate"),
                    max_input_tokens=tuning_params.get("max_input_tokens"),
                    max_output_tokens=tuning_params.get("max_output_tokens"),
                    num_epochs=tuning_params.get("num_epochs"),
                    tuning_type=tuning_params.get("tuning_type"),
                    verbalizer=tuning_params.get("verbalizer"),
                    auto_update_model=entity.get("auto_update_model"),
                )
                tuner._client = self.client

            case "ilab_tuning":
                tuner = ILabTuner(tuning_details["metadata"].get("name"), self.client)

            case "fine_tuning":
                tuning_params = entity["parameters"]
                tuner = FineTuner(
                    name=tuning_details["metadata"].get("name"),
                    task_id=tuning_params.get("task_id"),
                    description=tuning_details["metadata"].get("description"),
                    base_model=tuning_params.get("base_model", {}).get("model_id"),
                    num_epochs=tuning_params.get("num_epochs"),
                    learning_rate=tuning_params.get("learning_rate"),
                    batch_size=tuning_params.get("batch_size"),
                    max_seq_length=tuning_params.get("max_seq_length"),
                    accumulate_steps=tuning_params.get("accumulate_steps"),
                    verbalizer=tuning_params.get("verbalizer"),
                    response_template=tuning_params.get("response_template"),
                    gpu=tuning_params.get("gpu"),
                    peft_parameters=tuning_params.get("peft_parameters"),
                    auto_update_model=entity.get("auto_update_model"),
                    api_client=self.client,
                    gradient_checkpointing=tuning_params.get("gradient_checkpointing"),
                )
            case _:
                raise WMLClientError("Not supported tuning type")

        tuner.id = run_id
        return tuner

    def get_run_details(
        self, run_id: str | None = None, include_metrics: bool = False
    ) -> dict:
        """Get run details. If run_id is not supplied, the last run will be taken.

        :param run_id: ID of the run
        :type run_id: str, optional

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: configuration parameters of the run
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import TuneExperiment
            experiment = TuneExperiment(credentials, ...)

            experiment.runs.get_run_details(run_id='02bab973-ae83-4283-9d73-87b9fd462d35')
            experiment.runs.get_run_details()
        """
        if run_id is None:
            if self._is_fine_tuning_endpoint_available:
                try:
                    resources: list = self.client.training.get_details(  # type: ignore[assignment]
                        limit=1,
                        _internal=True,
                        _is_fine_tuning=True,
                    ).get(
                        "resources"
                    )
                    resources.extend(
                        self.client.training.get_details(
                            limit=1,
                            training_type="ilab",
                            _internal=True,
                            _is_fine_tuning=True,
                        ).get("resources", [])
                    )
                except ApiRequestFailure:
                    resources = self.client.training.get_details(  # type: ignore[assignment]
                        limit=1, training_type="prompt_tuning", _internal=True
                    ).get(
                        "resources"
                    )
                else:
                    resources.extend(
                        self.client.training.get_details(  # type: ignore[arg-type]
                            limit=1, training_type="prompt_tuning", _internal=True
                        ).get("resources")
                    )
            else:
                resources = self.client.training.get_details(  # type: ignore[assignment]
                    limit=1, training_type="prompt_tuning", _internal=True
                ).get(
                    "resources"
                )
            if len(resources) == 1:
                details = resources[0]
            elif len(resources) >= 2:
                timestamps = {}
                for i, r in enumerate(resources):
                    try:
                        timestamps[i] = datetime.fromisoformat(
                            r["metadata"]["modified_at"].replace("Z", "")
                        )
                    except KeyError:
                        timestamps[i] = datetime.fromisoformat(
                            r["metadata"]["created_at"].replace("Z", "")
                        )
                details = resources[max(timestamps, key=timestamps.__getitem__)]
            else:
                raise WMLClientError("There is no available training run to retrieve.")
        else:
            if self._is_fine_tuning_endpoint_available:
                try:
                    details = self.client.training.get_details(
                        training_id=run_id, _internal=True, _is_fine_tuning=True
                    )
                except ApiRequestFailure:
                    details = self.client.training.get_details(
                        training_id=run_id, _internal=True
                    )
            else:
                details = self.client.training.get_details(
                    training_id=run_id, _internal=True
                )

        if include_metrics:
            return details

        if details["entity"]["status"].get("metrics", False):
            del details["entity"]["status"]["metrics"]

        return details
