#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from logging import Logger
from typing import Any

import pandas as pd

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.utils import print_text_header_h1, StatusLogger
from ibm_watsonx_ai.wml_resource import WMLResource


def get_status(asked_object: Any) -> tuple[str, dict, dict]:
    run_details = asked_object.get_run_details()
    status = run_details["entity"].get("status", {})
    state = status.get("state")

    return state, status, run_details


def wait_for_run_finish(asked_object: Any, res_name: str, logger: Logger) -> dict:
    print_text_header_h1("Running '{}'".format(asked_object.id))

    state, status, run_details = get_status(asked_object)

    with StatusLogger(state) as status_logger:
        while state not in ["error", "completed", "completed_at", "canceled", "failed"]:
            state, status, run_details = get_status(asked_object)
            status_logger.log_state(state)

    if "completed" in state:
        print(
            "\n{} of '{}' finished successfully.".format(res_name, str(asked_object.id))
        )
    else:
        print(
            "\n{} of '{}' failed with status: '{}'.".format(
                res_name, asked_object.id, str(status)
            )
        )

    logger.debug("Response({}): {}".format(state, run_details))
    return run_details


class BaseRuns(WMLResource):
    def __init__(
        self,
        name: str,
        client: APIClient,
        url: str,
        filter: str | None = None,
        limit: int = 50,
    ) -> None:
        WMLResource.__init__(self, name, client)
        self._client = client
        self.url = url
        self.tuning_name = filter
        self.limit = limit

    def __call__(self, *, filter: str | None = None, limit: int = 50) -> BaseRuns:
        self.tuning_name = filter
        self.limit = limit
        return self

    def list(self) -> pd.DataFrame:
        columns = ["timestamp", "run_id", "state", "name"]

        pt_runs_details = self._get_all_run_details()

        records: list = []
        for run in pt_runs_details["resources"]:
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

        runs = pd.DataFrame(data=records, columns=columns)
        return runs.sort_values(by=["timestamp"], ascending=False)

    def _get_all_run_details(self, run_id: str | None = None) -> dict:
        url = self.url

        return self._get_artifact_details(
            base_url=url,
            id=run_id,
            limit=self.limit,
            resource_name="getting resources",
        )

    def get_run_details(self, run_id: str | None = None) -> dict:
        if run_id is not None:
            return self._get_all_run_details(run_id)
        else:
            return self._get_all_run_details()["resources"][0]
