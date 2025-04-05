#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any
from warnings import warn

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid

from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from pandas import DataFrame
    from ibm_watsonx_ai import APIClient


class TaskCredentials(WMLResource):
    """Store and manage your task credentials."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        task_credentials_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get task credentials details. If no task_credentials_id is passed, details for all task credentials
        will be returned.

        :param task_credentials_id: ID of task credentials to be fetched
        :type task_credentials_id: str, optional

        :param project_id: ID of project to be used for filtering
        :type project_id: str, optional

        :param space_id: ID of space to be used for filtering
        :type space_id: str, optional

        :return: created task credentials details
        :rtype: dict (if task_credentials_id is not None) or {"resources": [dict]} (if task_credentials_id is None)

        **Example:**

        .. code-block:: python

            task_credentials_details = client.task_credentials.get_details(task_credentials_id)

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Task Credentials API is supported on Cloud only.")

        task_credentials_id = _get_id_from_deprecated_uid(
            kwargs, task_credentials_id, "task_credentials", True
        )

        # TaskCredentials._validate_type(task_credentials_id, u'task_credentials_id', STR_TYPE, False)

        if task_credentials_id:
            response = requests.get(
                self._client.service_instance._href_definitions.get_task_credentials_href(
                    task_credentials_id
                ),
                headers=self._client._get_headers(),
            )

            return self._handle_response(
                200,
                "get task credentials details",
                response,
                _silent_response_logging=True,
            )
        else:
            params = {}

            if project_id := kwargs.get("project_id"):
                params["project_id"] = project_id
            elif space_id := kwargs.get("space_id"):
                params["space_id"] = space_id

            response = requests.get(
                self._client.service_instance._href_definitions.get_task_credentials_all_href(),
                params=params,
                headers=self._client._get_headers(),
            )

            return {
                "resources": self._handle_response(
                    200,
                    "get task credentials details",
                    response,
                    _silent_response_logging=True,
                ).get("credentials", {})
            }

    def store(
        self, name: str | None = None, description: str | None = None, **kwargs: Any
    ) -> dict:
        """Store current credentials using Task Credentials API to use with long run tasks. Supported only on Cloud.

        :param name: Name of the task credentials. Defaults to `Python API generated task credentials`
        :type name: str, optional

        :param description: Description of the task credentials. Defaults to `Python API generated task credentials`
        :type description: str, optional

        :return: A dictionary containing metadata of the stored task credentials.
        :rtype: dict

        **Example:**

        .. code-block:: python

            task_credentials_details = client.task_credentials.store()

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Task Credentials API is supported on Cloud only.")

        href = (
            self._client.service_instance._href_definitions.get_task_credentials_all_href()
        )

        if kwargs.get("project_id") is not None or kwargs.get("space_id") is not None:
            scope_fields_not_supported_warning = "Scope fields: project_id/space_id are not yet supported by Task Credentials Service."
            warn(scope_fields_not_supported_warning)

        if not self.list().empty:
            raise WMLClientError(
                "Task Credentials have already been stored. Use old or delete them."
            )

        creation_response = requests.post(
            href,
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
            json={
                "name": name if name else "Python API generated task credentials",
                "description": (
                    description
                    if description
                    else "Python API generated task credentials."
                ),
                "type": "iam_api_key",
                "secret": {"api_key": self._client.credentials.api_key},
            },
        )

        return self._handle_response(
            201,
            "creating task credentials",
            creation_response,
            _silent_response_logging=True,
        )

    def list(
        self,
        limit: int | None = None,
        **kwargs: Any,
    ) -> DataFrame:
        """Lists task credentials in table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed assets
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.task_credentials.list()

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Task Credentials API is supported on Cloud only.")

        details = self.get_details(**kwargs)

        task_credentials_details = details["resources"]
        task_credentials_values = [
            (m["name"], m["id"], m["scope"]) for m in task_credentials_details
        ]

        return self._list(
            task_credentials_values,
            ["NAME", "ASSET_ID", "TYPE"],
            limit,
        )

    @staticmethod
    def get_id(task_credentials_details: dict) -> str:
        """Get Unique Id of task credentials.

        :param task_credentials_details: metadata of the task credentials
        :type task_credentials_details: dict

        :return: Unique Id of task credentials
        :rtype: str

        **Example:**

        .. code-block:: python

            task_credentials_id = client.task_credentials.get_id(task_credentials_details)

        """
        return task_credentials_details["id"]

    def delete(self, task_credentials_id: str, **kwargs: Any) -> Literal["SUCCESS"]:
        """Delete a software specification.

        :param task_credentials_id: Unique Id of task credentials
        :type task_credentials_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.task_credentials.delete(task_credentials_id)

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Task Credentials API is supported on Cloud only.")

        task_credentials_id = _get_id_from_deprecated_uid(
            kwargs, task_credentials_id, "task_credentials"
        )

        TaskCredentials._validate_type(
            task_credentials_id, "task_credentials_id", str, True
        )

        params = self._client._params(skip_for_create=True)

        if project_id := kwargs.get("project_id"):
            params["project_id"] = project_id
        if space_id := kwargs.get("space_id"):
            params["space_id"] = space_id

        if project_id or space_id:
            scope_fields_not_supported_warning = "Scope fields: project_id/space_id are not yet supported by Task Credentials Service."
            warn(scope_fields_not_supported_warning)

        response = requests.delete(
            self._client.service_instance._href_definitions.get_task_credentials_href(
                task_credentials_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            return self._handle_response(
                200, "delete task credentials", response, _silent_response_logging=True
            )
        else:
            return self._handle_response(
                204, "delete task credentials", response, _silent_response_logging=True
            )
