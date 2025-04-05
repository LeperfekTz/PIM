#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Any, TYPE_CHECKING, Literal

from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import (
    CannotSetProjectOrSpace,
    ExceededLimitOfAPICalls,
)
from ibm_watsonx_ai.service_instance import ServiceInstance

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class Set(WMLResource):
    """Set a space_id or a project_id to be used in the subsequent actions."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def default_space(
        self, space_id: str | None = None, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Set a space ID.

        :param space_id: ID of the space to be used
        :type space_id: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example:**

        .. code-block:: python

            client.set.default_space(space_id)

        """
        space_id = _get_id_from_deprecated_uid(
            kwargs, space_id, "space", can_be_none=False
        )

        space_endpoint = (
            self._client.service_instance._href_definitions.get_platform_space_href(
                space_id
            )
        )

        space_details = self._client._session.get(
            space_endpoint, headers=self._client._get_headers()
        )
        if space_details.status_code == 404:
            error_msg = "Space with id '{}' does not exist".format(space_id)
            raise CannotSetProjectOrSpace(reason=error_msg)

        elif space_details.status_code == 200:
            self._client.default_space_id = space_id
            if self._client.default_project_id is not None:
                print("Unsetting the project_id ...")
            self._client.default_project_id = None
            self._client.project_type = None

            if self._client.CLOUD_PLATFORM_SPACES:
                instance_id = "not found"
                comp_obj_type = None
                space_details_json = space_details.json()
                if "compute" in space_details.json()["entity"].keys():
                    if "type" in space_details_json["entity"]["compute"][0]:
                        if space_details_json["entity"]["compute"][0]["type"] in [
                            "machine_learning",
                            "code-assistant",
                        ]:
                            instance_id = space_details_json["entity"]["compute"][0][
                                "guid"
                            ]
                            comp_obj_type = space_details_json["entity"]["compute"][0][
                                "type"
                            ]
                    self._client.service_instance = ServiceInstance(self._client)
                    self._client.service_instance._instance_id = instance_id
                    if comp_obj_type == "code-assistant":
                        self._client.WCA = True
                        self._client.service_instance.details = None
                    else:
                        self._client.service_instance._refresh_details = True
                else:
                    # It's possible that a previous space is used in the context of
                    # this client which had compute but this space doesn't have
                    self._client.service_instance = ServiceInstance(self._client)
                    self._client.service_instance.details = None
            return "SUCCESS"
        else:
            raise CannotSetProjectOrSpace(reason=space_details.text)

    # Setting project ID
    def default_project(self, project_id: str) -> Literal["SUCCESS"]:
        """Set a project ID.

        :param project_id: ID of the project to be used
        :type project_id: str

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example:**

        .. code-block:: python

            client.set.default_project(project_id)
        """

        if project_id is not None:
            self._client.default_project_id = project_id

            if self._client.default_space_id is not None:
                print("Unsetting the space_id ...")
            self._client.default_space_id = None

            project_endpoint = (
                self._client.service_instance._href_definitions.get_project_href(
                    project_id
                )
            )
            project_details = self._client._session.get(
                project_endpoint, headers=self._client._get_headers()
            )
            if project_details.status_code == 429:
                raise ExceededLimitOfAPICalls(
                    project_endpoint, reason=project_details.text
                )
            elif (
                project_details.status_code != 200
                and project_details.status_code != 204
            ):
                raise CannotSetProjectOrSpace(reason=project_details.text)
            else:
                self._client.project_type = project_details.json()["entity"]["storage"][
                    "type"
                ]
                if self._client.CLOUD_PLATFORM_SPACES:
                    instance_id = "not_found"
                    comp_obj_type = None
                    if "compute" in project_details.json()["entity"].keys():
                        for comp_obj in project_details.json()["entity"]["compute"]:
                            if comp_obj["type"] in [
                                "machine_learning",
                                "code-assistant",
                            ]:
                                comp_obj_type = comp_obj["type"]
                                instance_id = comp_obj["guid"]
                                break
                        self._client.service_instance = ServiceInstance(self._client)
                        self._client.service_instance._instance_id = instance_id
                        if comp_obj_type == "code-assistant":
                            self._client.service_instance.details = None
                            self._client.WCA = True
                        else:
                            self._client.service_instance._refresh_details = True
                    else:
                        # It`s possible that a previous project is used in the context of
                        # this client which had compute but this project doesn't have
                        self._client.service_instance = ServiceInstance(self._client)
                        self._client.service_instance.details = None
                else:
                    self._client.service_instance = ServiceInstance(self._client)
                return "SUCCESS"
        else:
            error_msg = "Project id cannot be None."
            raise CannotSetProjectOrSpace(reason=error_msg)
