#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.service_instance import ServiceInstance

from ibm_watsonx_ai.metanames import ProjectsMetaNames, MemberMetaNames

from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ResourceIdByNameNotFound,
    MultipleResourceIdByNameFound,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class Projects(WMLResource):
    """Store and manage projects."""

    ConfigurationMetaNames = ProjectsMetaNames()
    """MetaNames for projects creation."""

    MemberMetaNames = MemberMetaNames()
    """MetaNames for project members creation."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)
        self._client = client

    def _get_resources(
        self, url: str, op_name: str, params: dict | None = None
    ) -> dict:
        if params is not None and "limit" in params.keys():
            if params["limit"] < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif params["limit"] > 1000:
                raise WMLClientError("Limit cannot be larger than 1000.")

        if params is not None and len(params) > 0:
            response_get = requests.get(
                url, headers=self._client._get_headers(), params=params
            )

            return self._handle_response(200, op_name, response_get)
        else:

            resources = []

            while True:
                response_get = requests.get(url, headers=self._client._get_headers())

                result = self._handle_response(200, op_name, response_get)
                resources.extend(result["resources"])

                if "next" not in result:
                    break
                else:
                    url = self._credentials.url + result["next"]["href"]
                    if "start=invalid" in url:
                        break
            return {"resources": resources}

    def store(self, meta_props: dict) -> dict:
        """Create a project.

        :param meta_props: metadata of the project configuration. To see available meta names, use:

            .. code-block:: python

                client.projects.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored project
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.projects.ConfigurationMetaNames.NAME: "my project",
                client.projects.ConfigurationMetaNames.DESCRIPTION: "test project",
                client.projects.ConfigurationMetaNames.STORAGE: {
                    "type": "assetfiles"
                }
            }

            projects_details = client.projects.store(meta_props)
        """
        Projects._validate_type(meta_props, "meta_props", dict, True)

        if self.ConfigurationMetaNames.GENERATOR not in meta_props:
            meta_props[self.ConfigurationMetaNames.GENERATOR] = "Watsonx-Python-SDK"

        if "compute" in meta_props:
            if "name" not in meta_props["compute"]:
                raise WMLClientError("'name' is mandatory for 'COMPUTE'")
            temp_meta = meta_props["compute"]
            temp_meta.update({"type": "machine_learning"})

            meta_props["compute"] = temp_meta

        project_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        if "compute" in project_meta:
            project_meta["compute"] = [project_meta["compute"]]

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_transactional_projects_href(),
            headers=self._client._get_headers(),
            json=project_meta,
        )

        location = self._handle_response(
            201, "creating new project", creation_response
        )["location"]

        project_details = self.get_details(location.split("/")[-1])

        if "compute" in project_details["entity"].keys():
            instance_id = project_details["entity"]["compute"][0]["guid"]
            self._client.service_instance = ServiceInstance(self._client)
            self._client.service_instance._instance_id = instance_id

        return project_details

    @staticmethod
    def get_id(project_details: dict) -> str:
        """Get the project_id from the project details.

        :param project_details: metadata of the stored project
        :type project_details: dict

        :return: ID of the stored project
        :rtype: str

        **Example:**

        .. code-block:: python

            project_details = client.projects.store(meta_props)
            project_id = client.projects.get_id(project_details)
        """

        Projects._validate_type(project_details, "project_details", object, True)

        return WMLResource._get_required_element_from_dict(
            project_details, "project_details", ["metadata", "guid"]
        )

    def get_id_by_name(self, project_name: str) -> str:
        """Get the ID of a stored project by name.

        :param project_name: name of the stored project
        :type project_name: str

        :return: ID of the stored project
        :rtype: str

        **Example:**

        .. code-block:: python

            project_id = client.projects.get_id_by_name(project_name)

        """

        Projects._validate_type(project_name, "project_name", str, True)

        details = self.get_details(project_name=project_name)["resources"]

        if len(details) > 1:
            raise MultipleResourceIdByNameFound(project_name, "project")
        elif len(details) == 0:
            raise ResourceIdByNameNotFound(project_name, "project")

        return self.get_id(details[0])

    def delete(self, project_id: str) -> Literal["SUCCESS"]:
        """Delete a stored project.

        :param project_id: ID of the project
        :type project_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.projects.delete(project_id)
        """
        Projects._validate_type(project_id, "project_id", str, True)

        project_endpoint = self._client.service_instance._href_definitions.get_transactional_project_href(
            project_id
        )

        response_delete = requests.delete(
            project_endpoint, headers=self._client._get_headers()
        )

        response = self._handle_response(
            204, "project deletion", response_delete, False
        )

        print("DELETED")

        return "SUCCESS"

    def get_details(
        self,
        project_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool | None = False,
        get_all: bool | None = False,
        project_name: str | None = None,
    ) -> dict:
        """Get metadata of stored project(s).

        :param project_id: ID of the project
        :type project_id: str, optional
        :param limit: applicable when `project_id` is not provided, otherwise `limit` will be ignored
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all:  if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional
        :param project_name: name of the stored project, can be used only when `project_id` is None
        :type project_name: str, optional

        :return: metadata of stored project(s)
        :rtype:
            - **dict** - if project_id is not None
            - **{"resources": [dict]}** - if project_id is None

        **Example:**

        .. code-block:: python

            project_details = client.project.get_details(project_id)
            project_details = client.project.get_details(project_name)
            project_details = client.project.get_details(limit=100)
            project_details = client.project.get_details(limit=100, get_all=True)
            project_details = []
            for entry in client.project.get_details(limit=100, asynchronous=True, get_all=True):
                project_details.extend(entry)

        """
        Projects._validate_type(project_id, "project_id", str, False)

        href = self._client.service_instance._href_definitions.get_project_href(
            project_id
        )

        if project_id is not None:
            response_get = requests.get(href, headers=self._client._get_headers())

            return self._handle_response(200, "Get project", response_get)

        else:
            query_params = {"name": project_name} if project_name else None

            return self._get_with_or_without_limit(
                self._client.service_instance._href_definitions.get_projects_href(),
                100 if not limit or limit > 100 else limit,
                "projects",
                summary=False,
                pre_defined=False,
                skip_space_project_chk=True,
                query_params=query_params,
                _async=asynchronous,
                _all=get_all,
            )

    def list(
        self,
        limit: int | None = None,
        member: str | None = None,
        roles: str | None = None,
        project_type: str | None = None,
    ) -> DataFrame:
        """List stored projects in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional
        :param member: filters the result list, only includes projects where the user with a matching user ID
            is a member
        :type member: str, optional
        :param roles: a list of comma-separated project roles to use to filter the query results,
            must be used in conjunction with the "member" query parameter,
            available values : `admin`, `editor`, `viewer`
        :type roles: str, optional
        :param project_type: filter projects by their type, available types are 'cpd', 'wx', 'wca', 'dpx' and 'wxbi'
        :type project_type: str, optional

        :return: pandas.DataFrame with listed projects
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.projects.list()
        """

        Projects._validate_type(limit, "limit", int, False)
        href = self._client.service_instance._href_definitions.get_projects_href()

        params: dict[str, Any] = {}

        limit = 100 if not limit or limit > 100 else limit

        if member is not None:
            params.update({"member": member})

        if roles is not None:
            params.update({"roles": roles})

        if project_type is not None:
            params.update({"type": project_type})

        projects_resources = [
            m
            for r in self._get_with_or_without_limit(
                href,
                limit,
                "projects",
                summary=False,
                pre_defined=False,
                skip_space_project_chk=True,
                query_params=params,
                _async=True,
                _all=True,
            )
            for m in r["resources"]
        ]

        project_values = [
            (m["metadata"]["guid"], m["entity"]["name"], m["metadata"]["created_at"])
            for m in projects_resources
        ]

        table = self._list(project_values, ["ID", "NAME", "CREATED"], limit)
        return table

    def update(self, project_id: str, changes: dict) -> dict:
        """Update existing project metadata. 'STORAGE' cannot be updated. #TODO

        :param project_id: ID of the project with the definition to be updated
        :type project_id: str
        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated project
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.projects.ConfigurationMetaNames.NAME:"updated_project",
                client.projects.ConfigurationMetaNames.COMPUTE: {"name": "test_instance",
                                                               "crn": "v1:staging:public:pm-20-dev:us-south:a/09796a1b4cddfcc9f7fe17824a68a0f8:f1026e4b-77cf-4703-843d-c9984eac7272::"
                }
            }
            project_details = client.projects.update(project_id, changes=metadata)
        """
        if "storage" in changes:
            raise WMLClientError("STORAGE cannot be updated")

        if "generator" in changes:
            raise WMLClientError("GENERATOR cannot be updated")

        if "scope" in changes:
            raise WMLClientError("SCOPE cannot be updated")

        if "creator" in changes:
            raise WMLClientError("creator cannot be updated")

        if "creator_iam_id" in changes:
            raise WMLClientError("creator_iam_id cannot be updated")

        self._validate_type(project_id, "project_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self.get_details(project_id)

        if "compute" in changes:
            changes["compute"]["type"] = "machine_learning"

            payload_compute = []
            payload_compute.append(changes["compute"])
            changes["compute"] = payload_compute

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details["entity"], changes
        )

        payload = details["entity"]

        def modify(tree, path, value):
            if len(path) > 1:
                modify(tree[path[0]], path[1:], value)
            else:
                tree[path[0]] = value

        for r in patch_payload:
            path = r["path"].strip("/").split("/")
            modify(payload, path, r["value"])

        for key in ["storage", "generator", "scope", "creator", "creator_iam_id"]:
            if key in payload:
                payload.pop(key)

        href = self._client.service_instance._href_definitions.get_project_href(
            project_id
        )

        response = requests.patch(
            href, json=payload, headers=self._client._get_headers()
        )

        updated_details = self._handle_response(200, "projects patch", response)

        # Cloud Convergence
        if "compute" in updated_details["entity"].keys():
            instance_id = updated_details["entity"]["compute"][0]["guid"]
            self._client.service_instance = ServiceInstance(self._client)
            self._client.service_instance._instance_id = instance_id

        return updated_details

    #######SUPPORT FOR PROJECT MEMBERS

    def create_member(self, project_id: str, meta_props: dict) -> dict:
        """Create a member within a project.

        :param project_id: ID of the project with the definition to be updated
        :type project_id: str
        :param meta_props: metadata of the member configuration. To see available meta names, use:

            .. code-block:: python

                client.projects.MemberMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored member
        :rtype: dict

        .. note::
            * `role` can be any one of the following: "viewer", "editor", "admin"
            * `type` can be any one of the following: "user", "service"
            * `id` can be one of the following: service-ID or IAM-userID

        **Examples**

        .. code-block:: python

            metadata = {
                client.projects.MemberMetaNames.MEMBERS: [{"id":"IBMid-100000DK0B",
                                                         "type": "user",
                                                         "role": "admin" }]
            }
            members_details = client.projects.create_member(project_id=project_id, meta_props=metadata)

        .. code-block:: python

            metadata = {
                client.projects.MemberMetaNames.MEMBERS: [{"id":"iam-ServiceId-5a216e59-6592-43b9-8669-625d341aca71",
                                                         "type": "service",
                                                         "role": "admin" }]
            }
            members_details = client.projects.create_member(project_id=project_id, meta_props=metadata)
        """
        self._validate_type(project_id, "project_id", str, True)

        Projects._validate_type(meta_props, "meta_props", dict, True)

        meta = {}

        if "members" in meta_props:
            meta = meta_props
        elif "member" in meta_props:
            dictionary = meta_props["member"]
            payload = []
            payload.append(dictionary)
            meta["members"] = payload

        project_meta = self.MemberMetaNames._generate_resource_metadata(
            meta, with_validation=True, client=self._client
        )

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_projects_members_href(
                project_id
            ),
            headers=self._client._get_headers(),
            json=project_meta,
        )

        # TODO: Change response code one they change it to 201
        members_details = self._handle_response(
            200, "creating new members", creation_response
        )

        return members_details

    def get_member_details(self, project_id: str, user_name: str | None = None) -> dict:
        """Get metadata of a member associated with a project. If no user_name is passed, all members details will be returned.

        :param project_id: ID of that project with the definition to be updated
        :type project_id: str
        :param user_name: name of the member
        :type user_name: str, optional

        :return: metadata of the project member
        :rtype: dict

        **Example:**

        .. code-block:: python

            member_details = client.projects.get_member_details(project_id, "test@ibm.com")
            members_details = client.projects.get_member_details(project_id)
        """
        Projects._validate_type(project_id, "project_id", str, True)

        Projects._validate_type(user_name, "member_id", str, False)

        if user_name:
            href = self._client.service_instance._href_definitions.get_projects_member_href(
                project_id, user_name
            )
            response_get = requests.get(href, headers=self._client._get_headers())
            return self._handle_response(200, "Get project member", response_get)
        else:
            href = self._client.service_instance._href_definitions.get_projects_members_href(
                project_id
            )
            response_get = requests.get(href, headers=self._client._get_headers())
            return self._handle_response(200, "Get project members", response_get)

    def delete_member(self, project_id: str, user_name: str | None = None) -> str:
        """Delete a member associated with a project.

        :param project_id: ID of the project
        :type project_id: str
        :param user_name: name of the member
        :type user_name: str, optional

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example:**

        .. code-block:: python

            client.projects.delete_member(project_id, user_name)
        """
        Projects._validate_type(project_id, "project_id", str, True)
        Projects._validate_type(user_name, "user_name", str, False)

        member_endpoint = (
            self._client.service_instance._href_definitions.get_projects_member_href(
                project_id, user_name
            )
        )

        response_delete = requests.delete(
            member_endpoint, headers=self._client._get_headers()
        )

        print("DELETED")

        self._handle_response(204, "project member deletion", response_delete, False)

        return "SUCCESS"

    def update_member(self, project_id: str, user_name: str, changes: dict) -> dict:
        """Update the metadata of an existing member.

        :param project_id: ID of the project
        :type project_id: str
        :param user_name: name of the member to be updated
        :type user_name: str
        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated member
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.projects.MemberMetaNames.MEMBER: {"role": "editor"}
            }
            member_details = client.projects.update_member(project_id, user_name, changes=metadata)
        """
        self._validate_type(project_id, "project_id", str, True)
        self._validate_type(user_name, "user_name", str, True)

        self._validate_type(changes, "changes", dict, True)

        user_details = self.get_member_details(project_id, user_name)

        patch_request = []
        del user_details["type"]
        del user_details["state"]
        user_details.update(changes["member"])
        patch_request.append(user_details)

        # patching is different here, you just pass updated members but without `state` and `type`
        response = requests.patch(
            self._client.service_instance._href_definitions.get_projects_members_href(
                project_id
            ),
            json={"members": patch_request},
            headers=self._client._get_headers(),
        )

        updated_details = self._handle_response(200, "members patch", response)

        return updated_details

    def list_members(
        self,
        project_id: str,
        limit: int | None = None,
        identity_type: str | None = None,
        role: str | None = None,
        state: str | None = None,
    ) -> DataFrame:
        """Print the stored members of a project in a table format.

        :param project_id: ID of the project
        :type project_id: str
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param identity_type: filter the members by type
        :type identity_type: str, optional
        :param role: filter the members by role
        :type role: str, optional
        :param state: filter the members by state
        :type state: str, optional

        :return: pandas.DataFrame with listed members
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.projects.list_members(project_id)
        """
        self._validate_type(project_id, "project_id", str, True)

        params: dict[str, Any] = {}

        if limit is not None:
            params.update({"limit": limit})

        if identity_type is not None:
            params.update({"type": identity_type})

        if role is not None:
            params.update({"role": role})

        if state is not None:
            params.update({"state": state})

        href = (
            self._client.service_instance._href_definitions.get_projects_members_href(
                project_id
            )
        )

        member_resources = self._get_resources(href, "project members", params)[
            "resources"
        ]

        project_values = [
            (
                (m["id"], m["type"], m["role"], m["state"])
                if "state" in m
                else (m["id"], m["type"], m["role"], None)
            )
            for m in member_resources
        ]

        table = self._list(project_values, ["ID", "TYPE", "ROLE", "STATE"], limit)
        return table
