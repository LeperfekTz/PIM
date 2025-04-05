#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn

from ibm_watsonx_ai._wrappers import requests

from ibm_watsonx_ai.metanames import RemoteTrainingSystemMetaNames
from ibm_watsonx_ai.party_wrapper import Party
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame

FL_DEPRECATED_WARNING = "Federated Learning is deprecated and will be removed in IBM Cloud Pak for Data 5.2."


class RemoteTrainingSystem(WMLResource):
    """The RemoteTrainingSystem class represents a Federated Learning party and provides a list of identities
    that are permitted to join training as the RemoteTrainingSystem.
    """

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

        self._client = client
        self.ConfigurationMetaNames = RemoteTrainingSystemMetaNames()

    def store(self, meta_props: dict) -> dict:
        """Create a remote training system. `space_id` or `project_id` has to be provided.

        :param meta_props:  metadata, to see available meta names use
            ``client.remote_training_systems.ConfigurationMetaNames.get()``
        :type meta_props: dict

        :return: response json
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.remote_training_systems.ConfigurationMetaNames.NAME: "my-resource",
                client.remote_training_systems.ConfigurationMetaNames.TAGS: ["tag1", "tag2"],
                client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: {"name": "name", "region": "EU"}
                client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES: [{"id": "43689024", "type": "user"}],
                client.remote_training_systems.ConfigurationMetaNames.REMOTE_ADMIN: {"id": "43689020", "type": "user"}
            }
            client.set.default_space('3fc54cf1-252f-424b-b52d-5cdd9814987f')
            details = client.remote_training_systems.store(meta_props=metadata)
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(meta_props, "meta_props", dict, True)
        self._validate_input(meta_props)

        meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        if self._client.default_space_id is not None:
            meta["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            meta["project_id"] = self._client.default_project_id

        href = (
            self._client.service_instance._href_definitions.remote_training_systems_href()
        )

        creation_response = requests.post(
            href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=meta,
        )

        details = self._handle_response(
            expected_status_code=201,
            operationName="store remote training system specification",
            response=creation_response,
        )

        return details

    def _validate_input(self, meta_props: dict) -> None:
        if "name" not in meta_props:
            raise WMLClientError(
                "Its mandatory to provide 'NAME' in meta_props. Example: "
                "client.remote_training_systems.ConfigurationMetaNames.NAME"
            )

        if "allowed_identities" not in meta_props:
            raise WMLClientError(
                "Its mandatory to provide 'ALLOWED_IDENTITIES' in meta_props. Example: "
                "client.remote_training_systems.ConfigurationMetaNames.ALLOWED_IDENTITIES"
            )

        if "organization" in meta_props and "name" not in meta_props["organization"]:
            raise WMLClientError(
                "Its mandatory to provide 'name' for ORGANIZATION meta_prop. Eg: "
                "client.remote_training_systems.ConfigurationMetaNames.ORGANIZATION: "
                "{'name': 'org'} "
            )

    def delete(self, remote_training_systems_id: str) -> str:
        """Delete the given `remote_training_systems_id` definition. `space_id` or `project_id` has to be provided.

        :param remote_training_systems_id: identifier of the remote training system
        :type remote_training_systems_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.remote_training_systems.delete(remote_training_systems_id='6213cf1-252f-424b-b52d-5cdd9814956c')
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()
        RemoteTrainingSystem._validate_type(
            remote_training_systems_id, "remote_training_systems_id", str, True
        )

        href = (
            self._client.service_instance._href_definitions.remote_training_system_href(
                remote_training_systems_id
            )
        )

        delete_response = requests.delete(
            href, params=self._client._params(), headers=self._client._get_headers()
        )

        status = self._handle_response(
            expected_status_code=204,
            operationName="delete remote training system definition",
            response=delete_response,
            json_response=False,
        )

        if status == "SUCCESS":
            print("Remote training system deleted")

        return status

    def get_details(
        self,
        remote_training_system_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> dict:
        """Get metadata of a given remote training system. If `remote_training_system_id` is not
            metadata is returned for all remote training systems.

        :param remote_training_system_id: identifier of the remote training system
        :type remote_training_system_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: remote training system(s) metadata
        :rtype: dict (if remote_training_systems_id is not None) or {"resources": [dict]} (if remote_training_systems_id is None)


        **Examples**

        .. code-block:: python

            details = client.remote_training_systems.get_details(remote_training_systems_id)
            details = client.remote_training_systems.get_details()
            details = client.remote_training_systems.get_details(limit=100)
            details = client.remote_training_systems.get_details(limit=100, get_all=True)
            details = []
            for entry in client.remote_training_systems.get_details(limit=100, asynchronous=True, get_all=True):
                details.extend(entry)

        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(
            remote_training_system_id, "remote_training_systems_id", str, False
        )
        RemoteTrainingSystem._validate_type(limit, "limit", int, False)

        href = (
            self._client.service_instance._href_definitions.remote_training_systems_href()
        )

        if remote_training_system_id is None:
            return self._get_artifact_details(
                href,
                remote_training_system_id,
                limit,
                "remote_training_systems",
                _async=asynchronous,
                _all=get_all,
            )
        else:
            return self._get_artifact_details(
                href, remote_training_system_id, limit, "remote_training_systems"
            )

    def list(self, limit: int | None = None) -> DataFrame:
        """Lists stored remote training systems in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed remote training systems
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.remote_training_systems.list()
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        resources = self.get_details(get_all=self._should_get_all_values(limit))[
            "resources"
        ]

        values = [
            (m["metadata"]["id"], m["metadata"]["name"], m["metadata"]["created_at"])
            for m in resources
        ]

        return self._list(values, ["ID", "NAME", "CREATED"], limit)

    @staticmethod
    def get_id(remote_training_system_details: dict) -> str:
        """Get the ID of a remote training system.

        :param remote_training_system_details: metadata of the stored remote training system
        :type remote_training_system_details: dict

        :return: ID of the stored remote training system
        :rtype: str

        **Example:**

        .. code-block:: python

            details = client.remote_training_systems.get_details(remote_training_system_id)
            id = client.remote_training_systems.get_id(details)
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        RemoteTrainingSystem._validate_type(
            remote_training_system_details,
            "remote_training_system_details",
            object,
            True,
        )

        return WMLResource._get_required_element_from_dict(
            remote_training_system_details,
            "remote_training_system_details",
            ["metadata", "id"],
        )

    def update(self, remote_training_system_id: str, changes: dict) -> dict:
        """Update the existing metadata of a remote training system.

        :param remote_training_system_id: identifier of the remote training system
        :type remote_training_system_id: str
        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: updated remote training system details
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.remote_training_systems.ConfigurationMetaNames.NAME:"updated_remote_training_system"
            }
            details = client.remote_training_systems.update(remote_training_system_id, changes=metadata)

        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        self._validate_type(
            remote_training_system_id, "remote_training_system_id", str, True
        )
        self._validate_type(changes, "changes", dict, True)

        details = self.get_details(remote_training_system_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details["entity"], changes, with_validation=True
        )

        href = (
            self._client.service_instance._href_definitions.remote_training_system_href(
                remote_training_system_id
            )
        )

        response = requests.patch(
            href,
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        updated_details = self._handle_response(
            200, "remote training system patch", response
        )

        return updated_details

    def create_revision(self, remote_training_system_id: str) -> dict:
        """Create a new remote training system revision.

        :param remote_training_system_id: unique ID of the remote training system
        :type remote_training_system_id: str

        :return: details of the remote training system
        :rtype: dict

        **Example:**

        .. code-block:: python

            client.remote_training_systems.create_revision(remote_training_system_id)
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        RemoteTrainingSystem._validate_type(
            remote_training_system_id, "remote_training_system_id", str, False
        )

        href = (
            self._client.service_instance._href_definitions.remote_training_systems_href()
        )
        return self._create_revision_artifact(
            href, remote_training_system_id, "remote training system"
        )

    def get_revision_details(self, remote_training_system_id: str, rev_id: str) -> dict:
        """Get metadata from a specific revision of a stored remote system.

        :param remote_training_system_id: ID of the remote training system
        :type remote_training_system_id: str

        :param rev_id: unique ID of the remote system revision
        :type rev_id: str

        :return: metadata of the stored remote system revision
        :rtype: dict

        Example:

        .. code-block:: python

            details = client.remote_training_systems.get_details(remote_training_system_id, rev_id)
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(
            remote_training_system_id, "remote_training_system_id", str, True
        )
        RemoteTrainingSystem._validate_type(rev_id, "rev_id", str, True)

        href = (
            self._client.service_instance._href_definitions.remote_training_system_href(
                remote_training_system_id
            )
        )
        return self._get_with_or_without_limit(
            href,
            limit=None,
            op_name="remote_training_system_id",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def list_revisions(
        self, remote_training_system_id: str, limit: int | None = None
    ) -> DataFrame:
        """Print all revisions for a given remote_training_system_id in a table format.

        :param remote_training_system_id: unique ID of the stored remote system
        :type remote_training_system_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed remote training system revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.remote_training_systems.list_revisions(remote_training_system_id)
        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        self._client._check_if_either_is_set()

        RemoteTrainingSystem._validate_type(
            remote_training_system_id, "remote_training_system_id", str, True
        )

        href = self._client.service_instance._href_definitions.get_function_href(
            remote_training_system_id
        )

        resources = self._get_artifact_details(
            href + "/revisions",
            None,
            None,
            "remote system revisions",
            _all=self._should_get_all_values(limit),
        )["resources"]

        values = [
            (
                m["metadata"]["id"],
                m["metadata"]["rev"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in resources
        ]

        return self._list(values, ["ID", "REV", "NAME", "CREATED"], limit)

    def _validate_party_input(self, party_metadata: dict) -> None:

        if "data_handler" not in party_metadata:
            raise WMLClientError(
                "Its mandatory to provide 'DATA_HANDLER' in meta_props. Example: "
                "client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER"
            )

    def create_party(
        self, remote_training_system_id: str, party_metadata: dict
    ) -> Party:
        """Create a party object using the specified remote training system ID and the party metadata.

        :param remote_training_system_id: identifier of the remote training system
        :type remote_training_system_id: str
        :param party_metadata: the party configuration
        :type party_metadata: dict

        :return: a party object with the specified rts_id and configuration
        :rtype: Party

        **Examples**

        .. code-block:: python

            party_metadata = {
                client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
                    "info": {
                        "npz_file": "./data_party0.npz"
                    },
                    "name": "MnistTFDataHandler",
                    "path": "./mnist_keras_data_handler.py"
                },
                client.remote_training_systems.ConfigurationMetaNames.LOCAL_TRAINING: {
                    "name": "LocalTrainingHandler",
                    "path": "ibmfl.party.training.local_training_handler"
                },
                client.remote_training_systems.ConfigurationMetaNames.HYPERPARAMS: {
                    "epochs": 3
                },
            }
            party = client.remote_training_systems.create_party(remote_training_system_id, party_metadata)

        .. code-block:: python

            party_metadata = {
                client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER: {
                    "info": {
                        "npz_file": "./data_party0.npz"
                    },
                    "class": MnistTFDataHandler
                }
            }
            party = client.remote_training_systems.create_party(remote_training_system_id, party_metadata)

        """
        warn(FL_DEPRECATED_WARNING, category=DeprecationWarning)

        RemoteTrainingSystem._validate_type(
            remote_training_system_id, "remote_training_system_id", str, True
        )
        RemoteTrainingSystem._validate_type(
            party_metadata, "party_metadata", dict, True
        )
        self._validate_party_input(party_metadata)

        host = self._client.credentials.url.split("//")[1]
        party_config = {
            "aggregator": {"ip": host},
            "connection": {
                "info": {
                    "id": remote_training_system_id,
                }
            },
            "data": party_metadata[
                self._client.remote_training_systems.ConfigurationMetaNames.DATA_HANDLER
            ],
            "protocol_handler": {
                "name": "PartyProtocolHandler",
                "path": "ibmfl.party.party_protocol_handler",
            },
        }

        if "local_training" in party_metadata:
            party_config["local_training"] = party_metadata[
                self._client.remote_training_systems.ConfigurationMetaNames.LOCAL_TRAINING
            ]

        if "hyperparams" in party_metadata:
            party_config["hyperparams"] = party_metadata[
                self._client.remote_training_systems.ConfigurationMetaNames.HYPERPARAMS
            ]

        if "model" in party_metadata:
            party_config["model"] = party_metadata[
                self._client.remote_training_systems.ConfigurationMetaNames.MODEL
            ]

        return Party(client=self._client, config_dict=party_config)
