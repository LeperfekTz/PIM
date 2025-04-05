#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
import os
import json
import uuid
from typing import TYPE_CHECKING, Any
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.metanames import ModelDefinitionMetaNames
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    import pandas


class ModelDefinition(WMLResource):
    """Store and manage model definitions."""

    ConfigurationMetaNames = ModelDefinitionMetaNames()

    """MetaNames for model definition creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)
        self._ICP_PLATFORM_SPACES = client.ICP_PLATFORM_SPACES
        self.default_space_id = client.default_space_id

    def _generate_model_definition_document(
        self, meta_props: dict[str, str | dict]
    ) -> dict:
        doc: dict = {
            "metadata": {
                "name": "generated_name_" + str(uuid.uuid4()),
                "tags": ["generated_tag_" + str(uuid.uuid4())],
                "asset_type": "wml_model_definition",
                "origin_country": "us",
                "rov": {"mode": 0},
                "asset_category": "USER",
            },
            "entity": {
                "wml_model_definition": {
                    "ml_version": "4.0.0",
                    "version": "1.0",
                    "platform": {"name": "python", "versions": ["3.5"]},
                }
            },
        }

        if self.ConfigurationMetaNames.NAME in meta_props:
            doc["metadata"]["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            doc["metadata"]["description"] = meta_props[
                self.ConfigurationMetaNames.DESCRIPTION
            ]

        if self.ConfigurationMetaNames.VERSION in meta_props:
            doc["entity"]["wml_model_definition"]["version"] = meta_props[
                self.ConfigurationMetaNames.VERSION
            ]

        if self.ConfigurationMetaNames.PLATFORM in meta_props:
            doc["entity"]["wml_model_definition"]["platform"]["name"] = meta_props[self.ConfigurationMetaNames.PLATFORM]["name"]  # type: ignore[index]
            doc["entity"]["wml_model_definition"]["platform"]["versions"][
                0
            ] = meta_props[self.ConfigurationMetaNames.PLATFORM][
                "versions"
            ][  # type: ignore[index]
                0
            ]

        if self.ConfigurationMetaNames.COMMAND in meta_props:
            doc["entity"]["wml_model_definition"].update(
                {"command": meta_props[self.ConfigurationMetaNames.COMMAND]}
            )
        if self.ConfigurationMetaNames.CUSTOM in meta_props:
            doc["entity"]["wml_model_definition"].update(
                {"custom": meta_props[self.ConfigurationMetaNames.CUSTOM]}
            )

        return doc

    def store(self, model_definition: str, meta_props: dict[str, str | dict]) -> dict:
        """Create a model definition.

        :param meta_props: metadata of the model definition configuration. To see available meta names, use:

            .. code-block:: python

                client.model_definitions.ConfigurationMetaNames.get()

        :type meta_props: dict

        :param model_definition: path to the content file to be uploaded
        :type model_definition: str

        :return: metadata of the created model definition
        :rtype: dict

        **Example:**

        .. code-block:: python

            client.model_definitions.store(model_definition, meta_props)
        """

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        self.ConfigurationMetaNames._validate(meta_props)

        document = self._generate_model_definition_document(meta_props)

        model_definition_attachment_def = {
            "asset_type": "wml_model_definition",
            "name": "model_definition_attachment",
        }
        paramvalue = self._client._params()

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_model_definition_assets_href(),
            params=paramvalue,
            headers=self._client._get_headers(),
            json=document,
        )

        model_definition_details = self._handle_response(
            201, "creating new model_definition", creation_response
        )

        self._handle_response(201, "creating new attachment", creation_response)
        model_definition_id = model_definition_details["metadata"]["asset_id"]

        if self._client.CLOUD_PLATFORM_SPACES:
            model_definition_attachment_url = (
                self._client.service_instance._href_definitions.get_attachments_href(
                    model_definition_details["metadata"]["asset_id"]
                )
            )
        else:
            model_definition_attachment_url = (
                self._client.service_instance._href_definitions.get_model_definition_assets_href()
                + "/"
                + model_definition_details["metadata"]["asset_id"]
                + "/attachments"
            )

        attachment_response = requests.post(
            model_definition_attachment_url,
            params=paramvalue,
            headers=self._client._get_headers(),
            json=model_definition_attachment_def,
        )

        try:
            attachment_details = self._handle_response(
                201, "creating new model definition attachment", attachment_response
            )

            attachment_id = attachment_details["attachment_id"]
            attachment_status_json = json.loads(
                attachment_response.content.decode("utf-8")
            )
            model_definition_attachment_signed_url = attachment_status_json["url1"]
            model_definition_attachment_put_url = (
                self._client.credentials.url + model_definition_attachment_signed_url
            )

            with open(model_definition, "rb") as f:
                if self._ICP_PLATFORM_SPACES:
                    put_response = requests.put(
                        model_definition_attachment_put_url,
                        files={
                            "file": (model_definition, f, "application/octet-stream")
                        },
                    )
                else:
                    put_response = requests.put(
                        model_definition_attachment_signed_url,
                        data=open(model_definition, "rb").read(),
                    )

            if put_response.status_code != 201 and put_response.status_code != 200:
                self._handle_response(
                    200, "uploading a model_definition attachment file", put_response
                )

            complete_response = requests.post(
                self._client.service_instance._href_definitions.get_attachment_complete_href(
                    model_definition_id, attachment_id
                ),
                params=paramvalue,
                headers=self._client._get_headers(),
            )

            self._handle_response(
                200, "updating a model_definition status", complete_response
            )

            response = self._get_required_element_from_response(
                model_definition_details
            )

            entity = response["entity"]

            try:
                del entity["wml_model_definition"]["ml_version"]
            except KeyError:
                pass

            final_response = {"metadata": response["metadata"], "entity": entity}

            return final_response

        except Exception as e:
            try:
                self.delete(model_definition_id)
            finally:
                raise e

    def get_details(
        self,
        model_definition_id: str | None = None,
        limit: int | None = None,
        get_all: bool | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of a stored model definition. If no `model_definition_id` is passed,
        details for all model definitions are returned.

        :param model_definition_id: unique ID of the model definition
        :type model_definition_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of model definition
        :rtype: dict (if `model_definition_id` is not None)

        **Example:**

        .. code-block: python

            model_definition_details = client.model_definitions.get_details(model_definition_id)

        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition", True
        )

        def get_required_element_from_response(response: dict) -> dict:
            final_response = {
                "metadata": response["metadata"],
            }

            if "entity" in response:
                entity = response["entity"]

                try:
                    del entity["wml_model_definition"]["ml_version"]
                except KeyError:
                    pass

                final_response["entity"] = entity

            return final_response

        return self._get_asset_based_resource(
            model_definition_id,
            "wml_model_definition",
            get_required_element_from_response,
            limit=limit,
            get_all=get_all,
        )

    def download(
        self,
        model_definition_id: str | None,
        filename: str | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download the content of a model definition asset.

        :param model_definition_id: unique ID of the model definition asset to be downloaded
        :type model_definition_id: str
        :param filename: filename to be used for the downloaded file
        :type filename: str
        :param rev_id: revision ID
        :type rev_id: str, optional

        :return: path to the downloaded asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            client.model_definitions.download(model_definition_id, "model_definition_file")
        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )

        if filename is None:
            raise TypeError(
                "download() missing 1 required positional argument: 'filename'"
            )

        self._client._check_if_either_is_set()

        ModelDefinition._validate_type(
            model_definition_id, "model_definition_id", str, True
        )
        params = self._client._params()
        if rev_id is not None:
            ModelDefinition._validate_type(rev_id, "rev_id", str, False)
            params.update({"revision_id": rev_id})

        attachment_id = self._get_attachment_id(model_definition_id)
        artifact_content_url = (
            self._client.service_instance._href_definitions.get_attachment_href(
                model_definition_id, attachment_id
            )
        )
        if not self._ICP_PLATFORM_SPACES:
            response = requests.get(
                self._client.service_instance._href_definitions.get_attachment_href(
                    model_definition_id, attachment_id
                ),
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        else:
            response = requests.get(
                artifact_content_url,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        attachment_signed_url = response.json()["url"]
        if response.status_code == 200:
            if not self._ICP_PLATFORM_SPACES:
                att_response = requests.get(attachment_signed_url)
            else:
                att_response = requests.get(
                    self._credentials.url + attachment_signed_url
                )

            if att_response.status_code != 200:
                raise WMLClientError(
                    "Failure during {}.".format("downloading model_definition asset"),
                    att_response,
                )

            downloaded_asset = att_response.content
            try:
                with open(filename, "wb") as f:
                    f.write(downloaded_asset)
                print("Successfully saved asset content to file: '{}'".format(filename))
                return os.getcwd() + "/" + filename
            except IOError as e:
                raise WMLClientError(
                    "Saving asset with artifact_url: '{}' failed.".format(filename), e
                )
        else:
            raise WMLClientError(
                "Failed while downloading the asset " + model_definition_id
            )

    def delete(self, model_definition_id: str | None = None, **kwargs: Any) -> str:
        """Delete a stored model definition.

        :param model_definition_id: unique ID of the stored model definition
        :type model_definition_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.model_definitions.delete(model_definition_id)

        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        ModelDefinition._validate_type(
            model_definition_id, "model_definition_id", str, True
        )
        paramvalue = self._client._params()

        model_definition_endpoint = (
            self._client.service_instance._href_definitions.get_model_definition_assets_href()
            + "/"
            + model_definition_id
        )

        response_delete = requests.delete(
            model_definition_endpoint,
            params=paramvalue,
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204, "Model definition deletion", response_delete, False
        )

    def _get_required_element_from_response(self, response_data: dict) -> dict:

        WMLResource._validate_type(response_data, "model_definition_response", dict)
        revision_id = None

        try:
            href = ""
            if self._client.default_space_id is not None:
                new_el = {
                    "metadata": {
                        "space_id": response_data["metadata"]["space_id"],
                        "guid": response_data["metadata"]["asset_id"],
                        "asset_type": response_data["metadata"]["asset_type"],
                        "created_at": response_data["metadata"]["created_at"],
                        "last_updated_at": response_data["metadata"]["usage"][
                            "last_updated_at"
                        ],
                    },
                    "entity": response_data["entity"],
                }
                href = (
                    self._client.service_instance._href_definitions.get_base_asset_with_type_href(
                        response_data["metadata"]["asset_type"],
                        response_data["metadata"]["asset_id"],
                    )
                    + "?"
                    + "space_id="
                    + response_data["metadata"]["space_id"]
                )

            elif self._client.default_project_id is not None:
                new_el = {
                    "metadata": {
                        "project_id": response_data["metadata"]["project_id"],
                        "guid": response_data["metadata"]["asset_id"],
                        "asset_type": response_data["metadata"]["asset_type"],
                        "created_at": response_data["metadata"]["created_at"],
                        "last_updated_at": response_data["metadata"]["usage"][
                            "last_updated_at"
                        ],
                    },
                    "entity": response_data["entity"],
                }

                href = (
                    self._client.service_instance._href_definitions.get_base_asset_with_type_href(
                        response_data["metadata"]["asset_type"],
                        response_data["metadata"]["asset_id"],
                    )
                    + "?"
                    + "project_id="
                    + response_data["metadata"]["project_id"]
                )

            if "revision_id" in response_data["metadata"]:
                new_el["metadata"].update(
                    {"revision_id": response_data["metadata"]["revision_id"]}
                )
                revision_id = response_data["metadata"]["revision_id"]

            if "name" in response_data["metadata"]:
                new_el["metadata"].update({"name": response_data["metadata"]["name"]})

            if (
                "description" in response_data["metadata"]
                and response_data["metadata"]["description"]
            ):
                new_el["metadata"].update(
                    {"description": response_data["metadata"]["description"]}
                )

            if "href" in response_data["metadata"]:
                href_without_host = response_data["href"].split(".com")[-1]
                new_el["metadata"].update({"href": href_without_host})
            else:
                new_el["metadata"].update({"href": href})

            if "attachments" in response_data and response_data["attachments"]:
                new_el["metadata"].update(
                    {"attachment_id": response_data["attachments"][0]["id"]}
                )
            else:
                new_el["metadata"].update({"href": href})

            if "commit_info" in response_data["metadata"] and revision_id is not None:
                new_el["metadata"].update(
                    {
                        "revision_commit_date": response_data["metadata"][
                            "commit_info"
                        ]["committed_at"]
                    }
                )
            return new_el
        except Exception as e:
            raise WMLClientError(
                f"Failed to read Response from down-stream service: {response_data}"
            )

    def _get_attachment_id(
        self,
        model_definition_id: str | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )

        op_name = "getting attachment id "
        url = (
            self._client.service_instance._href_definitions.get_model_definition_assets_href()
            + "/"
            + model_definition_id
        )
        paramvalue = self._client._params()

        if rev_id is not None:
            paramvalue.update({"revision_id": rev_id})

        response_get = requests.get(
            url, params=paramvalue, headers=self._client._get_headers()
        )
        details = self._handle_response(200, op_name, response_get)
        try:
            attachment_id = details["attachments"][0]["id"]
        except KeyError:
            raise WMLClientError(
                f"No attachment exists for model definition (id={model_definition_id})."
            )

        return attachment_id

    def list(self, limit: int | None = None) -> pandas.DataFrame:
        """Return the stored model definition assets in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed model definitions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.model_definitions.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        href = (
            self._client.service_instance._href_definitions.get_model_definition_search_asset_href()
        )
        if limit is None:
            data: dict[str, Any] = {"query": "*:*"}
        else:
            ModelDefinition._validate_type(limit, "limit", int, False)
            data: dict[str, Any] = {  # type: ignore[no-redef]
                "query": "*:*",
                "limit": limit,
            }

        response = requests.post(
            href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=data,
        )

        self._handle_response(200, "model_definition assets", response)
        asset_details = self._handle_response(200, "model_definition assets", response)[
            "results"
        ]
        model_def_values = [
            (
                m["metadata"]["name"],
                m["metadata"]["asset_type"],
                m["metadata"]["asset_id"],
            )
            for m in asset_details
        ]

        table = self._list(
            model_def_values,
            ["NAME", "ASSET_TYPE", "ID"],
            limit,
        )
        return table

    def get_id(self, model_definition_details: dict) -> str:
        """Get the unique ID of a stored model definition asset.

        :param model_definition_details: metadata of the stored model definition asset
        :type model_definition_details: dict

        :return: unique ID of the stored model definition asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.model_definition.get_id(asset_details)

        """
        if "asset_id" in model_definition_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_definition_details,
                "model_definition_details",
                ["metadata", "asset_id"],
            )
        else:
            ModelDefinition._validate_type(
                model_definition_details, "model__definition_details", object, True
            )

            return WMLResource._get_required_element_from_dict(
                model_definition_details,
                "model_definition_details",
                ["metadata", "guid"],
            )

    def get_uid(self, model_definition_details: dict) -> str:
        """Get the UID of the stored model.

        *Deprecated:* Use ``get_id(model_definition_details)`` instead.

        :param model_definition_details: details of the stored model definition
        :type model_definition_details: dict

        :return: UID of the stored model definition
        :rtype: str

        **Example:**

        .. code-block:: python

            model_definition_uid = client.model_definitions.get_uid(model_definition_details)
        """
        get_uid_method_deprecated = "This method is deprecated, please use `get_id(model_definition_details)` instead"
        warn(get_uid_method_deprecated, category=DeprecationWarning)
        return ModelDefinition.get_id(self, model_definition_details)

    def get_href(self, model_definition_details: dict) -> str:
        """Get the href of a stored model definition.

        :param model_definition_details: details of the stored model definition
        :type model_definition_details: dict

        :return: href of the stored model definition
        :rtype: str

        **Example:**

        .. code-block:: python

            model_definition_id = client.model_definitions.get_href(model_definition_details)
        """
        if "asset_id" in model_definition_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_definition_details,
                "model_definition_details",
                ["metadata", "asset_id"],
            )
        else:
            ModelDefinition._validate_type(
                model_definition_details, "model__definition_details", object, True
            )
            # ModelDefinition._validate_type_of_details(model_definition_details, MODEL_DEFINITION_DETAILS_TYPE)

            return WMLResource._get_required_element_from_dict(
                model_definition_details,
                "model_definition_details",
                ["metadata", "href"],
            )

    def update(
        self,
        model_definition_id: str,
        meta_props: dict[str, str | dict] | None = None,
        file_path: str | None = None,
    ) -> dict:
        """Update the model definition with metadata, attachment, or both.

        :param model_definition_id: ID of the model definition
        :type model_definition_id: str
        :param meta_props: metadata of the model definition configuration to be updated
        :type meta_props: dict
        :param file_path: path to the content file to be uploaded
        :type file_path: str, optional

        :return: updated metadata of the model definition
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_definition_details = client.model_definition.update(model_definition_id, meta_props, file_path)
        """

        ModelDefinition._validate_type(
            model_definition_id, "model_definition_id", str, True
        )

        if meta_props is None and file_path is None:
            raise WMLClientError(
                "At least either meta_props or file_path has to be provided"
            )

        updated_details = None

        url = self._client.service_instance._href_definitions.get_asset_href(
            model_definition_id
        )

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   CAMS has meta and entity patching. 'name' and 'description' get stored in CAMS meta section
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the model_definition, delete it
        #   b. POST call to get signed URL for upload
        #   c. Upload to the signed URL
        #   d. Mark upload complete
        # STEP 4. Get the updated script record and return

        # STEP 1
        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of model_definition_id provided."
                )
            else:
                raise WMLClientError(
                    "Failure during {}.".format("getting script to update"), response
                )

        details = self._handle_response(200, "Get script details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, "meta_props", dict, True)

            props_for_asset_meta_patch = {}

            # Since we are dealing with direct asset apis, there can be metadata or entity patch or both
            if "name" in meta_props or "description" in meta_props:

                for key in meta_props:
                    if key == "name" or key == "description":
                        props_for_asset_meta_patch.update({key: meta_props[key]})

            meta_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
                details, props_for_asset_meta_patch, with_validation=True
            )

            props_for_asset_entity_patch = {}
            for key in meta_props:
                if key != "name" and key != "description":
                    props_for_asset_entity_patch.update({key: meta_props[key]})

            entity_patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
                details["entity"]["wml_model_definition"],
                props_for_asset_entity_patch,
                with_validation=True,
            )

            if meta_patch_payload:
                meta_patch_url = (
                    self._client.service_instance._href_definitions.get_asset_href(
                        model_definition_id
                    )
                )

                response_patch = requests.patch(
                    meta_patch_url,
                    json=meta_patch_payload,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                )

                updated_details = self._handle_response(
                    200, "script patch", response_patch
                )

            if entity_patch_payload:
                entity_patch_url = (
                    self._client.service_instance._href_definitions.get_asset_href(
                        model_definition_id
                    )
                    + "/attributes/wml_model_definition"
                )

                response_patch = requests.patch(
                    entity_patch_url,
                    json=entity_patch_payload,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                )

                updated_details = self._handle_response(
                    200, "script patch", response_patch
                )

        if file_path is not None:
            if "attachments" in details and details["attachments"]:
                current_attachment_id = details["attachments"][0]["id"]
            else:
                current_attachment_id = None

            # STEP 3
            attachments_response = self._update_attachment_for_assets(
                "wml_model_definition",
                model_definition_id,
                file_path,
                current_attachment_id,
            )

        if attachments_response is not None and "success" not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(
            model_definition_id
        )

        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of model_definition_id provided."
                )
            else:
                raise WMLClientError(
                    "Failure during {}.".format("getting script to update"), response
                )

        response = self._get_required_element_from_response(
            self._handle_response(200, "Get script details", response)
        )

        entity = response["entity"]

        try:
            del entity["wml_model_definition"]["ml_version"]
        except KeyError:
            pass

        final_response = {"metadata": response["metadata"], "entity": entity}

        return final_response

    def _update_msg(self, updated_details: dict | None) -> None:
        if updated_details is not None:
            print(
                "Could not update the attachment because of server error."
                " However metadata is updated. Try updating attachment again later"
            )
        else:
            raise WMLClientError(
                "Unable to update attachment because of server error. Try again later"
            )

    def create_revision(
        self, model_definition_id: str | None = None, **kwargs: Any
    ) -> dict:
        """Create a revision for the given model definition. Revisions are immutable once created.
        The metadata and attachment of the model definition is taken and a revision is created out of it.

        :param model_definition_id: ID of the model definition
        :type model_definition_id: str

        :return: revised metadata of the stored model definition
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_definition_revision = client.model_definitions.create_revision(model_definition_id)
        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )

        self._client._check_if_either_is_set()

        ModelDefinition._validate_type(
            model_definition_id, "model_definition_id", str, True
        )

        print("Creating model_definition revision...")

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(
                model_definition_id, "Model definition"
            )
        )

        entity = response["entity"]

        try:
            del entity["wml_model_definition"]["ml_version"]
        except KeyError:
            pass

        final_response = {"metadata": response["metadata"], "entity": entity}

        return final_response

    def get_revision_details(
        self,
        model_definition_id: str | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of a model definition.

        :param model_definition_id: ID of the model definition
        :type model_definition_id: str

        :param rev_id: ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.
        :type rev_id: str, optional

        :return: metadata of the stored model definition
        :rtype: dict

        **Example:**

        .. code-block:: python

            script_details = client.model_definitions.get_revision_details(model_definition_id, rev_id)
        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", True)

        op_name = "getting model_definition revision details"
        ModelDefinition._validate_type(
            model_definition_id, "model_definition_id", str, True
        )

        url = (
            self._client.service_instance._href_definitions.get_model_definition_assets_href()
            + "/"
            + model_definition_id
        )
        paramvalue = self._client._params()

        if rev_id is None:
            rev_id = "latest"

        paramvalue.update({"revision_id": rev_id})

        response_get = requests.get(
            url, params=paramvalue, headers=self._client._get_headers()
        )
        if response_get.status_code == 200:

            response = self._get_required_element_from_response(
                self._handle_response(200, op_name, response_get)
            )

            entity = response["entity"]

            try:
                del entity["wml_model_definition"]["ml_version"]
            except KeyError:
                pass

            final_response = {"metadata": response["metadata"], "entity": entity}

            return final_response
        else:
            return self._handle_response(200, op_name, response_get)

    def list_revisions(
        self,
        model_definition_id: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> pandas.DataFrame:
        """Return the stored model definition assets in a table format.

        :param model_definition_id: unique ID of the model definition
        :type model_definition_id: str
        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed model definitions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.model_definitions.list_revisions()

        """
        model_definition_id = _get_id_from_deprecated_uid(
            kwargs, model_definition_id, "model_definition"
        )

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        href = (
            self._client.service_instance._href_definitions.get_model_definition_assets_href()
            + "/"
            + model_definition_id
            + "/revisions"
        )
        params = self._client._params()

        if limit is not None:
            ModelDefinition._validate_type(limit, "limit", int, False)
            params.update({"limit": limit})
        response = requests.get(
            href, params=params, headers=self._client._get_headers()
        )
        self._handle_response(200, "model_definition revision assets", response)
        asset_details = self._handle_response(
            200, "model_definition revision assets", response
        )["results"]
        model_def_values = [
            (
                m["metadata"]["asset_id"],
                m["metadata"]["revision_id"],
                m["metadata"]["name"],
                m["metadata"]["asset_type"],
                m["metadata"]["commit_info"]["committed_at"],
            )
            for m in asset_details
        ]

        return self._list(
            model_def_values,
            ["ID", "REV_ID", "NAME", "ASSET_TYPE", "REVISION_COMMIT"],
            limit,
        )
