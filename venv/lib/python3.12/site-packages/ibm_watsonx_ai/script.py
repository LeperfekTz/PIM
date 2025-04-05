#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import os
from typing import Any, TYPE_CHECKING, TypeAlias
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.utils import (
    DATA_ASSETS_DETAILS_TYPE,
    modify_details_for_script_and_shiny,
)
from ibm_watsonx_ai.lifecycle import SpecStates
from ibm_watsonx_ai.metanames import ScriptMetaNames
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
    ForbiddenActionForGitBasedProject,
)

ListType: TypeAlias = list

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class Script(WMLResource):
    """Store and manage script assets."""

    ConfigurationMetaNames = ScriptMetaNames()
    """MetaNames for script assets creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def get_details(
        self,
        script_id: str | None = None,
        limit: int | None = None,
        get_all: bool | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get script asset details. If no script_id is passed, details for all script assets are returned.

        :param script_id: unique ID of the script
        :type script_id: str, optional

        :param limit:  limit number of fetched records
        :type limit: int, optional

        :param get_all:  if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: metadata of the stored script asset
        :rtype:
            - **dict** - if script_id is not None
            - **{"resources": [dict]}** - if script_id is None

        **Example:**

        .. code-block:: python

            script_details = client.script.get_details(script_id)

        """
        script_id = _get_id_from_deprecated_uid(
            kwargs, script_id, "script", can_be_none=True
        )

        def get_required_elements(response: dict[str, Any]) -> dict[str, Any]:
            response = modify_details_for_script_and_shiny(response)
            final_response = {
                "metadata": response["metadata"],
            }

            if "entity" in response:
                final_response["entity"] = response["entity"]

                try:
                    del final_response["entity"]["script"]["ml_version"]
                except KeyError:
                    pass

            return final_response

        return self._get_asset_based_resource(
            script_id, "script", get_required_elements, limit=limit, get_all=get_all
        )

    def store(self, meta_props: dict, file_path: str) -> dict:
        """Create a script asset and upload content to it.

        :param meta_props: name to be given to the script asset
        :type meta_props: dict

        :param file_path: path to the content file to be uploaded
        :type file_path: str

        :return: metadata of the stored script asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.script.ConfigurationMetaNames.NAME: 'my first script',
                client.script.ConfigurationMetaNames.DESCRIPTION: 'description of the script',
                client.script.ConfigurationMetaNames.SOFTWARE_SPEC_ID: '0cdb0f1e-5376-4f4d-92dd-da3b69aa9bda'
            }

            asset_details = client.script.store(meta_props=metadata, file_path="/path/to/file")

        """
        if self._client.project_type == "local_git_storage":
            raise ForbiddenActionForGitBasedProject(
                reason="Storing Scripts is not supported for git based project."
            )

        Script._validate_type(meta_props, "meta_props", dict, True)
        Script._validate_type(file_path, "file_path", str, True)
        script_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        name, extension = os.path.splitext(file_path)

        response = self._create_asset(script_meta, file_path, extension)

        entity = response["entity"]

        try:
            del entity["script"]["ml_version"]
        except KeyError:
            pass

        final_response = {"metadata": response["metadata"], "entity": entity}

        return final_response

    def _create_asset(
        self, script_meta: dict[str, Any], file_path: str, extension: str = ".py"
    ) -> dict[str, Any]:

        # Step1: Create a data asset
        name = script_meta["metadata"]["name"]

        if script_meta["metadata"].get("description") is not None:
            desc = script_meta["metadata"]["description"]
        else:
            desc = ""

        if script_meta.get("software_spec_uid") is not None:
            if script_meta["software_spec_uid"] == "":
                raise WMLClientError(
                    "Failed while creating a script asset, SOFTWARE_SPEC_ID cannot be empty"
                )

        if extension == ".py":
            lang = "python3"
        elif extension == ".R":
            lang = "R"
        else:
            raise WMLClientError(
                "This file type is not supported. It has to be either a python script(.py file ) or a "
                "R script"
            )

        # check if the software spec specified is base or derived and accordingly update the entity for asset creation

        # everything inside "if" should work.
        sw_spec_details = self._client.software_specifications.get_details(
            script_meta["software_spec_uid"]
        )

        if lang == "R":

            rscript_sw_specs = []  # names of supported r script software specifications
            retired_rscript_sw_specs = (
                []
            )  # names of retired r script software specifications
            constricted_rscript_sw_specs = (
                []
            )  # names of constricted r script software specifications
            deprecated_rscript_sw_specs = (
                []
            )  # names of deprecated r script software specifications

            if self._client.CPD_version >= 4.7:

                for sw_spec in self._client.software_specifications.get_details()[
                    "resources"
                ]:
                    if "life_cycle" in sw_spec["metadata"]:
                        sw_configuration = (
                            sw_spec["entity"]
                            .get("software_specification", {})
                            .get("software_configuration", {})
                        )
                        if (
                            sw_configuration.get("platform", {}).get("name") == "r"
                            and len(sw_configuration.get("included_packages", [])) > 1
                        ):
                            if (
                                SpecStates.RETIRED.value
                                in sw_spec["metadata"]["life_cycle"]
                            ):
                                retired_rscript_sw_specs.append(
                                    sw_spec["metadata"]["name"]
                                )
                            elif (
                                SpecStates.CONSTRICTED.value
                                in sw_spec["metadata"]["life_cycle"]
                            ):
                                constricted_rscript_sw_specs.append(
                                    sw_spec["metadata"]["name"]
                                )
                            elif (
                                SpecStates.DEPRECATED.value
                                in sw_spec["metadata"]["life_cycle"]
                            ):
                                deprecated_rscript_sw_specs.append(
                                    sw_spec["metadata"]["name"]
                                )
                            elif "rstudio" not in sw_spec["metadata"]["name"].lower():
                                rscript_sw_specs.append(sw_spec["metadata"]["name"])

            elif self._client.CPD_version == 4.6:
                rscript_sw_specs = ["runtime-22.2-r4.2"]
                deprecated_rscript_sw_specs = ["default_r3.6", "runtime-22.1-r3.6"]
            else:
                rscript_sw_specs = ["default_r3.6", "runtime-22.1-r3.6"]
                deprecated_rscript_sw_specs = []
            rscript_sw_spec_ids = [
                self._client.software_specifications.get_id_by_name(sw_name)
                for sw_name in rscript_sw_specs
            ]
            retired_sw_spec_ids = [
                self._client.software_specifications.get_id_by_name(sw_name)
                for sw_name in retired_rscript_sw_specs
            ]
            constricted_sw_spec_ids = [
                self._client.software_specifications.get_id_by_name(sw_name)
                for sw_name in constricted_rscript_sw_specs
            ]
            deprecated_sw_spec_ids = [
                self._client.software_specifications.get_id_by_name(sw_name)
                for sw_name in deprecated_rscript_sw_specs
            ]

            unsupported_dict = {
                SpecStates.RETIRED.value: retired_sw_spec_ids,
                SpecStates.CONSTRICTED.value: constricted_sw_spec_ids,
                SpecStates.DEPRECATED.value: deprecated_sw_spec_ids,
            }

            if (
                script_meta["software_spec_uid"]
                not in rscript_sw_spec_ids
                + deprecated_sw_spec_ids
                + constricted_sw_spec_ids
                + retired_sw_spec_ids
            ):
                raise WMLClientError(
                    f"For R scripts, only base software specs {', '.join(rscript_sw_specs)} "
                    "are supported. Specify "
                    "the id you get via "
                    "client.software_specifications.get_id_by_name(sw_name)"
                )
            warning_error_msg = (
                "Provided software spec is {key} for R scripts. "
                + f"Only base software specs {', '.join(rscript_sw_specs)} are supported."
                f" Specify the id you get via "
                f"client.software_specifications.get_id_by_name(sw_name)"
            )
            for key in unsupported_dict:
                if script_meta["software_spec_uid"] in unsupported_dict[key]:
                    match key:
                        case SpecStates.RETIRED.value:
                            raise WMLClientError(warning_error_msg.format(key=key))
                        case _:
                            software_spec_warning = warning_error_msg.format(key=key)
                            warn(software_spec_warning, category=RuntimeWarning)

        if sw_spec_details["entity"]["software_specification"]["type"] == "base":
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "script",
                    "origin_country": "us",
                    "asset_category": "USER",
                },
                "entity": {
                    "script": {
                        "ml_version": "4.0.0",
                        "language": {"name": lang},
                        "software_spec": {"base_id": script_meta["software_spec_uid"]},
                    }
                },
            }
        elif sw_spec_details["entity"]["software_specification"]["type"] == "derived":
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "script",
                    "origin_country": "us",
                    "asset_category": "USER",
                },
                "entity": {
                    "script": {
                        "ml_version": "4.0.0",
                        "language": {"name": lang},
                        "software_spec": {"id": script_meta["software_spec_uid"]},
                    }
                },
            }
        else:
            asset_meta = {
                "metadata": {
                    "name": name,
                    "description": desc,
                    "asset_type": "script",
                    "origin_country": "us",
                    "asset_category": "USER",
                },
                "entity": {
                    "script": {
                        "ml_version": "4.0.0",
                        "language": {"name": lang},
                        "software_spec": {"id": script_meta["software_spec_uid"]},
                    }
                },
            }

        # Step1: Create an asset
        print("Creating Script asset...")

        if self._client.CLOUD_PLATFORM_SPACES:
            creation_response = requests.post(
                self._client.service_instance._href_definitions.get_assets_href(),
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=asset_meta,
            )
        else:  # if self._client.ICP_PLATFORM_SPACES
            creation_response = requests.post(
                self._client.service_instance._href_definitions.get_data_assets_href(),
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=asset_meta,
            )

        asset_details = self._handle_response(
            201, "creating new asset", creation_response
        )

        # Step2: Create attachment
        if creation_response.status_code == 201:
            asset_id = asset_details["metadata"]["asset_id"]
            attachment_meta = {"asset_type": "script", "name": "attachment_" + asset_id}

            attachment_response = requests.post(
                self._client.service_instance._href_definitions.get_attachments_href(
                    asset_id
                ),
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=attachment_meta,
            )
            attachment_details = self._handle_response(
                201, "creating new attachment", attachment_response
            )
            if attachment_response.status_code == 201:
                attachment_id = attachment_details["attachment_id"]
                attachment_url = attachment_details["url1"]

                # Step3: Put content to attachment
                try:
                    with open(file_path, "rb") as f:
                        if not self._client.ICP_PLATFORM_SPACES:
                            put_response = requests.put(attachment_url, data=f.read())
                        else:
                            put_response = requests.put(
                                self._credentials.url + attachment_url,
                                files={"file": (name, f, "application/octet-stream")},
                            )
                except Exception as e:
                    deletion_response = requests.delete(
                        self._client.service_instance._href_definitions.get_data_asset_href(
                            asset_id
                        ),
                        params=self._client._params(),
                        headers=self._client._get_headers(),
                    )
                    print(deletion_response.status_code)
                    raise WMLClientError("Failed while reading a file.", e)

                if put_response.status_code == 201 or put_response.status_code == 200:

                    # Step4: Complete attachment
                    complete_response = requests.post(
                        self._client.service_instance._href_definitions.get_attachment_complete_href(
                            asset_id, attachment_id
                        ),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                    )

                    if complete_response.status_code == 200:
                        print("SUCCESS")
                        return self._get_required_element_from_response(asset_details)
                    else:
                        try:
                            self.delete(asset_id)
                        except:
                            pass
                        raise WMLClientError(
                            "Failed while creating a script asset. Try again."
                        )
                else:
                    try:
                        self.delete(asset_id)
                    except:
                        pass
                    raise WMLClientError(
                        "Failed while creating a script asset. Try again."
                    )
            else:
                print("SUCCESS")
                return self._get_required_element_from_response(asset_details)
        else:
            raise WMLClientError("Failed while creating a script asset. Try again.")

    def list(self, limit: int | None = None) -> DataFrame:
        """List stored scripts in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed scripts
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.script.list()
        """

        Script._validate_type(limit, "limit", int, False)
        href = self._client.service_instance._href_definitions.get_search_script_href()

        data: dict[str, Any] = {"query": "*:*"}
        if limit is not None:
            data.update({"limit": limit})

        response = requests.post(
            href,
            params=self._client._params(),
            headers=self._client._get_headers(),
            json=data,
        )

        self._handle_response(200, "list assets", response)
        asset_details = self._handle_response(200, "list assets", response)["results"]
        space_values = [
            (
                m["metadata"]["name"],
                m["metadata"]["asset_type"],
                m["metadata"]["asset_id"],
            )
            for m in asset_details
        ]

        table = self._list(space_values, ["NAME", "ASSET_TYPE", "ASSET_ID"], None)
        return table

    def download(
        self,
        asset_id: str | None = None,
        filename: str | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download the content of a script asset.

        :param asset_id: unique ID of the script asset to be downloaded
        :type asset_id: str

        :param filename: filename to be used for the downloaded file
        :type filename: str

        :param rev_id: revision ID
        :type rev_id: str, optional

        :return: path to the downloaded asset content
        :rtype: str

        **Example:**

        .. code-block:: python

            client.script.download(asset_id, "script_file")
        """
        if filename is None:
            raise TypeError(
                "download() missing 1 required positional argument: 'filename'"
            )

        asset_id = _get_id_from_deprecated_uid(
            kwargs, asset_id, "asset", can_be_none=False
        )
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)
        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        Script._validate_type(asset_id, "asset_id", str, True)
        Script._validate_type(rev_id, "rev_id", str, False)

        params = self._client._params()

        if rev_id is not None:
            params.update({"revision_id": rev_id})

        import urllib

        if not self._client.ICP_PLATFORM_SPACES:
            asset_response = requests.get(
                self._client.service_instance._href_definitions.get_asset_href(
                    asset_id
                ),
                params=params,
                headers=self._client._get_headers(),
            )
        else:
            asset_response = requests.get(
                self._client.service_instance._href_definitions.get_data_asset_href(
                    asset_id
                ),
                params=params,
                headers=self._client._get_headers(),
            )
        asset_details = self._handle_response(200, "get assets", asset_response)

        attachment_id = asset_details["attachments"][0]["id"]

        response = requests.get(
            self._client.service_instance._href_definitions.get_attachment_href(
                asset_id, attachment_id
            ),
            params=params,
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            attachment_signed_url = response.json()["url"]
            if "connection_id" in asset_details["attachments"][0]:
                att_response = requests.get(attachment_signed_url)
            else:
                if not self._client.ICP_PLATFORM_SPACES:
                    att_response = requests.get(attachment_signed_url)
                else:
                    att_response = requests.get(
                        self._credentials.url + attachment_signed_url
                    )
            if att_response.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading asset"), att_response
                )

            downloaded_asset = att_response.content
            try:
                with open(filename, "wb") as f:
                    f.write(downloaded_asset)
                print(
                    "Successfully saved data asset content to file: '{}'".format(
                        filename
                    )
                )
                return os.path.abspath(filename)
            except IOError as e:
                raise WMLClientError(
                    "Saving asset with artifact_url to local file: '{}' failed.".format(
                        filename
                    ),
                    e,
                )
        else:
            raise WMLClientError("Failed while downloading the asset " + asset_id)  # type: ignore

    @staticmethod
    def get_id(asset_details: dict) -> str:
        """Get the unique ID of a stored script asset.

        :param asset_details: metadata of the stored script asset
        :type asset_details: dict

        :return: unique ID of the stored script asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_id = client.script.get_id(asset_details)
        """
        Script._validate_type(asset_details, "asset_details", object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "data_assets_details", ["metadata", "guid"]
        )

    @staticmethod
    def get_href(asset_details: dict) -> str:
        """Get the URL of a stored script asset.

        :param asset_details: details of the stored script asset
        :type asset_details: dict

        :return: href of the stored script asset
        :rtype: str

        **Example:**

        .. code-block:: python

            asset_details = client.script.get_details(asset_id)
            asset_href = client.script.get_href(asset_details)

        """
        Script._validate_type(asset_details, "asset_details", object, True)
        Script._validate_type_of_details(asset_details, DATA_ASSETS_DETAILS_TYPE)

        return WMLResource._get_required_element_from_dict(
            asset_details, "asset_details", ["metadata", "href"]
        )

    def update(
        self,
        script_id: str | None = None,
        meta_props: dict | None = None,
        file_path: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update a script with metadata, attachment, or both.

        :param script_id: ID of the script
        :type script_id: str

        :param meta_props: changes for the script matadata
        :type meta_props: dict, optional

        :param file_path: file path to the new attachment
        :type file_path: str, optional

        :return: updated metadata of the script
        :rtype: dict

        **Example:**

        .. code-block:: python

            script_details = client.script.update(script_id, meta, content_path)
        """
        script_id = _get_id_from_deprecated_uid(
            kwargs, script_id, "script", can_be_none=False
        )

        Script._validate_type(script_id, "script_id", str, True)

        if meta_props is None and file_path is None:
            raise WMLClientError(
                "At least either meta_props or file_path has to be provided"
            )

        updated_details = None

        url = self._client.service_instance._href_definitions.get_asset_href(script_id)

        # STEPS
        # STEP 1. Get existing metadata
        # STEP 2. If meta_props provided, we need to patch meta
        #   a. Construct meta patch string and call /v2/assets/<asset_id> to patch meta
        #   b. Construct entity patch if required and call /v2/assets/<asset_id>/attributes/script to patch entity
        # STEP 3. If file_path provided, we need to patch the attachment
        #   a. If attachment already exists for the script, delete it
        #   b. POST call to get signed url for upload
        #   c. Upload to the signed url
        #   d. Mark upload complete
        # STEP 4. Get the updated script record and return

        # STEP 1
        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of script_id provided."
                )
            else:
                raise ApiRequestFailure(
                    "Failure during {}.".format("getting script to update"), response
                )

        details = self._handle_response(200, "Get script details", response)

        attachments_response = None

        # STEP 2a.
        # Patch meta if provided
        if meta_props is not None:
            self._validate_type(meta_props, "meta_props", dict, True)

            meta_patch_payload = []
            entity_patch_payload = []

            # Since we are dealing with direct asset apis, there can be metadata or entity patch or both
            if "name" in meta_props or "description" in meta_props:
                props_for_asset_meta_patch = {}

                for key in meta_props:
                    if key == "name" or key == "description":
                        props_for_asset_meta_patch.update({key: meta_props[key]})

                meta_patch_payload = (
                    self.ConfigurationMetaNames._generate_patch_payload(
                        details, props_for_asset_meta_patch, with_validation=True
                    )
                )
            # STEP 2b.
            if "software_spec_uid" in meta_props:
                if details["entity"]["script"]["software_spec"]:
                    entity_patch_payload = [
                        {
                            "op": "replace",
                            "path": "/software_spec/base_id",
                            "value": meta_props["software_spec_uid"],
                        }
                    ]
                else:
                    entity_patch_payload = [
                        {
                            "op": "add",
                            "path": "/software_spec",
                            "value": "{base_id:"
                            + meta_props["software_spec_uid"]
                            + "}",
                        }
                    ]

            if meta_patch_payload:
                meta_patch_url = (
                    self._client.service_instance._href_definitions.get_asset_href(
                        script_id
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
                        script_id
                    )
                    + "/attributes/script"
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
                "script", script_id, file_path, current_attachment_id
            )

        if attachments_response is not None and "success" not in attachments_response:
            self._update_msg(updated_details)

        # Have to fetch again to reflect updated asset and attachment ids
        url = self._client.service_instance._href_definitions.get_asset_href(script_id)

        response = requests.get(
            url, params=self._client._params(), headers=self._client._get_headers()
        )

        if response.status_code != 200:
            if response.status_code == 404:
                raise WMLClientError(
                    "Invalid input. Unable to get the details of script_id provided."
                )
            else:
                raise ApiRequestFailure(
                    "Failure during {}.".format("getting script to update"), response
                )

        response = self._get_required_element_from_response(
            self._handle_response(200, "Get script details", response)
        )

        entity = response["entity"]

        try:
            del entity["script"]["ml_version"]
        except KeyError:
            pass

        final_response = {"metadata": response["metadata"], "entity": entity}

        return final_response

    def _update_msg(self, updated_details: Any) -> None:
        if updated_details is not None:
            print(
                "Could not update the attachment because of server error."
                " However metadata is updated. Try updating attachment again later"
            )
        else:
            raise WMLClientError(
                "Unable to update attachment because of server error. Try again later"
            )

    def delete(self, asset_id: str | None = None, **kwargs: Any) -> dict | str:
        """Delete a stored script asset.

        :param asset_id: ID of the script asset
        :type asset_id: str

        :return: status ("SUCCESS" or "FAILED") if deleted synchronously or dictionary with response
        :rtype: str | dict

        **Example:**

        .. code-block:: python

            client.script.delete(asset_id)

        """
        asset_id = _get_id_from_deprecated_uid(
            kwargs, asset_id, "asset", can_be_none=False
        )

        Script._validate_type(asset_id, "asset_id", str, True)
        if self._if_deployment_exist_for_asset(asset_id):
            raise WMLClientError(
                "Cannot delete script that has existing deployments. Please delete all associated deployments and try again"
            )

        response = requests.delete(
            self._client.service_instance._href_definitions.get_asset_href(asset_id),
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if response.status_code == 200:
            return self._get_required_element_from_response(response.json())
        else:
            return self._handle_response(204, "delete assets", response)

    def create_revision(self, script_id: str | None = None, **kwargs: Any) -> dict:
        """Create a revision for the given script. Revisions are immutable once created.
        The metadata and attachment at `script_id` is taken and a revision is created out of it.

        :param script_id: ID of the script
        :type script_id: str

        :return: revised metadata of the stored script
        :rtype: dict

        **Example:**

        .. code-block:: python

            script_revision = client.script.create_revision(script_id)
        """
        script_id = _get_id_from_deprecated_uid(
            kwargs, script_id, "script", can_be_none=False
        )
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Script._validate_type(script_id, "script_id", str, True)

        print("Creating script revision...")

        response = self._get_required_element_from_response(
            self._create_revision_artifact_for_assets(script_id, "Script")
        )

        entity = response["entity"]

        try:
            del entity["script"]["ml_version"]
        except KeyError:
            pass

        final_response = {"metadata": response["metadata"], "entity": entity}

        return final_response

    def list_revisions(
        self, script_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> DataFrame:
        """Print all revisions for the given script ID in a table format.

        :param script_id: ID of the stored script
        :type script_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.script.list_revisions(script_id)
        """
        script_id = _get_id_from_deprecated_uid(
            kwargs, script_id, "script", can_be_none=False
        )
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Script._validate_type(script_id, "script_id", str, True)

        url = (
            self._client.service_instance._href_definitions.get_asset_href(script_id)
            + "/revisions"
        )

        # /v2/assets/{asset_id}/revisions returns 'results' object
        script_resources = self._get_with_or_without_limit(
            url,
            None,
            "List Script revisions",
            summary=None,
            pre_defined=None,
            _all=self._should_get_all_values(limit),
        )["resources"]

        script_values = [
            (
                m["metadata"]["asset_id"],
                m["metadata"]["revision_id"],
                m["metadata"]["name"],
                m["metadata"]["commit_info"]["committed_at"],
            )
            for m in script_resources
        ]

        table = self._list(
            script_values,
            ["ID", "REVISION_ID", "NAME", "REVISION_COMMIT"],
            limit,
        )
        return table

    def get_revision_details(
        self, script_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> ListType:
        """Get metadata of the script revision.

        :param script_id: ID of the script
        :type script_id: str

        :param rev_id: ID of the revision. If this parameter is not provided, it returns the latest revision. If there is no latest revision, it returns an error.
        :type rev_id: str, optional

        :return: metadata of the stored script(s)
        :rtype: list

        **Example:**

        .. code-block:: python

            script_details = client.script.get_revision_details(script_id, rev_id)
        """
        script_id = _get_id_from_deprecated_uid(
            kwargs, script_id, "script", can_be_none=False
        )
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)
        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_warning, category=DeprecationWarning)
            rev_id = str(rev_id)
        # For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()

        Script._validate_type(script_id, "script_id", str, True)
        Script._validate_type(rev_id, "rev_id", str, False)

        if rev_id is None:
            rev_id = "latest"

        url = self._client.service_instance._href_definitions.get_asset_href(script_id)

        resources = self._get_with_or_without_limit(
            url,
            limit=None,
            op_name="asset_revision",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )["resources"]
        responses = [
            self._get_required_element_from_response(resource) for resource in resources
        ]

        final_responses = []
        for response in responses:
            entity = response["entity"]

            try:
                del entity["script"]["ml_version"]
            except KeyError:
                pass

            final_responses.append({"metadata": response["metadata"], "entity": entity})

        return final_responses

    def _get_required_element_from_response(self, response_data: dict) -> dict:

        WMLResource._validate_type(response_data, "scripts", dict)

        revision_id = None
        metadata = {
            "name": response_data["metadata"]["name"],
            "guid": response_data["metadata"]["asset_id"],
            "href": response_data["href"],
            "asset_type": response_data["metadata"]["asset_type"],
            "created_at": response_data["metadata"]["created_at"],
            "last_updated_at": response_data["metadata"]["usage"]["last_updated_at"],
        }

        try:
            if self._client.default_space_id is not None:
                metadata["space_id"] = response_data["metadata"]["space_id"]

            elif self._client.default_project_id is not None:
                metadata["project_id"] = response_data["metadata"]["project_id"]

            if "description" in response_data["metadata"]:
                metadata.update(
                    {"description": response_data["metadata"]["description"]}
                )

            if "revision_id" in response_data["metadata"]:
                revision_id = response_data["metadata"]["revision_id"]
                metadata.update(
                    {"revision_id": response_data["metadata"]["revision_id"]}
                )

            if "attachments" in response_data and response_data["attachments"]:
                metadata.update(
                    {"attachment_id": response_data["attachments"][0]["id"]}
                )

            if "commit_info" in response_data["metadata"] and revision_id is not None:
                metadata.update(
                    {
                        "revision_commit_date": response_data["metadata"][
                            "commit_info"
                        ]["committed_at"]
                    }
                )

            new_el = {"metadata": metadata, "entity": response_data["entity"]}

            if self._client.default_project_id is not None:
                href_without_host = response_data["href"].split(".com")[-1]
                new_el["metadata"].update({"href": href_without_host})

            return new_el
        except Exception:
            raise WMLClientError(
                f"Failed to read Response from down-stream service: {response_data}"
            )
