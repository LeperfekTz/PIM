#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import logging
from typing import (
    TYPE_CHECKING,
    Callable,
    Any,
    TypeAlias,
    cast,
    Iterable,
    Literal,
    overload,
    Generator,
)

import json

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.utils import (
    get_type_of_details,
    next_resource_generator,
)
from ibm_watsonx_ai.wml_client_error import (
    MissingValue,
    WMLClientError,
    NoWMLCredentialsProvided,
    ApiRequestFailure,
    UnexpectedType,
    MissingMetaProp,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.sw_spec import SpecStates
    import pandas as pd
    import requests as _requests
    import httpx

    ArtifactDetailsType: TypeAlias = Generator | dict[str, Any]
    ResponseLike: TypeAlias = _requests.Response | httpx.Response

_DEFAULT_LIST_BATCH_SIZE_LIMIT = 200

httpx_handled = [
    "ibm_watsonx_ai.deployments",
    "ibm_watsonx_ai.foundation_models.ilab.taxonomies",
    "ibm_watsonx_ai.foundation_models.ilab.documents",
    "ibm_watsonx_ai.foundation_models.ilab.synthetic_data",
    "ibm_watsonx_ai.foundation_models_manager",
    "ibm_watsonx_ai.foundation_models.inference.fm_model_inference",
]


class WMLResource:
    def __init__(self, name: str, client: APIClient):
        self._logger = logging.getLogger(__name__)
        self._name = name
        WMLResource._validate_type(client, "client", object, True)
        if client.credentials is None:
            raise NoWMLCredentialsProvided
        WMLResource._validate_type(client.credentials, "credentials", Credentials, True)
        if not client.ICP_PLATFORM_SPACES:
            WMLResource._validate_type(client.token, "token", str, True)
        self._client = client
        self._credentials = client.service_instance._credentials

    @overload
    def _handle_response(
        self,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: Literal[True] = True,
        _silent_response_logging: bool = False,
        _field_to_hide: str | None = None,
    ) -> dict: ...

    @overload
    def _handle_response(
        self,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: Literal[False],
        _silent_response_logging: bool = ...,
        _field_to_hide: str | None = ...,
    ) -> str: ...

    def _handle_response(
        self,
        expected_status_code: int,
        operationName: str,
        response: ResponseLike,
        json_response: bool = True,
        _silent_response_logging: bool = False,
        _field_to_hide: str | None = None,
    ) -> dict | str:
        """Internal method for handling HTTP requests responses.

        :param _silent_response_logging: If True the whole response text is not visible in logging messages, defaults to False
        :type _silent_response_logging: bool, optional

        :param _field_to_hide: Determine what field in the response should be hide in logging, defaults to None
        :type _field_to_hide: str | None, optional
        """
        if "dele" in operationName or "cancel" in operationName:
            if response.status_code == expected_status_code:
                return "SUCCESS"
            else:
                msg = "{} failed. Reason: {}".format(
                    operationName,
                    response.text,
                )
                raise WMLClientError(msg)

        if response.status_code == expected_status_code:

            self._logger.info(
                "Successfully finished {} for url: '{}'".format(
                    operationName, response.url
                )
            )

            if self._logger.level <= logging.DEBUG:
                if _field_to_hide is not None:
                    replace_value = "..."
                    try:

                        def decode_dict(processed_dict: dict) -> dict:
                            if _field_to_hide in processed_dict.keys():
                                processed_dict[_field_to_hide] = replace_value
                            return processed_dict

                        response_text = json.dumps(
                            response.json(object_hook=decode_dict)
                        )
                    except Exception:
                        response_text = response.text
                else:
                    response_text = response.text

                self._logger.debug(
                    "Response({} {} {}){}".format(
                        response.request.method,
                        response.url,
                        response.status_code,
                        (f": {response_text}" if not _silent_response_logging else ""),
                    )
                )
            if json_response:
                try:
                    return response.json()
                except Exception as e:
                    raise WMLClientError(
                        f"Failure during parsing json response: '{response.text}'",
                        e,
                    )
            else:
                return response.text
        else:
            raise ApiRequestFailure(
                "Failure during {}.".format(operationName),
                response,
            )

    @staticmethod
    def _get_required_element_from_dict(
        el: dict, root_path: str, path: list[str]
    ) -> str:
        WMLResource._validate_type(el, root_path, dict)
        WMLResource._validate_type(root_path, "root_path", str)
        WMLResource._validate_type(path, "path", list)

        if path.__len__() < 1:
            raise WMLClientError("Unexpected path length: {}".format(path.__len__))

        try:
            new_el = el[path[0]]
            new_path = path[1:]
        except Exception as e:
            raise MissingValue(root_path + "." + str(path[0]), e)

        if path.__len__() > 1:
            return WMLResource._get_required_element_from_dict(
                new_el, root_path + "." + path[0], new_path
            )
        else:
            if new_el is None:
                raise MissingValue(root_path + "." + str(path[0]))

            return new_el

    def _get_asset_based_resource(
        self,
        asset_id: str,
        asset_type: str,
        get_required_element_from_response: Callable,
        limit: int | None = None,
        get_all: bool | None = None,
    ) -> dict:
        WMLResource._validate_type(asset_id, f"{asset_type}_id", str, False)

        if asset_id:
            response = requests.get(
                self._client.service_instance._href_definitions.get_data_asset_href(
                    asset_id
                ),
                params=self._client._params(),
                headers=self._client._get_headers(),
            )

            return get_required_element_from_response(
                self._handle_response(200, f"get {asset_type} details", response)
            )

        else:
            href = (
                self._client.service_instance._href_definitions.get_asset_search_href(
                    asset_type
                )
            )

            def get_chunk(data):
                response = requests.post(
                    href,
                    json=data,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                )

                return [
                    get_required_element_from_response(x)
                    for x in self._handle_response(
                        200, f"get {asset_type}s details", response
                    )[
                        "results"  # type: ignore[index]
                    ]
                ], response.json().get("next")

            data = {"query": "*:*"}

            result, data = get_chunk(data)

            if get_all:

                while data is not None and (limit is None or len(result) < limit):
                    res, data = get_chunk(data)
                    result.extend(res)

            return {"resources": result if limit is None else result[:limit]}

    @staticmethod
    def _validate_type(
        el: Any,
        el_name: str,
        expected_type: type | list[type],
        mandatory: bool = True,
        raise_error_for_list: bool = False,
    ) -> bool | None:
        if el_name is None:
            raise MissingValue("el_name")

        if type(el_name) is not str:
            raise UnexpectedType("el_name", str, type(el_name))

        if expected_type is None:
            raise MissingValue("expected_type")

        if type(expected_type) is not type and type(expected_type) is not list:
            raise UnexpectedType("expected_type", "type or list", type(expected_type))

        if type(mandatory) is not bool:
            raise UnexpectedType("mandatory", bool, type(mandatory))

        if mandatory and el is None:
            raise MissingValue(el_name)
        elif el is None:
            return None

        if type(expected_type) is list:
            try:
                next((x for x in expected_type if isinstance(el, x)))
                return True
            except StopIteration:
                if raise_error_for_list:
                    raise UnexpectedType(el_name, expected_type, type(el))
                return False  # keep for backward compatibilty
        else:
            if not isinstance(el, expected_type):
                raise UnexpectedType(el_name, expected_type, type(el))
        return None

    @staticmethod
    def _validate_meta_prop(
        meta_props: dict, name: str, expected_type: type, mandatory: bool = True
    ) -> None:
        if name in meta_props:
            WMLResource._validate_type(
                meta_props[name], "meta_props." + name, expected_type, mandatory
            )
        else:
            if mandatory:
                raise MissingMetaProp(name)

    @staticmethod
    def _validate_type_of_details(details: dict, expected_type: str | list) -> None:
        actual_type = get_type_of_details(details)

        if type(expected_type) is list:
            expected_types = expected_type
        else:
            expected_types = [expected_type]

        if not any([actual_type == exp_type for exp_type in expected_types]):
            logger = logging.getLogger(__name__)
            logger.debug(
                "Unexpected type of '{}', expected: '{}', actual: '{}', occured for details: {}".format(
                    "details", expected_type, actual_type, details
                )
            )
            raise UnexpectedType("details", expected_type, actual_type)

    @overload
    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> dict: ...

    @overload
    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> Generator: ...

    def _get_artifact_details(
        self,
        base_url: str,
        id: str | None,
        limit: int | None,
        resource_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> ArtifactDetailsType:
        op_name = "getting {} details".format(resource_name)

        if id is None:
            return self._get_with_or_without_limit(
                url=base_url,
                limit=limit,
                op_name=resource_name,
                summary=summary,
                pre_defined=pre_defined,
                query_params=query_params,
                _async=_async,
                _all=_all,
                _filter_func=_filter_func,
            )
        else:
            if query_params is None:
                params = self._client._params()
            else:
                params = query_params

            if "userfs" in params:
                params.pop("userfs")

            url = base_url + "/" + id

            request_handler = (
                self._client.httpx_client if self._name in httpx_handled else requests
            )

            response_get = request_handler.get(
                url=url,
                params=params,
                headers=self._client._get_headers(),
            )

            return self._handle_response(200, op_name, response_get)

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[False] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> dict: ...

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: Literal[True] = ...,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> Generator: ...

    @overload
    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> ArtifactDetailsType: ...

    def _get_with_or_without_limit(
        self,
        url: str,
        limit: int | None,
        op_name: str,
        summary: bool | None = None,
        pre_defined: bool | None = None,
        revision: str | None = None,
        skip_space_project_chk: bool = False,
        query_params: dict | None = None,
        _async: bool = False,
        _all: bool = False,
        _filter_func: Callable | None = None,
    ) -> ArtifactDetailsType:
        params = self._client._params(skip_space_project_chk)

        if self._name in httpx_handled:
            request_handler = self._client.httpx_client
            use_httpx = True
        else:
            request_handler = requests
            use_httpx = False

        if query_params is not None:
            params.update(query_params)

        if summary is False:
            params.update({"summary": "false"})

        if pre_defined is True:
            params.update({"system_runtimes": "true"})

        if "userfs" in params:
            params.pop("userfs")

        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            params.update({"limit": limit})
        else:
            params.update({"limit": 200})

        if revision is not None:
            if op_name == "asset_revision":
                params.update(
                    {"revision_id": revision}
                )  # CAMS assets api takes 'revision_id' query parameter
            else:
                params.update({"rev": revision})

        resources = []

        href = "/".join(url.split("/")[3:])
        url_2 = "/".join(url.split("/")[:3])

        resource_generator = next_resource_generator(
            self._client, url_2, href, params, _all, _filter_func, use_httpx
        )

        if _async:
            return resource_generator
        else:
            if _all:
                for entry in resource_generator:
                    resources.extend(entry["resources"])

                return {"resources": resources}

            else:
                response_get = request_handler.get(
                    url=url,
                    headers=self._client._get_headers(),
                    params=params,
                )

                result = cast(dict, self._handle_response(200, op_name, response_get))
                if "resources" in result:
                    resources.extend(result["resources"])

                elif "metadata" in result:
                    resources.extend([result])

                else:
                    resources.extend(cast(Iterable, result.get("results")))

                return {
                    "resources": _filter_func(resources) if _filter_func else resources
                }

    def _if_deployment_exist_for_asset(self, asset_id: str) -> bool:

        deployment_href = (
            self._client.service_instance._href_definitions.get_deployments_href()
            + "?asset_id="
            + asset_id
        )
        response_deployment = requests.get(
            deployment_href,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        deployment_json = cast(
            dict,
            self._handle_response(200, "Get deployment details", response_deployment),
        )
        resources = deployment_json.get("resources")
        if resources:
            return True
        else:
            return False

    def _list(
        self,
        values: list,
        header: list,
        limit: int | None,
        sort_by: str | None = "CREATED",
    ) -> pd.DataFrame:
        if sort_by is not None and sort_by in header:
            column_no = header.index(sort_by)
            values = sorted(values, key=lambda x: x[column_no], reverse=True)

        import pandas as pd

        if limit is None:
            return pd.DataFrame(values, columns=header)

        else:
            return pd.DataFrame(values[:limit], columns=header)

    def _create_revision_artifact(
        self, base_url: str, id: str, resource_name: str
    ) -> dict:
        op_name = "Creation revision for {}".format(resource_name)
        if self._client.default_project_id is not None:
            input_json = {"project_id": self._client.default_project_id}
        else:
            input_json = {"space_id": self._client.default_space_id}

        url = base_url + "/" + id + "/revisions"
        if self._client.CLOUD_PLATFORM_SPACES:
            params = self._client._params(skip_for_create=True)
            response = requests.post(
                url, headers=self._client._get_headers(), params=params, json=input_json
            )
        else:  # ICP_PLATFORM_SPACES
            response = requests.post(
                url,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True),
                json=input_json,
            )

        return self._handle_response(201, op_name, response)

    def _create_revision_artifact_for_assets(
        self, id: str, resource_name: str
    ) -> dict | str:
        op_name = "Creation revision for {}".format(resource_name)

        url = (
            self._client.service_instance._href_definitions.get_asset_href(id)
            + "/revisions"
        )
        commit_message = "Revision creation for " + resource_name + " " + id

        payload_json = {"commit_message": commit_message}

        # CAMS revision creation api takes space_id as a query parameter. Hence
        # params has to be passed

        response = requests.post(
            url,
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=payload_json,
        )

        return self._handle_response(201, op_name, response)

    def _update_attachment_for_assets(
        self,
        asset_type: str,
        asset_id: str,
        file_path: str,
        current_attachment_id: str | None = None,
    ) -> Literal[
        "error_in_marking_attachment_complete",
        "error_in_uploading_attachment",
        "error_in_getting_signed_url",
        "error_in_deleting_existing_attachment",
        "success",
    ]:

        if current_attachment_id is not None:
            # Delete existing attachment to upload new attachment
            attachments_id_url = (
                self._client.service_instance._href_definitions.get_asset_href(asset_id)
                + "/attachments/"
                + current_attachment_id
            )

            delete_attachment_response = requests.delete(
                attachments_id_url,
                headers=self._client._get_headers(),
                params=self._client._params(),
            )

        if (
            delete_attachment_response.status_code == 204
            or current_attachment_id is None
        ):
            attachment_meta = {
                "asset_type": asset_type,
                "name": "attachment_" + asset_id,
            }

            attachments_url = (
                self._client.service_instance._href_definitions.get_asset_href(asset_id)
                + "/attachments"
            )

            # STEP 3b.
            # Get the signed url from CAMS to upload the attachment
            attachment_response = requests.post(
                attachments_url,
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=attachment_meta,
            )

            attachment_details = cast(
                dict,
                self._handle_response(
                    201, "creating new attachment", attachment_response
                ),
            )
            if attachment_response.status_code == 201:
                attachment_id = attachment_details["attachment_id"]
                attachment_url = attachment_details["url1"]

                # STEP 3c.
                # Upload attachment
                with open(file_path, "rb") as f:
                    if not self._client.ICP_PLATFORM_SPACES:
                        put_response = requests.put(attachment_url, data=f.read())
                    else:
                        put_response = requests.put(
                            self._credentials.url + attachment_url,
                            files={
                                "file": (
                                    attachment_meta["name"],
                                    f,
                                    "application/octet-stream",
                                )
                            },
                        )

                if put_response.status_code == 201 or put_response.status_code == 200:
                    # STEP 3d.
                    # Mark attachment complete
                    complete_response = requests.post(
                        self._client.service_instance._href_definitions.get_attachment_complete_href(
                            asset_id, attachment_id
                        ),
                        headers=self._client._get_headers(),
                        params=self._client._params(),
                    )

                    if complete_response.status_code != 200:
                        self._logger.error(
                            "Error in marking attachment complete for asset {}".format(
                                asset_id
                            )
                        )
                        return "error_in_marking_attachment_complete"
                else:
                    self._logger.error(
                        "Error in uploading attachment for asset {}".format(asset_id)
                    )
                    return "error_in_uploading_attachment"
            else:
                self._logger.error(
                    "Error in getting signed url for attachment for asset {}".format(
                        asset_id
                    )
                )
                return "error_in_getting_signed_url"
        else:
            self._logger.error(
                "Error in deleting existing attachment {} for asset {}".format(
                    current_attachment_id, asset_id
                )
            )
            return "error_in_deleting_existing_attachment"

        return "success"

    def _get_and_cache_spec_ids_for_state(self, spec_state: SpecStates) -> list:
        if spec_state in self._client._spec_ids_per_state:
            return self._client._spec_ids_per_state[spec_state]

        url = (
            self._client.service_instance._href_definitions.get_software_specifications_list_href()
        )

        params = self._client._params()
        params["state"] = spec_state.value

        response_post = requests.get(
            url, params=params, headers=self._client._get_headers()
        )

        res = cast(
            dict,
            self._handle_response(200, "list software specifications", response_post),
        )

        spec_ids = [r["id"] for r in res["results"]]

        self._client._spec_ids_per_state[spec_state] = spec_ids

        return spec_ids

    @staticmethod
    def _should_get_all_values(limit: int | None):
        return limit is None or limit > _DEFAULT_LIST_BATCH_SIZE_LIMIT

    @staticmethod
    def _get_filter_func_by_spec_ids(spec_ids: list) -> Callable:
        def filter_func(resources: list) -> list:
            return [
                r
                for r in resources
                if r["entity"].get("software_spec", {}).get("id") in spec_ids
            ]

        return filter_func

    @staticmethod
    def _get_filter_func_by_artifact_name(artifact_name: str) -> Callable:
        def filter_func(resources: list) -> list:
            return [
                r
                for r in resources
                if r.get("metadata", {}).get("name") == artifact_name
            ]

        return filter_func
