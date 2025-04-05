#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
import os
from typing import TYPE_CHECKING, Callable, Any, cast, Literal
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import FunctionMetaNames
from ibm_watsonx_ai.utils import FUNCTION_DETAILS_TYPE, is_of_python_basic_type
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    UnexpectedType,
    ApiRequestFailure,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.lifecycle import SpecStates
    import pandas


class Functions(WMLResource):
    """Store and manage functions."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

        self.ConfigurationMetaNames = FunctionMetaNames()

    def store(self, function: str | Callable, meta_props: str | dict[str, Any]) -> dict:
        """Create a function.

        As a 'function' may be used one of the following:
            - filepath to gz file
            - 'score' function reference, where the function is the function which will be deployed
            - generator function, which takes no argument or arguments which all have primitive python default values
              and as result return 'score' function

        :param function: path to file with archived function content or function (as described above)
        :type function: str or function
        :param meta_props: meta data or name of the function, to see available meta names
            use ``client._functions.ConfigurationMetaNames.show()``
        :type meta_props: str or dict

        :return: stored function metadata
        :rtype: dict

        **Examples**

        The most simple use is (using `score` function):

        .. code-block:: python

            meta_props = {
                client._functions.ConfigurationMetaNames.NAME: "function",
                client._functions.ConfigurationMetaNames.DESCRIPTION: "This is ai function",
                client._functions.ConfigurationMetaNames.SOFTWARE_SPEC_UID: "53dc4cf1-252f-424b-b52d-5cdd9814987f"}

            def score(payload):
                values = [[row[0]*row[1]] for row in payload['values']]
                return {'fields': ['multiplication'], 'values': values}

            stored_function_details = client._functions.store(score, meta_props)

        Other, more interesting example is using generator function.
        In this situation it is possible to pass some variables:

        .. code-block:: python

            creds = {...}

            def gen_function(credentials=creds, x=2):
                def f(payload):
                    values = [[row[0]*row[1]*x] for row in payload['values']]
                    return {'fields': ['multiplication'], 'values': values}
                return f

            stored_function_details = client._functions.store(gen_function, meta_props)

        """
        self._client._check_if_either_is_set()

        import types

        Functions._validate_type(function, "function", [str, types.FunctionType], True)
        Functions._validate_type(meta_props, "meta_props", [dict, str], True)

        if type(meta_props) is str:
            meta_props = {self.ConfigurationMetaNames.NAME: meta_props}

        self.ConfigurationMetaNames._validate(meta_props)

        content_path, user_content_file, archive_name = self._prepare_function_content(
            function
        )

        try:

            function_metadata = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

            if self._client.default_space_id is not None:
                function_metadata["space_id"] = self._client.default_space_id
            elif self._client.default_project_id is not None:
                function_metadata["project_id"] = self._client.default_project_id
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/"
                    "client.set.default_project(<PROJECT_UID>) to proceed."
                )

            response_post = requests.post(
                self._client.service_instance._href_definitions.get_functions_href(),
                json=function_metadata,
                params=self._client._params(skip_for_create=True),
                headers=self._client._get_headers(),
            )

            details = self._handle_response(
                expected_status_code=201,
                operationName="saving function",
                response=response_post,
            )
            ##TODO_V4 Take care of this since the latest swagger endpoint is not working

            function_content_url = (
                self._client.service_instance._href_definitions.get_function_href(
                    details["metadata"]["id"]
                )
                + "/code"
            )

            put_header = self._client._get_headers(no_content_type=True)
            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    function_content_url,
                    data=data,
                    params=self._client._params(),
                    headers=put_header,
                )

        except Exception as e:
            raise e
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

        if response_definition_put.status_code != 201:
            self.delete(details["metadata"]["id"])
        self._handle_response(
            201, "saving function content", response_definition_put, json_response=False
        )

        return details

    def update(
        self,
        function_id: str | None = None,
        changes: dict | None = None,
        update_function: str | Callable | None = None,
        **kwargs: Any,
    ) -> dict:
        """Updates existing function metadata.

        :param function_id: ID of function which define what should be updated
        :type function_id: str
        :param changes: elements which should be changed, where keys are ConfigurationMetaNames
        :type changes: dict
        :param update_function: path to file with archived function content or function which should be changed
            for specific function_id, this parameter is valid only for CP4D 3.0.0
        :type update_function: str or function, optional

        **Example:**

        .. code-block:: python

            metadata = {
                client._functions.ConfigurationMetaNames.NAME: "updated_function"
            }

            function_details = client._functions.update(function_id, changes=metadata)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        # No additional checking for `changes` because of presence _validate_type(changes, ..., True)

        self._client._check_if_either_is_set()

        self._validate_type(function_id, "function_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self.get_details(function_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details["entity"], changes, with_validation=True
        )

        url = self._client.service_instance._href_definitions.get_function_href(
            function_id
        )
        headers = self._client._get_headers()
        response = requests.patch(
            url, json=patch_payload, params=self._client._params(), headers=headers
        )
        updated_details = self._handle_response(200, "function patch", response)

        if update_function is not None:
            self._update_function_content(function_id, update_function)

        return updated_details

    def _update_function_content(
        self,
        function_id: str | None = None,
        updated_function: str | Callable | None = None,
        **kwargs: Any,
    ) -> None:

        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        # No additional checking for `updated_function` because of presence _validate_type(updated_function, ..., True)

        import types

        Functions._validate_type(
            updated_function, "function", [str, types.FunctionType], True
        )

        updated_function = cast(str | Callable, updated_function)
        content_path, user_content_file, archive_name = self._prepare_function_content(
            updated_function
        )
        try:
            function_content_url = (
                self._client.service_instance._href_definitions.get_function_href(
                    function_id
                )
                + "/code"
            )

            put_header = self._client._get_headers(no_content_type=True)
            with open(content_path, "rb") as data:
                response_definition_put = requests.put(
                    function_content_url,
                    data=data,
                    params=self._client._params(),
                    headers=put_header,
                )
                if (
                    response_definition_put.status_code != 200
                    and response_definition_put.status_code != 204
                    and response_definition_put.status_code != 201
                ):
                    raise WMLClientError(
                        " Unable to update function content" + response_definition_put
                    )
        except Exception as e:
            raise e
        finally:
            try:
                os.remove(archive_name)  # type: ignore[arg-type]
            except Exception:
                pass

    def download(
        self,
        function_id: str | None = None,
        filename: str = "downloaded_function.gz",
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Download function content from Watson Machine Learning repository to local file.

        :param function_id: stored function ID
        :type function: str
        :param filename: name of local file to create, example: function_content.gz
        :type filename: str, optional


        :return: path to the downloaded function content
        :rtype: str

        **Example:**

        .. code-block:: python

            client._functions.download(function_id, 'my_func.tar.gz')

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", can_be_none=True)
        if rev_id is not None and isinstance(rev_id, int):
            rev_id_as_int_deprecated_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_deprecated_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        self._client._check_if_either_is_set()

        if os.path.isfile(filename):
            raise WMLClientError(
                "File with name: '{}' already exists.".format(filename)
            )

        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(filename, "filename", str, True)

        artifact_url = (
            self._client.service_instance._href_definitions.get_function_href(
                function_id
            )
        )

        artifact_content_url = (
            self._client.service_instance._href_definitions.get_function_href(
                function_id
            )
            + "/code"
        )

        try:
            if self._client.CLOUD_PLATFORM_SPACES:
                params = self._client._params()
                if rev_id is not None:
                    params.update({"rev": rev_id})

                r = requests.get(
                    artifact_content_url,
                    params=params,
                    headers=self._client._get_headers(),
                    stream=True,
                )
            else:
                params = self._client._params()

                if rev_id is not None:
                    params.update({"revision_id": rev_id})

                r = requests.get(
                    artifact_content_url,
                    params=params,
                    headers=self._client._get_headers(),
                    stream=True,
                )
            if r.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading function"), r
                )

            downloaded_model = r.content
            self._logger.info(
                "Successfully downloaded artifact with artifact_url: {}".format(
                    artifact_url
                )
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            raise WMLClientError(
                "Downloading function content with artifact_url: '{}' failed.".format(
                    artifact_url
                ),
                e,
            )

        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)
            print("Successfully saved function content to file: '{}'".format(filename))
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                "Saving function content with artifact_url: '{}' failed.".format(
                    filename
                ),
                e,
            )

    def delete(
        self, function_id: str | None = None, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Delete a stored function.

        :param function_id: stored function ID
        :type function_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client._functions.delete(function_id)
        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Functions._validate_type(function_id, "function_id", str, True)
        if self._if_deployment_exist_for_asset(function_id):
            raise WMLClientError(
                "Cannot delete function that has existing deployments. Please delete all associated deployments and try again"
            )

        function_endpoint = (
            self._client.service_instance._href_definitions.get_function_href(
                function_id
            )
        )
        self._logger.debug(
            "Deletion artifact function endpoint: {}".format(function_endpoint)
        )
        if not self._client.ICP_PLATFORM_SPACES:
            response_delete = requests.delete(
                function_endpoint,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        else:
            if Functions._if_deployment_exist_for_asset(self, function_id):
                raise WMLClientError(
                    "Cannot delete function that has existing deployments. Please delete all associated deployments and try again"
                )
            response_delete = requests.delete(
                function_endpoint,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        return self._handle_response(204, "function deletion", response_delete, False)

    def get_details(
        self,
        function_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        function_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of function(s). If neither function ID nor function name is specified,
        the metadata of all functions is returned.
        If only function name is specified, metadata of functions with the name is returned (if any).

        :param function_id: ID of the function
        :type: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `function_id` is None
        :type spec_state: SpecStates, optional

        :param function_name: name of the function, can be used only when `function_id` is None
        :type function_name: str, optional

        :return: metadata of the function
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state=True` may break set `limit`,
            returning less records than stated by set `limit`.

        **Examples**

        .. code-block:: python

            function_details = client._functions.get_details(function_id)
            function_details = client._functions.get_details(function_name='Sample_function')
            function_details = client._functions.get_details()
            function_details = client._functions.get_details(limit=100)
            function_details = client._functions.get_details(limit=100, get_all=True)
            function_details = []
            for entry in client._functions.get_details(limit=100, asynchronous=True, get_all=True):
                function_details.extend(entry)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function", True)

        Functions._validate_type(function_id, "function_uid", str, False)
        Functions._validate_type(limit, "limit", int, False)
        Functions._validate_type(asynchronous, "asynchronous", bool, False)
        Functions._validate_type(get_all, "get_all", bool, False)
        Functions._validate_type(spec_state, "spec_state", object, False)

        if limit and spec_state:
            print(
                "Warning: In current implementation setting `spec_state=True` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Functions._validate_type(function_id, "function_id", str, False)
        Functions._validate_type(limit, "limit", int, False)
        url = self._client.service_instance._href_definitions.get_functions_href()

        if function_id is None:
            if spec_state:
                filter_func = self._get_filter_func_by_spec_ids(
                    self._get_and_cache_spec_ids_for_state(spec_state)
                )
            elif function_name:
                filter_func = self._get_filter_func_by_artifact_name(function_name)
            else:
                filter_func = None

            return self._get_artifact_details(
                url,
                function_id,
                limit,
                "functions",
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )

        else:
            return self._get_artifact_details(url, function_id, limit, "functions")

    @staticmethod
    def get_id(function_details: dict) -> str:
        """Get ID of stored function.

        :param function_details: metadata of the stored function
        :type function_details: dict

        :return: ID of stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_id)
            function_id = client._functions.get_id(function_details)
        """

        Functions._validate_type(function_details, "function_details", object, True)
        if "asset_id" in function_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "asset_id"]
            )
        else:
            if "guid" in function_details["metadata"]:
                Functions._validate_type_of_details(
                    function_details, FUNCTION_DETAILS_TYPE
                )
                return WMLResource._get_required_element_from_dict(
                    function_details, "function_details", ["metadata", "guid"]
                )
            else:
                return WMLResource._get_required_element_from_dict(
                    function_details, "function_details", ["metadata", "id"]
                )

    @staticmethod
    def get_uid(function_details: dict) -> str:
        """Get UID of stored function.

        *Deprecated:* Use get_id(function_details) instead.

        :param function_details: metadata of the stored function
        :type function_details: dict

        :return: UID of stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_uid)
            function_uid = client._functions.get_uid(function_details)
        """
        get_uid_method_deprecated_warning = (
            "This method is deprecated, please use `get_id(function_details)` instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)

        return Functions.get_id(function_details)

    @staticmethod
    def get_href(function_details: dict) -> str:
        """Get the URL of a stored function.

        :param function_details: details of the stored function
        :type function_details: dict

        :return: href of the stored function
        :rtype: str

        **Example:**

        .. code-block:: python

            function_details = client.repository.get_function_details(function_id)
            function_url = client._functions.get_href(function_details)
        """
        Functions._validate_type(function_details, "function_details", object, True)
        if "asset_type" in function_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                function_details, "function_details", ["metadata", "href"]
            )

            # raise WMLClientError(u'This method is not supported for IBM Watson Studio Desktop. ')
        else:
            if "href" in function_details["metadata"]:
                Functions._validate_type_of_details(
                    function_details, FUNCTION_DETAILS_TYPE
                )
                return WMLResource._get_required_element_from_dict(
                    function_details, "function_details", ["metadata", "href"]
                )
            else:
                # Cloud Convergence
                return "/ml/v4/functions/{}".format(function_details["metadata"]["id"])

    def list(self, limit: int | None = None) -> pandas.DataFrame:
        """Return stored functions in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed functions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._functions.list()
        """
        ##For CP4D, check if either spce or project ID is set
        table = None

        self._client._check_if_either_is_set()

        function_resources = self.get_details(
            get_all=self._should_get_all_values(limit)
        )["resources"]

        sw_spec_info = {
            s["id"]: s
            for s in self._client.software_specifications.get_details(state_info=True)[
                "resources"
            ]
        }

        def get_spec_info(spec_id: str | None, prop: str) -> str:
            if spec_id and spec_id in sw_spec_info:
                return sw_spec_info[spec_id].get(prop, "")
            else:
                return ""

        function_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
                m["entity"]["type"] if "type" in m["entity"] else None,
                get_spec_info(m["entity"].get("software_spec", {}).get("id"), "state"),
                get_spec_info(
                    m["entity"].get("software_spec", {}).get("id"), "replacement"
                ),
            )
            for m in function_resources
        ]

        table = self._list(
            function_values,
            ["ID", "NAME", "CREATED", "TYPE", "SPEC_STATE", "SPEC_REPLACEMENT"],
            limit,
        )

        return table

    def clone(
        self,
        function_id: str | None = None,
        space_id: str | None = None,
        action: str = "copy",
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        raise WMLClientError(Messages.get_message(message_id="cloning_not_supported"))

    def create_revision(self, function_id: str | None = None, **kwargs: Any) -> dict:
        """Create a new function revision.

        :param function_id: unique ID of the function
        :type function_id: str

        :return: revised metadata of the stored function
        :rtype: dict

        **Example:**

        .. code-block:: python

            client._functions.create_revision(pipeline_id)
        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        Functions._validate_type(function_id, "function_id", str, False)

        url = self._client.service_instance._href_definitions.get_functions_href()
        return self._create_revision_artifact(url, function_id, "functions")

    def get_revision_details(
        self, function_id: str, rev_id: str, **kwargs: Any
    ) -> dict:
        """Get metadata of a specific revision of a stored function.

        :param function_id: definition of the stored function
        :type function_id: str

        :param rev_id: unique ID of the function revision
        :type rev_id: str

        :return: stored function revision metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            function_revision_details = client._functions.get_revision_details(function_id, rev_id)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")
        if rev_id is not None and isinstance(rev_id, int):
            rev_id_as_int_deprecated_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_deprecated_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        self._client._check_if_either_is_set()
        Functions._validate_type(function_id, "function_id", str, True)
        Functions._validate_type(rev_id, "rev_id", str, True)

        url = self._client.service_instance._href_definitions.get_function_href(
            function_id
        )
        return self._get_with_or_without_limit(
            url,
            limit=None,
            op_name="function",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def list_revisions(
        self, function_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        """Print all revisions for a given function ID in a table format.

        :param function_id: unique ID of the stored function
        :type function_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client._functions.list_revisions(function_id)

        """
        function_id = _get_id_from_deprecated_uid(kwargs, function_id, "function")

        self._client._check_if_either_is_set()

        Functions._validate_type(function_id, "function_id", str, True)

        url = self._client.service_instance._href_definitions.get_function_href(
            function_id
        )

        # CP4D logic is wrong. By passing "revisions" in second param above for _get_artifact_details()
        # it won't even consider limit value and also GUID gives only rev number, not actual guid

        function_resources = self._get_artifact_details(
            url + "/revisions",
            None,
            None,
            "function revisions",
            _all=self._should_get_all_values(limit),
        )["resources"]

        function_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["rev"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in function_resources
        ]

        table = self._list(
            function_values,
            ["ID", "REV", "NAME", "CREATED"],
            limit,
        )

        return table

    @staticmethod
    def _prepare_function_content(
        function: str | Callable,
    ) -> tuple[str, bool, str | None]:
        """Prepare function content for storing in the repository.
        If a Callable is passed, this function:
            - removes unnecessary indentation
            - validates and injects default parameters' values if ``function`` returns scoring function
            - creates an archive if ``function`` is a Callable

        :param function: _description_
        :type function: Union[str, Callable]

        :raises UnexpectedType: if any of ``function`` default parameters is not of basic Python type

        :raises WMLClientError: if ``function`` is defined incorrectly

        :return: path to compressed function source if archive is provided by user, name of the archive if not provided by user
        :rtype: tuple[str, bool, Optional[str]]
        """
        user_content_file = False
        archive_name = None

        if type(function) is str:
            content_path = function
            user_content_file = True
        else:
            function = cast(Callable, function)
            try:
                import inspect
                import gzip
                import uuid
                import re
                import shutil

                code = inspect.getsource(function).split("\n")
                r = re.compile(r"^ *")
                m = r.search(code[0])
                intend = m.group(0)  # type: ignore[union-attr]

                code = [line.replace(intend, "", 1) for line in code]

                args_spec = inspect.getfullargspec(function)

                defaults = args_spec.defaults if args_spec.defaults is not None else []
                args = args_spec.args if args_spec.args is not None else []

                if function.__name__ == "score":
                    code = "\n".join(code)
                    file_content = cast(str, code)
                elif len(args) == len(defaults):
                    for i, d in enumerate(defaults):
                        if not is_of_python_basic_type(d):
                            raise UnexpectedType(
                                args[i], "primitive python type", type(d)
                            )

                    args_pattern = ",".join(
                        [rf"\s*{arg}\s*=\s*(.+)\s*" for arg in args]
                    )
                    pattern = rf"^def {function.__name__}\s*\({args_pattern}\)\s*:"
                    code = "\n".join(code)
                    res = re.match(pattern, code)  # type: ignore[call-overload]

                    for i in range(len(defaults) - 1, -1, -1):
                        default = defaults[i]
                        code = (
                            code[: res.start(i + 1)]
                            + default.__repr__()
                            + code[res.end(i + 1) :]
                        )

                    code += f"\n\nscore = {function.__name__}()"

                    file_content = cast(str, code)
                else:
                    raise WMLClientError(
                        "Function passed is not 'score' function nor generator function. Generator function should have no arguments or all arguments with primitive python default values."
                    )

                tmp_uid = "tmp_python_function_code_{}".format(
                    str(uuid.uuid4()).replace("-", "_")
                )
                filename = "{}.py".format(tmp_uid)

                with open(filename, "w") as f:
                    f.write(file_content)

                archive_name = "{}.py.gz".format(tmp_uid)

                with open(filename, "rb") as f_in:
                    with gzip.open(archive_name, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                os.remove(filename)

                content_path = archive_name
            except Exception as e:
                try:
                    os.remove(filename)
                except Exception:
                    pass
                try:
                    os.remove(archive_name)  # type: ignore[arg-type]
                except Exception:
                    pass
                raise WMLClientError("Exception during getting function code.", e)

        return content_path, user_content_file, archive_name
