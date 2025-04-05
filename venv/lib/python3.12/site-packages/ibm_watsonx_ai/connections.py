#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import json

from typing import TYPE_CHECKING
from urllib.parse import unquote, quote
from warnings import warn

import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import ConnectionMetaNames
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    UnsupportedOperation,
    ApiRequestFailure,
)
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from collections.abc import Iterable
    import pandas as pd


class Connections(WMLResource):
    """Store and manage connections."""

    ConfigurationMetaNames = ConnectionMetaNames()
    """MetaNames for connection creation."""

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)

    def _get_required_element_from_response(self, response_data: dict) -> dict:

        WMLResource._validate_type(response_data, "connection_response", dict)

        new_el = {
            "metadata": {
                "id": response_data["metadata"]["asset_id"],
                "asset_type": response_data["metadata"]["asset_type"],
                "create_time": (
                    response_data["metadata"]["create_time"]
                    if "create_time" in response_data["metadata"]
                    else response_data["metadata"]["created_at"]
                ),
                "last_access_time": response_data["metadata"]["usage"].get(
                    "last_access_time"
                ),
            },
            "entity": {
                "datasource_type": (
                    response_data["entity"]["datasource_type"]
                    if "datasource_type" in response_data["entity"]
                    else response_data["entity"]["connection"]["datasource_type"]
                ),
                "name": (
                    response_data["entity"]["name"]
                    if "name" in response_data["entity"]
                    else response_data["metadata"]["name"]
                ),
            },
        }

        for el in ["description", "origin_country", "owner_id", "properties"]:
            if el in response_data["entity"]:
                new_el["entity"][el] = response_data["entity"].get(el)

        if self._client.default_space_id is not None:
            new_el["metadata"]["space_id"] = response_data["metadata"]["space_id"]

        elif self._client.default_project_id is not None:
            new_el["metadata"]["project_id"] = response_data["metadata"]["project_id"]

            if "href" in response_data["metadata"]:
                href_without_host = response_data["href"].split(".com")[-1]
                new_el["metadata"].update({"href": href_without_host})

        return new_el

    def get_details(self, connection_id: str | None = None) -> dict:
        """Get connection details for the given unique connection ID.
        If no connection_id is passed, details for all connections are returned.

        :param connection_id: unique ID of the connection
        :type connection_id: str

        :return: metadata of the stored connection
        :rtype: dict

        **Example:**

        .. code-block:: python

            connection_details = client.connections.get_details(connection_id)
            connection_details = client.connections.get_details()

        """
        self._client._check_if_either_is_set()
        Connections._validate_type(connection_id, "connection_id", str, False)

        header_param = self._client._get_headers()

        if self._client._iam_id:
            header_param["IBM-WDP-Impersonate"] = str(
                {"iam_id": str(self._client._iam_id)}
            )

        if connection_id:
            with self.requests_retry_session() as sess:
                response = sess.get(
                    self._client.service_instance._href_definitions.get_connection_by_id_href(
                        connection_id
                    ),
                    params=self._client._params(),
                    headers=header_param,
                )

                return self._get_required_element_from_response(
                    self._handle_response(
                        200,
                        "get connection details",
                        response,
                        _silent_response_logging=True,
                    )
                )
        else:
            with self.requests_retry_session() as sess:
                response = sess.post(
                    self._client.service_instance._href_definitions.get_asset_search_href(
                        "connection"
                    ),
                    json={"query": "*:*", "include": "entity"},
                    params=self._client._params(),
                    headers=header_param,
                )

                return {
                    "resources": [
                        self._get_required_element_from_response(r)
                        for r in self._handle_response(
                            200,
                            "get connection details",
                            response,
                            _silent_response_logging=True,
                        )["results"]
                    ]
                }

    def create(self, meta_props: dict) -> dict:
        """Create a connection. Examples of PROPERTIES field input:

        1. MySQL

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "3306",
                    "host": "host url",
                    "ssl": "false",
                    "username": "username"
                }

        2. Google BigQuery

            a. Method 1: Using service account json. The generated service account json can be provided as input as-is. Provide actual values in json. The example below is only indicative to show the fields. For information on how to generate the service account json, refer to Google BigQuery documentation.

                .. code-block:: python

                    client.connections.ConfigurationMetaNames.PROPERTIES: {
                        "type": "service_account",
                        "project_id": "project_id",
                        "private_key_id": "private_key_id",
                        "private_key": "private key contents",
                        "client_email": "client_email",
                        "client_id": "client_id",
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": "client_x509_cert_url"
                    }

            b. Method 2: Using OAuth Method. For information on how to generate a OAuth token, refer to Google BigQuery documentation.

                .. code-block:: python

                    client.connections.ConfigurationMetaNames.PROPERTIES: {
                        "access_token": "access token generated for big query",
                        "refresh_token": "refresh token",
                        "project_id": "project_id",
                        "client_secret": "This is your gmail account password",
                        "client_id": "client_id"
                    }

        3. MS SQL

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "1433",
                    "host": "host",
                    "username": "username"
                }

        4. Teradata

            .. code-block:: python

                client.connections.ConfigurationMetaNames.PROPERTIES: {
                    "database": "database",
                    "password": "password",
                    "port": "1433",
                    "host": "host",
                    "username": "username"
                }

        :param meta_props: metadata of the connection configuration. To see available meta names, use:

            .. code-block:: python

                client.connections.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: metadata of the stored connection
        :rtype: dict

        **Example:**

        .. code-block:: python

            sqlserver_data_source_type_id = client.connections.get_datasource_type_id_by_name('sqlserver')
            connections_details = client.connections.create({
                client.connections.ConfigurationMetaNames.NAME: "sqlserver connection",
                client.connections.ConfigurationMetaNames.DESCRIPTION: "connection description",
                client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: sqlserver_data_source_type_id,
                client.connections.ConfigurationMetaNames.PROPERTIES: { "database": "database",
                                                                        "password": "password",
                                                                        "port": "1433",
                                                                        "host": "host",
                                                                        "username": "username"}
            })

        """
        connection_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, with_validation=True, client=self._client
        )

        big_query_data_source_type_id = self.get_datasource_type_id_by_name("bigquery")

        # Either service acct json credentials can be given or oauth json can be given
        # If service acct json, then we need to create a newline json with "credentials" key
        if connection_meta["datasource_type"] == big_query_data_source_type_id:
            if "private_key" in connection_meta["properties"]:
                result = json.dumps(
                    connection_meta["properties"], separators=(",\n", ":")
                )
                newmap = {"credentials": result}
                connection_meta["properties"] = newmap

        connection_meta.update({"origin_country": "US"})
        # Step1  : Create an asset
        print(Messages.get_message(message_id="creating_connections"))

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_connections_href(),
            headers=self._client._get_headers(),
            json=connection_meta,
            params=self._client._params(),
        )
        try:
            connection_details = self._handle_response(
                201,
                "creating new connection",
                creation_response,
                _silent_response_logging=True,
            )
        except ApiRequestFailure as e:
            if creation_response.status_code == 400:
                datasource_type_id = connection_meta["datasource_type"]
                datasource_type_details = self.get_datasource_type_details_by_id(
                    datasource_type_id, connection_properties=True
                )
                connection_properties = datasource_type_details["entity"]["properties"][
                    "connection"
                ]
                properties_names = [
                    conn_property["name"] for conn_property in connection_properties
                ]
                raise ApiRequestFailure(
                    error_msg="Failure during {}.".format("creating new connection"),
                    response=creation_response,
                    reason=f"Incorrect connection properties for datasource type id: {datasource_type_id}. "
                    f"The following properties are correct: {properties_names}.",
                ) from e
            raise e

        if creation_response.status_code == 201:
            print(Messages.get_message(message_id="success"))
            return self._get_required_element_from_response(connection_details)
        else:
            raise WMLClientError(
                Messages.get_message(message_id="failed_while_creating_connections")
            )

    def delete(self, connection_id: str) -> str:
        """Delete a stored connection.

        :param connection_id: unique ID of the connection to be deleted
        :type connection_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.connections.delete(connection_id)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Connections._validate_type(connection_id, "connection_id", str, True)

        connection_endpoint = (
            self._client.service_instance._href_definitions.get_connection_by_id_href(
                connection_id
            )
        )
        response_delete = requests.delete(
            connection_endpoint,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(
            204,
            "connection deletion",
            response_delete,
            False,
            _silent_response_logging=True,
        )

    @staticmethod
    def get_uid(connection_details: dict) -> str:
        """Get the unique ID of a stored connection.

        *Deprecated:* Use ``Connections.get_id(details)`` instead.

        :param connection_details: metadata of the stored connection
        :type connection_details: dict

        :return: unique ID of the stored connection
        :rtype: str

        **Example:**

        .. code-block:: python

            connection_uid = client.connection.get_uid(connection_details)

        """
        get_uid_method_deprecated_warning = (
            "The method `Connections.get_uid` is deprecated, "
            "please use Connections.get_id() instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)

        return Connections.get_id(connection_details)

    @staticmethod
    def get_id(connection_details: dict) -> str:
        """Get ID of a stored connection.

        :param connection_details: metadata of the stored connection
        :type connection_details: dict

        :return: unique ID of the stored connection
        :rtype: str

        **Example:**

        .. code-block:: python

            connection_id = client.connection.get_id(connection_details)

        """
        Connections._validate_type(
            connection_details, "connection_details", object, True
        )

        return WMLResource._get_required_element_from_dict(
            connection_details, "connection_details", ["metadata", "id"]
        )

    def _get_datasource_details(self) -> list:
        datasource_details = []

        def get_res(url):
            response = requests.get(
                url,
                headers=self._client._get_headers(),
            )

            res = self._handle_response(
                200, "list datasource types", response, _silent_response_logging=True
            )["resources"]

            return res, response.json().get("next", {}).get("href")

        res, url = get_res(
            self._client.service_instance._href_definitions.get_connection_data_types_href()
        )
        datasource_details.extend(res)

        while url is not None:
            res, url = get_res(url)
            datasource_details.extend(res)

        return datasource_details

    def list_datasource_types(self) -> pd.DataFrame:
        """Print stored datasource types assets in a table format.

        :return: pandas.DataFrame with listed datasource types
        :rtype: pandas.DataFrame

        **Example:**
        https://test.cloud.ibm.com/apidocs/watsonx-ai#trainings-list

        .. code-block:: python

            client.connections.list_datasource_types()

        """
        datasource_details = self._get_datasource_details()

        space_values = [
            (
                m["entity"]["name"],
                m["metadata"]["asset_id"],
                m["entity"]["type"],
                m["entity"]["status"],
            )
            for m in datasource_details
        ]

        table = self._list(
            space_values, ["NAME", "DATASOURCE_ID", "TYPE", "STATUS"], None
        )
        return table

    def list(self) -> pd.DataFrame:
        """Return pd.DataFrame table with all stored connections in a table format.

        :return: pandas.DataFrame with listed connections
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.connections.list()

        """
        datasource_details = self.get_details()
        space_values = [
            (
                m["entity"]["name"],
                m["metadata"]["id"],
                m["metadata"]["create_time"],
                m["entity"]["datasource_type"],
            )
            for m in datasource_details["resources"]
        ]

        list_table = self._list(
            space_values, ["NAME", "ID", "CREATED", "DATASOURCE_TYPE_ID"], None
        )
        return list_table

    def list_uploaded_db_drivers(
        self,
    ) -> pd.DataFrame:
        """Return pd.DataFrame table with uploaded db driver jars in table a format. Supported for IBM Cloud Pak® for Data only.

        :return: pandas.DataFrame with listed uploaded db drivers
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.connections.list_uploaded_db_drivers()

        """
        if not self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError("Not supported on this environment.")

        try:
            if not self.get_uploaded_db_drivers():
                raise Exception("List empty for new api")
            table = self._list_uploaded_db_drivers_new_api()
        except:
            response = requests.get(
                self._client.service_instance._href_definitions.get_wsd_model_attachment_href()
                + "dbdrivers",
                headers=self._client._get_headers(no_content_type=True),
                params=self._client._params(),
            )
            jars = [[el["path"].split("/")[-1]] for el in response.json()["resources"]]

            table = self._list(jars, ["NAME"], None)
        return table

    def get_uploaded_db_drivers(self) -> dict[str, str]:
        """
        Get uploaded db driver jar names and paths.
        Supported for IBM Cloud Pak® for Data, version 4.6.1 and up.

        **Output**

        .. important::
             Returns dictionary containing name and path for connection files.\n
             **return type**: Dict[Str, Str]\n

        **Example:**

         >>> result = client.connections.get_uploaded_db_drivers()

        """
        if not self._client.ICP_PLATFORM_SPACES or self._client.CPD_version < 4.6:
            raise UnsupportedOperation(
                "Get uploaded db driver is supported only for IBM Cloud Pak® for Data, version 4.6.1 and later."
            )

        response = requests.get(
            self._client.service_instance._href_definitions.get_connections_files_href(),
            headers=self._client._get_headers(no_content_type=True),
        )
        result = self._handle_response(
            200, "get uploaded db drivers", response, _silent_response_logging=True
        )["resources"]
        return dict([(el["fileName"], el["url"]) for el in result])

    def _list_uploaded_db_drivers_new_api(self) -> pd.DataFrame:
        """List uploaded db driver jars. Supported for IBM Cloud Pak® for Data only.

        .. important::
            This method prints the uploaded db driver jar names and returns as pd.DataFrame.

        :return: pandas.DataFrame with listed uploaded db drivers
        :rtype: pandas.DataFrame

        **Example:**

        .. code-clock:: python

            client.connections._list_uploaded_db_drivers_new_api()

        """
        jars = [[name] for name in self.get_uploaded_db_drivers()]
        return self._list(jars, ["NAME"], None)

    def get_datasource_type_details_by_id(
        self, datasource_type_id: str, connection_properties: bool = False
    ) -> dict:
        """Get datasource type details for the given datasource type ID.

        :param datasource_type_id: ID of the datasource type
        :type datasource_type_id: str
        :param connection_properties: if True, the connection properties are included in the returned details. defaults to False
        :type connection_properties: bool

        :return: Datasource type details
        :rtype: dict

        **Example:**

        .. code-block:: python

            client.connections.get_datasource_type_details_by_id(datasource_type_id)

        """

        Connections._validate_type(datasource_type_id, "datasource_type", str, False)
        header_param = self._client._get_headers()
        params = {"connection_properties": connection_properties}

        response = requests.get(
            self._client.service_instance._href_definitions.get_connection_data_type_by_id_href(
                datasource_type_id
            ),
            params=params,
            headers=header_param,
        )

        return self._handle_response(
            200, "get datasource details", response, _silent_response_logging=True
        )

    def get_datasource_type_uid_by_name(self, name: str) -> str:
        """Get a stored datasource type ID for the given datasource type name.

        *Deprecated:* Use ``Connections.get_datasource_type_id_by_name(name)`` instead.

        :param name: name of datasource type
        :type name: str

        :return: ID of datasource type
        :rtype: str

        **Example:**

        .. code-block:: python

            client.connections.get_datasource_type_uid_by_name('cloudobjectstorage')

        """
        get_datasource_type_uid_deprecation_warning = (
            "This method is deprecated, "
            "please use get_datasource_type_id_by_name(name)"
        )
        warn(get_datasource_type_uid_deprecation_warning, category=DeprecationWarning)

        return self.get_datasource_type_id_by_name(name=name)

    def get_datasource_type_id_by_name(self, name: str) -> str:
        """Get a stored datasource type ID for the given datasource type name.

        :param name: name of datasource type
        :type name: str

        :return: ID of datasource type
        :rtype: str

        **Example:**

        .. code-block:: python

            client.connections.get_datasource_type_id_by_name('cloudobjectstorage')

        """
        Connections._validate_type(name, "name", str, True)

        datasource_details = self._get_datasource_details()

        datasource_id = None
        for i, ds_resource in enumerate(datasource_details):
            if ds_resource["entity"]["name"] == name:
                datasource_id = ds_resource["metadata"]["asset_id"]
        return datasource_id

    def get_write_mode_by_datasource_type(self, datasource_type: str) -> str:
        Connections._validate_type(datasource_type, "datasource_type", str, False)
        write_mode = "write_raw"  # default

        if not datasource_type:
            return write_mode

        header_param = self._client._get_headers()
        params = {"interaction_properties": "true"}

        with self.requests_retry_session() as sess:
            response = sess.get(
                self._client.service_instance._href_definitions.get_connection_data_type_by_id_href(
                    datasource_type
                ),
                params=params,
                headers=header_param,
            )

            datasource_details = self._handle_response(
                200, "get datasource details", response, _silent_response_logging=True
            )

        for val in datasource_details["entity"]["properties"]["target"][-1]["values"]:
            if val["value"] == "write_raw" or val["value"] == "insert":
                return val["value"]

        return write_mode

    def upload_db_driver(self, path: str) -> None:
        """Upload db driver jar. Supported for IBM Cloud Pak® for Data only, version 4.0.4 and up.

        :param path: path to the db driver jar file
        :type path: str

        **Example:**

        .. code-block:: python

            client.connections.upload_db_driver('example/path/db2jcc4.jar')

        """
        if not self._client.ICP_PLATFORM_SPACES:
            raise UnsupportedOperation(
                "Upload db driver is supported only for IBM Cloud Pak® for Data, version 4.0.4 and later."
            )

        try:
            self._upload_db_driver_new_api(path)
        except:
            driver_file_name = path.split("/")[-1]

            with open(path, "rb") as fdata:
                content_upload_url = (
                    self._client.service_instance._href_definitions.get_wsd_model_attachment_href()
                    + "dbdrivers/"
                    + quote(driver_file_name, safe="")
                )
                response = requests.put(
                    content_upload_url,
                    files={
                        "file": (
                            "native",
                            fdata,
                            "application/octet-stream",
                            {"Expires": "0"},
                        )
                    },
                    headers=self._client._get_headers(no_content_type=True),
                    params=self._client._params(),
                )

                self._client.repository._handle_response(
                    201,
                    "uploading db driver jar",
                    response,
                    _silent_response_logging=True,
                )

    def _upload_db_driver_new_api(self, path: str) -> None:
        """Upload a db driver jar. Supported for IBM Cloud Pak® for Data only, version 4.6.1 and later.

        :param path: path to the db driver jar
        :type path: str

        **Example:**

        .. code-block:: python

            client.connections._upload_db_driver_new_api('example/path/db2jcc4.jar')

        """
        if not self._client.ICP_PLATFORM_SPACES:
            raise UnsupportedOperation(
                "Upload db driver jar is supported only for IBM Cloud Pak® for Data only, version 4.6.1 and later."
            )

        driver_file_name = path.split("/")[-1]

        with open(path, "rb") as fdata:
            content_upload_url = self._client.service_instance._href_definitions.get_connections_file_href(
                quote(driver_file_name, safe="")
            )
            response = requests.post(
                content_upload_url,
                data=fdata,
                headers=self._client._get_headers(
                    content_type="application/octet-stream"
                ),
            )

            if response.status_code == 403:
                raise WMLClientError(
                    "User is missing [configure_platform] permission to upload new jar file."
                )

            self._client.repository._handle_response(
                200,
                "uploading db driver jar",
                response,
                json_response=False,
                _silent_response_logging=True,
            )

    def get_db_driver_url(self, name: str) -> str:
        # """
        # Get a signed db driver jar URL to be used during JDBC generic connection creation. The jar name passed as an argument needs to be uploaded into the system first.
        # Supported for IBM Cloud Pak for Data only, version 4.6.1 and above.
        #
        # :param name:  db driver jar name
        # :type name: str
        #
        # **Example:**
        #
        # .. code-block:: python
        #
        #     client.connections.get_db_driver_url('db2jcc4.jar')
        #
        # """
        if not self._client.ICP_PLATFORM_SPACES:
            raise UnsupportedOperation(
                "Get db driver jar is supported only for IBM Cloud Pak® for Data only, version 4.6.1 and later."
            )

        try:
            return self.get_uploaded_db_drivers()[name]
        except WMLClientError as e:
            raise e
        except:
            raise WMLClientError(f"Driver jar with name {name} not found.")

    def sign_db_driver_url(self, jar_name: str) -> str:
        """Get a signed db driver jar URL to be used during JDBC generic connection creation.
        The jar name passed as argument needs to be uploaded into the system first.
        Supported for IBM Cloud Pak® for Data only, version 4.0.4 and later.

        :param jar_name: name of db driver jar
        :type jar_name: str

        :return: URL of signed db driver
        :rtype: str

        **Example:**

        .. code-block:: python

            jar_uri = client.connections.sign_db_driver_url('db2jcc4.jar')

        """
        try:
            res = self.get_db_driver_url(jar_name)
            return res
        except:
            if not self._client.ICP_PLATFORM_SPACES:
                raise UnsupportedOperation(
                    "Get signed db driver jar url db driver is supported only  IBM Cloud Pak® for Data only, version 4.0.4 and later."
                )

            signed_url = (
                self._client.service_instance._href_definitions.get_wsd_model_attachment_href()
                + quote("dbdrivers/" + jar_name, safe="")
                + "/signed"
            )
            params = self._client._params()

            params["expires_in"] = 5000

            response = requests.post(
                signed_url,
                headers=self._client._get_headers(no_content_type=True),
                params=params,
            )

            self._client.repository._handle_response(
                201,
                "signing db driver url",
                response,
                json_response=False,
                _silent_response_logging=True,
            )

            return unquote(response.headers["Location"])

    @staticmethod
    def requests_retry_session(
        retries: int = 3,
        backoff_factor: float = 0.3,
        status_forcelist: Iterable[int] = (500, 502, 503, 504, 521, 524),
        session: requests.Session | None = None,
    ) -> requests.Session:
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry

        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
