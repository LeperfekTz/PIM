#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, Literal

from ibm_watsonx_ai.wml_client_error import (
    InvalidMultipleArguments,
    WMLClientError,
    UnexpectedType,
    InvalidValue,
)
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai._wrappers import requests
from ibm_watsonx_ai.helpers import DataConnection

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials
    import pandas


class TextExtractions(WMLResource):
    """Instantiate the Text Extraction service.

    :param credentials: credentials to the Watson Machine Learning instance
    :type credentials: Credentials, optional

    :param project_id: ID of the Watson Studio project, defaults to None
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space, defaults to None
    :type space_id: str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required, defaults to None
    :type api_client: APIClient, optional

    :raises InvalidMultipleArguments: raised if `space_id` and `project_id` or `credentials` and `api_client` are provided simultaneously
    :raises WMLClientError: raised if the CPD version is less than 5.0

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models.extractions import TextExtractions

       extraction = TextExtractions(
            credentials=Credentials(
                                api_key = IAM_API_KEY,
                                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
    ) -> None:

        if credentials is not None:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials)
        elif api_client is not None:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id is not None:
            self._client.set.default_space(space_id)
        elif project_id is not None:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.0:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    def run_job(
        self,
        document_reference: DataConnection,
        results_reference: DataConnection,
        steps: dict | None = None,
        results_format: Literal["json", "markdown"] = "json",
    ) -> dict:
        """Start a request to extract text and metadata from a document.

        :param document_reference: reference to the document in the bucket from which text will be extracted
        :type document_reference: DataConnection

        :param results_reference: reference to the location in the bucket where results will saved
        :type results_reference: DataConnection

        :param steps: steps for the text extraction pipeline, defaults to None
        :type steps: dict | None, optional

        :param results_format: results format for the text extraction, defaults to "json"
        :type results_format: Literal["json", "markdown"], optional

        :return: raw response from the server with the text extraction job details
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.metanames import TextExtractionsMetaNames
            from ibm_watsonx_ai.helpers import DataConnection, S3Location

            document_reference = DataConnection(
                connection_asset_id="<connection_id>",
                location=S3Location(bucket="<bucket_name>", path="path/to/file"),
                )

            results_reference = DataConnection(
                connection_asset_id="<connection_id>",
                location=S3Location(bucket="<bucket_name>", path="path/to/file"),
                )

            response = extraction.run_job(
                document_reference=document_reference,
                results_reference=results_reference,
                steps={
                    TextExtractionsMetaNames.OCR: {"languages_list": ["en", "fr"]},
                    TextExtractionsMetaNames.TABLE_PROCESSING: {"enabled": True},
                    },
                results_format="markdown"
            )

        """
        if not isinstance(document_reference, DataConnection):
            raise UnexpectedType(
                el_name="document_reference",
                expected_type=DataConnection,
                actual_type=type(document_reference),
            )
        elif not isinstance(results_reference, DataConnection):
            raise UnexpectedType(
                el_name="results_reference",
                expected_type=DataConnection,
                actual_type=type(results_reference),
            )

        TextExtractions._validate_type(steps, "steps", dict, False)
        payload: dict = {}

        if self._client.default_project_id is not None:
            payload.update({"project_id": self._client.default_project_id})
        elif self._client.default_space_id is not None:
            payload.update({"space_id": self._client.default_space_id})
        payload.update({"document_reference": document_reference._to_dict()})
        payload.update({"results_reference": results_reference._to_dict()})

        if steps is not None:
            payload.update({"steps": steps})

        if results_format == "json":
            payload.update({"assembly_json": {}})
        elif results_format == "markdown":
            payload.update({"assembly_md": {}})
        else:
            raise ValueError(
                "Incorrect results format provided. Only 'json' and 'markdown' are supported."
            )

        response = self._client.httpx_client.post(
            url=self._client.service_instance._href_definitions.get_text_extraction_href(),
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        return self._handle_response(201, "run_job", response)

    def list_jobs(self, limit: int | None = None) -> pandas.DataFrame:
        """List text extraction jobs. If limit is None, all jobs will be listed.

        :param limit: limit number of fetched records, defaults to None
        :type limit: int | None, optional

        :return: job information of a pandas DataFrame with text extraction
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            extraction.list_jobs()
        """
        import pandas

        columns = ["metadata.id", "metadata.created_at", "entity.results.status"]

        details = self.get_job_details(limit=limit)

        resources = details["resources"]
        data_normalize = pandas.json_normalize(resources)
        extraction_data = data_normalize.reindex(columns=columns)

        df_details: pandas.DataFrame = pandas.DataFrame(
            extraction_data, columns=columns
        )
        df_details.rename(
            columns={
                "metadata.id": "EXTRACTION_ID",
                "metadata.created_at": "CREATED",
                "entity.results.status": "STATUS",
            },
            inplace=True,
        )
        return df_details

    def get_job_details(
        self, extraction_id: str | None = None, limit: int | None = None
    ) -> dict:
        """Return text extraction job details. If `extraction_id` is None, returns the details of all text extraction jobs.

        :param extraction_id: ID of the text extraction job, defaults to None
        :type extraction_id: str | None, optional

        :param limit: limit number of fetched records, defaults to None
        :type limit: int | None, optional

        :return: details of the text extraction job
        :rtype: dict

        **Example:**

        .. code-block:: python

            extraction.get_job_details(extraction_id="<extraction_id>")

        """
        TextExtractions._validate_type(extraction_id, "extraction_id", str, False)
        if extraction_id is not None:
            response = self._client.httpx_client.get(
                url=self._client.service_instance._href_definitions.get_text_extraction_href()
                + f"/{extraction_id}",
                params=self._client._params(skip_userfs=True),
                headers=self._client._get_headers(),
            )
        else:
            _params: dict | None = None
            if limit is not None:
                if limit < 1 or limit > 200:
                    raise InvalidValue(
                        value_name="limit",
                        reason=f"The given value {limit} is not in the range <1, 200>",
                    )
                else:
                    _params = {"limit": limit}

            # TODO: pagination is not yet implemented
            response = self._client.httpx_client.get(
                url=self._client.service_instance._href_definitions.get_text_extraction_href(),
                params=(self._client._params(skip_userfs=True) | (_params or {})),
                headers=self._client._get_headers(),
            )
        return self._handle_response(200, "get_job_details", response)

    def delete_job(self, extraction_id: str) -> Literal["SUCCESS"]:
        """Delete a text extraction job.

        :return: return "SUCCESS" if the deletion succeeds
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction.delete_job(extraction_id="<extraction_id>")

        """
        TextExtractions._validate_type(extraction_id, "extraction_id", str, True)

        params = self._client._params(skip_userfs=True)
        params.update({"hard_delete": True})

        response = self._client.httpx_client.delete(
            url=self._client.service_instance._href_definitions.get_text_extraction_href()
            + f"/{extraction_id}",
            params=params,
            headers=self._client._get_headers(),
        )
        return self._handle_response(204, "delete_job", response)  # type: ignore[return-value]

    def cancel_job(self, extraction_id: str) -> Literal["SUCCESS"]:
        """Cancel a text extraction job.

        :return: return "SUCCESS" if the cancellation succeeds
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction.cancel_job(extraction_id="<extraction_id>")

        """
        TextExtractions._validate_type(extraction_id, "extraction_id", str, True)

        response = self._client.httpx_client.delete(
            url=self._client.service_instance._href_definitions.get_text_extraction_href()
            + f"/{extraction_id}",
            params=self._client._params(skip_userfs=True),
            headers=self._client._get_headers(),
        )
        return self._handle_response(204, "cancel_job", response)  # type: ignore[return-value]

    def get_results_reference(self, extraction_id: str) -> DataConnection:
        """Get a `DataConnection` instance that is a reference to the results stored on COS.

        :param extraction_id: ID of text extraction job
        :type extraction_id: str

        :return: location of the Data Connection to text extraction job results
        :rtype: DataConnection

        **Example:**

        .. code-block:: python

            results_reference = extraction.get_results_reference(extraction_id="<extraction_id>")

        """
        TextExtractions._validate_type(extraction_id, "extraction_id", str, True)

        job_details = self.get_job_details(extraction_id=extraction_id)

        results_reference = job_details.get("entity", {}).get("results_reference")
        data_conn = DataConnection._from_dict(results_reference)
        data_conn.set_client(self._client)
        return data_conn

    @staticmethod
    def get_id(extraction_details: dict) -> str:
        """Get the unique ID of a stored extraction request.

        :param extraction_details: metadata of the stored extraction
        :type extraction_details: dict

        :return: unique ID of the stored extraction request
        :rtype: str

        **Example:**

        .. code-block:: python

            extraction_details = extraction.get_job_details(extraction_id)
            extraction_id = extraction.get_id(extraction_details)

        """
        TextExtractions._validate_type(
            extraction_details, "extraction_details", dict, True
        )

        return WMLResource._get_required_element_from_dict(
            extraction_details, "extraction_details", ["metadata", "id"]
        )
