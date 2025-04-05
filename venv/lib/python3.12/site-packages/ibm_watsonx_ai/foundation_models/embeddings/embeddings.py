#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from __future__ import annotations
import os
from typing import TypeAlias, TYPE_CHECKING, Any
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from enum import Enum
from warnings import warn
import requests as _requests
import httpx

from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    InvalidMultipleArguments,
    ParamOutOfRange,
)
from .base_embeddings import BaseEmbeddings
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai._wrappers.requests import (
    _get_httpx_client,
)
import ibm_watsonx_ai._wrappers.requests as requests

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials

# Type Aliasses
ParamsType: TypeAlias = dict[str, str | dict[str, str]]
PayloadType: TypeAlias = dict[str, str | list[str] | ParamsType]


__all__ = ["Embeddings"]

# Defaults
MAX_INPUTS_LENGTH = 1000
DEFAULT_CONCURRENCY_LIMIT = 5

# Increase read and write timeout for embeddings generation
EMBEDDINGS_HTTPX_TIMEOUT = httpx.Timeout(
    read=30 * 60, write=30 * 60, connect=10, pool=30 * 60
)

# Do not change below, required by service
_RETRY_STATUS_CODES = [429, 500, 503, 504, 520]


class Embeddings(BaseEmbeddings, WMLResource):
    """Instantiate the embeddings service.

    :param model_id: the type of model to use
    :type model_id: str, optional

    :param params: parameters to use during generate requests, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames
    :type params: dict, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param verify: You can pass one of following as verify:

        * the path to a CA_BUNDLE file
        * the path of a directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param persistent_connection: defines whether to keep a persistent connection when evaluating the `generate`, 'embed_query', and 'embed_documents` methods with one prompt
                                  or batch of prompts that meet the length limit. For more details, see `Generate embeddings <https://cloud.ibm.com/apidocs/watsonx-ai#text-embeddings>`_.
                                  To close the connection, run `embeddings.close_persistent_connection()`, defaults to True. Added in 1.1.2.
    :type persistent_connection: bool, optional

    :param batch_size: Number of elements to be embedded sending in one call, defaults to 1000
    :type batch_size: int, optional

    :param concurrency_limit: number of requests to be sent in parallel when generating embedding vectors, max is 10, defaults to 5
    :type concurrency_limit: int, optional

    :param max_retries: number of retries performed when request was not successful and status code is in retry_status_codes, defaults to 10
    :type max_retries: int, optional

    :param delay_time: delay time to retry request, factor in exponential backoff formula: wx_delay_time * pow(2.0, attempt), defaults to 0.5s
    :type delay_time: float, optional

    :param retry_status_codes: list of status codes which will be considered for retry mechanism, defaults to [429, 503, 504, 520]
    :type retry_status_codes: list[int], optional


    .. note::
        When the ``credentials`` parameter is passed, one of these parameters is required: [``project_id``, ``space_id``].

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings
        from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
        from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

       embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
            EmbedParams.RETURN_OPTIONS: {
            'input_text': True
            }
        }

        embedding = Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            params=embed_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    def __init__(
        self,
        *,
        model_id: str,
        params: ParamsType | None = None,
        credentials: Credentials | dict[str, str] | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
        verify: bool | str | None = None,
        persistent_connection: bool = True,
        batch_size: int = MAX_INPUTS_LENGTH,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        max_retries: int | None = None,
        delay_time: float | None = None,
        retry_status_codes: list[int] | None = None,
    ) -> None:
        if isinstance(model_id, Enum):
            self.model_id = model_id.value
        else:
            self.model_id = model_id

        self.params = params

        if concurrency_limit is not DEFAULT_CONCURRENCY_LIMIT and (
            concurrency_limit > 10 or concurrency_limit < 1
        ):
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )
        self.concurrency_limit = concurrency_limit

        Embeddings._validate_type(params, "params", dict, False)
        Embeddings._validate_type(batch_size, "batch_size", int, False)

        if batch_size > MAX_INPUTS_LENGTH or batch_size < 1:
            raise ParamOutOfRange(
                param_name="batch_size", value=batch_size, min=1, max=MAX_INPUTS_LENGTH
            )
        else:
            self.batch_size = batch_size

        if credentials:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials, verify=verify)
        elif api_client:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id:
            self._client.set.default_space(space_id)
        elif project_id:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.0:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        self._persistent_connection = persistent_connection

        WMLResource.__init__(self, __name__, self._client)

        self._transport_params = requests._httpx_transport_params(self._client)

        if self._persistent_connection:
            self._http_client = self._client.httpx_client
        else:
            self._http_client = requests  # type: ignore[assignment]
            persistent_connection_warn = (
                "`persistent_connection` is deprecated and will be removed in future. "
            )
            warn(persistent_connection_warn, category=DeprecationWarning)

        # Set initially 8 requests per second as it is default for prod instances
        # if header "x-requests-limit-rate" is different capacity will be updated
        self.rate_limiter = requests.TokenBucket(rate=8, capacity=8)
        self.retry_status_codes = retry_status_codes
        self.max_retries = max_retries
        self.delay_time = delay_time

    def _generate_raw_response(
        self,
        generate_url: str,
        inputs: list[str],
        params: ParamsType | None = None,
        _http_client: httpx.Client | None = None,
    ) -> httpx.Response | _requests.Response:
        """Send request with post and return service response."""

        payload = self._prepare_payload(inputs, params)

        post_params: dict[str, Any] = dict(
            url=generate_url,
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )

        return self._post(_http_client, **post_params)

    def generate(
        self,
        inputs: list[str],
        params: ParamsType | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> dict:
        """Generate embeddings vectors for the given input with the given
        parameters. Returns a REST API response.

        :param inputs: list of texts for which embedding vectors will be generated
        :type inputs: list[str]
        :param params: MetaProps for the embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :param concurrency_limit: number of requests to be sent in parallel, max is 10, defaults to 5
        :type concurrency_limit: int, optional
        :return: scoring results containing generated embeddings vectors
        :rtype: dict
        """
        self._validate_type(inputs, "inputs", list, True)
        generate_url = (
            self._client.service_instance._href_definitions.get_fm_embeddings_href()
        )
        if concurrency_limit is not DEFAULT_CONCURRENCY_LIMIT and (
            concurrency_limit > 10 or concurrency_limit < 1
        ):
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )

        # '(concurrency_limit is DEFAULT_CONCURRENCY_LIMIT) == True' => NO SET
        if concurrency_limit is DEFAULT_CONCURRENCY_LIMIT:
            concurrency_limit = self.concurrency_limit
        # For batch of prompts use keep-alive connection even if persistent_connection=False
        http_client = (
            self._client.httpx_client
            if not self._persistent_connection
            else self._http_client
        )

        if len(inputs) > self.batch_size:
            inputs_split = [
                inputs[i : i + self.batch_size]
                for i in range(0, len(inputs), self.batch_size)
            ]

            def make_request(_inputs: list[str]) -> dict:
                self.rate_limiter.acquire()
                response = self._generate_raw_response(
                    generate_url=generate_url,
                    inputs=_inputs,
                    params=params,
                    _http_client=http_client,
                )
                rate_limit = int(response.headers.get("x-requests-limit-rate", 0))
                if rate_limit and rate_limit != self.rate_limiter.capacity:
                    self.rate_limiter.capacity = rate_limit

                rate_limit_remaining = int(
                    response.headers.get(
                        "x-requests-limit-remaining", self.rate_limiter.capacity
                    )
                )
                self.rate_limiter.adjust_tokens(rate_limit_remaining)
                return self._handle_response(
                    200,
                    "generate",
                    response,
                    _field_to_hide="embedding",
                )

            if (inputs_length := len(inputs_split)) <= concurrency_limit:
                with ThreadPoolExecutor(max_workers=inputs_length) as executor:
                    generated_responses = list(executor.map(make_request, inputs_split))
            else:
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    generated_responses = list(executor.map(make_request, inputs_split))

            def reduce_response(left: dict, right: dict) -> dict:
                import copy

                left_copy = copy.deepcopy(left)
                left_copy["results"].extend(right["results"])
                left_copy["input_token_count"] += right["input_token_count"]
                return left_copy

            return reduce(
                reduce_response, generated_responses[1:], generated_responses[0]
            )

        else:
            results = self._generate(generate_url, inputs, params)
        return results

    def embed_documents(
        self,
        texts: list[str],
        params: ParamsType | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
    ) -> list[list[float]]:
        """Returns list of embedding vectors for provided texts.

        :param texts: list of texts for which embedding vectors will be generated
        :type texts: list[str]
        :param params: MetaProps for the embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :param concurrency_limit: number of requests to be sent in parallel, max is 10, defaults to 5
        :type concurrency_limit: int, optional

        :return: list of embedding vectors
        :rtype: list[list[float]]

        **Example:**

        .. code-block:: python

            q = [
                "What is a Generative AI?",
                "Generative AI refers to a type of artificial intelligence that can original content."
                ]

            embedding_vectors = embedding.embed_documents(texts=q)
            print(embedding_vectors)
        """
        return [
            vector.get("embedding")
            for vector in self.generate(
                inputs=texts, params=params, concurrency_limit=concurrency_limit
            ).get("results", [{}])
        ]

    def embed_query(self, text: str, params: ParamsType | None = None) -> list[float]:
        """Returns an embedding vector for a provided text.

        :param text: text for which embedding vector will be generated
        :type text: str
        :param params: MetaProps for the embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :return: embedding vector
        :rtype: list[float]

        **Example:**

        .. code-block:: python

            q = "What is a Generative AI?"
            embedding_vector = embedding.embed_query(text=q)
            print(embedding_vector)
        """
        return (
            self.generate(inputs=[text], params=params)
            .get("results", [{}])[0]
            .get("embedding")
        )

    def _prepare_payload(
        self, inputs: list[str], params: ParamsType | None = None
    ) -> PayloadType:
        """Prepare payload based in provided inputs and params."""
        payload: PayloadType = {"model_id": self.model_id, "inputs": inputs}

        if params is not None:
            payload["parameters"] = params
        elif self.params is not None:
            payload["parameters"] = self.params

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        return payload

    def _generate(
        self,
        generate_url: str,
        inputs: list[str],
        params: ParamsType | None = None,
        _http_client: requests.HTTPXClient | None = None,
    ) -> dict:
        """Send request with post and return service response."""
        http_client = _http_client or self._http_client

        response_scoring = self._generate_raw_response(
            generate_url=generate_url,
            inputs=inputs,
            params=params,
            _http_client=http_client,
        )

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="embedding",
        )

    def to_dict(self) -> dict:
        data = super().to_dict()
        embeddings_args = {
            "model_id": self.model_id,
            "params": self.params,
            "credentials": self._client.credentials.to_dict(),
            "project_id": self._client.default_project_id,
            "space_id": self._client.default_space_id,
            "verify": os.environ.get("WML_CLIENT_VERIFY_REQUESTS"),
        }
        if self.batch_size != MAX_INPUTS_LENGTH:
            embeddings_args |= {"batch_size": self.batch_size}

        if self.concurrency_limit is not DEFAULT_CONCURRENCY_LIMIT:
            embeddings_args |= {"concurrency_limit": self.concurrency_limit}

        data.update(embeddings_args)

        return data

    def close_persistent_connection(self) -> None:
        """
        Only applicable if persistent_connection was set to True in Embeddings initialization.
        Calling this method closes the current `httpx.Client` and recreates a new `httpx.Client` with default values:
        timeout: httpx.Timeout(read=30 * 60, write=30 * 60, connect=10, pool=30 * 60)
        limit: httpx.Limits(max_connections=10, max_keepalive_connections=10, keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY)
        """
        if self._persistent_connection is not None and isinstance(
            self._http_client, httpx.Client
        ):
            self._http_client.close()
            self._client.httpx_client = _get_httpx_client(
                transport_params=self._transport_params,
                timeout=EMBEDDINGS_HTTPX_TIMEOUT,
            )
            self._http_client = self._client.httpx_client

    @requests._with_retry(retry_status_codes=_RETRY_STATUS_CODES)
    def _post(
        self, http_client: Any, *args: Any, **kwargs: Any
    ) -> httpx.Response | _requests.Response:
        return http_client.post(*args, **kwargs)
