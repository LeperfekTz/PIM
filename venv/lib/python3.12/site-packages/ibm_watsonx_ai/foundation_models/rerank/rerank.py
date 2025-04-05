#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
from typing import TYPE_CHECKING, TypedDict
from copy import deepcopy

from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.foundation_models.schema import (
    RerankParameters,
    BaseSchema,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials


class TextDict(TypedDict):
    text: str


class Rerank(WMLResource):
    """
    Rerank texts based on some queries.

    :param model_id: type of model to use
    :type model_id: str

    :param params: parameters to use during request generation
    :type params: dict, RerankParameters, optional

    :param credentials: credentials for the Watson Machine Learning instance
    :type credentials: Credentials or dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:

        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Rerank

        generate_params = {
            "truncate_input_tokens": 10
        }

        wx_ranker = Rerank(
            model_id="<RERANK MODEL>",
            params=generate_params,
            credentials=Credentials(
                api_key = IAM_API_KEY,
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id=project_id
        )

    """

    def __init__(
        self,
        model_id: str,
        params: dict | RerankParameters | None = None,
        credentials: Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
    ) -> None:

        self.model_id = model_id
        Rerank._validate_type(model_id, "model_id", str, True)

        self.params = params
        Rerank._validate_type(params, "params", [dict, RerankParameters], False, True)

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

        WMLResource.__init__(self, __name__, self._client)

    def generate(
        self,
        query: str,
        inputs: list[str | TextDict],
        params: dict | RerankParameters | None = None,
    ) -> dict:
        """
        Calling this method generates the following auditing event.

        :param query: The rank query.
        :type query: str

        :param inputs: The rank input strings.
        :type inputs: list[str], list[dict['text', str]]

        :param params:
        :type params: dict, RerankParameters, optional

        **Example:**

        .. code-block:: python

            query = "As a Youth, I craved excitement while in adulthood I followed Enthusiastic Pursuit."
            inputs = [
                "In my younger years, I often reveled in the excitement of spontaneous adventures and embraced the thrill of the unknown, whereas in my grownup life, I have come to appreciate the comforting stability of a well-established routine.",
                "As a young man, I frequently sought out exhilarating experiences, craving the adrenaline rush of lifes novelties, while as a responsible adult, I have come to understand the profound value of accumulated wisdom and life experience."
            ]
            response = wx_ranker.generate(query=query, inputs=inputs)

            # Print all response
            print(response)

        """

        self._client._check_if_either_is_set()

        self._validate_type(query, "query", str, True)
        self._validate_type(inputs, "inputs", list, True)

        if all(isinstance(el, str) for el in inputs):
            inputs_payload = [{"text": el_input} for el_input in inputs]
        else:
            inputs_payload = inputs  # type: ignore

        payload: dict = {
            "model_id": self.model_id,
            "query": query,
            "inputs": inputs_payload,
        }

        if params is not None:
            parameters = params

        elif self.params is not None:
            parameters = deepcopy(self.params)

        else:
            parameters = None

        if isinstance(parameters, BaseSchema):
            parameters = parameters.to_dict()

        if parameters:
            payload["parameters"] = parameters

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id
        response = self._client.httpx_client.post(
            url=self._client.service_instance._href_definitions.get_rerank_href(),
            json=payload,
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            headers=self._client._get_headers(),
        )
        return self._handle_response(200, "generate_rerank", response)
