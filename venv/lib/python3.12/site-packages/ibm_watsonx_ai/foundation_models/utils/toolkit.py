#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ResourceByNameNotFound,
    MissingToolRequiredProperties,
)
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai._wrappers import requests


class Tool(WMLResource):
    """Instantiate the utility agent tool.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    :param name: name of the tool
    :type name: str

    :param description: description of what the tool is used for
    :type description: str

    :param agent_description: the precise instruction to agent LLMs and should be treated as part of the system prompt,
    if not provided, `description` can be used in its place
    :type agent_description: str, optional

    :param input_schema: schema of the input that is provided when running the tool if applicable
    :param input_schema: dict, optional

    :param config_schema: schema of the config that is provided when running the tool if applicable
    :param config_schema: dict, optional

    """

    def __init__(
        self,
        api_client: APIClient,
        name: str,
        description: str,
        agent_description: str | None = None,
        input_schema: dict | None = None,
        config_schema: dict | None = None,
    ):
        self._client = api_client

        Tool._validate_type(name, "name", str)
        Tool._validate_type(input_schema, "input_schema", dict, False)
        Tool._validate_type(config_schema, "config_schema", dict, False)

        self.name = name
        self.description = description
        self.agent_description = agent_description
        self.input_schema = input_schema
        self.config_schema = config_schema

        if not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

        if self.input_schema is not None:
            self._input_schema_required = self.input_schema.get("required")

    def run(
        self,
        input: str | dict,
        config: dict | None = None,
    ) -> dict:
        """Runs a utility agent tool given `input`.

        :param input: input to be used when running tool
        :type input:
            - **str** - if running tool has no `input_schema`
            - **dict** - if running tool has `input_schema`

        :param config: configuration options that can be passed for some tools,
        must match the config schema for the tool
        :type config: dict, optional

        :return: the output from running the tool
        :rtype: dict

        **Example for the tool without input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search = toolkit.get_tool(tool_name='GoogleSearch')
            result = google_search.run(input="Search IBM")

        **Example for the tool with input schema:**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            weather_tool = toolkit.get_tool(tool_name='Weather')
            tool_input = {"name": "New York"}
            result = weather_tool.run(input=tool_input)

        """
        if self.input_schema is None:
            Tool._validate_type(input, "input", str)
        else:
            Tool._validate_type(input, "input", dict)
            if self._input_schema_required:
                if any(req not in input for req in self._input_schema_required):
                    raise MissingToolRequiredProperties(self._input_schema_required)

        payload = {
            "input": input,
            "tool_name": self.name,
        }

        if config and self.config_schema:
            payload.update({"config": config})  # type: ignore[dict-item]

        response = requests.post(
            url=self._client.service_instance._href_definitions.get_utility_agent_tools_run_href(),
            json=payload,
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "run tool", response)


class Toolkit(WMLResource):
    """Toolkit for utility agent tools.

    :param api_client: initialized APIClient object
    :type api_client: APIClient

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models.utils import Toolkit

        credentials = Credentials(
            url = "<url>",
            api_key = IAM_API_KEY
        )

        api_client = APIClient(credentials)
        toolkit = Toolkit(api_client=api_client)

    """

    def __init__(self, api_client: APIClient):

        self._client = api_client

        if not self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    def get_tools(self) -> list[dict]:
        """Get list of available utility agent tools.

        :return: list of available tools
        :rtype: list[dict]

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            tools = toolkit.get_tools()

        """
        response = requests.get(
            url=self._client.service_instance._href_definitions.get_utility_agent_tools_href(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(200, "getting utility agent tools", response).get(
            "resources", []
        )

    def get_tool(self, tool_name: str) -> Tool:
        """Get a utility agent tool with the given `tool_name`.

        :param tool_name: name of a specific tool
        :type tool_name: str

        :return: tool with a given name
        :rtype: Tool

        **Examples**

        .. code-block:: python

            toolkit = Toolkit(api_client=api_client)
            google_search: Tool = toolkit.get_tool(tool_name='GoogleSearch')

        """
        Toolkit._validate_type(tool_name, "tool_name", str)

        resources = self.get_tools()

        for r in resources:
            if r["name"] == tool_name:
                return Tool(
                    api_client=self._client,
                    name=tool_name,
                    description=r["description"],
                    agent_description=r.get("agent_description"),
                    input_schema=r.get("input_schema"),
                    config_schema=r.get("config_schema"),
                )
        raise ResourceByNameNotFound(tool_name, "utility agent tool")
