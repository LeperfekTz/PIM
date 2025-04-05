#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, Any, overload, Literal, TypeAlias, Mapping

if TYPE_CHECKING:
    import langchain
    from langchain.prompts import PromptTemplate as LcPromptTemplate
    from langchain.prompts import ChatPromptTemplate

import inspect
import copy
import pandas

from ibm_watsonx_ai.foundation_models.prompts.base_prompt_template import (
    BasePromptTemplate,
)
from ibm_watsonx_ai.foundation_models.prompts.base_prompt import (
    BasePrompt,
)
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    InvalidValue,
    InvalidMultipleArguments,
    PromptVariablesError,
)
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.foundation_models.utils.enums import (
    ModelTypes,
    PromptTemplateFormats,
)
from ibm_watsonx_ai.foundation_models.prompts.chat_prompt import ChatPrompt

ListType: TypeAlias = list


@dataclass
class PromptTemplateLock:
    """Storage for lock object."""

    locked: bool
    lock_type: str | None = None
    locked_by: str | None = None


class FreeformPromptTemplate(BasePromptTemplate):
    """Storage for Freeform prompt template asset parameters.

    :param prompt_id: ID of the prompt template, defaults to None.
    :type prompt_id: str, attribute setting not allowed

    :param created_at: time that the prompt was created (UTC), defaults to None.
    :type created_at: str, attribute setting not allowed

    :param lock: locked state of the asset, defaults to None.
    :type lock: PromptTemplateLock | None, attribute setting not allowed

    :param is_template: True if the prompt is a template, False otherwise; defaults to None.
    :type is_template: bool | None, attribute setting not allowed

    :param name: name of the prompt template, defaults to None.
    :type name: str, optional

    :param model_id: ID of the foundation model, defaults to None.
    :type model_id: ModelTypes | str | None, optional

    :param model_params: parameters of the model, defaults to None.
    :type model_params: dict, optional

    :param template_version: semantic version for tracking in IBM AI Factsheets, defaults to None.
    :type template_version: str, optional

    :param task_ids: list of task IDs, defaults to None.
    :type task_ids: list[str] | None, optional

    :param description: description of the prompt template asset, defaults to None.
    :type description: str, optional

    :param input_text: input text for the prompt, defaults to None.
    :type input_text: str, optional

    :param input_variables: input variables can be present in field `input_text`
                            and are identified by braces ('{' and '}'), defaults to None.
    :type input_variables: (list | dict[str, dict[str, str]]), optional

    :param validate_template: if True, the prompt template is validated for the presence of input variables, defaults to True.
    :type validate_template: bool, optional

    :raises ValidationError: raised when the set of input_variables is not consistent with the input variables present in the template.
                             Raised only when `validate_template` is set to True.

    **Examples**

    Example of an invalid Freeform prompt template:

    .. code-block:: python

        prompt_template = FreeformPromptTemplate(
            name="My freeform prompt",
            model_id="ibm/granite-13b-chat-v2",
            input_text='What are the most famous monuments in ?',
            input_variables=['country']
        )

        # Traceback (most recent call last):
        #    ...
        # ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

    Example of a valid Freeform prompt template:

    .. code-block:: python

        prompt_template = FreeformPromptTemplate(
            name="My freeform prompt",
            model_id="ibm/granite-13b-chat-v2"
            input_text='What are the most famous monuments in {country}?',
            input_variables=['country']
        )

    """

    _input_mode = "freeform"

    def __init__(
        self,
        name: str | None = None,
        model_id: ModelTypes | str | None = None,
        model_params: dict | None = None,
        template_version: str | None = None,
        task_ids: list[str] | None = None,
        description: str | None = None,
        input_text: str | None = None,
        input_variables: list | dict[str, dict[str, str]] | None = None,
        validate_template: bool = True,
    ) -> None:
        super().__init__(
            input_mode=self._input_mode,
            name=name,
            model_id=model_id,
            model_params=model_params,
            template_version=template_version,
            task_ids=task_ids,
            description=description,
            input_text=input_text,
            input_variables=input_variables,
        )

        # template validation
        if validate_template:
            self._validate_prompt(
                self.input_variables if self.input_variables else [],
                self.input_text if self.input_text is not None else "",
            )

    def _validation(self) -> None:
        """Validate the template structure.

        :raises ValidationError: raised when input_variables do not fit the placeholders in the input body.
        """
        input_variables = self.input_variables or []

        self._validate_prompt(
            input_variables,
            self.input_text or "",
        )


class PromptTemplate(BasePromptTemplate):
    """Parameter storage for a structured prompt template.

    :param prompt_id: ID of the prompt template, defaults to None.
    :type prompt_id: str, attribute setting not allowed

    :param created_at: time that the prompt was created (UTC), defaults to None.
    :type created_at: str, attribute setting not allowed

    :param lock: locked state of the asset, defaults to None.
    :type lock: PromptTemplateLock | None, attribute setting not allowed

    :param is_template: True if the prompt is a template, False otherwise; defaults to None.
    :type is_template: bool | None, attribute setting not allowed

    :param name: name of the prompt template, defaults to None.
    :type name: str, optional

    :param model_id: ID of the Foundation model, defaults to None.
    :type model_id: ModelTypes | str | None, optional

    :param model_params: parameters of the model, defaults to None.
    :type model_params: dict, optional

    :param template_version: semantic version for tracking in IBM AI Factsheets, defaults to None.
    :type template_version: str, optional

    :param task_ids: List of task IDs, defaults to None.
    :type task_ids: list[str] | None, optional

    :param description: description of the prompt template asset, defaults to None.
    :type description: str, optional

    :param input_text: input text for the prompt, defaults to None.
    :type input_text: str, optional

    :param input_variables: Input variables can be present in fields: `instruction`,
                            `input_prefix`, `output_prefix`, `input_text`, `examples`
                            and are identified by braces ('{' and '}'), defaults to None.
    :type input_variables: (list | dict[str, dict[str, str]]), optional

    :param instruction: instruction for the model, defaults to None.
    :type instruction: str, optional

    :param input_prefix: prefix string placed before the input text, defaults to None.
    :type input_prefix: str, optional

    :param output_prefix: prefix placed before the model response, defaults to None.
    :type output_prefix: str, optional

    :param examples: examples that might help the model adjust the response; [[input1, output1], ...], defaults to None.
    :type examples: list[list[str]], optional

    :param validate_template: if True, the prompt template is validated for the presence of input variables, defaults to True.
    :type validate_template: bool, optional

    :raises ValidationError: raised when the set of input_variables is not consistent with the input variables present in the template.
                             Raised only when `validate_template` is set to True.

    **Examples**

    Example of an invalid prompt template:

    .. code-block:: python

        prompt_template = PromptTemplate(
            name="My structured prompt",
            model_id="ibm/granite-13b-chat-v2"
            input_text='What are the most famous monuments in ?',
            input_variables=['country']
        )

        # Traceback (most recent call last):
        #     ...
        # ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

    Example of a valid prompt template:

    .. code-block:: python

        prompt_template = PromptTemplate(
            name="My structured prompt",
            model_id="ibm/granite-13b-chat-v2"
            input_text='What are the most famous monuments in {country}?',
            input_variables=['country']
        )

    """

    _input_mode = "structured"

    def __init__(
        self,
        name: str | None = None,
        model_id: ModelTypes | str | None = None,
        model_params: dict | None = None,
        template_version: str | None = None,
        task_ids: list[str] | None = None,
        description: str | None = None,
        input_text: str | None = None,
        input_variables: list | dict[str, dict[str, str]] | None = None,
        instruction: str | None = None,
        input_prefix: str | None = None,
        output_prefix: str | None = None,
        examples: list[list[str]] | None = None,
        validate_template: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_mode=self._input_mode,
            name=name,
            model_id=model_id,
            model_params=model_params,
            template_version=template_version,
            task_ids=task_ids,
            description=description,
            input_text=input_text,
            input_variables=input_variables,
        )

        self.instruction = instruction
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.examples = copy.deepcopy(examples) if examples is not None else examples

        supported_pt_kwargs = ["input_mode", "external_information"]
        unsupported_pt_keys = [
            key for key in kwargs.keys() if key not in supported_pt_kwargs
        ]
        if unsupported_pt_keys:
            raise WMLClientError(
                f"Unsupported kwargs: {', '.join(unsupported_pt_keys)}. "
                f"Supported kwargs are: {', '.join(supported_pt_kwargs)}."
            )

        for key, value in kwargs.items():
            if key == "input_mode":
                key = "_" + key
            setattr(self, key, value)

        # template validation
        if validate_template and not (getattr(self, "_input_mode") == "chat_mode"):
            self._validation()

    def _validation(self) -> None:
        """Validate the template structure.

        :raises ValidationError: raised when input_variables do not fit the placeholders in the input body.
        """
        input_variables = self.input_variables or []
        template_text = " ".join(
            filter(None, [self.instruction, self.input_prefix, self.output_prefix])
        )
        if self.examples:
            for example in self.examples:
                template_text += " ".join(example)

        self._validate_prompt(
            input_variables,
            template_text + (self.input_text or ""),
        )


class DetachedPromptTemplate(BasePromptTemplate):
    """Storage for detached prompt template parameters.

    :param prompt_id: ID of the prompt template, defaults to None.
    :type prompt_id: str, attribute setting not allowed

    :param created_at: time that the prompt was created (UTC), defaults to None.
    :type created_at: str, attribute setting not allowed

    :param lock: locked state of the asset, defaults to None.
    :type lock: PromptTemplateLock | None, attribute setting not allowed

    :param is_template: True if the prompt is a template, False otherwise; defaults to None.
    :type is_template: bool | None, attribute setting not allowed

    :param name: name of the prompt template, defaults to None.
    :type name: str, optional

    :param model_id: ID of the foundation model, defaults to None.
    :type model_id: ModelTypes | str | None, optional

    :param model_params: parameters of the model, defaults to None.
    :type model_params: dict, optional

    :param template_version: semantic version for tracking in IBM AI Factsheets, defaults to None.
    :type template_version: str, optional

    :param task_ids: list of task IDs, defaults to None.
    :type task_ids: list[str] | None, optional

    :param description: description of the prompt template asset, defaults to None.
    :type description: str, optional

    :param input_text: input text for the prompt, defaults to None.
    :type input_text: str, optional

    :param input_variables: input variables can be present in field: `input_text`
                            and are identified by braces ('{' and '}'), defaults to None.
    :type input_variables: (list | dict[str, dict[str, str]]), optional

    :param detached_prompt_id: ID of the external prompt, defaults to None
    :type detached_prompt_id: str | None, optional

    :param detached_model_id: ID of the external model, defaults to None
    :type detached_model_id: str | None, optional

    :param detached_model_provider: external model provider, defaults to None
    :type detached_model_provider: str | None, optional

    :param detached_prompt_url: URL for the external prompt, defaults to None
    :type detached_prompt_url: str | None, optional

    :param detached_prompt_additional_information: additional information of the external prompt, defaults to None
    :type detached_prompt_additional_information: list[dict[str, Any]] | None, optional

    :param detached_model_name: name of the external model, defaults to None
    :type detached_model_name: str | None, optional

    :param detached_model_url: URL for the external model, defaults to None
    :type detached_model_url: str | None, optional

    :param validate_template: if True, the prompt template is validated for the presence of input variables, defaults to True
    :type validate_template: bool, optional

    :param instruction: instruction for the model, defaults to None
    :type instruction: str, optional

    :param input_prefix: prefix string placed before the input text, defaults to None
    :type input_prefix: str, optional

    :param output_prefix: prefix placed before the model response, defaults to None
    :type output_prefix: str, optional

    :param examples: examples that might help the model adjust the response; [[input1, output1], ...], defaults to None
    :type examples: list[list[str]], optional

    :raises ValidationError: raised when the set of input_variables is not consistent with the input variables present in the template.
                             Raised only when `validate_template` is set to True.

    **Examples**

    Example of an invalid detached prompt template:

    .. code-block:: python

        prompt_template = DetachedPromptTemplate(
            name="My detached prompt",
            model_id="<some model>",
            input_text='What are the most famous monuments in ?',
            input_variables=['country'],
            detached_prompt_id="<prompt id>",
            detached_model_id="<model id>",
            detached_model_provider="<provider>",
            detached_prompt_url="<url>",
            detached_prompt_additional_information=[{"key":"value"}],
            detached_model_name="<model name>",
            detached_model_url ="<model url>"
        )

        # Traceback (most recent call last):
        #     ...
        # ValidationError: Invalid prompt template; check for mismatched or missing input variables. Missing input variable: {'country'}

    Example of a valid detached prompt template:

    .. code-block:: python

        prompt_template = DetachedPromptTemplate(
            name="My detached prompt",
            model_id="<some model>",
            input_text='What are the most famous monuments in {country}?',
            input_variables=['country'],
            detached_prompt_id="<prompt id>",
            detached_model_id="<model id>",
            detached_model_provider="<provider>",
            detached_prompt_url="<url>",
            detached_prompt_additional_information=[{"key":"value"}],
            detached_model_name="<model name>",
            detached_model_url ="<model url>"
        )

    """

    _input_mode = "detached"

    def __init__(
        self,
        name: str | None = None,
        model_id: ModelTypes | str | None = None,
        model_params: dict | None = None,
        template_version: str | None = None,
        task_ids: list[str] | None = None,
        description: str | None = None,
        input_text: str | None = None,
        input_variables: list | dict[str, dict[str, str]] | None = None,
        detached_prompt_id: str | None = None,
        detached_model_id: str | None = None,
        detached_model_provider: str | None = None,
        detached_prompt_url: str | None = None,
        detached_prompt_additional_information: list[dict[str, Any]] | None = None,
        detached_model_name: str | None = None,
        detached_model_url: str | None = None,
        validate_template: bool = True,
        instruction: str | None = None,
        input_prefix: str | None = None,
        output_prefix: str | None = None,
        examples: list[list[str]] | None = None,
    ) -> None:
        super().__init__(
            input_mode=self._input_mode,
            name=name,
            model_id=model_id,
            model_params=model_params,
            template_version=template_version,
            task_ids=task_ids,
            description=description,
            input_text=input_text,
            input_variables=input_variables,
        )
        self.instruction = instruction
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.examples = copy.deepcopy(examples) if examples is not None else examples

        self.detached_prompt_id = detached_prompt_id
        self.detached_model_id = detached_model_id
        self.detached_model_provider = detached_model_provider
        self.detached_prompt_url = detached_prompt_url
        self.detached_prompt_additional_information = (
            detached_prompt_additional_information
        )
        self.detached_model_name = detached_model_name
        self.detached_model_url = detached_model_url

        # template validation
        if validate_template:
            self._validation()

    def _validation(self) -> None:
        """Validate the template structure.

        :raises ValidationError: raised when input_variables do not fit the placeholders in the input body.
        """
        input_variables = self.input_variables or []
        template_text = " ".join(
            filter(None, [self.instruction, self.input_prefix, self.output_prefix])
        )
        if self.examples:
            for example in self.examples:
                template_text += " ".join(example)

        self._validate_prompt(
            input_variables,
            template_text + (self.input_text or ""),
        )


class PromptTemplateManager(WMLResource):
    """Instantiate the prompt template manager.

    :param credentials: credentials for the watsonx.ai instance
    :type credentials: Credentials or dict, optional

    :param project_id: ID of the project
    :type project_id: str, optional

    :param space_id: ID of the space
    :type space_id: str, optional

    :param verify: You can pass one of the following as verify:
        * the path to a CA_BUNDLE file
        * the path of directory with certificates of trusted CAs
        * `True` - default path to truststore will be taken
        * `False` - no verification will be made
    :type verify: bool or str, optional

    .. note::
        One of these parameters is required: ['project_id ', 'space_id']

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials

        from ibm_watsonx_ai.foundation_models.prompts import PromptTemplate, PromptTemplateManager

        prompt_mgr = PromptTemplateManager(
                        credentials=Credentials(
                            api_key=IAM_API_KEY,
                            url="https://us-south.ml.cloud.ibm.com"
                        ),
                        project_id="*****"
                        )

        prompt_template = PromptTemplate(name="My prompt",
                                         model_id='meta-llama/llama-3-3-70b-instruct',
                                         input_prefix="Human:",
                                         output_prefix="Assistant:",
                                         input_text="What is {object} and how does it work?",
                                         input_variables=['object'],
                                         examples=[['What is the Stock Market?',
                                                    'A stock market is a place where investors buy and sell shares of publicly traded companies.']])

        stored_prompt_template = prompt_mgr.store_prompt(prompt_template)
        print(stored_prompt_template.prompt_id)   # id of prompt template asset

    .. note::
        Here's an example of how you can pass variables to your deployed prompt template:

        .. code-block:: python

            from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

            meta_props = {
                client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT PROMPT TEMPLATE",
                client.deployments.ConfigurationMetaNames.ONLINE: {},
                client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: 'meta-llama/llama-3-3-70b-instruct'
                }

            deployment_details = client.deployments.create(stored_prompt_template.prompt_id, meta_props)

            client.deployments.generate_text(
                deployment_id=deployment_details["metadata"]["id"],
                params={
                    GenTextParamsMetaNames.PROMPT_VARIABLES: {
                        "object": "brain"
                    }
                }
            )

    """

    def __init__(
        self,
        credentials: Credentials | dict | None = None,
        *,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: str | bool | None = None,
        api_client: APIClient | None = None,
    ) -> None:
        self.project_id = project_id
        self.space_id = space_id
        if credentials:
            self._client = APIClient(credentials, verify=verify)
        elif api_client is not None:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if self.space_id is not None and self.project_id is not None:
            raise InvalidMultipleArguments(
                params_names_list=["project_id", "space_id"],
                reason="Both arguments were provided.",
            )
        if self.space_id is not None:
            self._client.set.default_space(self.space_id)
        elif self.project_id is not None:
            self._client.set.default_project(self.project_id)
        elif api_client is not None:
            if project_id := self._client.default_project_id:
                self.project_id = project_id
            elif space_id := self._client.default_space_id:
                self.space_id = space_id
            else:
                pass
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )

        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        WMLResource.__init__(self, __name__, self._client)

    @property
    def _params(self) -> dict[str, str]:
        """Request params"""
        if self.space_id is not None and self.project_id is not None:
            raise InvalidMultipleArguments(
                params_names_list=["project_id", "space_id"],
                reason="Both arguments were set.",
            )
        elif self.project_id is not None:
            return {"project_id": self.project_id}
        elif self.space_id is not None:
            return {"space_id": self.space_id}
        else:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the parameters were set.",
            )

    def _create_request_body(self, prompt_template: BasePrompt) -> dict:
        """Create a request body from a PromptTemplate object.

        :param prompt_template: PromptTemplate object based on which the request
                                body will be created.
        :type prompt_template: BasePrompt

        :return: Request body
        :rtype: dict
        """
        json_data: dict = {"prompt": dict()}
        if prompt_template.description is not None:
            json_data.update({"description": prompt_template.description})

        if (
            isinstance(prompt_template, BasePromptTemplate)
            and prompt_template.input_variables is not None
        ):

            PromptTemplateManager._validate_type(
                prompt_template.input_variables, "input_variables", [dict, list], False
            )
            if isinstance(prompt_template.input_variables, list):
                json_data.update(
                    {
                        "prompt_variables": {
                            key: {} for key in prompt_template.input_variables
                        }
                    }
                )
            else:
                json_data.update({"prompt_variables": prompt_template.input_variables})
        if prompt_template.task_ids is not None:
            PromptTemplateManager._validate_type(
                prompt_template.task_ids, "task_ids", list, False
            )
            json_data.update({"task_ids": prompt_template.task_ids})
        if (
            isinstance(prompt_template, BasePromptTemplate)
            and prompt_template.template_version is not None
        ):
            json_data.update(
                {"model_version": {"number": prompt_template.template_version}}
            )
        elif (
            isinstance(prompt_template, ChatPrompt)
            and prompt_template.prompt_version is not None
        ):
            json_data.update(
                {"model_version": {"number": prompt_template.prompt_version}}
            )
        if hasattr(prompt_template, "_input_mode"):
            json_data.update({"input_mode": prompt_template._input_mode})

        if (
            isinstance(prompt_template, BasePromptTemplate)
            and prompt_template.input_text is not None
        ):
            PromptTemplateManager._validate_type(
                prompt_template.input_text, "input_text", str, False
            )
            json_data["prompt"].update({"input": [[prompt_template.input_text, ""]]})

        PromptTemplateManager._validate_type(
            prompt_template.model_id, "model_id", str, True
        )
        if prompt_template.model_id is not None:
            json_data["prompt"].update({"model_id": prompt_template.model_id})

        if prompt_template.model_params is not None:
            PromptTemplateManager._validate_type(
                prompt_template.model_params, "model_parameters", dict, False
            )
            json_data["prompt"].update(
                {"model_parameters": prompt_template.model_params}
            )

        if hasattr(prompt_template, "external_information"):
            json_data["prompt"].update(
                {"external_information": prompt_template.external_information}
            )

        data: dict = dict()
        if isinstance(prompt_template, PromptTemplate | DetachedPromptTemplate):

            if prompt_template.instruction is not None:
                data.update({"instruction": prompt_template.instruction})

            if prompt_template.input_prefix is not None:
                data.update({"input_prefix": prompt_template.input_prefix})

            if prompt_template.output_prefix is not None:
                data.update({"output_prefix": prompt_template.output_prefix})
            if prompt_template.examples is not None:
                PromptTemplateManager._validate_type(
                    prompt_template.examples, "examples", list, False
                )
                data.update({"examples": prompt_template.examples})

            if isinstance(prompt_template, DetachedPromptTemplate):
                external_information: dict = dict()
                PromptTemplateManager._validate_type(
                    prompt_template.detached_prompt_id, "detached_prompt_id", str, True
                )
                PromptTemplateManager._validate_type(
                    prompt_template.detached_model_id, "detached_model_id", str, True
                )
                PromptTemplateManager._validate_type(
                    prompt_template.detached_model_provider,
                    "detached_model_provider",
                    str,
                    True,
                )
                external_information.update(
                    {
                        "external_prompt_id": prompt_template.detached_prompt_id,
                        "external_model_id": prompt_template.detached_model_id,
                        "external_model_provider": prompt_template.detached_model_provider,
                    }
                )

                if prompt_template.detached_prompt_additional_information is not None:
                    PromptTemplateManager._validate_type(
                        prompt_template.detached_prompt_url,
                        "detached_prompt_url",
                        str,
                        True,
                    )
                    external_information.update(
                        {
                            "external_prompt": {
                                "url": prompt_template.detached_prompt_url,
                                "additional_information": prompt_template.detached_prompt_additional_information,
                            }
                        }
                    )
                if (
                    prompt_template.detached_model_name is not None
                    or prompt_template.detached_model_url is not None
                ):
                    PromptTemplateManager._validate_type(
                        prompt_template.detached_model_url,
                        "detached_model_url",
                        str,
                        True,
                    )
                    PromptTemplateManager._validate_type(
                        prompt_template.detached_model_name,
                        "detached_model_name",
                        str,
                        True,
                    )
                    external_information.update(
                        {
                            "external_model": {
                                "url": prompt_template.detached_model_url,
                                "name": prompt_template.detached_model_name,
                            }
                        }
                    )

                json_data["prompt"].update(
                    {"external_information": external_information}
                )

        elif isinstance(prompt_template, ChatPrompt):

            if prompt_template.chat_items is not None:
                PromptTemplateManager._validate_type(
                    prompt_template.chat_items, "chat_items", list, True
                )
                json_data["prompt"].update({"chat_items": prompt_template.chat_items})

            if prompt_template.system_prompt is not None:
                PromptTemplateManager._validate_type(
                    prompt_template.system_prompt, "system_prompt", str, True
                )
                json_data["prompt"].update(
                    {"system_prompt": prompt_template.system_prompt}
                )

        json_data["prompt"].update({"data": data})
        return json_data

    def _from_json_to_prompt(
        self, response: dict
    ) -> FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt:
        """Convert json response to FreeformPromptTemplate or PromptTemplate object.

        :param response: Response body after request operation.
        :type response: dict

        :return: PromptTemplate object with given details.
        :rtype: FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt
        """
        prompt_field: dict = response.get("prompt", dict())
        data_field: dict = prompt_field.get("data", dict())
        prompt_template: (
            FreeformPromptTemplate
            | PromptTemplate
            | DetachedPromptTemplate
            | ChatPrompt
        )

        match response.get("input_mode"):
            case "freeform":
                prompt_template = FreeformPromptTemplate(
                    name=response.get("name"),
                    description=response.get("description"),
                    model_id=prompt_field.get("model_id"),
                    model_params=prompt_field.get("model_parameters"),
                    task_ids=response.get("task_ids"),
                    template_version=response.get("model_version", dict()).get(
                        "number"
                    ),
                    input_variables=response.get("prompt_variables"),
                    input_text=prompt_field.get("input", [[None, None]])[0][0],
                    validate_template=False,
                )
            case "detached":
                external_information_field: dict = prompt_field.get(
                    "external_information", {}
                )
                prompt_template = DetachedPromptTemplate(
                    name=response.get("name"),
                    description=response.get("description"),
                    model_id=prompt_field.get("model_id"),
                    model_params=prompt_field.get("model_parameters"),
                    task_ids=response.get("task_ids"),
                    template_version=response.get("model_version", dict()).get(
                        "number"
                    ),
                    input_variables=response.get("prompt_variables"),
                    input_text=prompt_field.get("input", [[None, None]])[0][0],
                    instruction=data_field.get("instruction"),
                    input_prefix=data_field.get("input_prefix"),
                    output_prefix=data_field.get("output_prefix"),
                    examples=data_field.get("examples"),
                    detached_prompt_id=external_information_field.get(
                        "external_prompt_id"
                    ),
                    detached_model_id=external_information_field.get(
                        "external_model_id"
                    ),
                    detached_model_provider=external_information_field.get(
                        "external_model_provider"
                    ),
                    detached_prompt_url=external_information_field.get(
                        "external_prompt", {}
                    ).get("url"),
                    detached_prompt_additional_information=external_information_field.get(
                        "external_prompt", {}
                    ).get(
                        "additional_information"
                    ),
                    detached_model_name=external_information_field.get(
                        "external_model", {}
                    ).get("name"),
                    detached_model_url=external_information_field.get(
                        "external_model", {}
                    ).get("url"),
                    validate_template=False,
                )

            case "chat":
                prompt_template = ChatPrompt(
                    name=response.get("name"),
                    description=response.get("description"),
                    model_id=prompt_field.get("model_id"),
                    model_params=prompt_field.get("model_parameters"),
                    task_ids=response.get("task_ids"),
                    prompt_version=response.get("model_version", dict()).get("number"),
                    chat_items=prompt_field.get("chat_items"),
                    system_prompt=prompt_field.get("system_prompt"),
                )

            case _:
                prompt_template = PromptTemplate(
                    name=response.get("name"),
                    description=response.get("description"),
                    model_id=prompt_field.get("model_id"),
                    model_params=prompt_field.get("model_parameters"),
                    task_ids=response.get("task_ids"),
                    template_version=response.get("model_version", dict()).get(
                        "number"
                    ),
                    input_variables=response.get("prompt_variables"),
                    input_text=prompt_field.get("input", [[None, None]])[0][0],
                    instruction=data_field.get("instruction"),
                    input_prefix=data_field.get("input_prefix"),
                    output_prefix=data_field.get("output_prefix"),
                    examples=data_field.get("examples"),
                    validate_template=False,
                )

        prompt_template._prompt_id = response.get("id")
        prompt_template._created_at = response.get("created_at")
        prompt_template._lock = PromptTemplateLock(
            **response.get("lock", {"locked": None, "locked_by": None})
        )
        prompt_template._is_template = response.get("is_template")

        return prompt_template

    def _get_details(self, limit: int | None = None) -> list:
        """Get details of all prompt templates. If limit is set to None,
        then all prompt templates are fetched.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: int | None

        :return: List of prompts metadata
        :rtype: List
        """
        headers = self._client._get_headers()
        url = self._client.service_instance._href_definitions.get_prompts_all_href()
        json_data: dict[str, int | str] = {
            "query": "asset.asset_type:wx_prompt",
            "sort": "-asset.created_at<string>",
        }
        if limit is not None:
            if limit < 1:
                raise WMLClientError("Limit cannot be lower than 1.")
            elif limit > 200:
                raise WMLClientError("Limit cannot be larger than 200.")

            json_data.update({"limit": limit})
        else:
            json_data.update({"limit": 200})
        prompts_list = []
        bookmark = True
        while bookmark is not None:
            response = self._client.httpx_client.post(
                url=url, json=json_data, headers=headers, params=self._params
            )
            details_json = self._handle_response(200, "Get next details", response)
            bookmark = details_json.get("next", {"href": None}).get("bookmark", None)
            prompts_list.extend(details_json.get("results", []))
            if limit is not None:
                break
            json_data.update({"bookmark": bookmark})
        return prompts_list

    def _change_lock(self, prompt_id: str, locked: bool, force: bool = False) -> dict:
        """Change the state of a prompt template lock.

        :param prompt_id: ID of the prompt template
        :type prompt_id: str

        :param locked: new state of the lock
        :type locked: bool

        :param force: force lock state overwrite, defaults to False.
        :type force: bool, optional

        :return: changed state of the lock
        :rtype: dict
        """
        headers = self._client._get_headers()
        params: Mapping[str, str | int | float | bool | None] = self._params | {
            "prompt_id": prompt_id,
            "force": force,
        }
        json_data = {"locked": locked}

        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}/lock"
        )
        response = self._client.httpx_client.put(
            url=url, json=json_data, headers=headers, params=params
        )

        return self._handle_response(200, "change_lock", response)

    @staticmethod
    def _prepare_lc_messages(
        system_prompt: str, chat_items: ListType | None
    ) -> ListType[tuple]:
        result = [("system", system_prompt)]

        mapping_to_lc = {
            "question": "human",
            "answer": "ai",
        }
        if chat_items:
            for el in chat_items:
                role = mapping_to_lc[el["type"]]
                content = el["content"]
                result.append((role, content))

        return result

    @overload
    def load_prompt(
        self,
        prompt_id: str,
        astype: Literal[PromptTemplateFormats.STRING, "string"],
        *,
        prompt_variables: dict[str, str] | None = None,
    ) -> str: ...

    @overload
    def load_prompt(
        self,
        prompt_id: str,
        astype: Literal[PromptTemplateFormats.LANGCHAIN, "langchain"],
        *,
        prompt_variables: dict[str, str] | None = None,
    ) -> LcPromptTemplate: ...

    @overload
    def load_prompt(
        self,
        prompt_id: str,
        astype: Literal[
            PromptTemplateFormats.PROMPTTEMPLATE, "prompt"
        ] = PromptTemplateFormats.PROMPTTEMPLATE,
        *,
        prompt_variables: dict[str, str] | None = None,
    ) -> (
        FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt
    ): ...

    def load_prompt(
        self,
        prompt_id: str,
        astype: PromptTemplateFormats | str = PromptTemplateFormats.PROMPTTEMPLATE,
        *,
        prompt_variables: dict[str, str] | None = None,
    ) -> (
        FreeformPromptTemplate
        | PromptTemplate
        | DetachedPromptTemplate
        | ChatPrompt
        | str
        | LcPromptTemplate
        | ChatPromptTemplate
    ):
        """Retrieve a prompt template asset.

        :param prompt_id: ID of the processed prompt template
        :type prompt_id: str

        :param astype: type of return object
        :type astype: PromptTemplateFormats

        :param prompt_variables: dictionary of input variables and values that will replace the input variables
        :type prompt_variables: dict[str, str]

        :return: prompt template asset
        :rtype: FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt | str | langchain.prompts.PromptTemplate

        **Example:**

        .. code-block:: python

            loaded_prompt_template = prompt_mgr.load_prompt(prompt_id)
            loaded_prompt_template_lc = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.LANGCHAIN)
            loaded_prompt_template_string = prompt_mgr.load_prompt(prompt_id, PromptTemplateFormats.STRING)
        """
        headers = self._client._get_headers()
        params = self._params | {"prompt_id": prompt_id}
        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}"
        )

        if isinstance(astype, PromptTemplateFormats):
            astype = astype.value

        if astype == "prompt":
            response = self._client.httpx_client.get(
                url=url, headers=headers, params=params
            )
            return self._from_json_to_prompt(
                self._handle_response(200, "_load_json_prompt", response)
            )
        elif astype in ("langchain", "string"):
            response = self._client.httpx_client.post(
                url=url + "/input", headers=headers, params=params
            )
            response_input = self._handle_response(200, "load_prompt", response).get(
                "input"
            )
            response_input = cast(str, response_input)
            if astype == "string":
                try:
                    return (
                        response_input
                        if prompt_variables is None
                        else response_input.format(**prompt_variables)
                    )
                except KeyError as key:
                    raise PromptVariablesError(str(key)) from key
            else:
                response = self._client.httpx_client.get(
                    url=url, headers=headers, params=params
                )
                response_json = self._handle_response(
                    200, "_load_json_prompt", response
                )

                if response_json.get("input_mode") == "chat":
                    from langchain.prompts import ChatPromptTemplate

                    system_prompt = response_json["prompt"].get("system_prompt")
                    chat_items = response_json["prompt"].get("chat_items")

                    messages = self._prepare_lc_messages(system_prompt, chat_items)

                    return ChatPromptTemplate(messages)
                else:
                    from langchain.prompts import PromptTemplate as LcPromptTemplate

                    return LcPromptTemplate.from_template(response_input)
        else:
            raise InvalidValue("astype")

    def list(self, *, limit: int | None = None) -> pandas.DataFrame:
        """List all available prompt templates in the DataFrame format.

        :param limit: limit number of fetched records, defaults to None.
        :type limit: int, optional

        :return: DataFrame of fundamental properties of available prompts.
        :rtype: pandas.core.frame.DataFrame

        **Example:**

        .. code-block:: python

            prompt_mgr.list(limit=5)    # list of 5 recent created prompt template assets

        .. hint::
            Additionally you can sort available prompt templates by "LAST MODIFIED" field.

            .. code-block:: python

                df_prompts = prompt_mgr.list()
                df_prompts.sort_values("LAST MODIFIED", ascending=False)

        """
        details = [
            "metadata.asset_id",
            "metadata.name",
            "metadata.created_at",
            "metadata.usage.last_updated_at",
        ]
        prompts_details = self._get_details(limit=limit)

        data_normalize = pandas.json_normalize(prompts_details)
        prompts_data = data_normalize.reindex(columns=details)

        df_details = pandas.DataFrame(prompts_data, columns=details)

        df_details.rename(
            columns={
                "metadata.asset_id": "ID",
                "metadata.name": "NAME",
                "metadata.created_at": "CREATED",
                "metadata.usage.last_updated_at": "LAST MODIFIED",
            },
            inplace=True,
        )

        return df_details

    def store_prompt(
        self,
        prompt_template: (
            FreeformPromptTemplate
            | PromptTemplate
            | DetachedPromptTemplate
            | ChatPrompt
            | langchain.prompts.PromptTemplate
        ),
    ) -> FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt:
        """Store a new prompt template.

        :param prompt_template: PromptTemplate to be stored.
        :type prompt_template: (FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt | langchain.prompts.PromptTemplate)

        :return: PromptTemplate object that is initialized with values provided in the server response object.
        :rtype: FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt
        """
        if isinstance(
            prompt_template,
            (
                PromptTemplate
                | FreeformPromptTemplate
                | DetachedPromptTemplate
                | ChatPrompt
            ),
        ):
            pass
        else:
            from langchain.prompts import PromptTemplate as LcPromptTemplate

            if isinstance(prompt_template, LcPromptTemplate):

                def get_metadata_value(
                    prompt_temp: LcPromptTemplate,
                    key: str,
                    default: ModelTypes | list | bool | str | None = None,
                    must_be_list: bool = False,
                    must_be_nested_list: bool = False,
                ) -> Any:
                    if (
                        hasattr(prompt_temp, "metadata")
                        and prompt_temp.metadata
                        and key in prompt_temp.metadata
                    ):
                        if must_be_list:
                            if isinstance(prompt_temp.metadata[key], str):
                                return [prompt_temp.metadata[key]]
                            return prompt_temp.metadata[key]
                        elif must_be_nested_list:
                            if isinstance(prompt_temp.metadata[key], str):
                                return [[prompt_temp.metadata[key]]]
                            elif isinstance(prompt_temp.metadata[key], list):
                                if isinstance(prompt_temp.metadata[key][0], str):
                                    return [prompt_temp.metadata[key]]
                                return prompt_temp.metadata[key]
                        else:
                            return prompt_temp.metadata[key]
                    else:
                        return default

                match get_metadata_value(prompt_template, "input_mode", "structured"):
                    case "structured":
                        prompt_template = PromptTemplate(
                            name=get_metadata_value(
                                prompt_template, "name", "My prompt"
                            ),
                            model_id=get_metadata_value(
                                prompt_template, "model_id", ModelTypes.FLAN_UL2
                            ),
                            model_params=get_metadata_value(
                                prompt_template, "model_params", None
                            ),
                            template_version=get_metadata_value(
                                prompt_template, "template_version", None
                            ),
                            task_ids=get_metadata_value(
                                prompt_template, "task_ids", None, must_be_list=True
                            ),
                            description=get_metadata_value(
                                prompt_template, "description", None
                            ),
                            input_text=get_metadata_value(
                                prompt_template, "input_text", prompt_template.template
                            ),
                            input_variables=get_metadata_value(
                                prompt_template,
                                "input_variables",
                                prompt_template.input_variables,
                            ),
                            instruction=get_metadata_value(
                                prompt_template, "instruction", None
                            ),
                            input_prefix=get_metadata_value(
                                prompt_template, "input_prefix", None
                            ),
                            output_prefix=get_metadata_value(
                                prompt_template, "output_prefix", None
                            ),
                            examples=get_metadata_value(
                                prompt_template,
                                "examples",
                                None,
                                must_be_nested_list=True,
                            ),
                            validate_template=get_metadata_value(
                                prompt_template, "validate_template", True
                            ),
                        )
                    case "freeform":
                        prompt_template = FreeformPromptTemplate(
                            name=get_metadata_value(
                                prompt_template, "name", "My prompt"
                            ),
                            model_id=get_metadata_value(
                                prompt_template, "model_id", ModelTypes.FLAN_UL2
                            ),
                            model_params=get_metadata_value(
                                prompt_template, "model_params", None
                            ),
                            template_version=get_metadata_value(
                                prompt_template, "template_version", None
                            ),
                            task_ids=get_metadata_value(
                                prompt_template, "task_ids", None, must_be_list=True
                            ),
                            description=get_metadata_value(
                                prompt_template, "description", None
                            ),
                            input_text=get_metadata_value(
                                prompt_template, "input_text", prompt_template.template
                            ),
                            input_variables=get_metadata_value(
                                prompt_template,
                                "input_variables",
                                prompt_template.input_variables,
                            ),
                            validate_template=get_metadata_value(
                                prompt_template, "validate_template", True
                            ),
                        )
                    case "chat":
                        prompt_template = ChatPrompt(
                            name=get_metadata_value(
                                prompt_template, "name", "My prompt"
                            ),
                            model_id=get_metadata_value(
                                prompt_template,
                                "model_id",
                                "meta-llama/llama-3-1-70b-instruct",
                            ),
                            model_params=get_metadata_value(
                                prompt_template, "model_params", None
                            ),
                            prompt_version=get_metadata_value(
                                prompt_template, "template_version", None
                            ),
                            task_ids=get_metadata_value(
                                prompt_template, "task_ids", None, must_be_list=True
                            ),
                            description=get_metadata_value(
                                prompt_template, "description", None
                            ),
                            chat_items=get_metadata_value(
                                prompt_template, "chat_items", None
                            ),
                            system_prompt=get_metadata_value(
                                prompt_template, "system_prompt", None
                            ),
                        )
                    case "detached":
                        prompt_template = DetachedPromptTemplate(
                            name=get_metadata_value(
                                prompt_template, "name", "My prompt"
                            ),
                            model_id=get_metadata_value(
                                prompt_template, "model_id", ModelTypes.FLAN_UL2
                            ),
                            model_params=get_metadata_value(
                                prompt_template, "model_params", None
                            ),
                            template_version=get_metadata_value(
                                prompt_template, "template_version", None
                            ),
                            task_ids=get_metadata_value(
                                prompt_template, "task_ids", None, must_be_list=True
                            ),
                            description=get_metadata_value(
                                prompt_template, "description", None
                            ),
                            input_text=get_metadata_value(
                                prompt_template, "input_text", prompt_template.template
                            ),
                            instruction=get_metadata_value(
                                prompt_template, "instruction", None
                            ),
                            input_prefix=get_metadata_value(
                                prompt_template, "input_prefix", None
                            ),
                            output_prefix=get_metadata_value(
                                prompt_template, "output_prefix", None
                            ),
                            examples=get_metadata_value(
                                prompt_template,
                                "examples",
                                None,
                                must_be_nested_list=True,
                            ),
                            input_variables=get_metadata_value(
                                prompt_template,
                                "input_variables",
                                prompt_template.input_variables,
                            ),
                            **{
                                param: get_metadata_value(
                                    prompt_template, param, prop.default
                                )
                                for param, prop in inspect.signature(
                                    DetachedPromptTemplate.__init__
                                ).parameters.items()
                                if param.startswith("detached")
                            },
                            validate_template=get_metadata_value(
                                prompt_template,
                                "validate_template",
                                True,
                            ),
                        )
                    case _:
                        raise WMLClientError(error_msg="Unsupported `input_mode`")
            else:
                raise WMLClientError(error_msg="Unsupported type for `prompt_template`")

        headers = self._client._get_headers()

        PromptTemplateManager._validate_type(
            prompt_template.name, "prompt_template.name", str, True
        )
        json_data: dict = {
            "name": prompt_template.name,
            "lock": {"locked": True},
            "input_mode": prompt_template._input_mode,
            "prompt": dict(),
        }

        json_data.update(self._create_request_body(prompt_template))

        url = self._client.service_instance._href_definitions.get_prompts_href(
            ga_api=self._client._use_pta_ga_api
        )
        response = self._client.httpx_client.post(
            url=url, json=json_data, headers=headers, params=self._params
        )
        res = self._handle_response(201, "store_prompt", response)

        return self._from_json_to_prompt(res)

    def delete_prompt(self, prompt_id: str, *, force: bool = False) -> str:
        """Remove a prompt template from a project or space.

        :param prompt_id: ID of the prompt template to be deleted
        :type prompt_id: str

        :param force: if True, then the prompt template is unlocked and then deleted, defaults to False.
        :type force: bool

        :return: status 'SUCCESS' if the prompt template is successfully deleted
        :rtype: str

        **Example:**

        .. code-block:: python

            prompt_mgr.delete_prompt(prompt_id)  # delete if asset is unlocked
        """
        if force:
            self.unlock(prompt_id)

        headers = self._client._get_headers()
        params = self._params | {"prompt_id": prompt_id}

        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}"
        )
        response = self._client.httpx_client.delete(
            url=url, headers=headers, params=params
        )

        return self._handle_response(204, "delete_prompt", response)  # type: ignore[return-value]

    def update_prompt(
        self,
        prompt_id: str,
        prompt_template: (
            FreeformPromptTemplate
            | PromptTemplate
            | DetachedPromptTemplate
            | ChatPrompt
        ),
    ) -> dict:
        """Update prompt template data.

        :param prompt_id: ID of the prompt template to be updated
        :type prompt_id: str

        :param prompt_template: prompt template with new data
        :type prompt_template: FreeformPromptTemplate | PromptTemplate | DetachedPromptTemplate | ChatPrompt

        :return: metadata of the updated deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            updated_prompt_template = PromptTemplate(name="New name")
            prompt_mgr.update_prompt(prompt_id, prompt_template)  # {'name': 'New name'} in metadata

        """
        headers = self._client._get_headers()
        params = self._params | {"prompt_id": prompt_id}

        new_body: dict = dict()
        current_prompt_template = self.load_prompt(prompt_id)
        if not isinstance(prompt_template, type(current_prompt_template)):
            raise TypeError(
                (
                    "Type of `prompt_template` is not consistent with"
                    " the input mode of the updated Prompt Template Asset: "
                    f"input_mode={current_prompt_template._input_mode}"
                )
            )

        for attribute in prompt_template.__dict__:
            if getattr(
                prompt_template, attribute
            ) is not None and not attribute.startswith("_"):
                setattr(
                    current_prompt_template,
                    attribute,
                    getattr(prompt_template, attribute),
                )

        if current_prompt_template.name is not None:
            new_body.update({"name": current_prompt_template.name})

        new_body.update(self._create_request_body(current_prompt_template))

        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}"
        )

        response = self._client.httpx_client.patch(
            url=url, json=new_body, headers=headers, params=params
        )
        return self._handle_response(200, "update_prompt", response)

    def get_lock(self, prompt_id: str) -> dict:
        """Get the current locked state of a prompt template.

        :param prompt_id: ID of the prompt template
        :type prompt_id: str

        :return: information about the locked state of a prompt template asset
        :rtype: dict

        **Example:**

        .. code-block:: python

            print(prompt_mgr.get_lock(prompt_id))
        """
        headers = self._client._get_headers()
        params = self._params | {"prompt_id": prompt_id}
        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}/lock"
        )

        response = self._client.httpx_client.get(
            url=url, headers=headers, params=params
        )

        return self._handle_response(200, "get_lock", response)

    def lock(self, prompt_id: str, force: bool = False) -> dict:
        """Lock a prompt template if it is unlocked and you have permission to lock it.

        :param prompt_id: ID of the prompt template
        :type prompt_id: str

        :param force: if True, lock is forcefully overwritten
        :type force: bool

        :return: locked prompt template
        :rtype: dict

        **Example:**

        .. code-block:: python

            prompt_mgr.lock(prompt_id)

        """
        return self._change_lock(prompt_id=prompt_id, locked=True, force=force)

    def unlock(self, prompt_id: str) -> dict:
        """Unlock a prompt template if it is locked and you have permission to unlock it.

        :param prompt_id: ID of the prompt template
        :type prompt_id: str

        :return: unlocked prompt template
        :rtype: dict

        **Example:**

        .. code-block:: python

            prompt_mgr.unlock(prompt_id)
        """
        # server returns status code 400 after trying to unlock unlocked prompt
        lock_state = self.get_lock(prompt_id)
        if lock_state["locked"]:
            return self._change_lock(prompt_id=prompt_id, locked=False, force=False)
        else:
            return lock_state

    def add_chat_items(self, prompt_id: str, chat_items: ListType[dict]) -> str:
        """Add a new chat items to a prompt.

        :param prompt_id: ID of the prompt template
        :type prompt_id: str

        :param chat_items: Chat items to be added to prompt
        :type chat_items: list[dict]

        :return: status ("SUCCESS" if succeeded)
        :rtype: str

        **Example:**

        .. code-block:: python

            prompt_mgr.add_chat_items(
                prompt_id="<PROMPT_ID>",
                chat_items=[
                    {
                        "type": "<CHAT ITEM TYPE>",
                        "content": "<CHAT ITEM CONTENT>",
                        "status": "<CHAT ITEM STATUS>",
                        "timestamp": <TIMESTAMP_AS_INT>,
                    },
                    {
                        "type": "<CHAT ITEM TYPE>",
                        "content": "<CHAT ITEM CONTENT>",
                        "status": "<CHAT ITEM STATUS>",
                        "timestamp": <TIMESTAMP_AS_INT>,
                    }
                ]
            )

        """
        headers = self._client._get_headers()
        params = self._params

        url = (
            self._client.service_instance._href_definitions.get_prompts_href(
                ga_api=self._client._use_pta_ga_api
            )
            + f"/{prompt_id}/chat_items"
        )

        response = self._client.httpx_client.post(
            url=url, json=chat_items, headers=headers, params=params
        )
        if response.status_code == 201:
            return "SUCCESS"
        else:
            return self._handle_response(
                201, "add_chat_items", response, json_response=False
            )
