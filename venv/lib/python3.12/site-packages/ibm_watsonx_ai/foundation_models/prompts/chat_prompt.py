#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.foundation_models.prompts.base_prompt import BasePrompt
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes


class ChatPrompt(BasePrompt):
    """Storage for chat prompt parameters.

    :param prompt_id: ID of the prompt, defaults to None.
    :type prompt_id: str, attribute setting not allowed

    :param created_at: time that the prompt was created (UTC), defaults to None.
    :type created_at: str, attribute setting not allowed

    :param lock: locked state of the asset, defaults to None.
    :type lock: PromptTemplateLock | None, attribute setting not allowed

    :param is_template: True if the prompt is a template, False otherwise; defaults to None.
    :type is_template: bool | None, attribute setting not allowed

    :param name: name of the prompt, defaults to None.
    :type name: str, optional

    :param model_id: ID of the foundation model, defaults to None.
    :type model_id: ModelTypes | str | None, optional

    :param model_params: parameters of the model, defaults to None.
    :type model_params: dict, optional

    :param prompt_version: semantic version for tracking in IBM AI Factsheets, defaults to None.
    :type prompt_version: str, optional

    :param task_ids: list of task IDs, defaults to None.
    :type task_ids: list[str] | None, optional

    :param description: description of the prompt asset, defaults to None.
    :type description: str, optional

    :param chat_items: chat items, defaults to None.
    :type chat_items: list[dict], optional

    :param system_prompt: system prompt used by model, defaults to None.
    :type system_prompt: str, optional


    **Examples**

    Example of a valid chat prompt:

    .. code-block:: python

        prompt = ChatPrompt(
            name="My chat prompt",
            model_id="<some model>",
            system_prompt="system prompt"
        )

    """

    _input_mode = "chat"

    def __init__(
        self,
        name: str | None = None,
        model_id: ModelTypes | str | None = None,
        model_params: dict | None = None,
        prompt_version: str | None = None,
        task_ids: list[str] | None = None,
        description: str | None = None,
        chat_items: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            input_mode=self._input_mode,
            name=name,
            model_id=model_id,
            model_params=model_params,
            prompt_version=prompt_version,
            task_ids=task_ids,
            description=description,
        )

        self.chat_items = chat_items
        self.system_prompt = system_prompt

    @property
    def prompt_version(self) -> str | None:
        return self._prompt_version

    @prompt_version.setter
    def prompt_version(self, value: str) -> None:
        self._prompt_version = value
