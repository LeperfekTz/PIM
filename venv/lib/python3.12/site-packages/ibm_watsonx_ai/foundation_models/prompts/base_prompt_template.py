#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import Iterable
from abc import abstractmethod

from ibm_watsonx_ai.foundation_models.prompts.base_prompt import BasePrompt
from ibm_watsonx_ai.wml_client_error import (
    ValidationError,
)
from ibm_watsonx_ai.foundation_models.utils.utils import TemplateFormatter


class BasePromptTemplate(BasePrompt):
    """Base class for Prompt Template Asset."""

    def __init__(
        self,
        input_mode: str,
        name: str | None = None,
        model_id: str | None = None,
        description: str | None = None,
        task_ids: list[str] | None = None,
        model_params: dict | None = None,
        template_version: str | None = None,
        input_text: str | None = None,
        input_variables: list | dict[str, dict[str, str]] | None = None,
    ):

        self.input_text = input_text
        self.input_variables = (
            input_variables.copy() if input_variables is not None else input_variables
        )

        BasePrompt.__init__(
            self,
            input_mode=input_mode,
            name=name,
            model_id=model_id,
            description=description,
            task_ids=task_ids,
            model_params=model_params,
        )

        self.template_version = template_version

    @property
    def template_version(self) -> str | None:
        return self._prompt_version

    @template_version.setter
    def template_version(self, value: str) -> None:
        self._prompt_version = value

    @abstractmethod
    def _validation(self) -> None:
        """Validate the consistency of the template structure with the provided input variables."""
        raise NotImplementedError

    def _validate_prompt(
        self, input_variables: Iterable[str], template_text: str
    ) -> None:
        """:raises ValidationError: When set of elements `input_variables` is not the same
        as set of placeholders in joined string input_text + template_text"""
        try:
            dummy_inputs = {input_variable: "wx" for input_variable in input_variables}
            TemplateFormatter().format(template_text, **dummy_inputs)
        except KeyError as key:
            raise ValidationError(
                str(key),
                additional_msg="One can turn off validation step setting `validate_template` to False.",
            )
