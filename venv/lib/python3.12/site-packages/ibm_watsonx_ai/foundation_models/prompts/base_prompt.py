#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterable
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .prompt_template import PromptTemplateLock
from datetime import datetime

from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes


class BasePrompt(ABC):
    """Base class for Prompt Asset."""

    def __init__(
        self,
        input_mode: str,
        name: str | None = None,
        model_id: str | None = None,
        description: str | None = None,
        task_ids: list[str] | None = None,
        model_params: dict | None = None,
        prompt_version: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.task_ids = task_ids.copy() if task_ids is not None else task_ids
        self.model_id: ModelTypes | str | None = model_id
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value
        self.model_params = (
            model_params.copy() if model_params is not None else model_params
        )
        self._prompt_version = prompt_version

        self._input_mode = input_mode
        self._prompt_id: str | None = None
        self._created_at: float | None = None
        self._lock: PromptTemplateLock | None = None
        self._is_template: bool | None = None

    @property
    def prompt_id(self) -> str | None:
        return self._prompt_id

    @property
    def created_at(self) -> str | None:
        if self._created_at is not None:
            return str(datetime.fromtimestamp(self._created_at / 1000)).split(".")[0]
        else:
            return None

    @property
    def lock(self) -> PromptTemplateLock | None:
        return self._lock

    @property
    def is_template(self) -> bool | None:
        return self._is_template

    def __repr__(self) -> str:
        args = [
            f"{key}={value!r}"
            for key, value in self.__dict__.items()
            if not key.startswith("_") and value is not None
        ]
        return f"{type(self).__name__}({ ', '.join(args)})"
