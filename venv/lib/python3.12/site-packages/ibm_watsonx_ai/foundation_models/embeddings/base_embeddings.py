#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import importlib
from typing import Any
from warnings import catch_warnings, simplefilter

from ibm_watsonx_ai.wml_client_error import UnexpectedKeyWordArgument


class BaseEmbeddings(ABC):
    """LangChain-like embedding function interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs."""
        raise NotImplementedError()

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Serialize Embeddings.

        :return: serializes this Embeddings so that it can be reconstructed by ``from_dict`` class method.
        :rtype: dict
        """
        return {"__class__": self.__class__.__name__, "__module__": self.__module__}

    @classmethod
    def from_dict(cls, data: dict, **kwargs: Any) -> BaseEmbeddings | None:
        """Deserialize ``BaseEmbeddings`` into a concrete one using arguments.

        :return: concrete Embeddings or None if data is incorrect
        :rtype: BaseEmbeddings | None
        """
        supported_kwargs = ["api_client"]
        if unsupported_kwargs := set(kwargs.keys()) - set(supported_kwargs):
            for kword in unsupported_kwargs:
                raise UnexpectedKeyWordArgument(
                    kword,
                    reason=f"{kword} is not supported as a keyword argument. Supported kwargs: {supported_kwargs}",
                )
        api_client = kwargs.get("api_client")

        data = copy.deepcopy(data)
        if isinstance(data, dict):
            class_type = data.pop("__class__", None)
            module_name = data.pop("__module__", None)

            if module_name:
                module = importlib.import_module(module_name)

                if class_type:
                    try:
                        cls = getattr(module, class_type)
                    except AttributeError:
                        raise AttributeError(
                            f"Module: {module} has no attribute {class_type}"
                        )

                    if cls:
                        if (
                            module_name
                            == "ibm_watsonx_ai.foundation_models.embeddings.embeddings"
                            and api_client is not None
                        ):
                            data.pop("credentials", None)
                            data["api_client"] = api_client

                        with catch_warnings(record=True):
                            simplefilter("ignore", category=DeprecationWarning)
                            return cls(**data)

        return None
