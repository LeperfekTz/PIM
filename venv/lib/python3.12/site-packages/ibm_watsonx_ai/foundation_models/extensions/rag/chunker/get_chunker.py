#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Literal, Any

from .base_chunker import BaseChunker
from .langchain_chunker import LangChainChunker


_supported_providers = ["langchain"]


def get_chunker(
    provider: Literal["langchain"], settings: dict[str, Any] | None = None
) -> BaseChunker:
    """
    Create and get complete chunker based on the provider and settings.

    :param provider: what is the source library to create Chunker instance
    :type provider: str

    :param settings: all the settings necessary to create chosen Chunker
    :type settings: dict[str, Any]

    :return: instance of BaseChunker that can split user's documents or text
    :rtype: BaseChunker
    """
    settings = settings or {}

    match provider:
        case "langchain":
            chunker = LangChainChunker(**settings)

        case _:
            raise ValueError(
                "{} provider is not supported! Use one of {}.".format(
                    provider, _supported_providers
                )
            )

    return chunker
