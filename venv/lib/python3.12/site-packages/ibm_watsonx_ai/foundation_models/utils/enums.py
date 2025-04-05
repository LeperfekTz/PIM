#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum

from ibm_watsonx_ai.utils.utils import StrEnum

__all__ = [
    "ModelTypes",
    "DecodingMethods",
    "PromptTuningTypes",
    "PromptTuningInitMethods",
    "TuneExperimentTasks",
    "PromptTemplateFormats",
    "EmbeddingTypes",
]


class ModelTypes(StrEnum):
    """

    .. deprecated:: 1.0.5
        Use :func:`TextModels` instead.

    Supported foundation models.

    .. note::
        You can check the current list of supported models types of various environments with
        :func:`get_model_specs() <ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_model_specs>` or
        by referring to the `watsonx.ai <https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx>`_
        documentation.

    """

    FLAN_T5_XXL = "google/flan-t5-xxl"
    FLAN_UL2 = "google/flan-ul2"
    MT0_XXL = "bigscience/mt0-xxl"
    GPT_NEOX = "eleutherai/gpt-neox-20b"
    MPT_7B_INSTRUCT2 = "ibm/mpt-7b-instruct2"
    STARCODER = "bigcode/starcoder"
    LLAMA_2_70B_CHAT = "meta-llama/llama-2-70b-chat"
    LLAMA_2_13B_CHAT = "meta-llama/llama-2-13b-chat"
    GRANITE_13B_INSTRUCT = "ibm/granite-13b-instruct-v1"
    GRANITE_13B_CHAT = "ibm/granite-13b-chat-v1"
    FLAN_T5_XL = "google/flan-t5-xl"
    GRANITE_13B_CHAT_V2 = "ibm/granite-13b-chat-v2"
    GRANITE_13B_INSTRUCT_V2 = "ibm/granite-13b-instruct-v2"
    ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT = "elyza/elyza-japanese-llama-2-7b-instruct"
    MIXTRAL_8X7B_INSTRUCT_V01_Q = "ibm-mistralai/mixtral-8x7b-instruct-v01-q"
    CODELLAMA_34B_INSTRUCT_HF = "codellama/codellama-34b-instruct-hf"
    GRANITE_20B_MULTILINGUAL = "ibm/granite-20b-multilingual"
    MERLINITE_7B = "ibm-mistralai/merlinite-7b"
    GRANITE_20B_CODE_INSTRUCT = "ibm/granite-20b-code-instruct"
    GRANITE_34B_CODE_INSTRUCT = "ibm/granite-34b-code-instruct"
    GRANITE_3B_CODE_INSTRUCT = "ibm/granite-3b-code-instruct"
    GRANITE_7B_LAB = "ibm/granite-7b-lab"
    GRANITE_8B_CODE_INSTRUCT = "ibm/granite-8b-code-instruct"
    LLAMA_3_70B_INSTRUCT = "meta-llama/llama-3-70b-instruct"
    LLAMA_3_8B_INSTRUCT = "meta-llama/llama-3-8b-instruct"
    MIXTRAL_8X7B_INSTRUCT_V01 = "mistralai/mixtral-8x7b-instruct-v01"


class DecodingMethods(Enum):
    """Supported decoding methods for text generation."""

    SAMPLE = "sample"
    GREEDY = "greedy"


class PromptTuningTypes:
    PT = "prompt_tuning"


class PromptTuningInitMethods:
    """Supported methods for prompt initialization in prompt tuning."""

    RANDOM = "random"
    TEXT = "text"
    # PRESET ?


class TuneExperimentTasks(Enum):
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    RETRIEVAL_AUGMENTED_GENERATION = "retrieval_augmented_generation"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    CODE_GENERATION_AND_CONVERSION = "code"
    EXTRACTION = "extraction"


class PromptTemplateFormats(Enum):
    """Supported formats of loaded prompt template."""

    PROMPTTEMPLATE = "prompt"
    STRING = "string"
    LANGCHAIN = "langchain"


class EmbeddingTypes(Enum):
    """
    .. deprecated:: 1.0.5
        Use :func:`EmbeddingModels` instead.

    Supported embedding models.

    .. note::
        You can check the current list of supported embeddings model types of various environments with
        :func:`get_embeddings_model_specs() <ibm_watsonx_ai.foundation_models_manager.FoundationModelsManager.get_embeddings_model_specs>`
        or by referring to the `watsonx.ai <https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models-embed.html?context=wx>`_
        documentation.
    """

    IBM_SLATE_30M_ENG = "ibm/slate-30m-english-rtrvr"
    IBM_SLATE_125M_ENG = "ibm/slate-125m-english-rtrvr"
