#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from ibm_watsonx_ai.foundation_models.extensions.rag import RAGPattern

# Defaults
WORD_TO_TOKEN_RATIO = 1.5


def estimate_tokens_count(
    text: str, word_to_token_ratio: float = WORD_TO_TOKEN_RATIO
) -> int:
    """Estimate the number of tokens in a given text.
    The token count is estimated using the number of words in the input text
    times a fixed factor estimating the number of tokens in a single word.

    :param text: the text to count the tokens for
    :type text: str

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :return: the count of the tokens in the text
    :rtype: int
    """
    words = text.split()
    return int(len(words) * word_to_token_ratio)


def build_prompt(
    prompt_template_text: str,
    context_template_text: str,
    question: str,
    reference_documents: list[str],
    model_max_input_tokens: int,
    word_to_token_ratio: float = WORD_TO_TOKEN_RATIO,
) -> str:
    """Build the input prompt from the prompt and context templates, and the inputs (question and reference documents).

    :param prompt_template_text: the text of the prompt template, used to create the RAG prompt
    :type prompt_template_text: str

    :param context_template_text: the text of the context template, used to format each reference document.
    :type context_template_text: str

    :param question: the question text that is to be part of the prompt
    :type question: str

    :param reference_documents: all the reference documents that are to be considered as part of the prompt.
        If the there are too many documents, or they are too long, the last documents will be omitted.
    :type reference_documents: list[str]

    :param model_max_input_tokens: the maximum number of input tokens supported by the model.
    :type model_max_input_tokens: int

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :return: the constructed prompt containing the instruction and model inputs (question and reference documents).
        The prompt length is under the maximal number of input tokens supported by the model (model_max_input_tokens).
        The prompt may contain only a subset of the reference documents, due to the limited input length.
    :rtype: str
    """
    if context_template_text:
        reference_documents = [
            context_template_text.format(document=reference_document)
            for reference_document in reference_documents
        ]

    selected_reference_documents = _select_reference_documents(
        prompt_template_text=prompt_template_text,
        question=question,
        reference_documents=reference_documents,
        model_max_input_tokens=model_max_input_tokens,
        word_to_token_ratio=word_to_token_ratio,
    )

    prompt_variables = {
        "question": question,
        "reference_documents": "\n".join(selected_reference_documents),
    }
    return prompt_template_text.format(**prompt_variables)


def _select_reference_documents(
    prompt_template_text: str,
    question: str,
    reference_documents: list[str],
    model_max_input_tokens: int,
    word_to_token_ratio: float = WORD_TO_TOKEN_RATIO,
) -> list[str]:
    """Select reference documents according to maximal number of input tokens supported by the model.
    Only using these selected references ensures that the constructed prompt fits (in terms of length) properly
    into the input window supported by the model.

    :param prompt_template_text: the text of the prompt template, used to create the RAG prompt
    :type prompt_template_text: str

    :param question: the question text that is to be part of the prompt
    :type question: str

    :param reference_documents: all the reference documents that are to be considered as part of the prompt.
        If the there are too many documents, or they are too long, the last documents will be omitted.
    :type reference_documents: list[str]

    :param model_max_input_tokens: the maximum number of input tokens supported by the model.
    :type model_max_input_tokens: int

    :param word_to_token_ratio: Constant representing the average number of tokens per word in a text, used for
        approximating the token count, defaults to 1.5
    :type word_to_token_ratio: float, optional

    :return: the reference documents that may be integrated into the prompt template, while maintaining
        the constraint on the model input window size.
    :rtype: list[str]
    """
    # The number of input tokens available after taking into account the prompt template
    # and the question
    available_input_tokens = (
        model_max_input_tokens
        - estimate_tokens_count(prompt_template_text, word_to_token_ratio)
        - estimate_tokens_count(question, word_to_token_ratio)
        # the placeholders will not be in the final prompt, so their token counts
        # should not be subtracted. count their token counts as available tokens.
        + estimate_tokens_count(RAGPattern.QUESTION_PLACEHOLDER, word_to_token_ratio)
        + estimate_tokens_count(
            RAGPattern.REFERENCE_DOCUMENTS_PLACEHOLDER, word_to_token_ratio
        )
    )

    selected_reference_documents = []
    for reference_document in reference_documents:
        # Select the current reference document if there are enough
        # available input tokens. Add +1 for the newline separator, used for
        # joining reference documents.
        tokens_required_for_reference_document = (
            estimate_tokens_count(reference_document, word_to_token_ratio) + 1
        )
        if tokens_required_for_reference_document <= available_input_tokens:
            available_input_tokens -= tokens_required_for_reference_document
            selected_reference_documents.append(reference_document)
        else:
            break

    return selected_reference_documents
