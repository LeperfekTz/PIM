#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


def default_inference_function(params=None):
    """
    Default function used in RAGPattern when no ``inference_function`` is provided.

    Input schema:
    payload = {
        client.deployments.ScoringMetaNames.INPUT_DATA: [
            {
                "values": ["question 1", "question 2"],
                "access_token": "<bearer_token>"
            }
        ]
    }

    Output schema:
    result = {
        'predictions': [
            {
                'fields': ['answer', 'reference_documents'],
                'values': [
                    ['answer 1', [ {'page_content': 'page content 1',
                                    'metadata':     'metadata 1'} ]],
                    ['answer 2', [ {'page_content': 'page content 2',
                                    'metadata':     'metadata 2'} ]]
                ]
            }
        ]
    }
    """
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever, VectorStore
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.prompt_builder import (
        build_prompt,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.utils import (
        get_max_input_tokens,
    )
    from ibm_watsonx_ai.wml_client_error import MissingValue

    client = APIClient(
        Credentials.from_dict(params["credentials"]),
        space_id=params["space_id"],
        project_id=params["project_id"],
    )
    vector_store = VectorStore.from_dict(client=client, data=params["vector_store"])
    retriever = Retriever.from_vector_store(
        vector_store=vector_store, init_parameters=params["retriever"]
    )
    model = ModelInference(api_client=client, **params["model"])

    build_prompt_additional_kwargs = dict(
        model_max_input_tokens=get_max_input_tokens(model=model, params=params),
        prompt_template_text=params["prompt_template_text"],
        context_template_text=params["context_template_text"],
    )

    word_to_token_ratio = params.get("word_to_token_ratio")
    if word_to_token_ratio is not None:
        build_prompt_additional_kwargs.update(
            {"word_to_token_ratio": word_to_token_ratio}
        )

    def score(payload):
        input_data = payload[client.deployments.ScoringMetaNames.INPUT_DATA]
        access_token = input_data[0].get("access_token")
        questions = input_data[0].get("values")

        if access_token is None:
            raise MissingValue(
                value_name="access_token",
                reason="Access token is required in scoring payload.",
            )

        if isinstance(questions, str):
            questions = [questions]

        client.set_token(access_token)

        result = {"predictions": [{"fields": ["answer", "reference_documents"]}]}
        all_prompts = []
        all_retrieved_docs = []

        for question in questions:
            retrieved_docs = retriever.retrieve(query=question)
            all_retrieved_docs.append(retrieved_docs)
            reference_documents = [doc.page_content for doc in retrieved_docs]

            prompt_input_text = build_prompt(
                question=question,
                reference_documents=reference_documents,
                **build_prompt_additional_kwargs,
            )
            all_prompts.append(prompt_input_text)

        answers = model.generate_text(prompt=all_prompts)

        predictions = [
            [
                answer,
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in retrieved_docs
                ],
            ]
            for answer, retrieved_docs in zip(answers, all_retrieved_docs)
        ]

        result["predictions"][0]["values"] = predictions

        return result

    return score
