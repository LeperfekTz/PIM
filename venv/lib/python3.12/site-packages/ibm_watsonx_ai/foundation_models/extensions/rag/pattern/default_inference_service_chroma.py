#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


def inference_service(context):
    """
     Default inference AI service function.

    Input schema:
    payload = {
       "messages":[
            {
                "role":"user",
                "content":"question_1"
            }
        ]
    }

    Output schema:
    result = {
        'choices': [
            {
                'index': 0,
                'message': {
                    'content': 'generated_content',
                    'role': 'assistant'
                },
                "reference_documents" : [
                            {
                                'sequence_number': [1, 2, 3],
                                'document_id': '<document_id>'
                            }
                        ]
            }
        ]
    }
    """
    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.helpers.connections import DataConnection
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.extensions.rag import Retriever, VectorStore
    from ibm_watsonx_ai.data_loaders.datasets.documents import DocumentsIterableDataset
    from ibm_watsonx_ai.foundation_models.extensions.rag.chunker.langchain_chunker import (
        LangChainChunker,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.pattern.prompt_builder import (
        build_prompt,
    )
    from ibm_watsonx_ai.foundation_models.extensions.rag.utils import (
        get_max_input_tokens,
    )

    client = APIClient(
        credentials=Credentials(
            url=REPLACE_THIS_CODE_WITH_CREDENTIALS_URL,
            token=context.generate_token(),
            name=REPLACE_THIS_CODE_WITH_CREDENTIALS_NAME,
            iam_serviceid_crn=REPLACE_THIS_CODE_WITH_CREDENTIALS_IAM_SERVICEID_CRN,
            projects_token=REPLACE_THIS_CODE_WITH_CREDENTIALS_PROJECTS_TOKEN,
            username=REPLACE_THIS_CODE_WITH_CREDENTIALS_USERNAME,
            instance_id=REPLACE_THIS_CODE_WITH_CREDENTIALS_INSTANCE_ID,
            version=REPLACE_THIS_CODE_WITH_CREDENTIALS_VERSION,
            bedrock_url=REPLACE_THIS_CODE_WITH_CREDENTIALS_BEDROCK_URL,
            platform_url=REPLACE_THIS_CODE_WITH_CREDENTIALS_PLATFORM_URL,
            proxies=REPLACE_THIS_CODE_WITH_CREDENTIALS_PROXIES,
            verify=REPLACE_THIS_CODE_WITH_CREDENTIALS_VERIFY,
        ),
        space_id=REPLACE_THIS_CODE_WITH_API_CLIENT_SPACE_ID,
        project_id=REPLACE_THIS_CODE_WITH_API_CLIENT_PROJECT_ID,
    )

    vector_store = VectorStore.from_dict(
        client=client, data=REPLACE_THIS_CODE_WITH_VECTOR_STORE_CALL
    )

    # Building knowledge base
    input_data_references = REPLACE_THIS_CODE_WITH_INPUT_DATA_REFERENCES

    dataset = DocumentsIterableDataset(
        connections=input_data_references,
        enable_sampling=False,
        api_client=client,
    )
    chunker = LangChainChunker(
        method=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_METHOD,
        chunk_size=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_CHUNK_SIZE,
        chunk_overlap=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_CHUNK_OVERLAP,
        encoding_name=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_ENCODING_NAME,
        model_name=REPLACE_THIS_CODE_WITH_LANGCHAIN_CHUNKER_MODEL_NAME,
    )

    documents = chunker.split_documents(dataset)
    vector_store.add_documents(documents)

    retriever = Retriever.from_vector_store(
        vector_store=vector_store, init_parameters=REPLACE_THIS_CODE_WITH_RETRIEVER
    )
    model = ModelInference(
        api_client=client,
        model_id=REPLACE_THIS_CODE_WITH_MODEL_MODEL_ID,
        deployment_id=REPLACE_THIS_CODE_WITH_MODEL_DEPLOYMENT_ID,
        params=REPLACE_THIS_CODE_WITH_MODEL_PARAMS,
    )

    build_prompt_additional_kwargs = dict(
        model_max_input_tokens=get_max_input_tokens(
            model=model,
            default_max_sequence_length=REPLACE_THIS_CODE_WITH_DEFAULT_MAX_SEQUENCE_LENGTH,
        ),
        prompt_template_text=REPLACE_THIS_CODE_WITH_PROMPT_TEMPLATE_TEXT,
        context_template_text=REPLACE_THIS_CODE_WITH_CONTEXT_TEMPLATE_TEXT,
    )

    word_to_token_ratio = REPLACE_THIS_CODE_WITH_WORD_TO_TOKEN_RATIO
    if word_to_token_ratio is not None:
        build_prompt_additional_kwargs.update(
            {"word_to_token_ratio": word_to_token_ratio}
        )

    def validate_messages(messages: list[dict]):
        if len(messages) != 1 or messages[-1]["role"] != "user":
            raise ValueError(
                "The messages must be represented as a list of dictionaries, each containing a single user role."
            )

    def generate(context):

        client.set_token(context.get_token())
        payload = context.get_json()

        messages = payload["messages"]

        validate_messages(messages=messages)

        question = messages[-1]["content"]

        retrieved_docs = retriever.retrieve(query=question)
        reference_documents = [doc.page_content for doc in retrieved_docs]

        prompt_input_text = build_prompt(
            question=question,
            reference_documents=reference_documents,
            **build_prompt_additional_kwargs,
        )
        answer = model.generate_text(prompt=prompt_input_text)

        execute_response = {
            "body": {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "reference_documents": [
                            {
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                            for doc in retrieved_docs
                        ],
                    }
                ]
            }
        }

        return execute_response

    def generate_stream(context):

        client.set_token(context.get_token())
        payload = context.get_json()
        messages = payload["messages"]

        validate_messages(messages=messages)

        question = messages[-1]["content"]

        retrieved_docs = retriever.retrieve(query=question)
        reference_documents = [doc.page_content for doc in retrieved_docs]

        prompt_input_text = build_prompt(
            question=question,
            reference_documents=reference_documents,
            **build_prompt_additional_kwargs,
        )
        response_stream = model.generate_text_stream(
            prompt=prompt_input_text, raw_response=True
        )

        # First delta contains role type
        chunk = next(response_stream)
        message = {
            "role": "assistant",
            "content": chunk["results"][0]["generated_text"],
        }

        chunk_response = {
            "choices": [
                {
                    "index": 0,
                    "delta": message,
                    "reference_documents": [
                        {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in retrieved_docs
                    ],
                }
            ]
        }
        chunk_response["choices"][0]["finish_reason"] = chunk["results"][0][
            "stop_reason"
        ]

        yield chunk_response

        for chunk in response_stream:

            message = {
                "content": chunk["results"][0]["generated_text"],
            }

            chunk_response = {"choices": [{"index": 0, "delta": message}]}
            chunk_response["choices"][0]["finish_reason"] = chunk["results"][0][
                "stop_reason"
            ]

            yield chunk_response

    return generate, generate_stream
