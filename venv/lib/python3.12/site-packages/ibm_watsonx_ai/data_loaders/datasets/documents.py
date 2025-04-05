#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = [
    "DocumentsIterableDataset",
]

import logging
import sys

from ibm_watsonx_ai.helpers import ContainerLocation

logger = logging.getLogger(__name__)

import os
from copy import copy

import pandas as pd
from typing import TYPE_CHECKING, Iterator, Any, Callable

from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.data_loaders.text_loader import (
    TextLoader,
    _asynch_download,
)
from ibm_watsonx_ai.helpers.remote_document import RemoteDocument
from ibm_watsonx_ai.utils.autoai.enums import DocumentsSamplingTypes, SamplingTypes
from ibm_watsonx_ai.utils.autoai.errors import InvalidSizeLimit

if TYPE_CHECKING:
    from ibm_watsonx_ai.helpers.connections import DataConnection

# Note: try to import torch lib if available, this fallback is done based on
# torch dependency removal request
try:
    from torch.utils.data import IterableDataset

except ImportError:
    IterableDataset: type = object  # type: ignore[no-redef]
# --- end note

DEFAULT_SAMPLE_SIZE_LIMIT = (
    1073741824  # 1GB in Bytes is verified later by _set_sample_size_limit
)
DEFAULT_SAMPLING_TYPE = SamplingTypes.FIRST_VALUES
DEFAULT_DOCUMENTS_SAMPLING_TYPE = DocumentsSamplingTypes.RANDOM


class DocumentsIterableDataset(IterableDataset):
    """
    This dataset is an Iterable stream of documents using an underneath Flight Service.
    It can download documents asynchronously and serve them to you from a generator.

    Supported types of documents:
        - **text/plain** (".txt" file extension) - plain structured text
        - **docx** (".docx" file extension) - standard Word style file
        - **pdf** (".pdf" file extension) - standard pdf document
        - **html** (".html" file extension) - saved html side
        - **markdown** (".md" file extension) - plain text formatted with markdown

    :param connections: list of connections to the documents
    :type connections: list[DataConnection]

    :param enable_sampling: if set to `True`, will enable sampling, default: True
    :type enable_sampling: bool

    :param sample_size_limit: upper limit for documents to be downloaded in bytes, default: 1 GB
    :type sample_size_limit: int

    :param sampling_type: a sampling strategy on how to read the data,
        check the `DocumentsSamplingTypes` enum class for more options
    :type sampling_type: str

    :param total_size_limit: upper limit for documents to be downloaded in Bytes, default: 1 GB,
        if more than one of: `total_size_limit`, `total_ndocs_limit` are set,
        then data are limited to the lower threshold.
    :type total_size_limit: int

    :param total_ndocs_limit: upper limit for documents to be downloaded in a number of rows,
        if more than one of: `total_size_limit`, `total_nrows_limit` are set,
        then data are limited to the lower threshold.
    :type total_ndocs_limit: int, optional

    :param benchmark_dataset: dataset of benchmarking data with IDs in the `document_ids` column corresponding
        to the names of documents in the `connections` list
    :type benchmark_dataset: pd.DataFrame, optional

    :param error_callback: error callback function, to handle the exceptions from document loading,
        as arguments are passed document_id and exception
    :type error_callback: function (str, Exception) -> None, optional

    :param api_client: initialized APIClient object with set project or space ID. If the DataConnection object in list
     connections does not have a set API client, then the api_client object is used for reading data.
    :type api_client: APIClient, optional


    **Example: default sampling - read up to 1 GB of random documents**

        .. code-block:: python

            connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

            iterable_dataset = DocumentsIterableDataset(connections=connections,
                                                        enable_sampling=True,
                                                        sampling_type='random',
                                                        sample_size_limit = 1GB)

    **Example: read all documents/no subsampling**

        .. code-block:: python

            connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

            iterable_dataset = DocumentsIterableDataset(connections=connections,
                                                        enable_sampling=False)

    **Example: context based sampling**

            .. code-block:: python

                connections = [DataConnection(data_asset_id='5d99c11a-2060-4ef6-83d5-dc593c6455e2')]

                iterable_dataset = DocumentsIterableDataset(connections=connections,
                                                            enable_sampling=True,
                                                            sampling_type='benchmark_driven',
                                                            sample_size_limit = 1GB,
                                                            benchmark_dataset=pd.DataFrame(
                                                                data={
                                                                    "question": [
                                                                        "What foundation models are available in watsonx.ai ?"
                                                                    ],
                                                                    "correct_answers": [
                                                                        [
                                                                            "The following models are available in watsonx.ai: ..."
                                                                        ]
                                                                    ],
                                                                    "correct_answer_document_ids": ["sample_pdf_file.pdf"],
                                                                }))

    """

    def __init__(
        self,
        *,
        connections: list[DataConnection],
        enable_sampling: bool = True,
        sample_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        sampling_type: str = DEFAULT_DOCUMENTS_SAMPLING_TYPE,
        total_size_limit: int = DEFAULT_SAMPLE_SIZE_LIMIT,
        total_ndocs_limit: int | None = None,
        benchmark_dataset: pd.DataFrame | None = None,
        error_callback: Callable[[str, Exception], None] = None,
        **kwargs: Any,
    ) -> None:
        from ibm_watsonx_ai.helpers import S3Location, AssetLocation, NFSLocation

        super().__init__()
        self.enable_sampling = enable_sampling
        self.sample_size_limit = sample_size_limit
        self.sampling_type = sampling_type
        self._set_size_limit(total_size_limit)
        self.total_ndocs_limit = total_ndocs_limit
        self.benchmark_dataset = benchmark_dataset
        self.error_callback = error_callback

        self._download_strategy = kwargs.get(
            "_download_strategy", "n_parallel"
        )  # expected values: "n_parallel", "sequential"

        api_client = kwargs.get("api_client", kwargs.get("_api_client"))

        if api_client is not None:
            for conn in connections:
                if conn._api_client is None:
                    conn.set_client(api_client)

        set_of_api_clients = set([conn._api_client for conn in connections])
        data_asset_id_name_mapping = {}

        if any([isinstance(conn.location, AssetLocation) for conn in connections]):

            for client in set_of_api_clients:
                for res in client.data_assets.get_details(get_all=True)["resources"]:
                    data_asset_id_name_mapping[res["metadata"]["asset_id"]] = res[
                        "metadata"
                    ]["resource_key"].split("/")[-1]

        def get_document_id(conn):
            if isinstance(conn.location, AssetLocation):
                if conn.location.id in data_asset_id_name_mapping:
                    return data_asset_id_name_mapping.get(conn.location.id)
                else:
                    raise WMLClientError(
                        f"The asset with id {conn.location.id} could not be found."
                    )
            else:
                return conn._get_filename()

        self.remote_documents = []

        for connection in connections:
            if isinstance(
                connection.location, (S3Location, ContainerLocation, NFSLocation)
            ):
                self.remote_documents.extend(
                    [
                        RemoteDocument(connection=c, document_id=get_document_id(c))
                        for c in connection._get_connections_from_folder()
                    ]
                )
            else:
                self.remote_documents.append(
                    RemoteDocument(
                        connection=connection, document_id=get_document_id(connection)
                    )
                )

        if len(set([doc.document_id for doc in self.remote_documents])) < len(
            self.remote_documents
        ):
            raise WMLClientError(
                "Not unique document file names passed in connections."
            )

    def _set_size_limit(self, size_limit: int) -> None:
        """If non-default value of total_size_limit was not passed,
        set Sample Size Limit based on T-Shirt size if code is run on training pod:
        For memory < 16 (T-Shirts: XS,S) default is 10MB,
        For memory < 32 & >= 16 (T-Shirts: M) default is 100MB,
        For memory = 32 (T-Shirt L) default is 0.7GB,
        For memory > 32 (T-Shirt XL) or runs outside pod default is 1GB.
        """
        self.total_size_limit: int | None
        from ibm_watsonx_ai.utils.autoai.connection import get_max_sample_size_limit

        max_tshirt_size_limit = (
            get_max_sample_size_limit() if os.getenv("MEM", False) else None
        )  # limit manual setting of sample size limit on autoai clusters #31527

        if self.enable_sampling:
            if max_tshirt_size_limit:
                if (
                    size_limit > max_tshirt_size_limit
                    and size_limit != DEFAULT_SAMPLE_SIZE_LIMIT
                ):
                    raise InvalidSizeLimit(size_limit, max_tshirt_size_limit)
                else:
                    self.total_size_limit = min(size_limit, max_tshirt_size_limit)
            else:
                self.total_size_limit = size_limit
        else:
            if size_limit == DEFAULT_SAMPLE_SIZE_LIMIT:
                self.total_size_limit = None  # do not limit reading if sampling is disabled, we want read all data
            else:
                self.total_size_limit = size_limit

    @staticmethod
    def _docs_context_sampling(
        remote_documents: list[RemoteDocument],
        benchmark_document_ids: list[str],
    ) -> list[RemoteDocument]:
        """Randomly sample documents from the benchmark set, then randomly from the rest up to a `size_upper_bound`.

        :param remote_documents: documents to sample from
        :type remote_documents: list[RemoteDocument]

        :param benchmark_document_ids: IDs of documents from the benchmark dataset
        :type benchmark_document_ids: list[str]

        :return: list of sampled documents
        :rtype: list[RemoteDocument]
        """
        sampled_documents = []
        benchmark_documents = [
            doc for doc in remote_documents if doc.document_id in benchmark_document_ids
        ]
        non_benchmark_documents = [
            doc
            for doc in remote_documents
            if doc.document_id not in benchmark_document_ids
        ]

        sampled_documents.extend(benchmark_documents)
        sampled_documents.extend(non_benchmark_documents)

        return sampled_documents

    @staticmethod
    def _docs_random_sampling(
        remote_documents: list[RemoteDocument],
    ) -> list[RemoteDocument]:
        """Randomly sample documents from `remote_documents` up to `size_upper_bound`.

        :param remote_documents: documents to sample from
        :type remote_documents: list[RemoteDocument]

        :return: list of sampled documents
        :rtype: list[RemoteDocument]
        """
        from random import shuffle

        sampling_order = list(range(len(remote_documents)))
        shuffle(sampling_order)

        return [remote_documents[i] for i in sampling_order]

    def __iter__(self) -> Iterator:
        """Iterate over documents."""
        size_limit = (
            self.sample_size_limit
            if self.sample_size_limit is not None and self.enable_sampling
            else self.total_size_limit
        )

        if self.enable_sampling:
            if self.sampling_type == DocumentsSamplingTypes.RANDOM:
                sampled_docs = self._docs_random_sampling(self.remote_documents)
            elif self.sampling_type == DocumentsSamplingTypes.BENCHMARK_DRIVEN:
                if self.benchmark_dataset is not None:
                    try:
                        benchmark_documents_ids = list(
                            set(
                                [
                                    y
                                    for x in self.benchmark_dataset[
                                        "correct_answer_document_ids"
                                    ].values
                                    for y in x
                                ]
                            )
                        )
                    except TypeError as e:
                        raise WMLClientError(
                            "Unable to collect `correct_answer_document_ids` from the benchmark dataset, "
                            f"due to invalid schema provided. Error: {e}"
                        )
                else:
                    raise ValueError(
                        "`benchmark_dataset` is mandatory for sample_type: DocumentsSamplingTypes.BENCHMARK_DRIVEN."
                    )

                sampled_docs = self._docs_context_sampling(
                    self.remote_documents, benchmark_documents_ids
                )
            else:
                raise ValueError(
                    f"Unsupported documents sampling type: {self.sampling_type}"
                )
        else:
            sampled_docs = copy(self.remote_documents)

        if self.total_ndocs_limit is not None:
            if len(sampled_docs) > self.total_ndocs_limit:
                logger.info(
                    f"Documents sampled with total_ndocs_limit param, "
                    + f"{len(sampled_docs[: self.total_ndocs_limit])} docs chosen "
                    + f"from {len(sampled_docs)} possible."
                )
            sampled_docs = sampled_docs[: self.total_ndocs_limit]

        match self._download_strategy:
            case "n_parallel":  # downloading documents entirely in parallel
                import multiprocessing.dummy as mp

                thread_no = min(5, len(sampled_docs))

                q_input = mp.Queue()
                qs_output = [mp.Queue() for _ in range(len(sampled_docs))]
                args = [(q_input, qs_output)] * thread_no

                for i, doc in enumerate(sampled_docs):
                    q_input.put((i, doc))

                with mp.Pool(thread_no) as pool:
                    from queue import Empty

                    pool.map_async(_asynch_download, args)

                    res_size = 0

                    for i in range(len(qs_output)):
                        try:
                            result = qs_output[i].get(timeout=10 * 60)
                        except Empty as e:
                            result = e

                        if isinstance(result, Exception):
                            e = result
                            doc_id = sampled_docs[i].document_id
                            logger.warning(f"Failed to download the file: `{doc_id}`")
                            if not isinstance(e, Empty):
                                logger.warning(e)
                            if self.error_callback:
                                self.error_callback(doc_id, e)
                        else:
                            doc = result
                            res_size += sys.getsizeof(doc.page_content)

                            if size_limit is not None and res_size > size_limit:
                                return

                            yield doc

            case _:  # "sequential" - simple sequential downloading
                res_size = 0
                for doc in sampled_docs:
                    try:
                        doc.download()

                        loaded_doc = TextLoader(doc).load()
                        res_size += len(loaded_doc.page_content.encode("utf-8"))

                        if size_limit is not None and res_size > size_limit:
                            return

                        yield loaded_doc
                    except Exception as e:
                        if self.error_callback:
                            self.error_callback(doc.document_id, e)
                        else:
                            raise e
