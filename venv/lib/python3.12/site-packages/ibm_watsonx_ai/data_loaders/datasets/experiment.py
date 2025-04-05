#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

__all__ = ["ExperimentIterableDataset"]

from ibm_watsonx_ai.data_loaders.datasets.tabular import TabularIterableDataset
from ibm_watsonx_ai.utils.autoai.enums import (
    SamplingTypes,
    DocumentsSamplingTypes,
)

DEFAULT_SAMPLE_SIZE_LIMIT = (
    1073741824  # 1GB in Bytes is verified later by _set_sample_size_limit
)
DEFAULT_REDUCED_SAMPLE_SIZE_LIMIT = 104857600  # 100MB in bytes
DEFAULT_SAMPLING_TYPE = SamplingTypes.FIRST_VALUES
DEFAULT_DOCUMENTS_SAMPLING_TYPE = DocumentsSamplingTypes.RANDOM

ExperimentIterableDataset = TabularIterableDataset
