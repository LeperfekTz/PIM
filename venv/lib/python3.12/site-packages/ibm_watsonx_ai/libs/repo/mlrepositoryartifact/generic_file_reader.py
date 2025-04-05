#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import os
import shutil
import logging

from ibm_watsonx_ai.libs.repo.mlrepository.artifact_reader import ArtifactReader
from ibm_watsonx_ai.libs.repo.util.compression_util import CompressionUtil
from ibm_watsonx_ai.libs.repo.util.unique_id_gen import uid_generate
from ibm_watsonx_ai.libs.repo.util.library_imports import LibraryChecker
from ibm_watsonx_ai.libs.repo.base_constants import *

logger = logging.getLogger(__name__)


class GenericFileReader(ArtifactReader):
    def __init__(self, compressed_archive_path):
        self.archive_path = compressed_archive_path

    def read(self):
        return self._open_stream()

    # This is a no. op. for GenericTarGZReader as we do not want to delete the
    # archive file.
    def close(self):
        pass

    def _open_stream(self):
        return open(self.archive_path, 'rb')
