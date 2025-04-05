#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging

from  ibm_watsonx_ai.libs.repo.mlrepository.artifact_reader import ArtifactReader

logger = logging.getLogger(__name__)


class RuntimesArtifactReader(ArtifactReader):
    def __init__(self, runtimespec_path):
        self.runtimespec_path = runtimespec_path

    def read(self):
        return self._open_stream()

    # This is a no. op. for RuntimeYmlFileReader as we do not want to delete the
    # archive file.
    def close(self):
        pass

    def _open_stream(self):
        return open(self.runtimespec_path, 'rt')
