#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.libs.repo.mlrepository import MetaProps
from ibm_watsonx_ai.libs.repo.mlrepository.wml_function_artifact import WmlFunctionArtifact
from ibm_watsonx_ai.libs.repo.mlrepositoryartifact.function_artifact_reader import FunctionArtifactReader

class FunctionArtifact(WmlFunctionArtifact):
    """
    Class of  function artifacts created with MLRepositoryCLient.

    """
    def __init__(self,
                 function,
                 uid=None,
                 name=None,
                 meta_props=MetaProps({}),):

        super(FunctionArtifact, self).__init__(uid, name, meta_props)

        self.function = function

    def reader(self):
        """
        Returns reader used for getting archive model content.

        :return: reader for TensorflowPipelineModelArtifact.pipeline.Pipeline
        :rtype: TensorflowPipelineReader
        """
        try:
            return self._reader
        except:
            self._reader = FunctionArtifactReader(self.function)
            return self._reader

    def _copy(self, uid=None, meta_props=None):
        if uid is None:
            uid = self.uid

        if meta_props is None:
            meta_props = self.meta

        return FunctionArtifact(
            self.function,
            uid=uid,
            name=self.name,
            meta_props=meta_props
        )
