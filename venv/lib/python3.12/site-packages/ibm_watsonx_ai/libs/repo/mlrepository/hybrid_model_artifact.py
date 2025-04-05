#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.libs.repo.mlrepository import ModelArtifact


class HybridModelArtifact(ModelArtifact):
    """
    Class representing Hybrid model artifact
    """
    def __init__(self, uid, name, meta_props):
        """
        Constructor for Hybrid model artifact
        :param uid: unique id for Hybrid model artifact
        :param name: name of the model
        :param metaprops: properties of the model and model artifact
        """
        super(HybridModelArtifact, self).__init__(uid, name, meta_props)
