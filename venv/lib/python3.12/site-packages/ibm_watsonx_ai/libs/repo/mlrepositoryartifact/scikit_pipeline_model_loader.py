#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from  ibm_watsonx_ai.libs.repo.mlrepositoryartifact.scikit_artifact_loader import ScikitArtifactLoader


class ScikitPipelineModelLoader(ScikitArtifactLoader):
    """
        Returns pipeline model instance associated with this model artifact.

        :return: pipeline model
        :rtype: scikit.learn.Pipeline
        """
    def model_instance(self,as_type=None):
        """
           :param as_type: string type referring to the model type to be returned.
           This parameter is applicable for xgboost models only.
           Currently accepts:
             'Booster': returns a model of type xgboost.Booster
             'XGBRegressor': returns a model of type xgboost.sklearn.XGBRegressor
           :return: returns a scikit model or an xgboost model of type xgboost.Booster
           or xgboost.sklearn.XGBRegressor
         """
        return self.load(as_type)
