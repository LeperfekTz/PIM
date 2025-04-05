#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import json
import os

from copy import deepcopy
from typing import TYPE_CHECKING, List, Union
from warnings import warn
from contextlib import redirect_stdout

from numpy import ndarray
from pandas import DataFrame

from ibm_watsonx_ai.helpers.connections import (
    DataConnection,
    S3Location,
    FSLocation,
    AssetLocation,
    ContainerLocation,
    DatabaseLocation,
)
from ibm_watsonx_ai.utils.autoai.enums import (
    RunStateTypes,
    PipelineTypes,
    TShirtSize,
    ClassificationAlgorithms,
    RegressionAlgorithms,
    DataConnectionTypes,
    ForecastingPipelineTypes,
    PredictionType,
    TimeseriesAnomalyPredictionPipelineTypes,
)
from ibm_watsonx_ai.utils.autoai.errors import (
    FitNotCompleted,
    MissingDataPreprocessingStep,
    DataSourceSizeNotSupported,
    TrainingDataSourceIsNotFile,
    NoneDataConnection,
    PipelineNotLoaded,
    ForecastingUnsupportedOperation,
    LibraryNotCompatible,
    InvalidDataAsset,
    TestDataNotPresent,
    FutureExogenousFeaturesNotSupported,
)
from ibm_watsonx_ai.utils.autoai.utils import try_import_lale, all_logging_disabled
from ibm_watsonx_ai.utils.autoai.connection import (
    validate_source_data_connections,
    validate_results_data_connection,
)
from ibm_watsonx_ai.utils import DisableWarningsLogger, WMLClientError
from ibm_watsonx_ai.messages.messages import Messages
from .base_auto_pipelines import BaseAutoPipelines
from ibm_watsonx_ai.experiment.autoai.engines import ServiceEngine

if TYPE_CHECKING:
    from ibm_watsonx_ai.experiment.autoai.engines import WMLEngine
    from ibm_watsonx_ai.utils.autoai.enums import Metrics, PredictionType, Transformers
    from sklearn.pipeline import Pipeline

__all__ = ["RemoteAutoPipelines"]


class RemoteAutoPipelines(BaseAutoPipelines):
    """RemoteAutoPipelines class for pipeline operation automation on Service.

    :param name: name for the AutoPipelines
    :type name: str

    :param prediction_type: type of the prediction
    :type prediction_type: PredictionType

    :param prediction_column: name of the target/label column
    :type prediction_column: str

    :param scoring: type of the metric to optimize with
    :type scoring: Metrics

    :param engine: engine for remote work on Service instance
    :type engine:  ServiceEngine or WMLEngine (deprecated)

    :param desc: description
    :type desc: str, optional

    :param holdout_size: percentage of the entire dataset to leave as a holdout, default 0.1
    :type holdout_size: float, optional

    :param max_num_daub_ensembles: maximum number (top-K ranked by DAUB model selection) of the selected algorithm,
        or estimator types, for example `LGBMClassifierEstimator`, `XGBoostClassifierEstimator`, or
        `LogisticRegressionEstimator` to use in pipeline composition, the default is `None` that means
        the true default value will be determined by the internal different algorithms, where only
        the highest ranked by model selection algorithm type is used
    :type max_num_daub_ensembles: int, optional

    :param train_sample_rows_test_size: training data sampling percentage
    :type train_sample_rows_test_size: float, optional

    :param include_only_estimators: list of estimators to include in computation process
    :type include_only_estimators: list[ClassificationAlgorithms or RegressionAlgorithms], optional

    :param cognito_transform_names: list of transformers to include in the feature enginnering computation process,
        see: AutoAI.Transformers
    :type cognito_transform_names: list[Transformers], optional

    :param csv_separator: the separator, or list of separators to try for separating columns in a CSV file,
        not used if the file_name is not a CSV file, default is ','
    :type csv_separator: list[str] or str, optional

    :param excel_sheet: name of the excel sheet to use, only use when xlsx file is an input,
        support for number of the sheet is deprecated, by default first sheet is used
    :type excel_sheet: str, optional

    :param encoding: encoding type for CSV training file
    :type encoding: str, optional

    :param positive_label: the positive class to report when binary classification, when multiclass or regression,
        this will be ignored
    :type positive_label: str, optional

    :param t_shirt_size: the size of the remote AutoAI POD instance (computing resources),
        only applicable to a remote scenario
    :type t_shirt_size: TShirtSize, optional

    :param time_ordered_data: defines user preference about time-based analise. If True, the analysis will
        consider the data as time-ordered and time-based. Supported only for regression.
    :type time_ordered_data: bool, optional

    :param feature_selector_mode: defines if feature selector should be triggered ["on", "off", "auto"],
                the "auto" mode analyzes the impact of removing insignificant features, if there is drop in accuracy,
                the PCA is applied to insignificant features, principal components describing variance in 30% or higher
                are selected in place of insignificant features and the model is evaluated again, if there is still drop
                in accuracy all features are used
                the "on" mode removes all insignificant features (0.0. importance), the feature selector is applied during
                cognito phase (applicable to pipelines with feature engineering stage)
    :type feature_selector_mode: str, optional
    """

    def __init__(
        self,
        name: str,
        prediction_type: "PredictionType",
        prediction_column: str,
        prediction_columns: List[str],
        timestamp_column_name: str,
        engine: Union["WMLEngine", "ServiceEngine"],
        scoring: "Metrics" = None,
        desc: str = None,
        holdout_size: float = None,
        max_num_daub_ensembles: int = None,
        t_shirt_size: "TShirtSize" = TShirtSize.M,
        train_sample_rows_test_size: float = None,
        include_only_estimators: List[
            Union["ClassificationAlgorithms", "RegressionAlgorithms"]
        ] = None,
        include_batched_ensemble_estimators: List[
            Union["BatchedClassificationAlgorithms", "BatchedRegressionAlgorithms"]
        ] = None,
        backtest_num: int = None,
        lookback_window: int = None,
        forecast_window: int = None,
        backtest_gap_length: int = None,
        cognito_transform_names: List["Transformers"] = None,
        csv_separator: Union[List[str], str] = ",",
        excel_sheet: Union[str, int] = None,
        encoding: str = "utf-8",
        positive_label: str = None,
        drop_duplicates: bool = True,
        outliers_columns: list = None,
        text_processing: bool = True,
        word2vec_feature_number: int = None,
        daub_give_priority_to_runtime: float = None,
        notebooks=False,
        autoai_pod_version=None,
        text_columns_names=None,
        n_parallel_data_connections=None,
        test_data_csv_separator: Union[List[str], str] = ",",
        test_data_excel_sheet: Union[str, int] = None,
        test_data_encoding: str = "utf-8",
        sampling_type=None,
        sample_size_limit=None,
        sample_rows_limit=None,
        sample_percentage_limit=None,
        number_of_batch_rows=None,
        categorical_imputation_strategy=None,
        numerical_imputation_strategy=None,
        numerical_imputation_value=None,
        imputation_threshold=None,
        fairness_info: dict = None,
        retrain_on_holdout: bool = True,
        feature_columns: List[str] = None,
        pipeline_types: List[
            Union[
                "ForecastingPipelineTypes", "TimeseriesAnomalyPredictionPipelineTypes"
            ]
        ] = None,
        supporting_features_at_forecast: bool = None,
        categorical_columns: list = None,
        numerical_columns: list = None,
        confidence_level: float = None,
        incremental_learning: bool = None,
        early_stop_enabled: bool = None,
        early_stop_window_size: int = None,
        time_ordered_data: bool = None,
        feature_selector_mode: str = None,
        **kwargs,
    ):

        # Deprecation of excel_sheet as number:
        if isinstance(excel_sheet, int) or isinstance(test_data_excel_sheet, int):
            excel_sheet_as_number_deprecated_warning = (
                "Support for excel sheet as number of the sheet (int) is deprecated! "
                "Please set excel sheet with name of the sheet."
            )
            warn(excel_sheet_as_number_deprecated_warning, category=DeprecationWarning)

        self.params = {
            "name": name,
            "desc": desc if desc else "",
            "prediction_type": (
                prediction_type if prediction_type != "timeseries" else "forecasting"
            ),
            "prediction_column": prediction_column,
            "prediction_columns": prediction_columns,
            "timestamp_column_name": timestamp_column_name,
            "scoring": scoring,
            "holdout_size": holdout_size,
            "max_num_daub_ensembles": max_num_daub_ensembles,
            "t_shirt_size": t_shirt_size,
            "train_sample_rows_test_size": train_sample_rows_test_size,
            "include_only_estimators": include_only_estimators,
            "include_batched_ensemble_estimators": include_batched_ensemble_estimators,
            "backtest_num": backtest_num,
            "lookback_window": lookback_window,
            "forecast_window": forecast_window,
            "backtest_gap_length": backtest_gap_length,
            "cognito_transform_names": cognito_transform_names,
            "csv_separator": csv_separator,
            "excel_sheet": excel_sheet,
            "encoding": encoding,
            "positive_label": positive_label,
            "drop_duplicates": drop_duplicates,
            "outliers_columns": outliers_columns,
            "notebooks": notebooks,
            "autoai_pod_version": autoai_pod_version,
            "text_processing": text_processing,
            "word2vec_feature_number": word2vec_feature_number,
            "daub_give_priority_to_runtime": daub_give_priority_to_runtime,
            "text_columns_names": text_columns_names,
            "sampling_type": sampling_type,
            "sample_size_limit": sample_size_limit,
            "sample_rows_limit": sample_rows_limit,
            "sample_percentage_limit": sample_percentage_limit,
            "number_of_batch_rows": number_of_batch_rows,
            "n_parallel_data_connections": n_parallel_data_connections,
            "test_data_csv_separator": test_data_csv_separator,
            "test_data_excel_sheet": test_data_excel_sheet,
            "test_data_encoding": test_data_encoding,
            "categorical_imputation_strategy": categorical_imputation_strategy,
            "numerical_imputation_strategy": numerical_imputation_strategy,
            "numerical_imputation_value": numerical_imputation_value,
            "imputation_threshold": imputation_threshold,
            "retrain_on_holdout": retrain_on_holdout,
            "feature_columns": feature_columns,
            "pipeline_types": pipeline_types,
            "supporting_features_at_forecast": supporting_features_at_forecast,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
            "confidence_level": confidence_level,
            "incremental_learning": incremental_learning,
            "early_stop_enabled": early_stop_enabled,
            "early_stop_window_size": early_stop_window_size,
            "time_ordered_data": time_ordered_data,
            "feature_selector_mode": feature_selector_mode,
        }

        if fairness_info:
            self.params["fairness_info"] = fairness_info

        self._engine: Union["WMLEngine", "ServiceEngine"] = engine
        self._engine.initiate_remote_resources(params=self.params, **kwargs)
        self.best_pipeline = None
        self._workspace = None

    def _get_engine(self) -> Union["WMLEngine", "ServiceEngine"]:
        """Return Engine for development purposes."""
        return self._engine

    ####################################################
    #   WML Pipeline Part / Parameters for AUtoAI POD  #
    ####################################################
    def get_params(self) -> dict:
        """Get configuration parameters of AutoPipelines.

        :return: AutoPipelines parameters
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.get_params()

            # Result:
            # {
            #     'name': 'test name',
            #     'desc': 'test description',
            #     'prediction_type': 'classification',
            #     'prediction_column': 'y',
            #     'scoring': 'roc_auc',
            #     'holdout_size': 0.1,
            #     'max_num_daub_ensembles': 1,
            #     't_shirt_size': 'm',
            #     'train_sample_rows_test_size': 0.8,
            #    'include_only_estimators': ["ExtraTreesClassifierEstimator",
            #                                "GradientBoostingClassifierEstimator",
            #                                "LGBMClassifierEstimator",
            #                                "LogisticRegressionEstimator",
            #                                "RandomForestClassifierEstimator",
            #                                "XGBClassifierEstimator"]
            # }
        """
        _params = self._engine.get_params().copy()
        del _params["autoai_pod_version"]
        del _params["notebooks"]

        return _params

    ###########################################################
    #   WML Training Part / Parameters for AUtoAI Experiment  #
    ###########################################################
    def fit(
        self,
        train_data: "DataFrame" = None,
        *,
        training_data_reference: List["DataConnection"] = None,
        training_results_reference: "DataConnection" = None,
        background_mode=False,
        test_data_references: List["DataConnection"] = None,
        training_data_references: List["DataConnection"] = None,
    ) -> dict:
        """Run a training process on Service of autoai on top of the training data referenced by DataConnection.

        :param training_data_reference: data storage connection details to inform where training data is stored,
            deprecated parameter, use `training_data_references` instead

        :param training_data_references: data storage connection details to inform where training data is stored,
            new version of `training_data_reference`
        :type training_data_references: list[DataConnection]

        :param training_results_reference: data storage connection details to store pipeline training results,
            not applicable on CP4D
        :type training_results_reference: DataConnection, optional

        :param background_mode: indicator if fit() method will run in background (async) or (sync)
        :type background_mode: bool, optional

        :param test_data_references: data storage connection details to inform where test / holdout data is stored
        :type test_data_references: list[DataConnection], optional

        :return: run details
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            from ibm_watsonx_ai.helpers import DataConnection, S3Location

            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.fit(
                training_data_reference=[DataConnection(
                    connection_asset_id=connection_id,
                    location=S3Location(
                        bucket='automl',
                        path='german_credit_data_biased_training.csv')
                    )
                )],
                DataConnection(
                    connection_asset_id=connection_id,
                    location=S3Location(
                        bucket='automl',
                        path='')
                    )
                ),
                background_mode=False)
        """
        if training_data_references is not None:
            training_data_reference = training_data_references

        if training_data_reference is None or not training_data_reference:
            raise NoneDataConnection("training_data_references")

        for conn in training_data_reference:
            if self._workspace.api_client.project_type == "local_git_storage":
                conn.location.userfs = "true"
            conn.set_client(self._workspace.api_client)
            # TODO: remove S3 implementation
            if conn.type == DataConnectionTypes.S3:
                conn._validate_cos_resource()

        training_data_reference = [
            new_conn
            for conn in training_data_reference
            for new_conn in conn._subdivide_connection()
        ]

        if isinstance(test_data_references, list):
            for conn in test_data_references:
                # Update test data ref with client object, experiment parameters
                if isinstance(conn, DataConnection):
                    if self._workspace.api_client.project_type == "local_git_storage":
                        conn.location.userfs = "true"
                    conn.set_client(self._workspace.api_client)
                    # TODO: remove S3 implementation
                    if conn.type == DataConnectionTypes.S3:
                        conn._validate_cos_resource()

                    conn.auto_pipeline_params = self._engine._auto_pipelines_parameters

        # note: update each training data connection with pipeline parameters for holdout split recreation
        for data_connection in training_data_reference:
            data_connection.auto_pipeline_params = (
                self._engine._auto_pipelines_parameters
            )

        if isinstance(train_data, DataFrame):
            if training_data_reference[0].type == "container":
                training_data_reference[0].write(
                    data=train_data,
                    remote_name=training_data_reference[0].location.path,
                )
            elif training_data_reference[0].type == "connection_asset":
                try:
                    # Cloud
                    training_data_reference[0].write(
                        data=train_data,
                        remote_name=training_data_reference[0].location.file_name,
                    )
                except AttributeError as e:
                    # CPD
                    raise WMLClientError(
                        "Writing `train_data` to the DataConnection object passed in .fit() function is not supported. Please upload the data first and then run training again, without `train_data` parameter."
                    ) from e

        elif train_data is None:
            pass

        else:
            raise TypeError("train_data should be of type pandas.DataFrame")

        # self._validate_training_data_size(training_data_reference)

        training_data_reference = validate_source_data_connections(
            training_data_reference, workspace=self._workspace, deployment=False
        )

        # note: for FSLocation we are creating asset and changing location to AssetLocation
        # so href is not set properly, setter on api_client will resolve that issue
        for conn in training_data_reference:
            conn.set_client(self._workspace.api_client)
        # --- end note

        training_results_reference = self.determine_result_reference(
            training_results_reference, training_data_reference, "default_autoai_out"
        )

        run_params = self._engine.fit(
            training_data_reference=training_data_reference,
            training_results_reference=training_results_reference,
            background_mode=background_mode,
            test_data_references=test_data_references,
        )

        for conn in training_data_reference:
            metrics = run_params["entity"]["status"].get("metrics", [])
            if metrics and metrics[-1]["context"].get("fairness"):
                conn.auto_pipeline_params["fairness_info"] = metrics[-1]["context"][
                    "fairness"
                ].get("info")

        return run_params

    def determine_result_reference(
        self, results_reference, data_references, result_path
    ):
        # note: if user did not provide results storage information, use default ones
        if results_reference is None:
            if (
                isinstance(data_references[0].location, S3Location)
                and not self._workspace.api_client.ICP_PLATFORM_SPACES
            ):
                results_reference = DataConnection(
                    connection=data_references[0].connection,
                    location=S3Location(
                        bucket=data_references[0].location.bucket, path="."
                    ),
                )

            elif (
                isinstance(data_references[0].location, AssetLocation)
                and not self._workspace.api_client.ICP_PLATFORM_SPACES
            ):
                connection_id = data_references[0].location._get_connection_id(
                    self._workspace.api_client
                )

                if connection_id is not None:
                    results_reference = DataConnection(
                        connection_asset_id=connection_id,
                        location=S3Location(
                            bucket=data_references[0].location._get_bucket(
                                self._workspace.api_client
                            ),
                            path=result_path,
                        ),
                    )

                else:  # set container output location when default DAta Asset is as a train ref
                    results_reference = DataConnection(
                        location=ContainerLocation(path=result_path)
                    )

            elif (
                isinstance(data_references[0].location, ContainerLocation)
                and not self._workspace.api_client.ICP_PLATFORM_SPACES
            ):
                results_reference = DataConnection(
                    location=ContainerLocation(path=result_path)
                )

            elif (
                isinstance(data_references[0].location, DatabaseLocation)
                and not self._workspace.api_client.ICP_PLATFORM_SPACES
            ):
                results_reference = DataConnection(
                    location=ContainerLocation(path=result_path)
                )

            else:
                location = FSLocation()
                client = (
                    self._engine._api_client
                    if isinstance(self._engine, ServiceEngine)
                    else self._engine._wml_client
                )
                if self._workspace.api_client.default_project_id is None:

                    location.path = location.path.format(
                        option="spaces", id=client.default_space_id
                    )

                else:
                    location.path = location.path.format(
                        option="projects", id=client.default_project_id
                    )
                results_reference = DataConnection(connection=None, location=location)
        elif getattr(results_reference, "type", False) == "fs":
            client = (
                self._engine._api_client
                if isinstance(self._engine, ServiceEngine)
                else self._engine._wml_client
            )
            if self._workspace.api_client.default_project_id is None:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="spaces", id=client.default_space_id
                    )
                )
            else:
                results_reference.location.path = (
                    results_reference.location.path.format(
                        option="projects", id=client.default_project_id
                    )
                )
        # -- end note
        if isinstance(results_reference.location, AssetLocation):
            if (
                results_reference.location._get_connection_id(
                    self._workspace.api_client
                )
                is None
            ):
                raise InvalidDataAsset(
                    reason="Please specify Data Asset pointing to connection e.g. COS as an output."
                )

        # note: results can be stored only on FS or COS
        if not isinstance(
            results_reference.location,
            (S3Location, FSLocation, AssetLocation, ContainerLocation),
        ):
            raise TypeError(
                "Unsupported results location type. Results referance can be stored"
                " only on S3Location or FSLocation or AssetLocation."
            )
        # -- end

        return results_reference

    #####################
    #   Run operations  #
    #####################
    def get_run_status(self) -> str:
        """Check status/state of initialized AutoPipelines run if ran in background mode.

        :return: run status details
        :rtype: dict

        **Example:**

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.get_run_status()

            # Result:
            # 'completed'
        """
        return self._engine.get_run_status()

    def get_run_details(self, include_metrics: bool = False, _internal=False) -> dict:
        """Get fit/run details.

        :param include_metrics: indicates to include metrics in the training details output
        :type include_metrics: bool, optional

        :return: AutoPipelineOptimizer fit/run details
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.get_run_details()
        """
        return self._engine.get_run_details(include_metrics=include_metrics)

    def cancel_run(self) -> None:
        """Cancels an AutoAI run."""
        self._engine.cancel_run()

    #################################
    #   Pipeline models operations  #
    #################################
    def summary(
        self, scoring: str = None, sort_by_holdout_score: bool = True
    ) -> "DataFrame":
        """Print AutoPipelineOptimizer Pipelines details (autoai trained pipelines).

        :param scoring: scoring metric which user wants to use to sort pipelines by,
            when not provided use optimized one
        :type scoring: string, optional

        :param sort_by_holdout_score: indicates if we want to sort pipelines by holdout metric or by training one,
            by default use holdout metric
        :type sort_by_holdout_score: bool, optional

        :return: computed pipelines and ML metrics
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.summary()

            # Result:
            #                training_normalized_gini_coefficient  ...  training_f1
            # Pipeline Name                                        ...
            # Pipeline_3                                 0.359173  ...     0.449197
            # Pipeline_4                                 0.359173  ...     0.449197
            # Pipeline_1                                 0.358124  ...     0.449057
            # Pipeline_2                                 0.358124  ...     0.449057
        """
        return self._engine.summary(scoring, sort_by_holdout_score)

    def get_pipeline_details(self, pipeline_name: str = None) -> dict:
        """Fetch specific pipeline details, eg. steps etc.

        :param pipeline_name: pipeline name eg. Pipeline_1, if not specified, best pipeline parameters will be fetched
        :type pipeline_name: str, optional

        :return: pipeline parameters
        :rtype: dict

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            remote_optimizer.get_pipeline_details()
            remote_optimizer.get_pipeline_details(pipeline_name='Pipeline_4')

            # Result:
            # {
            #     'composition_steps': ['TrainingDataset_full_4521_16', 'Split_TrainingHoldout',
            #                           'TrainingDataset_full_4068_16', 'Preprocessor_default', 'DAUB'],
            #     'pipeline_nodes': ['PreprocessingTransformer', 'GradientBoostingClassifierEstimator']
            # }
        """
        return self._engine.get_pipeline_details(pipeline_name=pipeline_name)

    def get_pipeline(
        self,
        pipeline_name: str = None,
        astype: "PipelineTypes" = PipelineTypes.LALE,
        persist: "bool" = False,
    ) -> Union["Pipeline", "TrainablePipeline"]:
        """Download specified pipeline from Service.

        :param pipeline_name: pipeline name, if you want to see the pipelines names, please use summary() method,
            if this parameter is None, the best pipeline will be fetched
        :type pipeline_name: str, optional

        :param astype: type of returned pipeline model, if not specified, lale type is chosen
        :type astype: PipelineTypes, optional

        :param persist: indicates if selected pipeline should be stored locally
        :type persist: bool, optional

        :return: Scikit-Learn pipeline
        :rtype: Pipeline or TrainablePipeline

        See also RemoteAutoPipelines.summary().

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            pipeline_1 = remote_optimizer.get_pipeline(pipeline_name='Pipeline_1')
            pipeline_2 = remote_optimizer.get_pipeline(pipeline_name='Pipeline_1', astype=AutoAI.PipelineTypes.LALE)
            pipeline_3 = remote_optimizer.get_pipeline(pipeline_name='Pipeline_1', astype=AutoAI.PipelineTypes.SKLEARN)
            type(pipeline_3)
            # <class 'sklearn.pipeline.Pipeline'>
            pipeline_4 = remote_optimizer.get_pipeline(pipeline_name='Pipeline_1', persist=True)
            # Selected pipeline stored under: "absolute_local_path_to_model/model.pickle"

        """
        try:
            if pipeline_name is None:
                pipeline_model, check_lale = self._engine.get_best_pipeline(
                    persist=persist
                )

            else:
                pipeline_model, check_lale = self._engine.get_pipeline(
                    pipeline_name=pipeline_name, persist=persist
                )

        except ForecastingUnsupportedOperation as e:
            raise e

        except LibraryNotCompatible as e:
            raise e

        except Exception as e:
            raise PipelineNotLoaded(
                pipeline_name if pipeline_name is not None else "best pipeline",
                reason=f"Pipeline with such a name probably does not exist. "
                f"Please make sure you specify correct pipeline name. Error: {e}",
            )

        if astype == PipelineTypes.SKLEARN:
            return pipeline_model

        elif astype == PipelineTypes.LALE:
            if check_lale:
                try_import_lale()
            from lale.helpers import import_from_sklearn_pipeline

            return import_from_sklearn_pipeline(pipeline_model)
        else:
            raise ValueError(
                "Incorrect value of 'astype'. "
                "Should be either PipelineTypes.SKLEARN or PipelineTypes.LALE"
            )

    def get_pipeline_notebook(
        self,
        pipeline_name: str = None,
        filename: str = None,
        insert_to_cell: bool = False,
    ) -> str:
        """Download specified pipeline notebook from Service.

        :param pipeline_name: pipeline name, if you want to see the pipelines names, please use summary() method,
            if this parameter is None, the best pipeline will be fetched
        :type pipeline_name: str, optional

        :param filename: filename under which the pipeline notebook will be saved
        :type filename: str, optional

        :param insert_to_cell: if run in jupyter notebook, if set inserts the notebook into cell below
        :type insert_to_cell: bool, optional

        :return: path to saved pipeline notebook
        :rtype: str

        See also RemoteAutoPipelines.summary().

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.experiment import AutoAI
            experiment = AutoAI(credentials, ...)
            remote_optimizer = experiment.optimizer(...)

            pipeline_notebook_path = remote_optimizer.get_pipeline_notebook(pipeline_name='Pipeline_1')

        """
        if pipeline_name is None:
            pipeline_name = self._engine.summary().index[0]

        path = self._engine.get_pipeline_notebook(pipeline_name, filename=filename)

        if insert_to_cell:
            with open(path, "r") as f:
                content = json.loads(f.read())

            def translate_cell_to_str(cell):
                if cell["cell_type"] == "code":
                    return "".join(cell["source"])
                elif cell["cell_type"] == "markdown":
                    return "\n".join(
                        ["# " + l for l in "".join(cell["source"]).split("\n")]
                    )
                else:
                    return ""

            result = "\n\n".join([translate_cell_to_str(c) for c in content["cells"]])

            import sys

            def get_optimizer_var_name():
                for name, module in sys.modules.items():
                    for varname, obj in module.__dict__.items():
                        if obj is self:
                            return varname

            if "'PUT_YOUR_APIKEY_HERE'" in result:
                try:
                    optimizer_var_name = get_optimizer_var_name()
                    result = result.replace(
                        "'PUT_YOUR_APIKEY_HERE'",
                        f"{optimizer_var_name}._workspace.credentials.api_key",
                    )

                except:
                    pass

            import IPython.core

            ipython = IPython.core.getipython.get_ipython()
            comment = "# generated by get_pipeline_notebook(insert_to_cell=True) from previous cell\n\n"
            ipython.set_next_input(comment + result, replace=False)

        return path

    # note: predict on top of the best computed pipeline, best pipeline is downloaded for the first time
    def predict(
        self,
        X: Union["DataFrame", "ndarray"] = None,
        observations: Union["DataFrame", "ndarray"] = None,
        supporting_features: Union["DataFrame", "ndarray"] = None,
    ) -> "ndarray":
        """Predict method called on top of the best fetched pipeline.

        :param X: test data for prediction
        :type X: numpy.ndarray or pandas.DataFrame

        :param observations: new observations of forecasting data that were used to train AutoAI model,
            supported only for forecasting pipelines
        :type observations: numpy.ndarray or pandas.DataFrame

        :param supporting_features: future values of exogenous features, supported only for forecasting pipelines
        :type supporting_features: numpy.ndarray or pandas.DataFrame

        :return: model predictions
        :rtype: numpy.ndarray
        """
        is_forecasting = self.params.get("prediction_type") in (
            PredictionType.FORECASTING,
            "timeseries",
            PredictionType.TIMESERIES_ANOMALY_PREDICTION,
        )

        if not is_forecasting and X is None:
            raise ValueError("X parameter is required in predict method.")
        if is_forecasting:
            if X is not None and observations is not None:
                raise ValueError(
                    "For forecasting the  parameter X should not be used. Please use observations instead of X."
                )
            else:
                X = observations if observations is not None else X

        if self.best_pipeline is None:
            # note: automatically download the best computed pipeline
            if self.get_run_status() == RunStateTypes.COMPLETED:
                self.best_pipeline, _ = self._engine.get_best_pipeline()
            else:
                raise FitNotCompleted(
                    self._engine._current_run_id,
                    reason="Please check the run status with run_status() method.",
                )
            # --- end note

        additional_params = {}
        if supporting_features is not None:
            if is_forecasting:
                additional_params.update(
                    {
                        "supporting_features": (
                            supporting_features
                            if isinstance(supporting_features, ndarray)
                            else supporting_features.values
                        )
                    }
                )
            else:
                raise FutureExogenousFeaturesNotSupported(
                    self.params.get("prediction_type")
                )

        if isinstance(X, DataFrame) or isinstance(X, ndarray):
            return self.best_pipeline.predict(
                X if isinstance(X, ndarray) else X.values, **additional_params
            )
        elif X is None and is_forecasting:
            return self.best_pipeline.predict()
        else:
            raise TypeError(
                "X should be either of type pandas.DataFrame or numpy.ndarray"
            )

    # --- end note

    def get_data_connections(self) -> List["DataConnection"]:
        """Create DataConnection objects for further user usage
            (eg. to handle data storage connection or to recreate autoai holdout split).

        :return: list of DataConnection with populated optimizer parameters
        :rtype: list[DataConnection]
        """

        optimizer_parameters = self.get_params()
        run_details = self.get_run_details(include_metrics=True, _internal=True)

        user_holdout_exists = (
            True if run_details["entity"].get("test_data_references") else False
        )

        training_data_references = run_details["entity"]["training_data_references"]

        data_connections = [
            DataConnection._from_dict(_dict=data_connection)
            for data_connection in training_data_references
        ]

        for (
            data_connection
        ) in data_connections:  # note: populate DataConnections with optimizer params
            data_connection.auto_pipeline_params = deepcopy(optimizer_parameters)
            data_connection.set_client(
                self._engine._api_client
                if isinstance(self._engine, ServiceEngine)
                else self._engine._wml_client
            )
            data_connection._run_id = self._engine._current_run_id
            data_connection._user_holdout_exists = user_holdout_exists

            metrics = run_details["entity"]["status"].get("metrics", [])
            if metrics and metrics[-1]["context"].get("fairness"):
                data_connection.auto_pipeline_params["fairness_info"] = metrics[-1][
                    "context"
                ]["fairness"].get("info")

        return data_connections

    def get_test_data_connections(self) -> List["DataConnection"]:
        """Create DataConnection objects for further user usage (To recreate autoai holdout that user specified).

        :return: list of DataConnection with populated optimizer parameters
        :rtype: list[DataConnection]
        """
        optimizer_parameters = self.get_params()
        run_details = self.get_run_details(_internal=True)

        if not run_details["entity"].get("test_data_references"):
            raise TestDataNotPresent(
                reason="User specified test data was not present in this experiment. "
                "Try to use 'with_holdout_split' parameter for original "
                "training_data_references to retrieve test data."
            )

        test_data_references = run_details["entity"]["test_data_references"]

        data_connections = [
            DataConnection._from_dict(_dict=data_connection)
            for data_connection in test_data_references
        ]

        for (
            data_connection
        ) in data_connections:  # note: populate DataConnections with optimizer params
            data_connection.auto_pipeline_params = deepcopy(optimizer_parameters)
            data_connection.set_client(
                self._engine._api_client
                if isinstance(self._engine, ServiceEngine)
                else self._engine._wml_client
            )
            data_connection._run_id = self._engine._current_run_id
            data_connection._test_data = True
            data_connection._user_holdout_exists = True

        return data_connections
