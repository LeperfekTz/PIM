#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
import os
import copy
import time
from warnings import warn, catch_warnings, simplefilter
import shutil
from typing import (
    TYPE_CHECKING,
    TypeAlias,
    Any,
    BinaryIO,
    Generator,
    overload,
    cast,
    Literal,
)

from ibm_watsonx_ai.libs.repo.mlrepositoryartifact import MLRepositoryArtifact
from ibm_watsonx_ai.libs.repo.mlrepository import MetaProps
import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.helpers import DataConnection

from ibm_watsonx_ai.utils import (
    MODEL_DETAILS_TYPE,
    load_model_from_directory,
    is_lale_pipeline,
)
from ibm_watsonx_ai.metanames import ModelMetaNames, LibraryMetaNames
from ibm_watsonx_ai.utils.autoai.utils import (
    download_request_json,
    load_file_from_file_system_nonautoai,
    init_cos_client,
    check_if_ts_pipeline_is_winner,
    prepare_auto_ai_model_to_publish_normal_scenario,
)

from ibm_watsonx_ai.utils.deployment.errors import ModelPromotionFailed, PromotionFailed
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ApiRequestFailure,
    UnexpectedType,
)
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.libs.repo.util.compression_util import CompressionUtil
from ibm_watsonx_ai.libs.repo.util.unique_id_gen import uid_generate
from ibm_watsonx_ai.href_definitions import (
    API_VERSION,
    SPACES,
    PIPELINES,
    LIBRARIES,
    RUNTIMES,
)
from ibm_watsonx_ai.utils.autoai.utils import (
    get_autoai_run_id_from_experiment_metadata,
    prepare_auto_ai_model_to_publish,
)
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid


if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.sw_spec import SpecStates
    import pandas
    import numpy
    import pyspark
    import requests as Requests

    LabelColumnNamesType: TypeAlias = (
        numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str]
    )
    TrainingDataType: TypeAlias = (
        pandas.DataFrame | numpy.ndarray | pyspark.sql.Dataframe | list
    )
    TrainingTargetType: TypeAlias = (
        pandas.DataFrame | pandas.Series | numpy.ndarray | list
    )
    FeatureNamesArrayType: TypeAlias = numpy.ndarray | list

PipelineType: TypeAlias = Any
MLModelType: TypeAlias = Any


class Models(WMLResource):
    """Store and manage models."""

    ConfigurationMetaNames = ModelMetaNames()
    """MetaNames for models creation."""

    LibraryMetaNames = LibraryMetaNames()
    """MetaNames for libraries creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

        if self._client.ICP_PLATFORM_SPACES:
            self.default_space_id = client.default_space_id

    def _save_library_archive(
        self, ml_pipeline: pyspark.ml.pipeline.PipelineModel
    ) -> str:

        id_length = 20
        gen_id = uid_generate(id_length)
        temp_dir_name = "{}".format("library" + gen_id)

        temp_dir = os.path.join(".", temp_dir_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        ml_pipeline.write().overwrite().save(temp_dir)
        archive_path = self._compress_artifact(temp_dir)
        shutil.rmtree(temp_dir)
        return archive_path

    def _compress_artifact(self, compress_artifact: str) -> str:
        tar_filename = "{}_content.tar".format("library")
        gz_filename = "{}.gz".format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, ".", tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _create_pipeline_input(
        self, lib_href: str, name: str, space_id: str | None = None
    ) -> dict[str, Any]:

        metadata = {
            self._client.pipelines.ConfigurationMetaNames.NAME: name
            + "_"
            + uid_generate(8),
            self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "dlaas_only",
                "pipelines": [
                    {
                        "id": "dlaas_only",
                        "runtime_ref": "spark",
                        "nodes": [
                            {
                                "id": "repository",
                                "type": "model_node",
                                "inputs": [],
                                "outputs": [],
                                "parameters": {
                                    "training_lib_href": (
                                        lib_href
                                        if self._client.ICP_PLATFORM_SPACES
                                        else f"{lib_href}/content"
                                    )
                                },
                            }
                        ],
                    }
                ],
            },
        }

        if space_id is not None:
            metadata.update(
                {self._client.pipelines.ConfigurationMetaNames.SPACE_ID: space_id}
            )

        if self._client.default_project_id is not None:
            metadata.update(
                {"project": {"href": "/v2/projects/" + self._client.default_project_id}}
            )
        return metadata

    def _tf2x_load_model_instance(self, model_id: str) -> Any:
        artifact_url = (
            self._client.service_instance._href_definitions.get_model_last_version_href(
                model_id
            )
        )
        params = self._client._params()
        id_length = 20
        gen_id = uid_generate(id_length)

        # Step1 :  Download the model content

        params.update({"content_format": "native"})
        artifact_content_url = str(artifact_url + "/download")
        r = requests.get(
            artifact_content_url,
            params=params,
            headers=self._client._get_headers(),
            stream=True,
        )

        if r.status_code != 200:
            raise ApiRequestFailure("Failure during {}.".format("downloading model"), r)

        downloaded_model = r.content
        self._logger.info(
            "Successfully downloaded artifact with artifact_url: {}".format(
                artifact_url
            )
        )

        # Step 2 :  copy the downloaded tar.gz in to a temp folder
        try:
            temp_dir_name = "{}".format("hdfs" + gen_id)

            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            gz_filename = temp_dir + "/download.tar.gz"
            tar_filename = temp_dir + "/download.tar"
            with open(gz_filename, "wb") as f:
                f.write(downloaded_model)

        except IOError as e:
            raise WMLClientError(
                "Saving model with artifact_url: '{}' failed.".format(model_id), e
            )

        # create model instance based on the type using load_model
        try:
            CompressionUtil.decompress_file_gzip(
                gzip_filepath=gz_filename, filepath=tar_filename
            )
            CompressionUtil.extract_tar(tar_filename, temp_dir)
            os.remove(tar_filename)
            import tensorflow as tf
            import glob

            h5format = True
            if not glob.glob(temp_dir + "/sequential_model.h5"):
                h5format = False
            if h5format is True:
                model_instance = tf.keras.models.load_model(
                    temp_dir + "/sequential_model.h5", custom_objects=None, compile=True
                )
                return model_instance
            elif glob.glob(temp_dir + "/saved_model.pb"):
                model_instance = tf.keras.models.load_model(
                    temp_dir, custom_objects=None, compile=True
                )
                return model_instance
            else:
                raise WMLClientError(
                    "Load model with artifact_url: '{}' failed.".format(model_id)
                )

        except IOError as e:
            raise WMLClientError(
                "Saving model with artifact_url: '{}' failed.".format(model_id), e
            )

    def _publish_from_object(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )
        if (
            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
            and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. It is mandatory to provide RUNTIME_ID or "
                "SOFTWARE_SPEC_ID in meta_props. RUNTIME_ID is deprecated"
            )
        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            runtime_id_deprecated_warning = (
                "RUNTIME_ID is deprecated and will be removed in future. "
                "Instead, please use SOFTWARE_SPEC_ID."
            )
            warn(runtime_id_deprecated_warning, category=DeprecationWarning)
        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                if pipeline is None or training_data is None:
                    raise WMLClientError(
                        "pipeline and training_data are expected for spark models."
                    )
                name = meta_props[self.ConfigurationMetaNames.NAME]
                version = "1.0"
                platform = {"name": "python", "versions": ["3.6"]}
                library_tar = self._save_library_archive(pipeline)
                model_definition_props = {
                    self._client.model_definitions.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                    self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                }
                training_lib = self._client.model_definitions.store(
                    library_tar, model_definition_props
                )
                lib_href = self._client.model_definitions.get_href(training_lib)
                lib_href = lib_href.split("?", 1)[0]  # temp fix for stripping space_id

                pipeline_metadata = self._create_pipeline_input(
                    lib_href, name, space_id=None
                )
                pipeline_save = self._client.pipelines.store(pipeline_metadata)

                pipeline_href = self._client.pipelines.get_href(pipeline_save)

                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "href": pipeline_href
                }

                if (
                    self.ConfigurationMetaNames.SPACE_ID in meta_props
                    and meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                    is not None
                ):
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                        "href": API_VERSION
                        + SPACES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                    }
                else:
                    if self._client.default_project_id is not None:
                        meta_props["project"] = {
                            "href": "/v2/projects/" + self._client.default_project_id
                        }

                if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.RUNTIME_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID] = {
                        "href": API_VERSION
                        + RUNTIMES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID]
                    }

                if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                    meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_ID)

                if self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.TRAINING_LIB_ID,
                        str,
                        False,
                    )
                    meta_props[
                        self._client.repository.ModelMetaNames.TRAINING_LIB_ID
                    ] = {
                        "href": API_VERSION
                        + LIBRARIES
                        + "/"
                        + meta_props[
                            self._client.repository.ModelMetaNames.TRAINING_LIB_ID
                        ]
                    }

                meta_data = MetaProps(meta_props)

                model_artifact = MLRepositoryArtifact(
                    model,
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    training_data=training_data,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                if (
                    self.ConfigurationMetaNames.SPACE_ID in meta_props
                    and meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                    is not None
                ):
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                        "href": API_VERSION
                        + SPACES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                    }
                if self._client.ICP_PLATFORM_SPACES:
                    if self._client.default_space_id is not None:
                        meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                            "href": API_VERSION
                            + SPACES
                            + "/"
                            + self._client.default_space_id
                        }
                    else:
                        if self._client.default_project_id is not None:
                            meta_props["project"] = {
                                "href": "/v2/projects/"
                                + self._client.default_project_id
                            }
                        else:
                            raise WMLClientError(
                                "It is mandatory is set the space or Project. \
                             Use client.set.default_space(<SPACE_ID>) to set the space or Use client.set.default_project(<PROJECT_ID)"
                            )

                if self._client.default_space_id is not None:
                    meta_props["space_id"] = self._client.default_space_id
                else:
                    if self._client.default_project_id is not None:
                        meta_props["project_id"] = self._client.default_project_id
                    else:
                        raise WMLClientError(
                            "It is mandatory is set the space or Project. \
                            Use client.set.default_space(<SPACE_ID>) to set the space or"
                            " Use client.set.default_project(<PROJECT_ID)"
                        )

                if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.RUNTIME_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID] = {
                        "href": API_VERSION
                        + RUNTIMES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID]
                    }
                if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                    if self._client.CPD_version:
                        self._validate_meta_prop(
                            meta_props,
                            self.ConfigurationMetaNames.SOFTWARE_SPEC_ID,
                            str,
                            False,
                        )
                        meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                            "id": meta_props[
                                self.ConfigurationMetaNames.SOFTWARE_SPEC_ID
                            ]
                        }
                    else:
                        meta_props.pop(self.ConfigurationMetaNames.SOFTWARE_SPEC_ID)

                if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                        "href": API_VERSION
                        + PIPELINES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
                    }

                if self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.TRAINING_LIB_ID,
                        str,
                        False,
                    )
                    meta_props[
                        self._client.repository.ModelMetaNames.TRAINING_LIB_ID
                    ] = {
                        "href": API_VERSION
                        + LIBRARIES
                        + "/"
                        + meta_props[
                            self._client.repository.ModelMetaNames.TRAINING_LIB_ID
                        ]
                    }

                meta_data = MetaProps(meta_props)
                model_artifact = MLRepositoryArtifact(
                    model,
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    training_data=training_data,
                    training_target=training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            if self._client.ICP_PLATFORM_SPACES:
                query_param_for_repo_client = self._client._params()

            else:
                query_param_for_repo_client = None
            saved_model = self._client.repository._ml_repository_client.models.save(
                model_artifact, query_param=query_param_for_repo_client
            )
        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)
        else:
            return self.get_details("{}".format(saved_model.uid))

    def _get_subtraining_object(
        self, trainingobject: dict[str, Any], subtrainingId: str
    ) -> dict[str, Any]:
        subtrainings = trainingobject["resources"]
        for each_training in subtrainings:
            if each_training["metadata"]["guid"] == subtrainingId:
                return each_training
        raise WMLClientError("The subtrainingId " + subtrainingId + " is not found")

    ##TODO not yet supported

    def _publish_from_training(
        self,
        model_id: str,
        subtrainingId: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        round_number: int | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""

        model_meta = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )

        try:
            details = self._client.training.get_details(model_id, _internal=True)

        except ApiRequestFailure as e:
            raise UnexpectedType(
                "model parameter", "model path / training_id", model_id
            )
        model_type = ""

        ##Check if the training is created from pipeline or experiment
        if "pipeline" in details["entity"]:
            pipeline_id = details["entity"]["pipeline"]["id"]
            if "model_type" in details["entity"]["pipeline"]:
                model_type = details["entity"]["pipeline"]["model_type"]

        if "experiment" in details["entity"]:
            url = self._credentials.url + "/ml/v4/trainings?parent_id=" + model_id

            details_parent = requests.get(
                url, params=self._client._params(), headers=self._client._get_headers()
            )
            details_json = self._handle_response(
                200, "Get training details", details_parent
            )
            subtraining_object = self._get_subtraining_object(
                details_json, subtrainingId
            )
            model_meta.update(
                {"import": subtraining_object["entity"]["results_reference"]}
            )
            if "pipeline" in subtraining_object["entity"]:
                pipeline_id = subtraining_object["entity"]["pipeline"]["id"]
                if "model_type" in subtraining_object["entity"]["pipeline"]:
                    model_type = subtraining_object["entity"]["pipeline"]["model_type"]
        else:
            model_meta.update({"import": details["entity"]["results_reference"]})

        if "pipeline" in details["entity"] or "experiment" in details["entity"]:
            if "experiment" in details["entity"]:
                url = self._credentials.url + "/ml/v4/trainings?parent_id=" + model_id

                details_parent = requests.get(
                    url,
                    params=self._client._params(),
                    headers=self._client._get_headers(),
                )
                details_json = self._handle_response(
                    200, "Get training details", details_parent
                )
                subtraining_object = self._get_subtraining_object(
                    details_json, subtrainingId
                )
                if "pipeline" in subtraining_object["entity"]:
                    definition_details = self._client.pipelines.get_details(pipeline_id)
                    runtime_id = (
                        definition_details["entity"]["document"]["runtimes"][0]["name"]
                        + "_"
                        + definition_details["entity"]["document"]["runtimes"][0][
                            "version"
                        ].split("-")[0]
                        + "-py3"
                    )
                    if model_type == "":
                        model_type = (
                            definition_details["entity"]["document"]["runtimes"][0][
                                "name"
                            ]
                            + "_"
                            + definition_details["entity"]["document"]["runtimes"][0][
                                "version"
                            ].split("-")[0]
                        )

                    if self.ConfigurationMetaNames.TYPE not in meta_props:
                        model_meta.update({"type": model_type})

                    if self.ConfigurationMetaNames.RUNTIME_ID not in meta_props:
                        model_meta.update(
                            {"runtime": {"href": "/v4/runtimes/" + runtime_id}}
                        )
            else:
                definition_details = self._client.pipelines.get_details(pipeline_id)
                if model_type == "":
                    model_type = (
                        definition_details["entity"]["document"]["runtimes"][0]["name"]
                        + "_"
                        + definition_details["entity"]["document"]["runtimes"][0]
                        .get("version", "0.1")
                        .split("-")[0]
                    )

                if self.ConfigurationMetaNames.TYPE not in meta_props:
                    model_meta.update({"type": model_type})

        if label_column_names:
            model_meta["label_column"] = label_column_names[0]

        if (
            details.get("entity").get("status").get("state") == "failed"
            or details.get("entity").get("status").get("state") == "pending"
        ):
            raise WMLClientError(
                "Training is not successfully completed for the given training_id. Please check the status of training run. Training should be completed successfully to store the model"
            )

        model_dir = model_id
        if "federated_learning" in details["entity"] and round_number is not None:
            if (
                not details.get("entity")
                .get("federated_learning")
                .get("save_intermediate_models", False)
            ):
                raise WMLClientError(
                    "The Federated Learning experiment was not configured to save intermediate models"
                )

            rounds = details.get("entity").get("federated_learning").get("rounds")
            if (
                isinstance(round_number, int)
                and 0 < round_number
                and round_number <= rounds
            ):
                if round_number < rounds:
                    # intermediate models
                    model_dir = model_dir + "_" + str(round_number)
            else:
                raise WMLClientError(
                    "Invalid input. round_number should be an int between 1 and {}".format(
                        rounds
                    )
                )

        asset_url = (
            details["entity"]["results_reference"]["location"]["assets_path"]
            + "/"
            + model_dir
            + "/resources/wml_model/request.json"
        )

        if self._client.ICP_PLATFORM_SPACES:
            try:
                asset_parts = asset_url.split("/")
                asset_url = "/".join(asset_parts[asset_parts.index("assets") + 1 :])
                request_str = (
                    load_file_from_file_system_nonautoai(
                        api_client=self._client, file_path=asset_url
                    )
                    .read()
                    .decode()
                )

                import json

                if json.loads(request_str).get("code") == 404:
                    raise Exception("Not found file.")
            except Exception:
                asset_url = (
                    "trainings/"
                    + model_id
                    + "/assets/"
                    + model_dir
                    + "/resources/wml_model/request.json"
                )
                request_str = (
                    load_file_from_file_system_nonautoai(
                        api_client=self._client, file_path=asset_url
                    )
                    .read()
                    .decode()
                )
        else:
            if len(details["entity"]["results_reference"]["connection"]) > 1:
                cos_client = init_cos_client(
                    details["entity"]["results_reference"]["connection"]
                )
                bucket = details["entity"]["results_reference"]["location"]["bucket"]
            else:

                results_reference = DataConnection._from_dict(
                    details["entity"]["results_reference"]
                )
                results_reference.set_client(self._client)
                results_reference._check_if_connection_asset_is_s3()
                results_reference = results_reference._to_dict()
                cos_client = init_cos_client(results_reference["connection"])
                bucket = results_reference["location"]["bucket"]
            cos_client.meta.client.download_file(
                Bucket=bucket, Filename="request.json", Key=asset_url
            )
            with open("request.json", "r") as f:
                request_str = f.read()
        import json

        request_json: dict[str, dict] = json.loads(request_str)
        request_json["name"] = meta_props[self.ConfigurationMetaNames.NAME]
        request_json["content_location"]["connection"] = details["entity"][
            "results_reference"
        ]["connection"]
        if "space_id" in model_meta:
            request_json["space_id"] = model_meta["space_id"]

        else:
            request_json["project_id"] = model_meta["project_id"]

        if "label_column" in model_meta:
            request_json["label_column"] = model_meta["label_column"]

        if "pipeline" in request_json:
            request_json.pop("pipeline")  # not needed for other space
        if "training_data_references" in request_json:
            request_json.pop("training_data_references")
        if "software_spec" in request_json:
            request_json.pop("software_spec")
            request_json.update(
                {
                    "software_spec": {
                        "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                    }
                }
            )

        params = {}
        params.update({"version": self._client.version_param})
        creation_response = requests.post(
            self._credentials.url + "/ml/v4/models",
            headers=self._client._get_headers(),
            json=request_json,
            params=params,
        )
        model_details = self._handle_response(
            202, "creating new model", creation_response
        )
        model_id = model_details["metadata"]["id"]
        return self.get_details(model_id)

    def _store_autoAI_model(
        self,
        model_path: str,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store trained model from object storage into Watson Machine Learning repository on IBM Cloud."""
        model_meta = self.ConfigurationMetaNames._generate_resource_metadata(
            meta_props, client=self._client
        )
        # For V4 cloud prepare the metadata
        if "autoai_sdk" in model_path:
            input_payload = meta_props

        else:
            input_payload = copy.deepcopy(
                self._create_cloud_model_payload(
                    model_meta,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            )

        params = {}
        params.update({"version": self._client.version_param})
        url = self._credentials.url + "/ml/v4/models"

        if label_column_names:
            input_payload["label_column"] = label_column_names[0]

        creation_response = requests.post(
            url, params=params, headers=self._client._get_headers(), json=input_payload
        )
        if creation_response.status_code == 201:
            model_details = self._handle_response(
                201, "creating new model", creation_response
            )
        else:
            model_details = self._handle_response(
                202, "creating new model", creation_response
            )
        model_id = model_details["metadata"]["id"]

        if "entity" in model_details:
            start_time = time.time()
            elapsed_time = 0.0
            while (
                model_details["entity"].get("content_import_state") == "running"
                and elapsed_time < 60
            ):
                time.sleep(2)
                elapsed_time = time.time() - start_time
                model_details = self.get_details(model_id)

        return self.get_details(model_id)

    def _publish_from_file(
        self,
        model: str,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        ver: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """Store saved model into Watson Machine Learning repository on IBM Cloud."""
        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props:
            raise WMLClientError(
                "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in metaprop."
            )

        if ver == True:
            # check if artifactid is passed
            Models._validate_type(artifactid, "artifactid", str, True)
            return self._publish_from_archive(
                model,
                meta_props,
                version=ver,
                artifactid=artifactid,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )

        def is_xml(model_filepath: str) -> bool:
            if os.path.splitext(os.path.basename(model_filepath))[-1] == ".pmml":
                raise WMLClientError(
                    "The file name has an unsupported extension. Rename the file with a .xml extension."
                )
            return os.path.splitext(os.path.basename(model_filepath))[-1] == ".xml"

        def is_h5(model_filepath: str) -> bool:
            return os.path.splitext(os.path.basename(model_filepath))[-1] == ".h5"

        def is_json(model_filepath: str) -> bool:
            return os.path.splitext(os.path.basename(model_filepath))[-1] == ".json"

        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        import tarfile
        import zipfile

        model_filepath = model
        if os.path.isdir(model):
            # TODO this part is ugly, but will work. In final solution this will be removed
            if "tensorflow" in meta_props[self.ConfigurationMetaNames.TYPE]:
                # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                if os.path.basename(model) == "":
                    model = os.path.dirname(model)
                filename = os.path.basename(model) + ".tar.gz"
                current_dir = os.getcwd()
                os.chdir(model)
                target_path = os.path.dirname(model)

                with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
                    tar.add(".")

                os.chdir(current_dir)
                model_filepath = os.path.join(target_path, filename)
                if (
                    tarfile.is_tarfile(model_filepath)
                    or zipfile.is_zipfile(model_filepath)
                    or is_xml(model_filepath)
                ):
                    return self._publish_from_archive(
                        model_filepath,
                        meta_props,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                else:
                    raise WMLClientError(
                        "Invalid model file path  specified for: '{}'.".format(
                            meta_props[self.ConfigurationMetaNames.TYPE]
                        )
                    )
            else:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.TYPE, str, True
                )
                if "caffe" in meta_props[self.ConfigurationMetaNames.TYPE]:
                    raise WMLClientError(
                        "Invalid model file path  specified for: '{}'.".format(
                            meta_props[self.ConfigurationMetaNames.TYPE]
                        )
                    )

                loaded_model = load_model_from_directory(
                    meta_props[self.ConfigurationMetaNames.TYPE], model
                )
                if self._client.CLOUD_PLATFORM_SPACES:
                    saved_model = self._publish_from_object_cloud(
                        loaded_model,
                        meta_props,
                        training_data,
                        training_target,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                else:
                    saved_model = self._publish_from_object(
                        loaded_model,
                        meta_props,
                        training_data,
                        training_target,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                return saved_model

        elif is_xml(model_filepath):
            try:
                # New V4 cloud flow
                input_meta_data = copy.deepcopy(
                    self._create_cloud_model_payload(
                        meta_props,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )
                )
                meta_data = MetaProps(input_meta_data)

                model_artifact = MLRepositoryArtifact(
                    str(model_filepath),
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )

                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param_for_repo_client
                )
            except Exception as e:
                raise WMLClientError("Publishing model failed.", e)
            else:
                return self.get_details(saved_model.uid)  # type: ignore[attr-defined]

        elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(model_filepath):
            return self._publish_from_archive(
                model,
                meta_props,
                feature_names=feature_names,
                label_column_names=label_column_names,
            )
        elif (
            is_json(model_filepath)
            and self.ConfigurationMetaNames.TYPE in meta_props
            and meta_props[self.ConfigurationMetaNames.TYPE]
            in [f"xgboost_{version}" for version in ("1.3", "1.5")]
        ):

            # validation
            with open(model, "r") as file:
                try:
                    import json

                    json.loads(file.read())
                except Exception:
                    raise WMLClientError(
                        "Json file has invalid content. Please, validate if it was generated with xgboost>=1.3."
                    )

            output_filename = model.replace(".json", ".tar.gz")

            try:
                with tarfile.open(output_filename, "w:gz") as tar:
                    tar.add(model, arcname=os.path.basename(model))
                return self._publish_from_archive(
                    output_filename,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            finally:
                os.remove(output_filename)
        else:
            raise WMLClientError(
                "Saving trained model in repository failed. '{}' file does not have valid format".format(
                    model_filepath
                )
            )

    def _create_model_payload(
        self,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        payload = {
            "name": meta_props[self.ConfigurationMetaNames.NAME],
        }
        if self.ConfigurationMetaNames.TAGS in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TAGS, list, False
            )
            payload.update(
                {
                    self.ConfigurationMetaNames.TAGS: meta_props[
                        self.ConfigurationMetaNames.TAGS
                    ]
                }
            )
        if (
            self.ConfigurationMetaNames.SPACE_ID in meta_props
            and meta_props[self._client.repository.ModelMetaNames.SPACE_ID] is not None
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
            )
            payload.update(
                {
                    self.ConfigurationMetaNames.SPACE_ID: {
                        "href": API_VERSION
                        + SPACES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                    }
                }
            )

        if self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                meta_props[self._client.repository.ModelMetaNames.SPACE_ID] = {
                    "href": API_VERSION + SPACES + "/" + self._client.default_space_id
                }
            else:
                if self._client.default_project_id is not None:
                    payload.update(
                        {
                            "project": {
                                "href": "/v2/projects/"
                                + self._client.default_project_id
                            }
                        }
                    )
                else:
                    raise WMLClientError(
                        "It is mandatory is set the space. Use client.set.default_space(<SPACE_ID>) to set the space."
                    )

        if self.ConfigurationMetaNames.RUNTIME_ID in meta_props:
            payload.update(
                {
                    self.ConfigurationMetaNames.RUNTIME_ID: {
                        "href": API_VERSION
                        + RUNTIMES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.RUNTIME_ID]
                    }
                }
            )

        if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
            )
            payload.update(
                {
                    self.ConfigurationMetaNames.PIPELINE_ID: {
                        "href": API_VERSION
                        + PIPELINES
                        + "/"
                        + meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID]
                    }
                }
            )

        if self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TRAINING_LIB_ID, str, False
            )
            payload.update(
                {
                    self.ConfigurationMetaNames.TRAINING_LIB_ID: {
                        "href": API_VERSION
                        + LIBRARIES
                        + "/"
                        + meta_props[
                            self._client.repository.ModelMetaNames.TRAINING_LIB_ID
                        ]
                    }
                }
            )

        if self.ConfigurationMetaNames.DESCRIPTION in meta_props:
            payload.update(
                {"description": meta_props[self.ConfigurationMetaNames.DESCRIPTION]}
            )

        if self.ConfigurationMetaNames.TYPE in meta_props:
            payload.update({"type": meta_props[self.ConfigurationMetaNames.TYPE]})

        if (
            self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES in meta_props
            and meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES]
            is not None
        ):
            payload.update(
                {
                    "training_data_references": meta_props[
                        self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
                    ]
                }
            )

        if (
            self.ConfigurationMetaNames.IMPORT in meta_props
            and meta_props[self.ConfigurationMetaNames.IMPORT] is not None
        ):
            payload.update({"import": meta_props[self.ConfigurationMetaNames.IMPORT]})
        if (
            self.ConfigurationMetaNames.CUSTOM in meta_props
            and meta_props[self.ConfigurationMetaNames.CUSTOM] is not None
        ):
            payload.update({"custom": meta_props[self.ConfigurationMetaNames.CUSTOM]})
        if (
            self.ConfigurationMetaNames.DOMAIN in meta_props
            and meta_props[self.ConfigurationMetaNames.DOMAIN] is not None
        ):
            payload.update({"domain": meta_props[self.ConfigurationMetaNames.DOMAIN]})

        if (
            self.ConfigurationMetaNames.HYPER_PARAMETERS in meta_props
            and meta_props[self.ConfigurationMetaNames.HYPER_PARAMETERS] is not None
        ):
            payload.update(
                {
                    "hyper_parameters": meta_props[
                        self.ConfigurationMetaNames.HYPER_PARAMETERS
                    ]
                }
            )
        if (
            self.ConfigurationMetaNames.METRICS in meta_props
            and meta_props[self.ConfigurationMetaNames.METRICS] is not None
        ):
            payload.update({"metrics": meta_props[self.ConfigurationMetaNames.METRICS]})

        input_schema = []
        output_schema = []
        if (
            self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.INPUT_DATA_SCHEMA, dict, False
            )
            input_schema = [meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]]

        if (
            self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None
        ):
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA, dict, False
            )
            output_schema = [meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]]

        if len(input_schema) != 0 or len(output_schema) != 0:
            payload.update(
                {"schemas": {"input": input_schema, "output": output_schema}}
            )

        if label_column_names:
            payload["label_column"] = label_column_names[0]

        return payload

    def _publish_from_archive(
        self,
        path_to_archive: str,
        meta_props: dict[str, Any],
        version: bool = False,
        artifactid: str | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )

        def is_xml(model_filepath: str) -> bool:
            return os.path.splitext(os.path.basename(model_filepath))[-1] == ".xml"

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
        )
        payload = self._create_cloud_model_payload(
            meta_props,
            feature_names=feature_names,
            label_column_names=label_column_names,
        )
        retry = True
        while retry:
            retry = False
            response = requests.post(
                url,
                json=payload,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            if (
                response.status_code == 400
                and "hybrid_pipeline_software_specs" in response.text
                and "hybrid_pipeline_software_specs" in payload
            ):
                payload.pop("hybrid_pipeline_software_specs")
                retry = True
            else:
                if response.status_code == 400 and (
                    "Invalid request entity: Unsupported software specification"
                    in response.text
                    or "Unsupported model type and software specification combination"
                    in response.text
                ):
                    sw_spec_id = meta_props[
                        self.ConfigurationMetaNames.SOFTWARE_SPEC_ID
                    ]
                    sw_spec_details = self._client.software_specifications.get_details(
                        sw_spec_id
                    )
                    sw_spec = sw_spec_details.get("metadata").get("name")
                    spec_lifecycle = sw_spec_details.get("metadata").get(
                        "life_cycle", {}
                    )

                    if replacement := spec_lifecycle.get("replacement_name"):
                        replacement_str = f" Use replacement software specification instead: {replacement}"
                    else:
                        replacement_str = ""

                    if spec_lifecycle.get("retired"):
                        if retired_since := spec_lifecycle.get("retired_since_version"):
                            retired_software_spec_warning = f"Software specification `{sw_spec}` is retired since version {retired_since}.{replacement_str}"
                            warn(retired_software_spec_warning, PendingDeprecationWarning)  # fmt: skip
                        else:
                            retired_software_spec_warning = f"Software specification `{sw_spec}` is retired.{replacement_str}"
                            warn(retired_software_spec_warning, PendingDeprecationWarning)  # fmt: skip
                    elif spec_lifecycle.get("deprecated"):
                        if deprecated_since := spec_lifecycle.get(
                            "deprecated_since_version"
                        ):
                            deprecated_software_spec_warning = f"Software specification `{sw_spec}` is deprecated since version {deprecated_since}.{replacement_str}"
                            warn(deprecated_software_spec_warning, DeprecationWarning)
                        else:
                            deprecated_software_spec_warning = f"Software specification `{sw_spec}` is deprecated.{replacement_str}"
                            warn(deprecated_software_spec_warning, DeprecationWarning)

                result = self._handle_response(201, "creating model", response)
                model_id = self._get_required_element_from_dict(
                    result, "model_details", ["metadata", "id"]
                )

        url = (
            self._client.service_instance._href_definitions.get_published_model_href(
                model_id
            )
            + "/content"
        )
        with open(path_to_archive, "rb") as f:
            qparams = self._client._params()
            if is_xml(path_to_archive):
                qparams.update({"content_format": "coreML"})
                response = requests.put(
                    url,
                    data=f,
                    params=qparams,
                    headers=self._client._get_headers(content_type="application/xml"),
                )
            else:
                if not self._client.ICP_PLATFORM_SPACES:
                    qparams.update({"content_format": "native"})
                    qparams.update({"version": self._client.version_param})
                    model_type = meta_props[self.ConfigurationMetaNames.TYPE]
                    # update the content path for the Auto-ai model.
                    if model_type == "wml-hybrid_0.1":
                        response = self._upload_autoai_model_content(f, url, qparams)
                    else:
                        # other type of models
                        response = requests.put(
                            url,
                            data=f,
                            params=qparams,
                            headers=self._client._get_headers(
                                content_type="application/octet-stream"
                            ),
                        )
                else:
                    qparams.update({"content_format": "native"})
                    qparams.update({"version": self._client.version_param})
                    model_type = meta_props[self.ConfigurationMetaNames.TYPE]
                    # update the content path for the Auto-ai model.
                    if model_type == "wml-hybrid_0.1":
                        response = self._upload_autoai_model_content(f, url, qparams)
                    else:
                        response = requests.put(
                            url,
                            data=f,
                            params=qparams,
                            headers=self._client._get_headers(
                                content_type="application/octet-stream"
                            ),
                        )
            if response.status_code != 200 and response.status_code != 201:
                self.delete(model_id)
            self._handle_response(201, "uploading model content", response, False)

            if version == True:
                return self._client.repository.get_details(
                    str(artifactid) + "/versions/" + model_id
                )
            return self.get_details(model_id)

    def _publish_empty_model_asset(
        self,
        meta_props: dict[str, Any],
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        """
        The method creates model asset without uploading model content.
        """

        model_payload = self._create_cloud_model_payload(
            meta_props,
            label_column_names=label_column_names,
        )

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_published_models_href(),
            headers=self._client._get_headers(),
            params=self._client._params(),
            json=model_payload,
        )

        model_details = self._handle_response(
            201, "creating new model", creation_response
        )

        return model_details

    def store(
        self,
        model: MLModelType = None,
        meta_props: dict[str, Any] | None = None,
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        version: bool = False,
        artifactid: str | None = None,
        feature_names: (
            numpy.ndarray[Any, numpy.dtype[numpy.str_]] | list[str] | None
        ) = None,
        label_column_names: LabelColumnNamesType | None = None,
        subtrainingId: str | None = None,
        round_number: int | None = None,
        experiment_metadata: dict[str, Any] | None = None,
        training_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a model.

        :ref:`Here<save_models>` you can explore how to save external models in correct format.

        :param model: Can be one of following:

            - The train model object:\n
                - scikit-learn
                - xgboost
                - spark (PipelineModel)
            - path to saved model in format:\n
                - tensorflow / keras (.tar.gz)
                - pmml (.xml)
                - scikit-learn (.tar.gz)
                - spss (.str)
                - spark (.tar.gz)
                - xgboost (.tar.gz)
            - directory containing model file(s):\n
                - scikit-learn
                - xgboost
                - tensorflow
            - unique ID of the trained model
        :type model: str (for filename or path) or object (corresponding to model type)
        :param meta_props: metadata of the models configuration. To see available meta names, use:

            .. code-block:: python

                client._models.ConfigurationMetaNames.get()

        :type meta_props: dict, optional
        :param training_data: Spark DataFrame supported for spark models. Pandas dataframe, numpy.ndarray or array
            supported for scikit-learn models
        :type training_data: spark dataframe, pandas dataframe, numpy.ndarray or array, optional
        :param training_target: array with labels required for scikit-learn models
        :type training_target: array, optional
        :param pipeline: pipeline required for spark mllib models
        :type pipeline: object, optional
        :param feature_names: feature names for the training data in case of Scikit-Learn/XGBoost models,
            this is applicable only in the case where the training data is not of type - pandas.DataFrame
        :type feature_names: numpy.ndarray or list, optional
        :param label_column_names: label column names of the trained Scikit-Learn/XGBoost models
        :type label_column_names: numpy.ndarray or list, optional
        :param round_number: round number of a Federated Learning experiment that has been configured to save
            intermediate models, this applies when model is a training id
        :type round_number: int, optional
        :param experiment_metadata: metadata retrieved from the experiment that created the model
        :type experiment_metadata: dict, optional
        :param training_id: Run id of AutoAI or TuneExperiment experiment.
        :type training_id: str, optional

        :return: metadata of the created model
        :rtype: dict

        .. note::

            * For a keras model, model content is expected to contain a .h5 file and an archived version of it.

            * `feature_names` is an optional argument containing the feature names for the training data
              in case of Scikit-Learn/XGBoost models. Valid types are numpy.ndarray and list.
              This is applicable only in the case where the training data is not of type - pandas.DataFrame.

            * If the `training_data` is of type pandas.DataFrame and `feature_names` are provided,
              `feature_names` are ignored.

            * For INPUT_DATA_SCHEMA meta prop use list even when passing single input data schema. You can provide
              multiple schemas as dictionaries inside a list.

        **Examples**

        .. code-block:: python

            stored_model_details = client._models.store(model, name)

        In more complicated cases you should create proper metadata, similar to this one:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('scikit-learn_0.23-py3.7')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'scikit-learn_0.23'
            }

        In case when you want to provide input data schema of the model, you can provide it as part of meta:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('spss-modeler_18.1')

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'customer satisfaction prediction model',
                client._models.ConfigurationMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client._models.ConfigurationMetaNames.TYPE: 'spss-modeler_18.1',
                client._models.ConfigurationMetaNames.INPUT_DATA_SCHEMA: [{'id': 'test',
                                                                      'type': 'list',
                                                                      'fields': [{'name': 'age', 'type': 'float'},
                                                                                 {'name': 'sex', 'type': 'float'},
                                                                                 {'name': 'fbs', 'type': 'float'},
                                                                                 {'name': 'restbp', 'type': 'float'}]
                                                                      },
                                                                      {'id': 'test2',
                                                                       'type': 'list',
                                                                       'fields': [{'name': 'age', 'type': 'float'},
                                                                                  {'name': 'sex', 'type': 'float'},
                                                                                  {'name': 'fbs', 'type': 'float'},
                                                                                  {'name': 'restbp', 'type': 'float'}]
                }]
            }

        ``store()`` method used with a local tar.gz file that contains a model:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_tar_gz, meta_props=metadata, training_data=None)

        ``store()`` method used with a local directory that contains model files:

        .. code-block:: python

            stored_model_details = client._models.store(path_to_model_directory, meta_props=metadata, training_data=None)

        ``store()`` method used with the ID of a trained model:

        .. code-block:: python

            stored_model_details = client._models.store(trained_model_id, meta_props=metadata, training_data=None)

        ``store()`` method used with a pipeline that was generated by an AutoAI experiment:

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction model stored from object'
            }
            stored_model_details = client._models.store(pipeline_model, meta_props=metadata, experiment_metadata=experiment_metadata)

        .. code-block:: python

            metadata = {
                client._models.ConfigurationMetaNames.NAME: 'AutoAI prediction Pipeline_1 model'
            }
            stored_model_details = client._models.store(model="Pipeline_1", meta_props=metadata, training_id = training_id)

        Example of storing a prompt tuned model:

        .. code-block:: python

            stored_model_details = client._models.store(training_id = prompt_tuning_run_id)

        Example of storing a custom foundation model:

        .. code-block:: python

            sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.0')

            metadata = {
                client.repository.ModelMetaNames.NAME: 'custom FM asset',
                client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
                client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0
            }
            stored_model_details = client.repository.store_model(model='mistralai/Mistral-7B-Instruct-v0.2', meta_props=metadata)

        """
        from ibm_watsonx_ai.foundation_models.utils.utils import (
            is_training_prompt_tuning,
        )

        if (
            self._client.default_space_id is None
            and self._client.default_project_id is None
        ):
            raise WMLClientError(
                "It is mandatory is set the space or project. Use client.set.default_space(<SPACE_ID>) to set the space or client.set.default_project(<PROJECT_ID>)."
            )

        if type(meta_props) is dict and (
            "project" in meta_props or "space" in meta_props
        ):
            raise WMLClientError(
                "'project' (MetaNames.PROJECT_ID) and 'space' (MetaNames.SPACE_ID) meta names are deprecated and considered as invalid. Instead use client.set.default_space(<SPACE_ID>) to set the space or client.set.default_project(<PROJECT_ID>)."
            )

        custom_model = False
        curated_model = False
        base_foundation_model = False

        if isinstance(meta_props, dict) and meta_props.get(
            self.ConfigurationMetaNames.TYPE, ""
        ).startswith("custom_foundation_model"):
            if self._client.CPD_version < 4.8:
                raise WMLClientError(
                    Messages.get_message(">= 4.8", message_id="invalid_cpd_version")
                )

            custom_model = True

        elif isinstance(meta_props, dict) and meta_props.get(
            self.ConfigurationMetaNames.TYPE, ""
        ).startswith("curated_foundation_model"):
            if not self._client.CLOUD_PLATFORM_SPACES:
                raise WMLClientError(
                    error_msg="Deploy on Demand is unsupported for this release."
                )

            curated_model = True

        elif isinstance(meta_props, dict) and meta_props.get(
            self.ConfigurationMetaNames.TYPE, ""
        ).startswith("base_foundation_model"):
            if self._client.CLOUD_PLATFORM_SPACES:
                raise WMLClientError(
                    error_msg="Deployment of Base Foundation Model is unsupported for this release."
                )
            if self._client.CPD_version < 5.0:
                raise WMLClientError(
                    Messages.get_message(">= 5.0", message_id="invalid_cpd_version")
                )

            base_foundation_model = True

        is_prompt_tuned_training = is_training_prompt_tuning(
            training_id, api_client=self._client
        )

        Models._validate_type(
            meta_props,
            "meta_props",
            dict,
            mandatory=not is_prompt_tuned_training,
        )

        is_do_model = (
            (meta_props or {})
            .get(self.ConfigurationMetaNames.TYPE, "")
            .startswith("do-")
        )  # For DO model can be None, see #38648

        Models._validate_type(
            model,
            "model",
            object,
            mandatory=not (is_prompt_tuned_training or is_do_model),
        )

        if meta_props is None:
            meta_props = {}

        meta_props = copy.deepcopy(meta_props)
        if meta_props.get(self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES):
            meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES] = [
                p._to_dict() if type(p) == DataConnection else p
                for p in meta_props[
                    self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
                ]
            ]
            training_data_ref = meta_props[
                self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
            ][0]
            if "type" in training_data_ref:

                training_data_references_type = training_data_ref["type"]

                supported_training_data_references_types = {
                    "connection_asset",
                    "data_asset",
                    "container",
                    "url",
                }
                if self._client.ICP_PLATFORM_SPACES:
                    supported_training_data_references_types.add("fs")

                if (
                    training_data_references_type
                    not in supported_training_data_references_types
                ):
                    raise ValueError(
                        f"Incorrect training data references type: '{training_data_references_type}'. "
                        f"Supported values: {supported_training_data_references_types}"
                    )

        if (
            self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES in meta_props
            and self.ConfigurationMetaNames.INPUT_DATA_SCHEMA not in meta_props
            and meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][0].get(
                "schema"
            )
        ):

            try:
                if not meta_props.get(
                    self.ConfigurationMetaNames.LABEL_FIELD
                ) and meta_props[self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES][
                    0
                ][
                    "schema"
                ].get(
                    "fields"
                ):
                    fields = meta_props[
                        self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
                    ][0]["schema"]["fields"]
                    target_fields = [
                        f["name"]
                        for f in fields
                        if f["metadata"].get("modeling_role") == "target"
                    ]
                    if target_fields:
                        meta_props[self.ConfigurationMetaNames.LABEL_FIELD] = (
                            target_fields[0]
                        )

                def is_field_non_label(f: dict[str, Any]) -> bool:
                    if meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD):
                        return (
                            f["name"]
                            != meta_props[self.ConfigurationMetaNames.LABEL_FIELD]
                        )
                    else:
                        return True

                if meta_props.get(self.ConfigurationMetaNames.LABEL_FIELD):
                    input_data_schema = {
                        "fields": [
                            f
                            for f in meta_props[
                                self.ConfigurationMetaNames.TRAINING_DATA_REFERENCES
                            ][0].get("schema")["fields"]
                            if is_field_non_label(f)
                        ],
                        "type": "struct",
                        "id": "1",
                    }

                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] = (
                        input_data_schema
                    )
            except Exception:
                pass

        # note: do not validate metaprops when we have them from training microservice (always correct)
        if (model is None) or (isinstance(model, str) and "autoai_sdk" in model):
            pass
        elif experiment_metadata or training_id:
            # note: if experiment_metadata are not None it means that the model is created from experiment,
            # and all required information are known from the experiment metadata and the origin
            Models._validate_type(meta_props, "meta_props", dict, True)
            Models._validate_type(meta_props["name"], "meta_props.name", str, True)
        else:
            self.ConfigurationMetaNames._validate(meta_props)

        if "frameworkName" in meta_props:
            framework_name = meta_props["frameworkName"].lower()
            if version == True and (
                framework_name == "mllib" or framework_name == "wml"
            ):
                raise WMLClientError(
                    "Unsupported framework name: '{}' for creating a model version".format(
                        framework_name
                    )
                )

        if training_id and is_training_prompt_tuning(training_id, self._client):
            # import here to avoid circular import
            from ibm_watsonx_ai.foundation_models.utils.utils import load_request_json

            model_request_json = load_request_json(
                run_id=training_id, api_client=self._client
            )
            if meta_props:
                model_request_json.update(meta_props)

            creation_response = requests.post(
                self._client.service_instance._href_definitions.get_published_models_href(),
                headers=self._client._get_headers(),
                params=self._client._params(),
                json=model_request_json,
            )
            model_details = self._handle_response(
                202, "creating new model", creation_response
            )
            model_id = model_details["metadata"]["id"]

            # note: wait till content_import_state is done
            if "entity" in model_details:
                start_time = time.time()
                elapsed_time = 0.0
                while (
                    model_details["entity"].get("content_import_state") == "running"
                    and elapsed_time < 60
                ):
                    time.sleep(2)
                    elapsed_time = time.time() - start_time
                    model_details = self.get_details(model_id)
            # --- end note
            return self.get_details(model_id)

        if model is None:
            saved_model = self._publish_empty_model_asset(meta_props)

        elif model is not None and custom_model:
            saved_model = self._store_custom_foundation_model(model, meta_props)
        elif model is not None and curated_model:
            saved_model = self._store_type_of_foundation_model(
                model, meta_props, "curated"
            )
        elif model is not None and base_foundation_model:
            saved_model = self._store_type_of_foundation_model(
                model, meta_props, "base"
            )
        elif not isinstance(model, str):
            if version == True:
                raise WMLClientError(
                    "Unsupported type: object for param model. Supported types: path to saved model, training ID"
                )
            else:
                if experiment_metadata or training_id:

                    if experiment_metadata:

                        training_id = get_autoai_run_id_from_experiment_metadata(
                            experiment_metadata
                        )

                    # Note: validate if training_id is from AutoAI experiment
                    run_params = self._client.training.get_details(
                        training_id=training_id, _internal=True
                    )
                    pipeline_id = run_params["entity"].get("pipeline", {}).get("id")
                    pipeline_nodes_list = (
                        self._client.pipelines.get_details(pipeline_id)["entity"]
                        .get("document", {})
                        .get("pipelines", [])
                        if pipeline_id
                        else []
                    )
                    if (
                        len(pipeline_nodes_list) == 0
                        or pipeline_nodes_list[0]["id"] != "autoai"
                    ):
                        raise WMLClientError(
                            "Parameter training_id or experiment_metadata is not connected to AutoAI training"
                        )

                    if is_lale_pipeline(model):
                        model = model.export_to_sklearn_pipeline()

                    with catch_warnings():
                        simplefilter("ignore", category=DeprecationWarning)
                        schema, artifact_name = prepare_auto_ai_model_to_publish(
                            pipeline_model=model,
                            run_params=run_params,
                            run_id=training_id,
                            api_client=self._client,
                        )

                    new_meta_props = {
                        self._client.repository.ModelMetaNames.TYPE: "wml-hybrid_0.1",
                        self._client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: self._client.software_specifications.get_id_by_name(
                            "hybrid_0.1"
                        ),
                    }

                    results_reference = DataConnection._from_dict(
                        run_params["entity"]["results_reference"]
                    )
                    results_reference.set_client(self._client)

                    request_json = download_request_json(
                        run_params=run_params,
                        model_name=self._get_last_run_metrics_name(
                            training_id=cast(str, training_id)
                        ),
                        api_client=self._client,
                        results_reference=results_reference,
                    )
                    if request_json:
                        if "schemas" in request_json:
                            new_meta_props["schemas"] = request_json["schemas"]
                        if "hybrid_pipeline_software_specs" in request_json:
                            new_meta_props["hybrid_pipeline_software_specs"] = (
                                request_json["hybrid_pipeline_software_specs"]
                            )

                    if experiment_metadata:
                        if "prediction_column" in experiment_metadata:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.LABEL_FIELD
                            ] = experiment_metadata.get("prediction_column")

                        if "training_data_references" in experiment_metadata:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                            ] = [
                                e._to_dict() if isinstance(e, DataConnection) else e
                                for e in experiment_metadata.get(
                                    "training_data_references", []
                                )
                            ]
                            if (
                                len(
                                    new_meta_props[
                                        self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                                    ]
                                )
                                > 0
                            ):
                                new_meta_props[
                                    self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                                ][0]["schema"] = schema

                        if "test_data_references" in experiment_metadata:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.TEST_DATA_REFERENCES
                            ] = [
                                e._to_dict() if isinstance(e, DataConnection) else e
                                for e in experiment_metadata.get(
                                    "test_data_references", []
                                )
                            ]
                    else:  # if training_id
                        label_column = None
                        pipeline_details = self._client.pipelines.get_details(
                            run_params["entity"]["pipeline"]["id"]
                        )
                        for node in pipeline_details["entity"]["document"]["pipelines"][
                            0
                        ]["nodes"]:
                            if "automl" in node["id"] or "autoai" in node["id"]:
                                label_column = (
                                    node.get("parameters", {})
                                    .get("optimization", {})
                                    .get("label", None)
                                )

                        if label_column is not None:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.LABEL_FIELD
                            ] = label_column

                        # TODO Is training_data_references and test_data_references needed in meta props??
                        if "training_data_references" in run_params["entity"]:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                            ] = run_params["entity"]["training_data_references"]

                        if "test_data_references" in run_params["entity"]:
                            new_meta_props[
                                self._client.repository.ModelMetaNames.TRAINING_DATA_REFERENCES
                            ] = run_params["entity"]["test_data_references"]

                    if run_params["entity"].get("pipeline", {}).get("id"):
                        new_meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_ID
                        ] = run_params["entity"]["pipeline"]["id"]

                    new_meta_props.update(meta_props)

                    saved_model = self._client.repository.store_model(
                        model=artifact_name, meta_props=new_meta_props
                    )
                else:
                    saved_model = self._publish_from_object_cloud(
                        model=model,
                        meta_props=meta_props,
                        training_data=training_data,
                        training_target=training_target,
                        pipeline=pipeline,
                        feature_names=feature_names,
                        label_column_names=label_column_names,
                    )

        else:
            if (
                model.endswith(".pickle") or model.endswith("pipeline-model.json")
            ) and os.path.sep in model:
                # AUTO AI Trained model
                # pipeline-model.json is needed for OBM + KB
                saved_model = self._store_autoAI_model(
                    model_path=model,
                    meta_props=meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            elif model.startswith("Pipeline_") and (experiment_metadata or training_id):
                if experiment_metadata:

                    training_id = get_autoai_run_id_from_experiment_metadata(
                        experiment_metadata
                    )

                # Note: validate if training_id is from AutoAI experiment
                run_params = self._client.training.get_details(
                    training_id=training_id, _internal=True
                )

                # raise an error when TS pipeline is discarded one
                check_if_ts_pipeline_is_winner(details=run_params, model_name=model)

                # Note: We need to fetch credentials when 'container' is the type
                if run_params["entity"]["results_reference"]["type"] == "container":
                    data_connection = DataConnection._from_dict(
                        _dict=run_params["entity"]["results_reference"]
                    )
                    data_connection.set_client(self._client)
                else:
                    data_connection = None
                # --- end note

                (
                    artifact_name,
                    model_props,
                ) = prepare_auto_ai_model_to_publish_normal_scenario(
                    pipeline_model=model,
                    run_params=run_params,
                    run_id=training_id,
                    api_client=self._client,
                    result_reference=data_connection,
                )
                model_props.update(meta_props)

                saved_model = self._client.repository.store_model(
                    model=artifact_name, meta_props=model_props
                )

            elif (
                (os.path.sep in model) or os.path.isfile(model) or os.path.isdir(model)
            ):
                if not os.path.isfile(model) and not os.path.isdir(model):
                    raise WMLClientError(
                        "Invalid path: neither file nor directory exists under this path: '{}'.".format(
                            model
                        )
                    )
                saved_model = self._publish_from_file(
                    model=model,
                    meta_props=meta_props,
                    training_data=training_data,
                    training_target=training_target,
                    ver=version,
                    artifactid=artifactid,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
            else:
                saved_model = self._publish_from_training(
                    model_id=model,
                    meta_props=meta_props,
                    subtrainingId=subtrainingId,  # type: ignore[arg-type]
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                    round_number=round_number,
                )
        if (
            "system" in saved_model
            and "warnings" in saved_model["system"]
            and saved_model["system"]["warnings"]
        ):
            if saved_model["system"]["warnings"] is not None:
                message = saved_model["system"]["warnings"][0].get("message", None)
                print("Note: Warnings!! : ", message)
        return saved_model

    def update(
        self,
        model_id: str | None = None,
        meta_props: dict | None = None,
        update_model: MLModelType = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update an existing model.

        :param model_id: ID of model to be updated
        :type model_id: str

        :param meta_props: new set of meta_props to be updated
        :type meta_props: dict, optional

        :param update_model: archived model content file or path to directory that contains the archived model file
            that needs to be changed for the specific model_id
        :type update_model: object or model, optional

        :return: updated metadata of the model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client.models.update(model_id, update_model=updated_content)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        Models._validate_type(model_id, "model_id", str, False)
        model_id = cast(str, model_id)
        Models._validate_type(meta_props, "meta_props", dict, True)

        if meta_props is not None:  # TODO
            # raise WMLClientError('Meta_props update unsupported.')
            self._validate_type(meta_props, "meta_props", dict, True)

            url = self._client.service_instance._href_definitions.get_published_model_href(
                model_id
            )

            response = requests.get(
                url, params=self._client._params(), headers=self._client._get_headers()
            )

            if response.status_code != 200:
                if response.status_code == 404:
                    raise WMLClientError(
                        "Invalid input. Unable to get the details of model_id provided."
                    )
                else:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("getting model to update"), response
                    )

            details = self._handle_response(200, "Get model details", response)
            model_type = details["entity"]["type"]
            # update the content path for the Auto-ai model.
            if model_type == "wml-hybrid_0.1" and update_model is not None:
                # The only supported format is a zip file containing `pipeline-model.json` and pickled model compressed
                # to tar.gz format.
                if not update_model.endswith(".zip"):
                    raise WMLClientError(
                        "Invalid model content. The model content file should bre zip archive containing"
                        ' ".pickle.tar.gz" file or "pipline-model.json", for the model type\'{}\'.'.format(
                            model_type
                        )
                    )

            # with validation should be somewhere else, on the begining, but when patch will be possible
            patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
                details["entity"], meta_props, with_validation=True
            )
            response_patch = requests.patch(
                url,
                json=patch_payload,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
            updated_details = self._handle_response(
                200, "model version patch", response_patch
            )
            if update_model is not None:
                self._update_model_content(model_id, details, update_model)
            return updated_details

        return self.get_details(model_id)

    def load(self, artifact_id: str | None, **kwargs: Any) -> Any:
        """Load a model from the repository to object in a local environment.

        :param artifact_id: ID of the stored model
        :type artifact_id: str

        :return: trained model
        :rtype: object

        **Example:**

        .. code-block:: python

            model = client.models.load(model_id)
        """
        artifact_id = _get_id_from_deprecated_uid(kwargs, artifact_id, "artifact")
        artifact_id = cast(str, artifact_id)
        Models._validate_type(artifact_id, "artifact_id", str, False)
        # check if this is tensorflow 2.x model type
        model_details = self.get_details(artifact_id)
        if "wml-hybrid" in model_details.get("entity", {}).get("type", ""):
            raise WMLClientError(
                "The use of the load() method is restricted and not permitted for AutoAI models."
            )
        if model_details.get("entity", {}).get("type", "").startswith("tensorflow_2."):
            return self._tf2x_load_model_instance(artifact_id)
        try:
            # Cloud Convergence: CHK IF THIS CONDITION IS CORRECT since loaded_model
            # functionality below
            if (
                self._client.default_space_id is None
                and self._client.default_project_id is None
            ):
                raise WMLClientError(
                    "It is mandatory is set the space or project. \
                    Use client.set.default_space(<SPACE_ID>) to set the space or client.set.default_project(<PROJECT_ID>)."
                )
            else:
                query_param = self._client._params()
                loaded_model = self._client.repository._ml_repository_client.models._get_v4_cloud_model(
                    artifact_id, query_param=query_param
                )

            loaded_model = loaded_model.model_instance()
            self._logger.info(
                "Successfully loaded artifact with artifact_id: {}".format(artifact_id)
            )
            return loaded_model
        except Exception as e:
            raise WMLClientError(
                "Loading model with artifact_id: '{}' failed.".format(artifact_id), e
            )

    def download(
        self,
        model_id: str | None,
        filename: str = "downloaded_model.tar.gz",
        rev_id: str | None = None,
        format: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Download a model from the repository to local file.

        :param model_id: ID of the stored model
        :type model_id: str

        :param filename: name of local file to be created
        :type filename: str, optional

        :param rev_id: ID of the revision
        :type rev_id: str, optional

        :param format: format of the content
        :type format: str, optional

        **Example:**

        .. code-block:: python

            client.models.download(model_id, 'my_model.tar.gz')
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        model_id = cast(str, model_id)
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev", True)
        if os.path.isfile(filename):
            raise WMLClientError(
                "File with name: '{}' already exists.".format(filename)
            )

        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(filename, "filename", str, True)

        if filename.endswith(".json"):
            is_json = True
            json_filename = filename
            import uuid

            filename = f"tmp_{uuid.uuid4()}.tar.gz"
        else:
            is_json = False

        artifact_url = (
            self._client.service_instance._href_definitions.get_model_last_version_href(
                model_id
            )
        )
        params = self._client._params()
        try:
            url = self._client.service_instance._href_definitions.get_published_model_href(
                model_id
            )
            model_get_response = requests.get(
                url, params=self._client._params(), headers=self._client._get_headers()
            )

            model_details = self._handle_response(200, "get model", model_get_response)
            if rev_id is not None:
                params.update({"revision_id": rev_id})

            model_type = model_details["entity"]["type"]
            if (
                model_type.startswith("keras_")
                or model_type.startswith("scikit-learn_")
                or model_type.startswith("xgboost_")
            ) and format is not None:
                Models._validate_type(format, "format", str, False)
                if str(format).upper() == "COREML":
                    params.update({"content_format": "coreml"})
                else:
                    params.update({"content_format": "native"})
            else:
                params.update({"content_format": "native"})
            artifact_content_url = str(artifact_url + "/download")
            if model_details["entity"]["type"] == "wml-hybrid_0.1":
                self._download_auto_ai_model_content(
                    model_id, artifact_content_url, filename
                )
                print("Successfully saved model content to file: '{}'".format(filename))
                return os.getcwd() + "/" + filename
            else:
                r = requests.get(
                    artifact_content_url,
                    params=params,
                    headers=self._client._get_headers(),
                    stream=True,
                )

            if r.status_code != 200:
                raise ApiRequestFailure(
                    "Failure during {}.".format("downloading model"), r
                )

            downloaded_model = r.content
            self._logger.info(
                "Successfully downloaded artifact with artifact_url: {}".format(
                    artifact_url
                )
            )
        except WMLClientError as e:
            raise e
        except Exception as e:
            if artifact_url is not None:
                raise WMLClientError(
                    "Downloading model with artifact_url: '{}' failed.".format(
                        artifact_url
                    ),
                    e,
                )
            else:
                raise WMLClientError("Downloading model failed.", e)
        finally:
            if is_json:
                try:
                    os.remove(filename)
                except Exception:
                    pass
        try:
            with open(filename, "wb") as f:
                f.write(downloaded_model)

            if is_json:
                import tarfile

                tar = tarfile.open(filename, "r:gz")
                file_name = tar.getnames()[0]
                if not file_name.endswith(".json"):
                    raise WMLClientError("Downloaded model is not json.")
                tar.extractall()
                tar.close()
                os.rename(file_name, json_filename)

                os.remove(filename)
                filename = json_filename

            print("Successfully saved model content to file: '{}'".format(filename))
            return os.getcwd() + "/" + filename
        except IOError as e:
            raise WMLClientError(
                "Saving model with artifact_url: '{}' failed.".format(filename), e
            )

    def delete(self, model_id: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Delete a model from the repository.

        :param model_id: ID of the stored model
        :type model_id: str

        **Example:**

        .. code-block:: python

            client.models.delete(model_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        Models._validate_type(model_id, "model_id", str, False)

        model_endpoint = (
            self._client.service_instance._href_definitions.get_published_model_href(
                model_id
            )
        )

        self._logger.debug(
            "Deletion artifact model endpoint: {}".format(model_endpoint)
        )
        if not self._client.ICP_PLATFORM_SPACES:
            if self._if_deployment_exist_for_asset(model_id):
                raise WMLClientError(
                    "Cannot delete model that has existing deployments. Please delete all associated deployments and try again"
                )

            response_delete = requests.delete(
                model_endpoint,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        else:
            # check if the model as a corresponding deployment
            if self._if_deployment_exist_for_asset(model_id):
                raise WMLClientError(
                    "Cannot delete model that has existing deployments. Please delete all associated deployments and try again"
                )
            response_delete = requests.delete(
                model_endpoint,
                params=self._client._params(),
                headers=self._client._get_headers(),
            )
        return self._handle_response(204, "model deletion", response_delete, False)

    @overload
    def get_details(
        self,
        model_id: str = "",
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    @overload
    def get_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator: ...

    def get_details(
        self,
        model_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | Generator:
        """Get metadata of stored models. If neither model ID nor model name is specified,
        the metadata of all models is returned.
        If only model name is specified, metadata of models with the name is returned (if any).

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `model_id` is None
        :type spec_state: SpecStates, optional

        :param model_name: name of the stored model, definition, or pipeline, can be used only when `model_id` is None
        :type model_name: str, optional

        :return: metadata of the stored model(s)
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        .. note::
            In current implementation setting `spec_state=True` may break set `limit`,
            returning less records than stated by set `limit`.

        **Example:**

        .. code-block:: python

            model_details = client.models.get_details(model_id)
            models_details = client.models.get_details(model_name='Sample_model')
            models_details = client.models.get_details()
            models_details = client.models.get_details(limit=100)
            models_details = client.models.get_details(limit=100, get_all=True)
            models_details = []
            for entry in client.models.get_details(limit=100, asynchronous=True, get_all=True):
                models_details.extend(entry)

        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model", True)
        if limit and spec_state:
            spec_state_setting_warning = (
                "Warning: In current implementation setting `spec_state=True` may break set `limit`, "
                "returning less records than stated by set `limit`."
            )
            warn(spec_state_setting_warning, category=DeprecationWarning)

        ##For CP4D, check if either space or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, False)
        Models._validate_type(limit, "limit", int, False)

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
        )

        if model_id is None:
            if spec_state:
                filter_func = self._get_filter_func_by_spec_ids(
                    self._get_and_cache_spec_ids_for_state(spec_state)
                )
            elif model_name:
                filter_func = self._get_filter_func_by_artifact_name(model_name)
            else:
                filter_func = None

            return self._get_artifact_details(
                url,
                model_id,
                limit,
                "models",
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )

        else:
            return self._get_artifact_details(url, model_id, limit, "models")

    @staticmethod
    def get_href(model_details: dict[str, Any]) -> str:
        """Get the URL of a stored model.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: URL of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_url = client.models.get_href(model_details)
        """

        Models._validate_type(model_details, "model_details", object, True)

        if "asset_id" in model_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "href"]
            )
        else:
            if "id" not in model_details["metadata"]:
                Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(
                    model_details, "model_details", ["metadata", "href"]
                )
            else:
                model_id = WMLResource._get_required_element_from_dict(
                    model_details, "model_details", ["metadata", "id"]
                )
                return "/ml/v4/models/" + model_id

    @staticmethod
    def get_uid(model_details: dict[str, Any]) -> str:
        """Get the UID of a stored model.

        *Deprecated:* Use ``get_id(model_details)`` instead.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: UID of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_uid = client.models.get_uid(model_details)
        """
        get_uid_method_deprecated_warning = (
            "This method is deprecated, please use Models.get_id(model_details) instead"
        )
        warn(get_uid_method_deprecated_warning, category=DeprecationWarning)
        return Models.get_id(model_details)

    @staticmethod
    def get_id(model_details: dict[str, Any]) -> str:
        """Get the ID of a stored model.

        :param model_details: details of the stored model
        :type model_details: dict

        :return: ID of the stored model
        :rtype: str

        **Example:**

        .. code-block:: python

            model_id = client.models.get_id(model_details)
        """
        Models._validate_type(model_details, "model_details", object, True)

        if "asset_id" in model_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                model_details, "model_details", ["metadata", "asset_id"]
            )
        else:
            if "id" not in model_details["metadata"]:
                Models._validate_type_of_details(model_details, MODEL_DETAILS_TYPE)
                return WMLResource._get_required_element_from_dict(
                    model_details, "model_details", ["metadata", "guid"]
                )
            else:
                return WMLResource._get_required_element_from_dict(
                    model_details, "model_details", ["metadata", "id"]
                )

    def list(
        self,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
    ) -> pandas.DataFrame | Generator:
        """List stored models in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :return: pandas.DataFrame with listed models or generator if `asynchronous` is set to True
        :rtype: pandas.DataFrame | Generator

        **Example:**

        .. code-block:: python

            client.models.list()
            client.models.list(limit=100)
            client.models.list(limit=100, get_all=True)
            [entry for entry in client.models.list(limit=100, asynchronous=True, get_all=True)]
        """

        ##For CP4D, check if either spce or project ID is set
        def process_resources(self: Any, model_resources: dict) -> pandas.DataFrame:
            model_resources = model_resources["resources"]
            sw_spec_info = {
                s["id"]: s
                for s in self._client.software_specifications.get_details(
                    state_info=True
                )["resources"]
            }

            def get_spec_info(spec_id: str, prop: str) -> str:
                if spec_id and spec_id in sw_spec_info:
                    return sw_spec_info[spec_id].get(prop, "")
                else:
                    return ""

            model_values = [
                (
                    m["metadata"]["id"],
                    m["metadata"]["name"],
                    m["metadata"]["created_at"],
                    m["entity"]["type"],
                    get_spec_info(
                        m["entity"].get("software_spec", {}).get("id"), "state"
                    ),
                    get_spec_info(
                        m["entity"].get("software_spec", {}).get("id"), "replacement"
                    ),
                )
                for m in model_resources
            ]

            table = self._list(
                model_values,
                ["ID", "NAME", "CREATED", "TYPE", "SPEC_STATE", "SPEC_REPLACEMENT"],
                limit,
            )
            return table

        self._client._check_if_either_is_set()
        if asynchronous:
            return (
                process_resources(self, model_resources)  # type: ignore[arg-type]
                for model_resources in self.get_details(
                    limit=limit, asynchronous=asynchronous, get_all=get_all
                )
            )

        else:
            model_resources = self.get_details(limit=limit, get_all=get_all)
            return process_resources(self, model_resources)

    def create_revision(
        self, model_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Create a revision for a given model ID.

        :param model_id: ID of the stored model
        :type model_id: str

        :return: revised metadata of the stored model
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client.models.create_revision(model_id)
        """
        ##For CP4D, check if either spce or project ID is set
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, False)

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
        )
        return self._create_revision_artifact(url, model_id, "models")

    def list_revisions(
        self, model_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> pandas.DataFrame:
        """Print all revisions for the given model ID in a table format.

        :param model_id: unique ID of the stored model
        :type model_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.models.list_revisions(model_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Models._validate_type(model_id, "model_id", str, True)

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
            + "/"
            + model_id
        )

        model_resources = self._get_artifact_details(
            url,
            "revisions",
            None,
            "model revisions",
            _all=self._should_get_all_values(limit),
        )["resources"]

        model_values = [
            (m["metadata"]["rev"], m["metadata"]["name"], m["metadata"]["created_at"])
            for m in model_resources
        ]

        table = self._list(model_values, ["REV", "NAME", "CREATED"], limit)
        return table

    def get_revision_details(
        self, model_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Get metadata of a stored model's specific revision.

        :param model_id: ID of the stored model, definition, or pipeline
        :type model_id: str

        :param rev_id: unique ID of the stored model revision
        :type rev_id: str

        :return: metadata of the stored model(s)
        :rtype: dict

        **Example:**

        .. code-block:: python

            model_details = client.models.get_revision_details(model_id, rev_id)
        """
        model_id = _get_id_from_deprecated_uid(kwargs, model_id, "model")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")

        if isinstance(rev_id, int):
            rev_id_as_int_warning = "`rev_id` parameter type as int is deprecated, please convert to str instead"
            warn(rev_id_as_int_warning, category=DeprecationWarning)
            rev_id = str(rev_id)

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()
        Models._validate_type(model_id, "model_id", str, True)
        Models._validate_type(rev_id, "rev_id", str, True)

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
            + "/"
            + model_id
        )
        return self._get_with_or_without_limit(
            url,
            limit=None,
            op_name="model",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    def promote(
        self, model_id: str, source_project_id: str, target_space_id: str
    ) -> dict[str, Any] | None:
        """Promote a model from a project to space. Supported only for IBM Cloud Pak® for Data.

        *Deprecated:* Use `client.spaces.promote(asset_id, source_project_id, target_space_id)` instead.
        """
        promote_model_method_deprecated_warning = (
            "Note: Function `client.repository.promote_model(model_id, source_project_id, target_space_id)` "
            "has been deprecated. Use `client.spaces.promote(asset_id, source_project_id, target_space_id)` instead."
        )
        warn(promote_model_method_deprecated_warning, category=DeprecationWarning)
        try:
            return self._client.spaces.promote(
                model_id, source_project_id, target_space_id
            )
        except PromotionFailed as e:
            raise ModelPromotionFailed(
                e.project_id, e.space_id, e.promotion_response, str(e.reason)
            )

    def _update_model_content(
        self, model_id: str, updated_details: dict[str, Any], update_model: Any
    ) -> None:

        model = copy.copy(update_model)
        model_type = updated_details["entity"]["type"]

        def is_xml(model_filepath: str) -> bool:
            if os.path.splitext(os.path.basename(model_filepath))[-1] == ".pmml":
                raise WMLClientError(
                    "The file name has an unsupported extension. Rename the file with a .xml extension."
                )
            return os.path.splitext(os.path.basename(model_filepath))[-1] == ".xml"

        import tarfile
        import zipfile

        model_filepath = model

        if "scikit-learn_" in model_type or "mllib_" in model_type:
            meta_props = updated_details["entity"]
            meta_data = MetaProps(meta_props)
            name = updated_details["metadata"]["name"]
            model_artifact = MLRepositoryArtifact(
                update_model, name=name, meta_props=meta_data, training_data=None
            )
            model_artifact.uid = model_id
            query_params = self._client._params()
            query_params.update({"content_format": "native"})
            self._client.repository._ml_repository_client.models.upload_content(
                model_artifact, query_param=query_params, no_delete=True
            )
        else:
            if (
                (os.path.sep in update_model)
                or os.path.isfile(update_model)
                or os.path.isdir(update_model)
            ):
                if not os.path.isfile(update_model) and not os.path.isdir(update_model):
                    raise WMLClientError(
                        "Invalid path: neither file nor directory exists under this path: '{}'.".format(
                            model
                        )
                    )

            if os.path.isdir(model):
                if "tensorflow" in model_type:
                    # TODO currently tar.gz is required for tensorflow - the same ext should be supported for all frameworks
                    if os.path.basename(model) == "":
                        model = os.path.dirname(update_model)
                    filename = os.path.basename(update_model) + ".tar.gz"
                    current_dir = os.getcwd()
                    os.chdir(model)
                    target_path = os.path.dirname(model)

                    with tarfile.open(os.path.join("..", filename), mode="w:gz") as tar:
                        tar.add(".")

                    os.chdir(current_dir)
                    model_filepath = os.path.join(target_path, filename)
                    if (
                        tarfile.is_tarfile(model_filepath)
                        or zipfile.is_zipfile(model_filepath)
                        or is_xml(model_filepath)
                    ):
                        path_to_archive = model_filepath
                else:
                    if "caffe" in model_type:
                        raise WMLClientError(
                            "Invalid model file path  specified for: '{}'.".format(
                                model_type
                            )
                        )
                    loaded_model = load_model_from_directory(model_type, model)
                    path_to_archive = loaded_model
            elif is_xml(model_filepath):
                path_to_archive = model_filepath
            elif tarfile.is_tarfile(model_filepath) or zipfile.is_zipfile(
                model_filepath
            ):
                path_to_archive = model_filepath

            else:
                raise WMLClientError(
                    "Saving trained model in repository failed. '{}' file does not have valid format".format(
                        model_filepath
                    )
                )

            url = (
                self._client.service_instance._href_definitions.get_published_model_href(
                    model_id
                )
                + "/content"
            )
            with open(path_to_archive, "rb") as f:
                if is_xml(path_to_archive):
                    response = requests.put(
                        url,
                        data=f,
                        params=self._client._params(),
                        headers=self._client._get_headers(
                            content_type="application/xml"
                        ),
                    )
                else:
                    qparams = self._client._params()

                    if model_type.startswith("wml-hybrid_0"):
                        response = self._upload_autoai_model_content(f, url, qparams)
                    else:
                        qparams.update({"content_format": "native"})
                        response = requests.put(
                            url,
                            data=f,
                            params=qparams,
                            headers=self._client._get_headers(
                                content_type="application/octet-stream"
                            ),
                        )
                self._handle_response(201, "uploading model content", response, False)

    def _create_cloud_model_payload(
        self,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)
        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        else:
            if self._client.default_project_id is not None:
                metadata.update({"project_id": self._client.default_project_id})
            else:
                raise WMLClientError(
                    "It is mandatory is set the space or Project. \
                 Use client.set.default_space(<SPACE_ID>) to set the space or"
                    " Use client.set.default_project(<PROJECT_ID)"
                )

        if (
            self.ConfigurationMetaNames.RUNTIME_ID in meta_props
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input.  RUNTIME_ID is not supported in cloud environment. Specify SOFTWARE_SPEC_ID"
            )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
            )
            metadata.update(
                {
                    self.ConfigurationMetaNames.SOFTWARE_SPEC_ID: {
                        "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                    }
                }
            )

        if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
            )
            metadata.update(
                {
                    self.ConfigurationMetaNames.PIPELINE_ID: {
                        "id": meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_ID
                        ]
                    }
                }
            )

        if self.ConfigurationMetaNames.MODEL_DEFINITION_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.MODEL_DEFINITION_ID, str, False
            )
            metadata.update(
                {
                    self.ConfigurationMetaNames.MODEL_DEFINITION_ID: {
                        "id": meta_props[
                            self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                        ]
                    }
                }
            )

        if (
            self.ConfigurationMetaNames.IMPORT in meta_props
            and meta_props[self.ConfigurationMetaNames.IMPORT] is not None
        ):
            print(
                "WARNING: Invalid input. IMPORT is not supported in cloud environment."
            )

        if (
            self.ConfigurationMetaNames.TRAINING_LIB_ID in meta_props
            and meta_props[self.ConfigurationMetaNames.IMPORT] is not None
        ):
            print(
                "WARNING: Invalid input. TRAINING_LIB_ID is not supported in cloud environment."
            )

        input_schema = []
        output_schema = []
        if (
            self.ConfigurationMetaNames.INPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA] is not None
        ):
            if isinstance(
                meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA], list
            ):
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.INPUT_DATA_SCHEMA,
                    list,
                    False,
                )
                input_schema = meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
            else:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.INPUT_DATA_SCHEMA,
                    dict,
                    False,
                )
                input_schema = [
                    meta_props[self.ConfigurationMetaNames.INPUT_DATA_SCHEMA]
                ]
            metadata.pop(self.ConfigurationMetaNames.INPUT_DATA_SCHEMA)

        if (
            self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA in meta_props
            and meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA] is not None
        ):
            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith("do-"):
                if isinstance(
                    meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA], dict
                ):
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                        dict,
                        False,
                    )
                    output_schema = [
                        meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                    ]
                else:
                    self._validate_meta_prop(
                        meta_props,
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                        list,
                        False,
                    )
                    output_schema = meta_props[
                        self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA
                    ]
            else:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA,
                    dict,
                    False,
                )
                output_schema = [
                    meta_props[self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA]
                ]
            metadata.pop(self.ConfigurationMetaNames.OUTPUT_DATA_SCHEMA)

        if len(input_schema) != 0 or len(output_schema) != 0:
            metadata.update(
                {"schemas": {"input": input_schema, "output": output_schema}}
            )

        if label_column_names:
            metadata["label_column"] = label_column_names[0]

        return metadata

    def _publish_from_object_cloud(
        self,
        model: MLModelType,
        meta_props: dict[str, Any],
        training_data: TrainingDataType | None = None,
        training_target: TrainingTargetType | None = None,
        pipeline: PipelineType | None = None,
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Store model from object in memory into Watson Machine Learning repository on Cloud."""
        self._validate_meta_prop(
            meta_props, self.ConfigurationMetaNames.NAME, str, True
        )
        if (
            self.ConfigurationMetaNames.RUNTIME_ID in meta_props
            and self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
        ):
            raise WMLClientError(
                "Invalid input. RUNTIME_ID is no longer supported, instead of that provide SOFTWARE_SPEC_ID in meta_props."
            )
        else:
            if (
                self.ConfigurationMetaNames.SOFTWARE_SPEC_ID not in meta_props
                and self.ConfigurationMetaNames.RUNTIME_ID not in meta_props
            ):
                raise WMLClientError(
                    "Invalid input. It is mandatory to provide SOFTWARE_SPEC_ID in meta_props."
                )
        try:
            if "pyspark.ml.pipeline.PipelineModel" in str(type(model)):
                if pipeline is None or training_data is None:
                    raise WMLClientError(
                        "pipeline and training_data are expected for spark models."
                    )
                name = meta_props[self.ConfigurationMetaNames.NAME]
                version = "1.0"
                platform = {"name": "python", "versions": ["3.6"]}
                library_tar = self._save_library_archive(pipeline)
                model_definition_props = {
                    self._client.model_definitions.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.model_definitions.ConfigurationMetaNames.VERSION: version,
                    self._client.model_definitions.ConfigurationMetaNames.PLATFORM: platform,
                }
                model_definition_details = self._client.model_definitions.store(
                    library_tar, model_definition_props
                )
                model_definition_id = self._client.model_definitions.get_id(
                    model_definition_details
                )
                # create a pipeline for model definition
                pipeline_metadata = {
                    self._client.pipelines.ConfigurationMetaNames.NAME: name
                    + "_"
                    + uid_generate(8),
                    self._client.pipelines.ConfigurationMetaNames.DOCUMENT: {
                        "doc_type": "pipeline",
                        "version": "2.0",
                        "primary_pipeline": "dlaas_only",
                        "pipelines": [
                            {
                                "id": "dlaas_only",
                                "runtime_ref": "spark",
                                "nodes": [
                                    {
                                        "id": "repository",
                                        "type": "model_node",
                                        "inputs": [],
                                        "outputs": [],
                                        "parameters": {
                                            "model_definition": {
                                                "id": model_definition_id
                                            }
                                        },
                                    }
                                ],
                            }
                        ],
                    },
                }

                pipeline_save = self._client.pipelines.store(pipeline_metadata)
                pipeline_id = self._client.pipelines.get_id(pipeline_save)
                meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                    "id": pipeline_id
                }
            else:
                if self.ConfigurationMetaNames.PIPELINE_ID in meta_props:
                    self._validate_meta_prop(
                        meta_props, self.ConfigurationMetaNames.PIPELINE_ID, str, False
                    )
                    meta_props[self._client.repository.ModelMetaNames.PIPELINE_ID] = {
                        "id": meta_props[
                            self._client.repository.ModelMetaNames.PIPELINE_ID
                        ]
                    }

            if (
                self.ConfigurationMetaNames.SPACE_ID in meta_props
                and meta_props[self._client.repository.ModelMetaNames.SPACE_ID]
                is not None
            ):
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SPACE_ID, str, False
                )
                meta_props["space_id"] = meta_props[
                    self._client.repository.ModelMetaNames.SPACE_ID
                ]
                meta_props.pop(self.ConfigurationMetaNames.SPACE_ID)
            else:
                if self._client.default_project_id is not None:
                    meta_props["project_id"] = self._client.default_project_id
                else:
                    meta_props["space_id"] = self._client.default_space_id

            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
                )
                meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID] = {
                    "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                }

            if self.ConfigurationMetaNames.MODEL_DEFINITION_ID in meta_props:
                self._validate_meta_prop(
                    meta_props,
                    self.ConfigurationMetaNames.MODEL_DEFINITION_ID,
                    str,
                    False,
                )
                meta_props[
                    self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                ] = {
                    "id": meta_props[
                        self._client.repository.ModelMetaNames.MODEL_DEFINITION_ID
                    ]
                }

            if str(meta_props[self.ConfigurationMetaNames.TYPE]).startswith(
                "tensorflow_"
            ):
                saved_model = self._store_tf_model(
                    model,
                    meta_props,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
                return saved_model
            else:
                meta_data = MetaProps(meta_props)
                model_artifact = MLRepositoryArtifact(
                    model,
                    name=str(meta_props[self.ConfigurationMetaNames.NAME]),
                    meta_props=meta_data,
                    training_data=training_data,
                    training_target=training_target,
                    feature_names=feature_names,
                    label_column_names=label_column_names,
                )
                query_param_for_repo_client = self._client._params()
                saved_model = self._client.repository._ml_repository_client.models.save(
                    model_artifact, query_param=query_param_for_repo_client
                )
                return self.get_details("{}".format(saved_model.uid))  # type: ignore[attr-defined]

        except Exception as e:
            raise WMLClientError("Publishing model failed.", e)

    def _upload_autoai_model_content(
        self, file: str | BinaryIO, url: str, qparams: dict[str, Any]
    ) -> Requests.Response:
        import zipfile
        import json

        node_ids = None
        with zipfile.ZipFile(file, "r") as zipObj:
            # Get a list of all archived file names from the zip
            listOfFileNames = zipObj.namelist()
            t1 = zipObj.extract("pipeline-model.json")
            with open(t1, "r") as f2:
                data = json.load(f2)
                # note: we can have multiple nodes (OBM + KB)
                node_ids = [
                    node.get("id") for node in data.get("pipelines")[0].get("nodes")
                ]

            if node_ids is None:
                raise WMLClientError(
                    "Invalid pipline-model.json content file. There is no node id value found"
                )

            qparams.update({"content_format": "native"})
            qparams.update({"name": "pipeline-model.json"})

            if "pipeline_node_id" in qparams.keys():
                qparams.pop("pipeline_node_id")

            response = requests.put(
                url,
                data=open(t1, "rb").read(),
                params=qparams,
                headers=self._client._get_headers(content_type="application/json"),
            )

            listOfFileNames.remove("pipeline-model.json")

            # note: the file order is importand, should be OBM model first then KB model
            for fileName, node_id in zip(listOfFileNames, node_ids):
                # Check filename endswith json
                if fileName.endswith(".tar.gz") or fileName.endswith(".zip"):
                    # Extract a single file from zip
                    qparams.update({"content_format": "pipeline-node"})
                    qparams.update({"pipeline_node_id": node_id})
                    qparams.update({"name": fileName})
                    t2 = zipObj.extract(fileName)
                    response = requests.put(
                        url,
                        data=open(t2, "rb").read(),
                        params=qparams,
                        headers=self._client._get_headers(
                            content_type="application/octet-stream"
                        ),
                    )
        return response

    def _download_auto_ai_model_content(
        self, model_id: str, content_url: str, filename: str
    ) -> None:
        import zipfile

        with zipfile.ZipFile(filename, "w") as zip:
            # writing each file one by one
            pipeline_model_file = "pipeline-model.json"
            with open(pipeline_model_file, "wb") as f:
                params = self._client._params()
                params.update({"content_format": "native"})
                r = requests.get(
                    content_url,
                    params=params,
                    headers=self._client._get_headers(),
                    stream=True,
                )
                if r.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading model"), r
                    )
                self._logger.info(
                    "Successfully downloaded artifact pipeline_model.json artifact_url: {}".format(
                        content_url
                    )
                )
                f.write(r.content)
            f.close()
            zip.write(pipeline_model_file)
            mfilename = "model_" + model_id + ".pickle.tar.gz"
            with open(mfilename, "wb") as f:
                params1 = self._client._params()
                params1.update({"content_format": "pipeline-node"})
                res = requests.get(
                    content_url,
                    params=params1,
                    headers=self._client._get_headers(),
                    stream=True,
                )
                if res.status_code != 200:
                    raise ApiRequestFailure(
                        "Failure during {}.".format("downloading model"), r
                    )
                f.write(res.content)
                self._logger.info(
                    "Successfully downloaded artifact with artifact_url: {}".format(
                        content_url
                    )
                )
            f.close()
            zip.write(mfilename)

    def _store_tf_model(
        self,
        model: Any,
        meta_props: dict[str, Any],
        feature_names: FeatureNamesArrayType | None = None,
        label_column_names: LabelColumnNamesType | None = None,
    ) -> dict[str, Any]:
        # Model type is
        import tensorflow as tf

        url = (
            self._client.service_instance._href_definitions.get_published_models_href()
        )
        id_length = 20
        gen_id = uid_generate(id_length)

        tf_meta = None
        options = None
        signature = None
        save_format = None
        include_optimizer = None
        if (
            "tf_model_params" in meta_props
            and meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS] is not None
        ):
            tf_meta = copy.deepcopy(
                meta_props[self.ConfigurationMetaNames.TF_MODEL_PARAMS]
            )
            save_format = tf_meta.get("save_format")
            options = tf_meta.get("options")
            signature = tf_meta.get("signature")
            include_optimizer = tf_meta.get("include_optimizer")

        if save_format == "tf" or (
            save_format is None
            and "tensorflow.python.keras.engine.training.Model" in str(type(model))
        ):
            temp_dir_name = "{}".format("pb" + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            import tensorflow as tf

            tf.saved_model.save(model, temp_dir, signatures=signature, options=options)

        elif save_format == "h5" or (
            save_format is None
            and "tensorflow.python.keras.engine.sequential.Sequential"
            in str(type(model))
        ):
            temp_dir_name = "{}".format("hdfs" + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            model_file = temp_dir + "/sequential_model.h5"
            tf.keras.models.save_model(
                model,
                model_file,
                include_optimizer=include_optimizer,
                save_format="h5",
                signatures=None,
                options=options,
            )
        elif (
            isinstance(model, str)
            and os.path.splitext(os.path.basename(model))[-1] == ".h5"
        ):
            temp_dir_name = "{}".format("hdfs" + gen_id)
            temp_dir = temp_dir_name
            if not os.path.exists(temp_dir):
                import shutil

                os.makedirs(temp_dir)
                shutil.copy2(model, temp_dir)
        else:

            raise WMLClientError(
                "Saving the tensorflow model requires the model of either tf format or h5 format for Sequential model."
            )

        path_to_archive = self._model_content_compress_artifact(temp_dir_name, temp_dir)
        payload = copy.deepcopy(meta_props)
        if label_column_names:
            payload["label_column"] = label_column_names[0]

        response = requests.post(
            url,
            json=payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )
        result = self._handle_response(201, "creating model", response)
        model_id = self._get_required_element_from_dict(
            result, "model_details", ["metadata", "id"]
        )

        url = (
            self._client.service_instance._href_definitions.get_published_model_href(
                model_id
            )
            + "/content"
        )
        with open(path_to_archive, "rb") as f:
            qparams = self._client._params()

            qparams.update({"content_format": "native"})
            qparams.update({"version": self._client.version_param})
            # update the content path for the Auto-ai model.

            response = requests.put(
                url,
                data=f,
                params=qparams,
                headers=self._client._get_headers(
                    content_type="application/octet-stream"
                ),
            )
            if response.status_code != 200 and response.status_code != 201:
                self.delete(model_id)
            self._handle_response(201, "uploading model content", response, False)

            if os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
                os.remove(path_to_archive)
            return self.get_details(model_id)

    def _store_custom_foundation_model(
        self, model: str, meta_props: dict[str, Any]
    ) -> dict[str, Any]:
        # Store custom foundation model

        payload = self._create_custom_model_payload(model=model, meta_props=meta_props)

        creation_response = requests.post(
            url=self._client.service_instance._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
            json=payload,
        )

        if creation_response.status_code == 201:
            model_details = self._handle_response(
                201, "creating new model", creation_response
            )
        else:
            model_details = self._handle_response(
                202, "creating new model", creation_response
            )
        model_id = model_details["metadata"]["id"]

        if "entity" in model_details:
            start_time = time.time()
            elapsed_time = 0.0
            while (
                model_details["entity"].get("content_import_state") == "running"
                and elapsed_time < 60
            ):
                time.sleep(2)
                elapsed_time = time.time() - start_time
                model_details = self.get_details(model_id)
        return self.get_details(model_id)

    def _store_type_of_foundation_model(
        self,
        model: str,
        meta_props: dict[str, Any],
        model_type: Literal["curated", "base"],
    ) -> dict[str, Any]:
        # Store curated/base foundation model

        payload = self._create_type_of_model_payload(
            model=model, meta_props=meta_props, model_type=model_type
        )

        creation_response = requests.post(
            url=self._client.service_instance._href_definitions.get_published_models_href(),
            params=self._client._params(skip_for_create=True),
            headers=self._client._get_headers(),
            json=payload,
        )

        if creation_response.status_code == 201:
            model_details = self._handle_response(
                201, "creating new model", creation_response
            )
        else:
            model_details = self._handle_response(
                202, "creating new model", creation_response
            )
        model_id = model_details["metadata"]["id"]

        if "entity" in model_details:
            start_time = time.time()
            elapsed_time = 0.0
            while (
                model_details["entity"].get("content_import_state") == "running"
                and elapsed_time < 60
            ):
                time.sleep(2)
                elapsed_time = time.time() - start_time
                model_details = self.get_details(model_id)
        return model_details

    def _create_custom_model_payload(
        self, model: str, meta_props: dict[str, Any]
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)

        Models._validate_type(model, "model", str, True)

        if self.ConfigurationMetaNames.TYPE in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )

        if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
            )
            metadata.update(
                {
                    self.ConfigurationMetaNames.SOFTWARE_SPEC_ID: {
                        "id": meta_props[self.ConfigurationMetaNames.SOFTWARE_SPEC_ID]
                    }
                }
            )

        metadata.update(
            {self.ConfigurationMetaNames.FOUNDATION_MODEL: {"model_id": model}}
        )

        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        else:
            if self._client.default_project_id is not None:
                metadata.update({"project_id": self._client.default_project_id})
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="it_is_mandatory_to_set_the_space_project_id"
                    )
                )

        if self._client.CLOUD_PLATFORM_SPACES:
            if self.ConfigurationMetaNames.MODEL_LOCATION not in meta_props:
                raise WMLClientError("model_location missing in meta_props")

            conn_id = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "connection_id"
            )
            bucket = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "bucket"
            )
            path = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "file_path"
            )
            conn_type = meta_props[self.ConfigurationMetaNames.MODEL_LOCATION].get(
                "type"
            )

            if conn_id is None:
                raise WMLClientError("connection_id missing in meta_props")

            if bucket is None:
                raise WMLClientError("bucket missing in meta_props")

            if path is None:
                raise WMLClientError("file_path missing in meta_props")

            model_location = {
                "type": conn_type if conn_type else "connection_asset",
                "connection": {"id": conn_id},
                "location": {"bucket": bucket, "file_path": path},
            }

            metadata[self.ConfigurationMetaNames.FOUNDATION_MODEL][
                self.ConfigurationMetaNames.MODEL_LOCATION
            ] = model_location

            if self.ConfigurationMetaNames.MODEL_LOCATION in metadata:
                del metadata[self.ConfigurationMetaNames.MODEL_LOCATION]

        return metadata

    def _create_type_of_model_payload(
        self,
        model: str,
        meta_props: dict[str, Any],
        model_type: Literal["curated", "base"],
    ) -> dict[str, Any]:
        metadata = copy.deepcopy(meta_props)

        Models._validate_type(model, "model", str, True)

        if self.ConfigurationMetaNames.TYPE in meta_props:
            self._validate_meta_prop(
                meta_props, self.ConfigurationMetaNames.TYPE, str, True
            )

        if model_type == "curated":
            if not model.endswith(f"-curated"):
                model += f"-curated"

        elif model_type == "base":
            if self.ConfigurationMetaNames.SOFTWARE_SPEC_ID in meta_props:
                self._validate_meta_prop(
                    meta_props, self.ConfigurationMetaNames.SOFTWARE_SPEC_ID, str, True
                )
                metadata.update(
                    {
                        self.ConfigurationMetaNames.SOFTWARE_SPEC_ID: {
                            "id": meta_props[
                                self.ConfigurationMetaNames.SOFTWARE_SPEC_ID
                            ]
                        }
                    }
                )

        metadata.update(
            {self.ConfigurationMetaNames.FOUNDATION_MODEL: {"model_id": model}}
        )

        if self._client.default_space_id is not None:
            metadata["space_id"] = self._client.default_space_id
        else:
            if self._client.default_project_id is not None:
                metadata.update({"project_id": self._client.default_project_id})
            else:
                raise WMLClientError(
                    Messages.get_message(
                        message_id="it_is_mandatory_to_set_the_space_project_id"
                    )
                )

        return metadata

    def _model_content_compress_artifact(
        self, type_name: str, compress_artifact: str
    ) -> str:
        tar_filename = "{}_content.tar".format(type_name)
        gz_filename = "{}.gz".format(tar_filename)
        CompressionUtil.create_tar(compress_artifact, ".", tar_filename)
        CompressionUtil.compress_file_gzip(tar_filename, gz_filename)
        os.remove(tar_filename)
        return gz_filename

    def _is_h5(self, model_filepath: str) -> bool:
        return os.path.splitext(os.path.basename(model_filepath))[-1] == ".h5"

    def _get_last_run_metrics_name(self, training_id: str) -> str:
        run_metrics = self._client.training.get_metrics(training_id=training_id)
        last_run_metric = run_metrics[-1]
        metrics_name = (
            last_run_metric.get("context").get("intermediate_model").get("name")
        )
        return metrics_name
