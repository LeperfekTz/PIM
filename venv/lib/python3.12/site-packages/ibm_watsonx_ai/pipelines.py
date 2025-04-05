#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal

from ibm_watsonx_ai._wrappers import requests

from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import PipelineMetanames
from ibm_watsonx_ai.utils import PIPELINE_DETAILS_TYPE
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from pandas import DataFrame


class Pipelines(WMLResource):
    """Store and manage pipelines."""

    ConfigurationMetaNames = PipelineMetanames()
    """MetaNames for pipelines creation."""

    def __init__(self, client: APIClient) -> None:
        WMLResource.__init__(self, __name__, client)

    def _generate_pipeline_document(self, meta_props: dict) -> dict:
        if self._client.ICP_PLATFORM_SPACES:
            doc: dict[str, Any] = {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "wmla_only",
                "pipelines": [
                    {
                        "id": "wmla_only",
                        "runtime_ref": "hybrid",
                        "nodes": [
                            {
                                "id": "training",
                                "type": "model_node",
                                "op": "dl_train",
                                "runtime_ref": "DL_WMLA",
                                "inputs": [],
                                "outputs": [],
                                "parameters": {
                                    "name": "pipeline",
                                    "description": "Pipeline - Python client",
                                },
                            }
                        ],
                    }
                ],
                "schemas": [
                    {"id": "schema1", "fields": [{"name": "text", "type": "string"}]}
                ],
            }
        else:
            doc = {
                "doc_type": "pipeline",
                "version": "2.0",
                "primary_pipeline": "dlaas_only",
                "pipelines": [
                    {
                        "id": "dlaas_only",
                        "runtime_ref": "hybrid",
                        "nodes": [
                            {
                                "id": "training",
                                "type": "model_node",
                                "op": "dl_train",
                                "runtime_ref": "DL",
                                "inputs": [],
                                "outputs": [],
                                "parameters": {
                                    "name": "tf-mnist",
                                    "description": "Simple MNIST model implemented in TF",
                                },
                            }
                        ],
                    }
                ],
                "schemas": [
                    {"id": "schema1", "fields": [{"name": "text", "type": "string"}]}
                ],
            }

        if self.ConfigurationMetaNames.COMMAND in meta_props:
            doc["pipelines"][0]["nodes"][0]["parameters"]["command"] = meta_props[
                self.ConfigurationMetaNames.COMMAND
            ]
        if self.ConfigurationMetaNames.RUNTIMES in meta_props:
            doc["runtimes"] = meta_props[self.ConfigurationMetaNames.RUNTIMES]
            if self._client.ICP_PLATFORM_SPACES:
                doc["runtimes"][0]["id"] = "DL_WMLA"
            else:
                doc["runtimes"][0]["id"] = "DL"

        if self.ConfigurationMetaNames.COMPUTE in meta_props:
            doc["pipelines"][0]["nodes"][0]["parameters"]["compute"] = meta_props[
                self.ConfigurationMetaNames.COMPUTE
            ]
        return doc

    def store(self, meta_props: dict, **kwargs: Any) -> dict:
        """Create a pipeline.

        :param meta_props: metadata of the pipeline configuration. To see available meta names, use:

            .. code-block:: python

                client.pipelines.ConfigurationMetaNames.get()

        :type meta_props: dict

        :return: stored pipeline metadata
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.pipelines.ConfigurationMetaNames.NAME: 'my_training_definition',
                client.pipelines.ConfigurationMetaNames.DOCUMENT: {"doc_type":"pipeline",
                                                                   "version": "2.0",
                                                                   "primary_pipeline": "dlaas_only",
                                                                   "pipelines": [{"id": "dlaas_only",
                                                                                  "runtime_ref": "hybrid",
                                                                                  "nodes": [{"id": "training",
                                                                                             "type": "model_node",
                                                                                             "op": "dl_train",
                                                                                             "runtime_ref": "DL",
                                                                                             "inputs": [],
                                                                                             "outputs": [],
                                                                                             "parameters": {"name": "tf-mnist",
                                                                                                            "description": "Simple MNIST model implemented in TF",
                                                                                                            "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                                                                                                            "compute": {"name": "k80","nodes": 1},
                                                                                                            "training_lib_href": "/v4/libraries/64758251-bt01-4aa5-a7ay-72639e2ff4d2/content"
                                                                                             },
                                                                                             "target_bucket": "wml-dev-results"
                                                                                  }]
                                                                   }]
                }
            }
            pipeline_details = client.pipelines.store(training_definition_filepath, meta_props=metadata)

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        # quick support for COS credentials instead of local path
        # TODO add error handling and cleaning (remove the file)
        Pipelines._validate_type(meta_props, "meta_props", dict, True)

        if self.ConfigurationMetaNames.DOCUMENT in meta_props:
            pipeline_meta = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )
        else:
            document = self._generate_pipeline_document(meta_props)
            meta_props[self.ConfigurationMetaNames.DOCUMENT] = document
            pipeline_meta = self.ConfigurationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

        if self._client.ICP_PLATFORM_SPACES:
            if self._client.default_space_id is not None:
                pipeline_meta["space"] = {
                    "href": "/v4/spaces/" + self._client.default_space_id
                }
            elif self._client.default_project_id is not None:
                pipeline_meta["project"] = {
                    "href": "/v2/projects/" + self._client.default_project_id
                }
            else:
                raise WMLClientError(
                    "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_UID>)/client.set.default_project(<PROJECT_UID>) to proceed."
                )

        if self._client.default_space_id is not None:
            pipeline_meta["space_id"] = self._client.default_space_id
        elif self._client.default_project_id is not None:
            pipeline_meta["project_id"] = self._client.default_project_id
        else:
            raise WMLClientError(
                "It is mandatory to set the space/project id. Use client.set.default_space(<SPACE_ID>)/client.set.default_project(<PROJECT_ID>) to proceed."
            )

        # add kwargs into optimization section at the very end of preparing payload
        try:
            for p in pipeline_meta[self.ConfigurationMetaNames.DOCUMENT]["pipelines"]:
                for n in p["nodes"]:
                    params = n["parameters"]["optimization"]
                    params.update(kwargs)
                    n["parameters"]["optimization"] = params
        except:
            pass

        creation_response = requests.post(
            self._client.service_instance._href_definitions.get_pipelines_href(),
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True),
            json=pipeline_meta,
        )

        pipeline_details = self._handle_response(
            201, "creating new pipeline", creation_response
        )

        return pipeline_details

    def create_revision(self, pipeline_id: str | None = None, **kwargs: Any) -> dict:
        """Create a new pipeline revision.

        :param pipeline_id: unique ID of the pipeline
        :type pipeline_id: str

        :return: details of the pipeline revision
        :rtype: dict

        **Example:**

        .. code-block:: python

            client.pipelines.create_revision(pipeline_id)

        """
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")

        Pipelines._validate_type(pipeline_id, "pipeline_id", str, False)

        url = self._client.service_instance._href_definitions.get_pipelines_href()
        return self._create_revision_artifact(url, pipeline_id, "pipelines")

    def update(
        self,
        pipeline_id: str | None = None,
        changes: dict | None = None,
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update metadata of an existing pipeline.

        :param pipeline_id: unique ID of the pipeline to be updated
        :type pipeline_id: str
        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict
        :param rev_id: revision ID of the pipeline
        :type rev_id: str

        :return: metadata of the updated pipeline
        :rtype: dict

        **Example:**

        .. code-block:: python

            metadata = {
                client.pipelines.ConfigurationMetaNames.NAME: "updated_pipeline"
            }
            pipeline_details = client.pipelines.update(pipeline_id, changes=metadata)

        """
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        if changes is None:
            raise TypeError("Missing required positional argument 'changes'")

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        self._validate_type(pipeline_id, "pipeline_id", str, True)
        self._validate_type(changes, "changes", dict, True)

        details = self._client.pipelines.get_details(pipeline_id)

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            details["entity"], changes, with_validation=True
        )

        url = self._client.service_instance._href_definitions.get_pipeline_href(
            pipeline_id
        )

        response = requests.patch(
            url,
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        updated_details = self._handle_response(200, "pipeline patch", response)

        return updated_details

    def delete(
        self, pipeline_id: str | None = None, **kwargs: Any
    ) -> Literal["SUCCESS"]:
        """Delete a stored pipeline.

        :param pipeline_id: unique ID of the pipeline
        :type pipeline_id: str

        :return: status "SUCCESS" if deletion is successful
        :rtype: Literal["SUCCESS"]

        **Example:**

        .. code-block:: python

            client.pipelines.delete(pipeline_id)

        """
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_id, "pipeline_id", str, True)

        pipeline_endpoint = (
            self._client.service_instance._href_definitions.get_pipeline_href(
                pipeline_id
            )
        )

        response_delete = requests.delete(
            pipeline_endpoint,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "pipeline deletion", response_delete, False)

    def get_details(
        self,
        pipeline_id: str | None = None,
        limit: int | None = None,
        asynchronous: bool | None = False,
        get_all: bool | None = False,
        pipeline_name: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get metadata of stored pipeline(s). If neither pipeline ID nor pipeline name is specified,
        the metadata of all pipelines is returned.
        If only pipeline name is specified, metadata of pipelines with the name is returned (if any).

        :param pipeline_id: ID of the pipeline
        :type pipeline_id: str, optional
        :param limit: limit number of fetched records
        :type limit: int, optional
        :param asynchronous: if `True`, it will work as a generator
        :type asynchronous: bool, optional
        :param get_all: if `True`, it will get all entries in 'limited' chunks
        :type get_all: bool, optional
        :param pipeline_name: name of the pipeline, can be used only when `pipeline_id` is None
        :type pipeline_name: str, optional

        :return: metadata of pipeline(s)
        :rtype: dict (if ID is not None) or {"resources": [dict]} (if ID is None)

        **Example:**

        .. code-block:: python

            pipeline_details = client.pipelines.get_details(pipeline_id)
            pipeline_details = client.pipelines.get_details(pipeline_name='Sample_pipeline')
            pipeline_details = client.pipelines.get_details()
            pipeline_details = client.pipelines.get_details(limit=100)
            pipeline_details = client.pipelines.get_details(limit=100, get_all=True)
            pipeline_details = []
            for entry in client.pipelines.get_details(limit=100, asynchronous=True, get_all=True):
                pipeline_details.extend(entry)

        """
        pipeline_id = _get_id_from_deprecated_uid(
            kwargs, pipeline_id, "pipeline", can_be_none=True
        )

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_id, "pipeline_id", str, False)
        Pipelines._validate_type(limit, "limit", int, False)
        url = self._client.service_instance._href_definitions.get_pipelines_href()

        if pipeline_id is None:
            filter_func = (
                self._get_filter_func_by_artifact_name(pipeline_name)
                if pipeline_name
                else None
            )
            return self._get_artifact_details(
                url,
                pipeline_id,
                limit,
                "pipelines",
                summary=False,
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )
        else:
            return self._get_artifact_details(
                url, pipeline_id, limit, "pipeline", summary=False
            )

    def get_revision_details(
        self, pipeline_id: str | None = None, rev_id: str | None = None, **kwargs: Any
    ) -> dict:
        """Get metadata of a pipeline revision.

        :param pipeline_id: ID of the stored pipeline
        :type pipeline_id: str

        :param rev_id: revision ID of the stored pipeline
        :type rev_id: str

        :return: revised metadata of the stored pipeline
        :rtype: dict

        **Example:**

        .. code-block:: python

            pipeline_details = client.pipelines.get_revision_details(pipeline_id, rev_id)

        .. note::
            `rev_id` parameter is not applicable in Cloud platform.
        """
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")
        rev_id = _get_id_from_deprecated_uid(kwargs, rev_id, "rev")

        Pipelines._validate_type(pipeline_id, "pipeline_id", str, True)
        Pipelines._validate_type(rev_id, "rev_id", str, True)

        url = self._client.service_instance._href_definitions.get_pipeline_href(
            pipeline_id
        )

        return self._get_with_or_without_limit(
            url,
            limit=None,
            op_name="pipeline",
            summary=None,
            pre_defined=None,
            revision=rev_id,
        )

    @staticmethod
    def get_href(pipeline_details: dict) -> str:
        """Get the href from pipeline details.

        :param pipeline_details: metadata of the stored pipeline
        :type pipeline_details: dict

        :return: href of the pipeline
        :rtype: str

        **Example:**

        .. code-block:: python

            pipeline_details = client.pipelines.get_details(pipeline_id)
            pipeline_href = client.pipelines.get_href(pipeline_details)

        """
        Pipelines._validate_type(pipeline_details, "pipeline_details", object, True)

        if "asset_type" in pipeline_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                pipeline_details, "pipeline_details", ["metadata", "href"]
            )
        else:
            if "href" in pipeline_details["metadata"]:
                Pipelines._validate_type_of_details(
                    pipeline_details, PIPELINE_DETAILS_TYPE
                )
                return WMLResource._get_required_element_from_dict(
                    pipeline_details, "pipeline_details", ["metadata", "href"]
                )
            else:
                pipeline_id = WMLResource._get_required_element_from_dict(
                    pipeline_details, "pipeline_details", ["metadata", "id"]
                )
                return "/ml/v4/pipelines/" + pipeline_id

    @staticmethod
    def get_id(pipeline_details: dict) -> str:
        """Get the pipeline ID from pipeline details.

        :param pipeline_details: metadata of the stored pipeline
        :type pipeline_details: dict

        :return: unique ID of the pipeline
        :rtype: str

        **Example:**

        .. code-block:: python

            pipeline_id = client.pipelines.get_id(pipeline_details)

        """
        Pipelines._validate_type(pipeline_details, "pipeline_details", object, True)
        if "asset_id" in pipeline_details["metadata"]:
            return WMLResource._get_required_element_from_dict(
                pipeline_details, "pipeline_details", ["metadata", "asset_id"]
            )
        else:
            if "id" not in pipeline_details["metadata"]:
                Pipelines._validate_type_of_details(
                    pipeline_details, PIPELINE_DETAILS_TYPE
                )

            return WMLResource._get_required_element_from_dict(
                pipeline_details, "pipeline_details", ["metadata", "id"]
            )

    def list(self, limit: int | None = None) -> DataFrame:
        """List stored pipelines in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed pipelines
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.pipelines.list()

        """
        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        pipeline_resources = self.get_details(
            get_all=self._should_get_all_values(limit)
        )["resources"]

        pipeline_values = [
            (
                m["metadata"]["id"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in pipeline_resources
        ]

        table = self._list(
            pipeline_values,
            ["ID", "NAME", "CREATED"],
            limit,
        )

        return table

    def list_revisions(
        self, pipeline_id: str | None = None, limit: int | None = None, **kwargs: Any
    ) -> DataFrame:
        """List all revision for a given pipeline ID in a table format.

        :param pipeline_id: unique ID of the stored pipeline
        :type pipeline_id: str

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed revisions
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.pipelines.list_revisions(pipeline_id)

        """
        pipeline_id = _get_id_from_deprecated_uid(kwargs, pipeline_id, "pipeline")

        ##For CP4D, check if either spce or project ID is set
        self._client._check_if_either_is_set()

        Pipelines._validate_type(pipeline_id, "pipeline_id", str, True)

        url = self._client.service_instance._href_definitions.get_pipeline_href(
            pipeline_id
        )

        pipeline_resources = self._get_artifact_details(
            url,
            "revisions",
            None,
            "pipeline revisions",
            _all=self._should_get_all_values(limit),
        )["resources"]

        pipeline_values = [
            (
                m["metadata"]["rev"],
                m["metadata"]["name"],
                m["metadata"]["created_at"],
            )
            for m in pipeline_resources
        ]

        table = self._list(
            pipeline_values,
            ["REV", "NAME", "CREATED"],
            limit,
        )

        return table

    def clone(
        self,
        pipeline_id: str | None = None,
        space_id: str | None = None,
        action: str | None = "copy",
        rev_id: str | None = None,
        **kwargs: Any,
    ) -> dict:
        raise WMLClientError(Messages.get_message(message_id="cloning_not_supported"))
