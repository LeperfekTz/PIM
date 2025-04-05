#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from __future__ import annotations
from typing import (
    Literal,
    Iterable,
    Callable,
    Any,
    cast,
    TYPE_CHECKING,
    NoReturn,
    Generator,
    TypeAlias,
    AsyncGenerator,
)
import numpy as np
import json
from warnings import warn
from enum import Enum

from ibm_watsonx_ai.utils import (
    print_text_header_h1,
    print_text_header_h2,
    StatusLogger,
)
from ibm_watsonx_ai.utils.utils import _get_id_from_deprecated_uid
from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    MissingValue,
    InvalidValue,
    ApiRequestFailure,
)

from ibm_watsonx_ai.href_definitions import is_id
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.metanames import (
    ScoringMetaNames,
    DecisionOptimizationMetaNames,
    DeploymentMetaNames,
)
from ibm_watsonx_ai.libs.repo.util.library_imports import LibraryChecker
from ibm_watsonx_ai.utils.autoai.utils import all_logging_disabled

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.lifecycle import SpecStates
    import pandas

lib_checker = LibraryChecker()

ListType: TypeAlias = list


class Deployments(WMLResource):
    """Deploy and score published artifacts (models and functions)."""

    DEFAULT_CONCURRENCY_LIMIT = 8

    class HardwareRequestSizes(str, Enum):
        """
        An enum class that represents the different hardware request sizes
        available.
        """

        Small = "gpu_s"
        Medium = "gpu_m"
        Large = "gpu_l"

    def __init__(self, client: APIClient):
        WMLResource.__init__(self, __name__, client)
        self.ConfigurationMetaNames = DeploymentMetaNames()
        self.ScoringMetaNames = ScoringMetaNames()
        self.DecisionOptimizationMetaNames = DecisionOptimizationMetaNames()

    def _deployment_status_errors_handling(
        self, deployment_details: dict, operation_name: str, deployment_id: str
    ) -> NoReturn:
        try:
            if "failure" in deployment_details["entity"]["status"]:
                errors = deployment_details["entity"]["status"]["failure"]["errors"]
                for error in errors:
                    if type(error) == str:
                        try:
                            error_obj = json.loads(error)
                            print(error_obj["message"])
                        except:
                            print(error)
                    elif type(error) == dict:
                        print(error["message"])
                    else:
                        print(error)
                raise WMLClientError(
                    "Deployment "
                    + operation_name
                    + " failed for deployment id: "
                    + deployment_id
                    + ". Errors: "
                    + str(errors)
                )
            else:
                print(deployment_details["entity"]["status"])
                raise WMLClientError(
                    "Deployment "
                    + operation_name
                    + " failed for deployment id: "
                    + deployment_id
                    + ". Error: "
                    + str(deployment_details["entity"]["status"]["state"])
                )
        except WMLClientError as e:
            raise e
        except Exception as e:
            self._logger.debug("Deployment " + operation_name + " failed: " + str(e))
            print(deployment_details["entity"]["status"]["failure"])
            raise WMLClientError(
                "Deployment "
                + operation_name
                + " failed for deployment id: "
                + deployment_id
                + "."
            )

    # TODO model_id and artifact_id should be changed to artifact_id only
    def create(
        self,
        artifact_id: str | None = None,
        meta_props: dict | None = None,
        rev_id: str | None = None,
        **kwargs: dict,
    ) -> dict:
        """Create a deployment from an artifact. An artifact is a model or function that can be deployed.

        :param artifact_id: ID of the published artifact (the model or function ID)
        :type artifact_id: str

        :param meta_props: meta props. To see the available list of meta names, use:

            .. code-block:: python

                client.deployments.ConfigurationMetaNames.get()

        :type meta_props: dict, optional

        :param rev_id: revision ID of the deployment
        :type rev_id: str, optional

        :return: metadata of the created deployment
        :rtype: dict

        **Example:**

        .. code-block:: python

            meta_props = {
                client.deployments.ConfigurationMetaNames.NAME: "SAMPLE DEPLOYMENT NAME",
                client.deployments.ConfigurationMetaNames.ONLINE: {},
                client.deployments.ConfigurationMetaNames.HARDWARE_SPEC : { "id":  "e7ed1d6c-2e89-42d7-aed5-8sb972c1d2b"},
                client.deployments.ConfigurationMetaNames.SERVING_NAME : 'sample_deployment'
            }
            deployment_details = client.deployments.create(artifact_id, meta_props)

        """
        artifact_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=artifact_id, resource_name="artifact"
        )
        # Backward compatibility in past `rev_id` was an int.
        if isinstance(rev_id, int):
            rev_id_as_int_deprecated = (
                "`rev_id` parameter type as int is deprecated, "
                "please convert to str instead"
            )
            warn(rev_id_as_int_deprecated, category=DeprecationWarning)
            rev_id = str(rev_id)

        Deployments._validate_type(artifact_id, "artifact_id", str, True)

        if self._client.ICP_PLATFORM_SPACES:
            predictionUrl = self._credentials.url

        if meta_props is None:
            raise WMLClientError("Invalid input. meta_props can not be empty.")

        if self._client.CLOUD_PLATFORM_SPACES and "r_shiny" in meta_props:
            raise WMLClientError("Shiny is not supported in this release")

        if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            base_model_id = meta_props.get(self.ConfigurationMetaNames.BASE_MODEL_ID)

            if isinstance(base_model_id, ModelTypes):
                meta_props[self.ConfigurationMetaNames.BASE_MODEL_ID] = (
                    base_model_id.value
                )

        metaProps = self.ConfigurationMetaNames._generate_resource_metadata(meta_props)

        if (
            "serving_name" in str(metaProps)
            and meta_props.get("serving_name", False)
            and "r_shiny" in str(metaProps)
        ):
            if "parameters" in metaProps["r_shiny"]:
                metaProps["r_shiny"]["parameters"]["serving_name"] = meta_props[
                    "serving_name"
                ]
            else:
                metaProps["r_shiny"]["parameters"] = {
                    "serving_name": meta_props["serving_name"]
                }
            if "online" in metaProps:
                del metaProps["online"]

        if "wml_instance_id" in meta_props:
            metaProps.update({"wml_instance_id": meta_props["wml_instance_id"]})

        ##Check if default space is set
        metaProps["asset"] = (
            metaProps.get("asset") if metaProps.get("asset") else {"id": artifact_id}
        )
        if rev_id is not None:
            metaProps["asset"].update({"rev": rev_id})

        if self._client.default_project_id:
            metaProps["project_id"] = self._client.default_project_id
        else:
            metaProps["space_id"] = self._client.default_space_id

        # note: checking if artifact_id points to prompt_template
        if self._client.CPD_version >= 4.8 or self._client.CLOUD_PLATFORM_SPACES:
            with all_logging_disabled():
                try:
                    from ibm_watsonx_ai.foundation_models.prompts import (
                        PromptTemplateManager,
                    )

                    model_id = (
                        PromptTemplateManager(api_client=self._client)
                        .load_prompt(artifact_id)
                        .model_id
                    )
                except Exception:
                    pass  # Foundation models scenario should not impact other ML models' deployment scenario.
                else:
                    metaProps.pop("asset")
                    metaProps["prompt_template"] = {"id": artifact_id}
                    if (
                        DeploymentMetaNames.BASE_MODEL_ID not in metaProps
                        and DeploymentMetaNames.BASE_DEPLOYMENT_ID not in metaProps
                    ):
                        metaProps.update({DeploymentMetaNames.BASE_MODEL_ID: model_id})
        # --- end note

        url = self._client.service_instance._href_definitions.get_deployments_href()

        response = self._client.httpx_client.post(
            url,
            json=metaProps,
            params=self._client._params(),  # version is mandatory
            headers=self._client._get_headers(),
        )

        ## Post Deployment call executed
        if response.status_code == 202:
            deployment_details = response.json()

            if kwargs.get("background_mode"):
                background_mode_turned_on_warning = (
                    "Background mode is turn on and deployment scoring will be available only when status of deployment will be `ready`. "
                    "To check deployment status run `client.deployment.get_details(deployment_id)"
                )
                warn(background_mode_turned_on_warning)
                return deployment_details
            else:

                if self._client.ICP_PLATFORM_SPACES:
                    if "online_url" in deployment_details["entity"]["status"]:
                        scoringUrl = (
                            deployment_details.get("entity")
                            .get("status")
                            .get("online_url")
                            .get("url")
                            .replace("https://ibm-nginx-svc:443", predictionUrl)
                        )
                        deployment_details["entity"]["status"]["online_url"][
                            "url"
                        ] = scoringUrl

                deployment_id = self.get_id(deployment_details)

                import time

                print_text_header_h1(
                    "Synchronous deployment creation for id: '{}' started".format(
                        artifact_id
                    )
                )

                status = deployment_details["entity"]["status"]["state"]

                notifications = []

                with StatusLogger(status) as status_logger:
                    while True:
                        time.sleep(5)
                        deployment_details = self._client.deployments.get_details(
                            deployment_id, _silent=True
                        )
                        # this is wrong , needs to update for ICP
                        if "system" in deployment_details:
                            notification = deployment_details["system"]["warnings"][0][
                                "message"
                            ]
                            if notification not in notifications:
                                print("\nNote: " + notification)
                                notifications.append(notification)

                        status = deployment_details["entity"]["status"]["state"]
                        status_logger.log_state(status)
                        if status != "DEPLOY_IN_PROGRESS" and status != "initializing":
                            break
                if status == "DEPLOY_SUCCESS" or status == "ready":
                    print("")
                    print_text_header_h2(
                        "Successfully finished deployment creation, deployment_id='{}'".format(
                            deployment_id
                        )
                    )
                    return deployment_details
                else:
                    print_text_header_h2("Deployment creation failed")
                    self._deployment_status_errors_handling(
                        deployment_details, "creation", deployment_id
                    )
        else:
            error_msg = "Deployment creation failed"
            reason = response.text
            print(reason)
            print_text_header_h2(error_msg)
            raise WMLClientError(
                error_msg + ". Error: " + str(response.status_code) + ". " + reason
            )

    @staticmethod
    def get_uid(deployment_details: dict) -> str:
        """Get deployment_uid from the deployment details.

        *Deprecated:* Use ``get_id(deployment_details)`` instead.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment UID that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_uid = client.deployments.get_uid(deployment)

        """
        get_uid_deprecated_warning = (
            "`get_uid()` is deprecated and will be removed in future. "
            "Instead, please use `get_id()`."
        )
        warn(get_uid_deprecated_warning, category=DeprecationWarning)
        return Deployments.get_id(deployment_details)

    @staticmethod
    def get_id(deployment_details: dict) -> str:
        """Get the deployment ID from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment ID that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_id = client.deployments.get_id(deployment)

        """
        Deployments._validate_type(deployment_details, "deployment_details", dict, True)

        try:
            if "id" in deployment_details["metadata"]:
                id = deployment_details.get("metadata", {}).get("id")
            else:
                id = deployment_details.get("metadata", {}).get("guid")
        except Exception as e:
            raise WMLClientError(
                "Getting deployment ID from deployment details failed.", e
            )

        if id is None:
            raise MissingValue("deployment_details.metadata.id")

        return id

    @staticmethod
    def get_href(deployment_details: dict) -> str:
        """Get deployment_href from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: deployment href that is used to manage the deployment
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_href = client.deployments.get_href(deployment)

        """
        Deployments._validate_type(deployment_details, "deployment_details", dict, True)

        try:
            if "href" in deployment_details["metadata"]:
                url = deployment_details.get("metadata", {}).get("href")
            else:
                url = "/ml/v4/deployments/{}".format(
                    deployment_details["metadata"]["id"]
                )
        except Exception as e:
            raise WMLClientError(
                "Getting deployment url from deployment details failed.", e
            )

        if url is None:
            raise MissingValue("deployment_details.metadata.href")

        return url

    def _get_serving_name_info(self, serving_name: str) -> tuple:
        """Get info about the serving name

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about the serving name: (<status_code>, <response json if any>)
        :rtype: tuple

        **Example:**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available('test')

        """
        params = {
            "serving_name": serving_name,
            "conflict": "true",
            "version": self._client.version_param,
        }

        url = self._client.service_instance._href_definitions.get_deployments_href()
        res = self._client.httpx_client.get(
            url, headers=self._client._get_headers(), params=params
        )

        if res.status_code == 409:
            response = res.json()
        else:
            response = None

        return (res.status_code, response)

    def is_serving_name_available(self, serving_name: str) -> bool:
        """Check if the serving name is available for use.

        :param serving_name: serving name that filters deployments
        :type serving_name: str

        :return: information about whether the serving name is available
        :rtype: bool

        **Example:**

        .. code-block:: python

            is_available = client.deployments.is_serving_name_available('test')

        """
        status_code, _ = self._get_serving_name_info(serving_name)

        return status_code != 409

    def get_details(
        self,
        deployment_id: str | None = None,
        serving_name: str | None = None,
        limit: int | None = None,
        asynchronous: bool = False,
        get_all: bool = False,
        spec_state: SpecStates | None = None,
        _silent: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Get information about deployment(s).
        If deployment_id is not passed, all deployment details are returned.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str, optional

        :param serving_name: serving name that filters deployments
        :type serving_name: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param asynchronous: if True, it will work as a generator
        :type asynchronous: bool, optional

        :param get_all: if True, it will get all entries in 'limited' chunks
        :type get_all: bool, optional

        :param spec_state: software specification state, can be used only when `deployment_id` is None
        :type spec_state: SpecStates, optional

        :return: metadata of the deployment(s)
        :rtype: dict (if deployment_id is not None) or {"resources": [dict]} (if deployment_id is None)

        **Example:**

        .. code-block:: python

            deployment_details = client.deployments.get_details(deployment_id)
            deployment_details = client.deployments.get_details(deployment_id=deployment_id)
            deployments_details = client.deployments.get_details()
            deployments_details = client.deployments.get_details(limit=100)
            deployments_details = client.deployments.get_details(limit=100, get_all=True)
            deployments_details = []
            for entry in client.deployments.get_details(limit=100, asynchronous=True, get_all=True):
                deployments_details.extend(entry)

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs,
            resource_id=deployment_id,
            resource_name="deployment",
            can_be_none=True,
        )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, False)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(
                "'deployment_id' is not an id: '{}'".format(deployment_id)
            )

        url = self._client.service_instance._href_definitions.get_deployments_href()

        query_params = self._client._params()

        if serving_name:
            query_params["serving_name"] = serving_name

        if deployment_id is None:
            filter_func = (
                self._get_filter_func_by_spec_state(spec_state) if spec_state else None
            )

            deployment_details = self._get_artifact_details(
                base_url=url,
                id=deployment_id,
                limit=limit,
                resource_name="deployments",
                query_params=query_params,
                _async=asynchronous,
                _all=get_all,
                _filter_func=filter_func,
            )
        else:
            deployment_details = self._get_artifact_details(
                url,
                deployment_id,
                limit,
                "deployments",
                query_params=query_params,
            )

        if (
            not isinstance(deployment_details, Generator)
            and "system" in deployment_details
            and not _silent
        ):
            print("Note: " + deployment_details["system"]["warnings"][0]["message"])

        return deployment_details

    @staticmethod
    def get_scoring_href(deployment_details: dict) -> str:
        """Get scoring URL from deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: scoring endpoint URL that is used to make scoring requests
        :rtype: str

        **Example:**

        .. code-block:: python

            scoring_href = client.deployments.get_scoring_href(deployment)

        """

        Deployments._validate_type(deployment_details, "deployment", dict, True)

        scoring_url = None
        try:
            url = deployment_details["entity"]["status"].get("online_url")
            if url is not None:
                scoring_url = deployment_details["entity"]["status"]["online_url"][
                    "url"
                ]
            else:
                raise MissingValue(
                    "Getting scoring url for deployment failed. This functionality is  available only for sync deployments"
                )

        except Exception as e:
            raise WMLClientError(
                "Getting scoring url for deployment failed. This functionality is  available only for sync deployments",
                e,
            )

        if scoring_url is None:
            raise MissingValue("scoring_url missing in online_predictions")
        return scoring_url

    @staticmethod
    def get_serving_href(deployment_details: dict) -> str:
        """Get serving URL from the deployment details.

        :param deployment_details: metadata of the deployment
        :type deployment_details: dict

        :return: serving endpoint URL that is used to make scoring requests
        :rtype: str

        **Example:**

        .. code-block:: python

            scoring_href = client.deployments.get_serving_href(deployment)

        """

        Deployments._validate_type(deployment_details, "deployment", dict, True)

        try:
            serving_name = (
                deployment_details["entity"]["online"]
                .get("parameters")
                .get("serving_name")
            )
            serving_url = [
                url
                for url in deployment_details["entity"]
                .get("status")
                .get("serving_urls")
                if serving_name == url.split("/")[-2]
            ][0]

            if serving_url:
                return serving_url
            else:
                raise MissingValue(
                    "Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name."
                )

        except Exception as e:
            raise WMLClientError(
                "Getting serving url for deployment failed. This functionality is available only for sync deployments with serving name.",
                e,
            )

    def delete(self, deployment_id: str | None = None, **kwargs: Any) -> str:
        """Delete a deployment.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.deployments.delete(deployment_id)

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=deployment_id, resource_name="deployment"
        )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if deployment_id is not None and not is_id(deployment_id):
            raise WMLClientError(
                "'deployment_id' is not an id: '{}'".format(deployment_id)
            )

        deployment_url = (
            self._client.service_instance._href_definitions.get_deployment_href(
                deployment_id
            )
        )

        response_delete = self._client.httpx_client.delete(
            deployment_url,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        return self._handle_response(204, "deployment deletion", response_delete, False)

    def score(
        self, deployment_id: str, meta_props: dict, transaction_id: str | None = None
    ) -> dict:
        """Make scoring requests against the deployed artifact.

        :param deployment_id: unique ID of the deployment to be scored
        :type deployment_id: str

        :param meta_props: meta props for scoring, use ``client.deployments.ScoringMetaNames.show()`` to view the list of ScoringMetaNames
        :type meta_props: dict

        :param transaction_id: transaction ID to be passed with the records during payload logging
        :type transaction_id: str, optional

        :return: scoring result that contains prediction and probability
        :rtype: dict

        .. note::

                * *client.deployments.ScoringMetaNames.INPUT_DATA* is the only metaname valid for sync scoring.
                * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA:
                [{'fields':
                    ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                    'values': [
                        ['M',23,'Single','Student'],
                        ['M',55,'Single','Executive']
                    ]
                }]
            }
            predictions = client.deployments.score(deployment_id, scoring_payload)

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if meta_props.get(self.ScoringMetaNames.INPUT_DATA) is None:
            raise WMLClientError(
                "Scoring data input 'ScoringMetaNames.INPUT_DATA' is mandatory for synchronous "
                "scoring"
            )

        scoring_data = meta_props[self.ScoringMetaNames.INPUT_DATA]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name="pandas")
                import pandas as pd

                scoring_values = each_score_request["values"]
                # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                if isinstance(scoring_values, pd.DataFrame):
                    scoring_values = scoring_values.where(
                        pd.notnull(scoring_values), None
                    )
                    fields_names = scoring_values.columns.values.tolist()
                    values = scoring_values.values.tolist()

                    try:
                        values[pd.isnull(values)] = None

                        # note: above code fails when there is no null values in a dataframe
                    except TypeError:
                        pass

                    each_score_request["values"] = values
                    if fields_names is not None:
                        each_score_request["fields"] = fields_names

                ## If payload is a numpy dataframe

                elif isinstance(scoring_values, np.ndarray):

                    values = scoring_values.tolist()
                    each_score_request["values"] = values

                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

        payload = {}

        payload["input_data"] = score_payload

        if meta_props.get(self.ScoringMetaNames.SCORING_PARAMETERS) is not None:
            payload["scoring_parameters"] = meta_props.get(
                self.ScoringMetaNames.SCORING_PARAMETERS
            )

        headers = self._client._get_headers()

        if transaction_id is not None:
            headers.update({"x-global-transaction-id": transaction_id})

        scoring_url = (
            self._credentials.url
            + "/ml/v4/deployments/"
            + deployment_id
            + "/predictions"
        )

        params = self._client._params()
        del params["space_id"]
        response_scoring = self._client.httpx_client.post(
            scoring_url,
            json=payload,
            params=params,  # version parameter is mandatory
            headers=headers,
        )

        return self._handle_response(200, "scoring", response_scoring)

        #########################################

    def get_download_url(self, deployment_details: dict) -> str:
        """Get deployment_download_url from the deployment details.

        :param deployment_details: created deployment details
        :type deployment_details: dict

        :return: deployment download URL that is used to get file deployment (for example: Core ML)
        :rtype: str

        **Example:**

        .. code-block:: python

            deployment_url = client.deployments.get_download_url(deployment)

        """
        if self._client.ICP_PLATFORM_SPACES:
            raise WMLClientError(
                "Downloading virtual deployment is no longer supported in Cloud Pak for Data, versions 3.5 and later."
            )

        if self._client.CLOUD_PLATFORM_SPACES:
            raise WMLClientError(
                "Downloading virtual deployment is no longer supported in Cloud Pak for Data as a Service."
            )

        Deployments._validate_type(deployment_details, "deployment_details", dict, True)
        try:
            virtual_deployment_detaails = (
                deployment_details.get("entity", {})
                .get("status", {})
                .get("virtual_deployment_downloads")
            )
            if virtual_deployment_detaails is not None:
                url = virtual_deployment_detaails[0].get("url")
            else:
                url = None
        except Exception as e:
            raise WMLClientError(
                "Getting download url from deployment details failed.", e
            )

        if url is None:
            raise MissingValue(
                "deployment_details.entity.virtual_deployment_downloads.url"
            )

        return url

    def list(
        self, limit: int | None = None, artifact_type: str | None = None
    ) -> pandas.DataFrame:
        """Returns deployments in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :param artifact_type: return only deployments with the specified artifact_type
        :type artifact_type: str, optional

        :return: pandas.DataFrame with the listed deployments
        :rtype: pandas.DataFrame

        **Example:**

        .. code-block:: python

            client.deployments.list()

        """
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        details = self.get_details(get_all=self._should_get_all_values(limit))

        resources = details["resources"]

        values = []
        index = 0

        sw_spec_info = {
            s["id"]: s
            for s in self._client.software_specifications.get_details(state_info=True)[
                "resources"
            ]
        }

        def enrich_asset_with_type(asset_details: dict, asset_type: str) -> dict:
            if asset_type:
                asset_details["metadata"]["asset_type"] = asset_type

            return asset_details

        asset_info = {
            el["metadata"]["id"]: enrich_asset_with_type(el, asset_type)
            for asset_type, resources in {
                "model": self._client._models.get_details(get_all=True),
                "function": self._client._functions.get_details(get_all=True),
            }.items()
            for el in resources["resources"]
        }

        def get_spec_info(spec_id: str, prop: str) -> str:
            if spec_id and spec_id in sw_spec_info:
                return sw_spec_info[spec_id].get(prop, "")
            else:
                return ""

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            # Ideally this needs to be on the server side
            if limit is not None and index == limit:
                break

            spec_id = (
                asset_info.get(
                    m["entity"].get("asset", m["entity"].get("prompt_template"))["id"],
                    {},
                )
                .get("entity", {})
                .get("software_spec", {})
                .get("id")
            )

            if (
                artifact_type
                and m["entity"].get("deployed_asset_type", "unknown") != artifact_type
            ):
                pass  # filter by artifact_type
            else:
                values.append(
                    (
                        (
                            m["metadata"]["guid"]
                            if "guid" in m["metadata"]
                            else m["metadata"]["id"]
                        ),
                        m["entity"]["name"],
                        m["entity"]["status"]["state"],
                        m["metadata"]["created_at"],
                        m["entity"].get("deployed_asset_type", "unknown"),
                        get_spec_info(spec_id, "state"),
                        get_spec_info(spec_id, "replacement"),
                    )
                )

            index = index + 1

        table = self._list(
            values,
            [
                "ID",
                "NAME",
                "STATE",
                "CREATED",
                "ARTIFACT_TYPE",
                "SPEC_STATE",
                "SPEC_REPLACEMENT",
            ],
            limit,
        )

        return table

    def list_jobs(self, limit: int | None = None) -> pandas.DataFrame:
        """Return the async deployment jobs in a table format.

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: pandas.DataFrame with listed deployment jobs
        :rtype: pandas.DataFrame

        .. note::

            This method list only async deployment jobs created for WML deployment.

        **Example:**

        .. code-block:: python

            client.deployments.list_jobs()

        """

        details = self.get_job_details(limit=limit)
        resources = details["resources"]
        values = []
        index = 0

        for m in resources:
            # Deployment service currently doesn't support limit querying
            # As a workaround, its filtered in python client
            if limit is not None and index == limit:
                break

            if "scoring" in m["entity"]:
                state = m["entity"]["scoring"]["status"]["state"]
            else:
                state = m["entity"]["decision_optimization"]["status"]["state"]

            deploy_id = m["entity"]["deployment"]["id"]
            values.append(
                (m["metadata"]["id"], state, m["metadata"]["created_at"], deploy_id)
            )

            index = index + 1

        table = self._list(
            values, ["JOB-ID", "STATE", "CREATED", "DEPLOYMENT-ID"], limit
        )

        return table

    def _get_deployable_asset_type(self, details: dict) -> str:
        url = details["entity"]["asset"]["id"]
        if "model" in url:
            return "model"
        elif "function" in url:
            return "function"
        else:
            return "unknown"

    def update(
        self,
        deployment_id: str | None = None,
        changes: dict | None = None,
        **kwargs: Any,
    ) -> dict | None:
        """Updates existing deployment metadata. If ASSET is patched, then 'id' field is mandatory
        and it starts a deployment with the provided asset id/rev. Deployment ID remains the same.

        :param deployment_id: unique ID of deployment to be updated
        :type deployment_id: str

        :param changes: elements to be changed, where keys are ConfigurationMetaNames
        :type changes: dict

        :return: metadata of the updated deployment
        :rtype: dict or None

        **Examples**

        .. code-block:: python

            metadata = {client.deployments.ConfigurationMetaNames.NAME:"updated_Deployment"}
            updated_deployment_details = client.deployments.update(deployment_id, changes=metadata)

            metadata = {client.deployments.ConfigurationMetaNames.ASSET: {  "id": "ca0cd864-4582-4732-b365-3165598dc945",
                                                                            "rev":"2" }}
            deployment_details = client.deployments.update(deployment_id, changes=metadata)

        """
        deployment_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=deployment_id, resource_name="deployment"
        )
        if changes is None:
            raise TypeError(
                "update() missing 1 required positional argument: 'changes'"
            )

        Deployments._validate_type(changes, "changes", dict, True)
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            self._client._check_if_space_is_set()

        ret202 = False

        ## In case of passing 'AUTO_ONLINE_DEPLOYMENT' as true, we need to poll for deployment to be either 'deploy_success' or 'update_success'.

        Deployments._validate_type(deployment_id, "deployment_id", str, True)

        if ("asset" in changes and not changes["asset"]) and (
            "prompt_template" in changes and not changes["prompt_template"]
        ):
            msg = "ASSET/PROMPT_TEMPLATE cannot be empty. 'id' and 'rev' (only ASSET) fields are supported. 'id' is mandatory"
            print(msg)
            raise WMLClientError(msg)

        patch_job = (
            changes.get("asset") is not None
            or self.ConfigurationMetaNames.PROMPT_TEMPLATE in changes
            or self.ConfigurationMetaNames.SERVING_NAME in changes
            or self.ConfigurationMetaNames.OWNER in changes
        )

        patch_job_field = None
        if patch_job:
            if changes.get("asset") is not None:
                patch_job_field = "ASSET"
            elif self.ConfigurationMetaNames.PROMPT_TEMPLATE in changes:
                patch_job_field = "PROMPT_TEMPLATE"
            elif self.ConfigurationMetaNames.SERVING_NAME in changes:
                patch_job_field = "SERVING_NAME"
            elif self.ConfigurationMetaNames.OWNER in changes:
                patch_job_field = "OWNER"

            if patch_job_field is None:
                raise WMLClientError("Unexpected patch job element.")

        if patch_job and (len(changes) > 1):
            msg = (
                f"When {patch_job_field} is being updated/patched, other fields cannot be updated. If other fields are to be "
                f"updated, try without {patch_job_field} update. {patch_job_field} update triggers deployment with the new asset retaining "
                "the same deployment_id"
            )
            print(msg)
            raise WMLClientError(msg)

        deployment_details = self.get_details(deployment_id)
        serving_name_change = False
        new_serving_name = None
        if self.ConfigurationMetaNames.SERVING_NAME in changes:
            new_serving_name = changes.pop(self.ConfigurationMetaNames.SERVING_NAME)
            serving_name_change = True

        patch_payload = self.ConfigurationMetaNames._generate_patch_payload(
            deployment_details, changes, with_validation=True
        )

        if serving_name_change:
            replace = "serving_name" in deployment_details["entity"].get("online").get(
                "parameters", []
            )
            patch_payload.append(
                {
                    "op": "replace" if replace else "add",
                    "path": "/online/parameters",
                    "value": {"serving_name": new_serving_name},
                }
            )

        ## As auto_online_deployment and auto_redeploy values are passed as 'bool' but service needs them in 'str' format to patch.
        for ele in patch_payload:
            if (
                "auto_online_deployment" in ele["path"]
                or "auto_redeploy" in ele["path"]
            ):
                ele["value"] = str(ele["value"]).lower()

        url = self._client.service_instance._href_definitions.get_deployment_href(
            deployment_id
        )

        response = self._client.httpx_client.patch(
            url,
            json=patch_payload,
            params=self._client._params(),
            headers=self._client._get_headers(),
        )

        if patch_job and response.status_code == 202:
            updated_details = self._handle_response(
                202, "deployment asset patch", response
            )

            ret202 = True

            print(
                f"Since {patch_job_field} is patched, deployment with new asset id/rev is being started. "
                "Monitor the status using deployments.get_details(deployment_id) api"
            )
        elif response.status_code == 202:
            updated_details = self._handle_response(202, "deployment scaling", response)
            ret202 = True
        else:
            updated_details = self._handle_response(200, "deployment patch", response)

        if "auto_online_deployment" in changes:
            if response is not None:
                if response.status_code == 200:
                    deployment_details = self.get_details(deployment_id)

                    import time

                    print_text_header_h1(
                        " deployment update for id: '{}' started".format(deployment_id)
                    )

                    status = deployment_details["entity"]["status"]["state"]

                    with StatusLogger(status) as status_logger:
                        while True:
                            time.sleep(5)
                            deployment_details = self.get_details(deployment_id)
                            status = deployment_details["entity"]["status"]["state"]
                            status_logger.log_state(status)

                            if (
                                status != "DEPLOY_IN_PROGRESS"
                                and status != "UPDATE_IN_PROGRESS"
                            ):
                                break

                    if status == "DEPLOY_SUCCESS" or status == "UPDATE_SUCCESS":
                        print("")
                        print_text_header_h2(
                            "Successfully finished deployment update, deployment_id='{}'".format(
                                deployment_id
                            )
                        )
                        return deployment_details
                    else:
                        print_text_header_h2("Deployment update failed")
                        if deployment_id is not None:
                            self._deployment_status_errors_handling(
                                deployment_details, "update", deployment_id
                            )
                else:
                    error_msg = "Deployment update failed"
                    reason = response.text
                    print(reason)
                    print_text_header_h2(error_msg)
                    raise WMLClientError(
                        error_msg
                        + ". Error: "
                        + str(response.status_code)
                        + ". "
                        + reason
                    )

        if not ret202:
            return updated_details

        return None

    ## Below functions are for async scoring. They are just dummy functions.
    def _score_async(
        self,
        deployment_id: str,
        scoring_payload: dict,
        transaction_id: str | None = None,
        retention: int | None = None,
    ) -> str | dict:

        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(scoring_payload, "scoring_payload", dict, True)
        headers = self._client._get_headers()

        if transaction_id is not None:
            headers.update({"x-global-transaction-id": transaction_id})
        # making change - connection keep alive
        scoring_url = (
            self._client.service_instance._href_definitions.get_async_deployment_job_href()
        )
        params = self._client._params()

        if not self._client.ICP_PLATFORM_SPACES and retention is not None:
            if not isinstance(retention, int) or retention < -1:
                raise TypeError(
                    "`retention` takes integer values greater or equal than -1."
                )
            params.update({"retention": retention})

        response_scoring = self._client.httpx_client.post(
            scoring_url, params=params, json=scoring_payload, headers=headers
        )

        return self._handle_response(202, "scoring asynchronously", response_scoring)

    def create_job(
        self,
        deployment_id: str,
        meta_props: dict,
        retention: int | None = None,
        transaction_id: str | None = None,
        _asset_id: str | None = None,
    ) -> str | dict:
        """Create an asynchronous deployment job.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param meta_props: metaprops. To see the available list of metanames,
            use ``client.deployments.ScoringMetaNames.get()``
            or ``client.deployments.DecisionOptimizationmetaNames.get()``

        :type meta_props: dict

        :param retention: how many job days job meta should be retained,
            takes integer values >= -1, supported only on Cloud
        :type retention: int, optional

        :param transaction_id: transaction ID to be passed with the payload
        :type transaction_id: str, optional

        :return: metadata of the created async deployment job
        :rtype: dict or str

        .. note::

            * The valid payloads for scoring input are either list of values, pandas or numpy dataframes.

        **Example:**

        .. code-block:: python

            scoring_payload = {client.deployments.ScoringMetaNames.INPUT_DATA: [{'fields': ['GENDER','AGE','MARITAL_STATUS','PROFESSION'],
                                                                                     'values': [['M',23,'Single','Student'],
                                                                                                ['M',55,'Single','Executive']]}]}
            async_job = client.deployments.create_job(deployment_id, scoring_payload)

        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(meta_props, "meta_props", dict, True)

        if _asset_id:
            Deployments._validate_type(_asset_id, "_asset_id", str, True)
            # We assume that _asset_id is the id of the asset that was deployed
            # in the deployment with id deployment_id, and we save one REST call
            asset = _asset_id
        else:
            deployment_details = self.get_details(deployment_id)
            asset = deployment_details["entity"]["asset"]["id"]

        do_model = False
        asset_details = self._client.data_assets.get_details(asset)
        if (
            "wml_model" in asset_details["entity"]
            and "type" in asset_details["entity"]["wml_model"]
        ):
            if "do" in asset_details["entity"]["wml_model"]["type"]:
                do_model = True

        flag = 0  ## To see if it is async scoring or DecisionOptimization Job
        if do_model:
            payload = self.DecisionOptimizationMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )
            flag = 1
        else:
            payload = self.ScoringMetaNames._generate_resource_metadata(
                meta_props, with_validation=True, client=self._client
            )

        scoring_data = None
        if "scoring" in payload and "input_data" in payload["scoring"]:
            scoring_data = payload["scoring"]["input_data"]

        if (
            "decision_optimization" in payload
            and "input_data" in payload["decision_optimization"]
        ):
            scoring_data = payload["decision_optimization"]["input_data"]

        if scoring_data is not None:
            score_payload = []
            for each_score_request in scoring_data:
                lib_checker.check_lib(lib_name="pandas")
                import pandas as pd

                if "values" in each_score_request:
                    scoring_values = each_score_request["values"]
                    # Check feature types, currently supporting pandas df, numpy.ndarray, python lists and Dmatrix
                    if isinstance(scoring_values, pd.DataFrame):
                        fields_names = scoring_values.columns.values.tolist()
                        values = scoring_values.where(
                            pd.notnull(scoring_values), None
                        ).values.tolist()  # replace nan with None

                        each_score_request["values"] = values
                        if fields_names is not None:
                            each_score_request["fields"] = fields_names

                    ## If payload is a numpy dataframe

                    elif isinstance(scoring_values, np.ndarray):
                        # replace nan with None
                        values = np.where(
                            pd.notnull(scoring_values), scoring_values, None
                        ).tolist()  # type: ignore[call-overload]
                        each_score_request["values"] = values

                score_payload.append(each_score_request)

            ##See if it is scoring or DecisionOptimizationJob

            if flag == 0:
                payload["scoring"]["input_data"] = score_payload
            if flag == 1:
                payload["decision_optimization"]["input_data"] = score_payload

        import copy

        if "input_data_references" in meta_props:
            Deployments._validate_type(
                meta_props.get("input_data_references"),
                "input_data_references",
                list,
                True,
            )
            modified_input_data_references = False
            input_data = copy.deepcopy(meta_props.get("input_data_references"))
            input_data = cast(Iterable[Any], input_data)
            for i, input_data_fields in enumerate(input_data):
                if "connection" not in input_data_fields:
                    modified_input_data_references = True
                    input_data_fields.update({"connection": {}})
            if modified_input_data_references:
                if "scoring" in payload:
                    payload["scoring"].update({"input_data_references": input_data})
                else:
                    payload["decision_optimization"].update(
                        {"input_data_references": input_data}
                    )

        if "output_data_reference" in meta_props:
            Deployments._validate_type(
                meta_props.get("output_data_reference"),
                "output_data_reference",
                dict,
                True,
            )

            output_data = copy.deepcopy(meta_props.get("output_data_reference"))
            output_data = cast(dict, output_data)
            if (
                "connection" not in output_data
            ):  # and output_data.get('connection', None) is not None:
                output_data.update({"connection": {}})
                payload["scoring"].update({"output_data_reference": output_data})

        if "output_data_references" in meta_props:
            Deployments._validate_type(
                meta_props.get("output_data_references"),
                "output_data_references",
                list,
                True,
            )
            output_data = copy.deepcopy(meta_props.get("output_data_references"))
            modified_output_data_references = False
            output_data = cast(Iterable[Any], output_data)
            for i, output_data_fields in enumerate(output_data):
                if "connection" not in output_data_fields:
                    modified_output_data_references = True
                    output_data_fields.update({"connection": {}})
            if modified_output_data_references and "decision_optimization" in payload:
                payload["decision_optimization"].update(
                    {"output_data_references": output_data}
                )

        payload.update({"deployment": {"id": deployment_id}})
        if "hardware_spec" in meta_props:
            payload.update(
                {"hardware_spec": meta_props[self.ConfigurationMetaNames.HARDWARE_SPEC]}
            )
        if "hybrid_pipeline_hardware_specs" in meta_props:
            payload.update(
                {
                    "hybrid_pipeline_hardware_specs": meta_props[
                        self.ConfigurationMetaNames.HYBRID_PIPELINE_HARDWARE_SPECS
                    ]
                }
            )

        payload.update({"space_id": self._client.default_space_id})

        if "name" not in payload:
            import uuid

            payload.update({"name": "name_" + str(uuid.uuid4())})

        return self._score_async(
            deployment_id, payload, transaction_id=transaction_id, retention=retention
        )

    def get_job_details(
        self,
        job_id: str | None = None,
        include: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get information about deployment job(s).
        If deployment job_id is not passed, all deployment jobs details are returned.

        :param job_id: unique ID of the job
        :type job_id: str, optional

        :param include: fields to be retrieved from 'decision_optimization'
            and 'scoring' section mentioned as value(s) (comma separated) as output response fields
        :type include: str, optional

        :param limit: limit number of fetched records
        :type limit: int, optional

        :return: metadata of deployment job(s)
        :rtype: dict (if job_id is not None) or {"resources": [dict]} (if job_id is None)

        **Example:**

        .. code-block:: python

            deployment_details = client.deployments.get_job_details()
            deployments_details = client.deployments.get_job_details(job_id=job_id)

        """
        job_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=job_id, resource_name="job", can_be_none=True
        )
        if job_id is not None:
            Deployments._validate_type(job_id, "job_id", str, True)
        url = (
            self._client.service_instance._href_definitions.get_async_deployment_job_href()
        )

        params = self._client._params()
        if include:
            params["include"] = include

        return self._get_artifact_details(
            base_url=url,
            id=job_id,
            limit=limit,
            resource_name="async deployment job" if job_id else "async deployment jobs",
            query_params=params,
        )

    def get_job_status(self, job_id: str) -> dict:
        """Get the status of a deployment job.

        :param job_id: unique ID of the deployment job
        :type job_id: str

        :return: status of the deployment job
        :rtype: dict

        **Example:**

        .. code-block:: python

            job_status = client.deployments.get_job_status(job_id)

        """

        job_details = self.get_job_details(job_id)

        if "scoring" not in job_details["entity"]:
            return job_details["entity"]["decision_optimization"]["status"]
        return job_details["entity"]["scoring"]["status"]

    def get_job_id(self, job_details: dict) -> str:
        """Get the unique ID of a deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: unique ID of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_id=job_id)
            job_status = client.deployments.get_job_id(job_details)

        """
        return job_details["metadata"]["id"]

    def get_job_uid(self, job_details: dict) -> str:
        """Get the unique ID of a deployment job.

        *Deprecated:* Use ``get_job_id(job_details)`` instead.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: unique ID of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_uid=job_uid)
            job_status = client.deployments.get_job_uid(job_details)

        """
        get_job_uid_deprecated_warning = (
            "`get_job_uid()` is deprecated and will be removed in future. "
            "Instead, please use `get_job_id()`."
        )
        warn(get_job_uid_deprecated_warning, category=DeprecationWarning)
        return self.get_job_id(job_details)

    def get_job_href(self, job_details: dict) -> str:
        """Get the href of a deployment job.

        :param job_details: metadata of the deployment job
        :type job_details: dict

        :return: href of the deployment job
        :rtype: str

        **Example:**

        .. code-block:: python

            job_details = client.deployments.get_job_details(job_id=job_id)
            job_status = client.deployments.get_job_href(job_details)

        """
        return "/ml/v4/deployment_jobs/{}".format(job_details["metadata"]["id"])

    def delete_job(
        self, job_id: str | None = None, hard_delete: bool = False, **kwargs: Any
    ) -> str:
        """Delete a deployment job that is running. This method can also delete metadata
        details of completed or canceled jobs when hard_delete parameter is set to True.

        :param job_id: unique ID of the deployment job to be deleted
        :type job_id: str

        :param hard_delete: specify `True` or `False`:

            `True` - To delete the completed or canceled job.

            `False` - To cancel the currently running deployment job.

        :type hard_delete: bool, optional


        :return: status ("SUCCESS" or "FAILED")
        :rtype: str

        **Example:**

        .. code-block:: python

            client.deployments.delete_job(job_id)

        """
        job_id = _get_id_from_deprecated_uid(
            kwargs=kwargs, resource_id=job_id, resource_name="job"
        )
        Deployments._validate_type(job_id, "job_id", str, True)

        if job_id is not None and not is_id(job_id):
            raise WMLClientError("'job_id' is not an id: '{}'".format(job_id))

        url = self._client.service_instance._href_definitions.get_async_deployment_jobs_href(
            job_id
        )

        params = self._client._params()

        if hard_delete is True:
            params.update({"hard_delete": "true"})

        response_delete = self._client.httpx_client.delete(
            url, headers=self._client._get_headers(), params=params
        )

        return self._handle_response(
            204, "deployment async job deletion", response_delete, False
        )

    def _get_filter_func_by_spec_state(self, spec_state: SpecStates) -> Callable:
        def filter_func(resources: list) -> list[str]:
            asset_ids = [
                i["metadata"]["id"]
                for key, value in {
                    "model": self._client._models.get_details(
                        get_all=True, spec_state=spec_state
                    ),
                    "function": self._client._functions.get_details(
                        get_all=True, spec_state=spec_state
                    ),
                }.items()
                for i in value["resources"]
            ]

            return [
                r
                for r in resources
                if r["entity"].get("asset", {}).get("id") in asset_ids
            ]

        return filter_func

    def _get_model_inference(
        self,
        deployment_id: str,
        inference_type: Literal["text", "text_stream", "chat", "chat_stream"],
        params: dict | None = None,
    ) -> Any:
        """Based on provided deployment_id and params get ModelInference object.
        Verify that the deployment with the given deployment_id has generating methods.
        """
        # Import ModelInference here to avoid circular import error
        from ibm_watsonx_ai.foundation_models.inference import ModelInference

        match inference_type:
            case "text":
                generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_href(
                    deployment_id=deployment_id, item="text"
                )
            case "text_stream":
                if self._client._use_fm_ga_api:
                    generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_stream_href(
                        deployment_id=deployment_id
                    )
                else:  # Remove on CPD 5.0 release
                    generated_url = self._client.service_instance._href_definitions.get_fm_deployment_generation_href(
                        deployment_id=deployment_id, item="text_stream"
                    )
            case "chat":
                generated_url = self._client.service_instance._href_definitions.get_fm_deployment_chat_href(
                    deployment_id=deployment_id
                )
            case "chat_stream":
                generated_url = self._client.service_instance._href_definitions.get_fm_deployment_chat_stream_href(
                    deployment_id=deployment_id
                )
            case _:
                raise InvalidValue(
                    value_name="inference_type",
                    reason=f"Available types: 'text', 'text_stream', 'chat', 'chat_stream', got: {inference_type}.",
                )

        inference_url_list = [
            url.get("url")
            for url in self.get_details(deployment_id, _silent=True)["entity"]
            .get("status", {})
            .get("inference", {})
        ]
        if not inference_url_list:
            inference_url_list = (
                self.get_details(deployment_id, _silent=True)["entity"]
                .get("status", {})
                .get("serving_urls", [])
            )
        if inference_type in ["text", "text_stream"]:
            if generated_url not in inference_url_list:
                for inference_url in inference_url_list:  # Remove on CPD 5.0 release
                    if (
                        "v1-beta/deployments" not in inference_url
                    ):  # Remove on CPD 5.0 release
                        raise WMLClientError(
                            Messages.get_message(
                                deployment_id,
                                message_id="fm_deployment_has_not_inference_for_generation",
                            )
                        )

        return ModelInference(
            deployment_id=deployment_id, params=params, api_client=self._client
        )

    def generate(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: bool | None = None,
        guardrails_pii_params: bool | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
    ) -> dict:
        """Generate a raw response with `prompt` for given `deployment_id`.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: prompt needed for text generation. If deployment_id points to the Prompt Template asset, then the prompt argument must be None, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param concurrency_limit: number of requests to be sent in parallel, maximum is 10
        :type concurrency_limit: int, optional

        :param async_mode: If True, then yield results asynchronously (using generator). In this case both the prompt and
                           the generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: scoring result containing generated content
        :rtype: dict
        """
        d_inference = self._get_model_inference(deployment_id, "text", params)
        return d_inference.generate(
            prompt=prompt,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            params=params,
            async_mode=async_mode,
            validate_prompt_variables=validate_prompt_variables,
        )

    def generate_text(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: bool | None = None,
        guardrails_pii_params: bool | None = None,
        concurrency_limit: int = DEFAULT_CONCURRENCY_LIMIT,
        validate_prompt_variables: bool = True,
    ) -> str:
        """Given the selected deployment (deployment_id), a text prompt as input, and the parameters and concurrency_limit,
        the selected inference will generate a completion text as generated_text response.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: the prompt string or list of strings. If the list of strings is passed, requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param raw_response: returns the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param concurrency_limit: number of requests to be sent in parallel, maximum is 10
        :type concurrency_limit: int, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """
        d_inference = self._get_model_inference(deployment_id, "text", params)
        return d_inference.generate_text(
            prompt=prompt,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            params=params,
            validate_prompt_variables=validate_prompt_variables,
        )

    def generate_text_stream(
        self,
        deployment_id: str,
        prompt: str | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: bool | None = None,
        guardrails_pii_params: bool | None = None,
        validate_prompt_variables: bool = True,
    ) -> str:
        """Given the selected deployment (deployment_id), a text prompt as input and parameters,
        the selected inference will generate a streamed text as generate_text_stream.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param prompt: the prompt string, defaults to None
        :type prompt: str, optional

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict, optional

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True, then potentially hateful, abusive, and/or profane language (HAP) was detected
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool, optional

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: generated content
        :rtype: str

        .. note::
            By default only the first occurance of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        """
        d_inference = self._get_model_inference(deployment_id, "text_stream", params)
        return d_inference.generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    def chat(
        self, deployment_id: str, messages: ListType[dict], context: str | None = None
    ) -> dict:
        d_inference = self._get_model_inference(deployment_id, "chat")

        return d_inference.chat(messages=messages, context=context)

    def chat_stream(
        self, deployment_id: str, messages: ListType[dict], context: str | None = None
    ) -> Generator:
        d_inference = self._get_model_inference(deployment_id, "chat_stream")

        return d_inference.chat_stream(messages=messages, context=context)

    async def achat(
        self, deployment_id: str, messages: ListType[dict], context: str | None = None
    ) -> dict:
        d_inference = self._get_model_inference(deployment_id, "chat")

        return await d_inference.achat(messages=messages, context=context)

    async def achat_stream(
        self, deployment_id: str, messages: ListType[dict], context: str | None = None
    ) -> AsyncGenerator:
        d_inference = self._get_model_inference(deployment_id, "chat_stream")

        return await d_inference.achat_stream(messages=messages, context=context)

    def run_ai_service(self, deployment_id: str, ai_service_payload: dict) -> Any:
        """Execute an AI service by providing a scoring payload.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :return: response of the AI service
        :rtype: Any

        .. note::
            * By executing this class method, a POST request is performed.
            * In case of `method not allowed` error, try sending requests directly to your deployed ai service.
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        scoring_url = (
            self._client.service_instance._href_definitions.get_deployment_href(
                deployment_id
            )
            + "/ai_service"
        )
        response_scoring = self._client.httpx_client.post(
            url=scoring_url,
            json=ai_service_payload,
            params=self._client._params(
                skip_for_create=True, skip_userfs=True
            ),  # version parameter is mandatory
            headers=self._client._get_headers(),
        )

        error_msg = "POST is not supported using this method. Send requests directly to the deployed ai_service."
        reason = response_scoring.text

        if response_scoring.status_code == 405:
            raise WMLClientError(
                error_msg
                + " Error: "
                + str(response_scoring.status_code)
                + ". "
                + reason
            )

        return self._handle_response(200, "AI Service run", response_scoring)

    def run_ai_service_stream(
        self, deployment_id: str, ai_service_payload: dict
    ) -> Generator:
        """Execute an AI service by providing a scoring payload.

        :param deployment_id: unique ID of the deployment
        :type deployment_id: str

        :param ai_service_payload: AI service payload to be passed to generate the method
        :type ai_service_payload: dict

        :return: stream of the response of the AI service
        :rtype: Generator
        """
        Deployments._validate_type(deployment_id, "deployment_id", str, True)
        Deployments._validate_type(ai_service_payload, "ai_service_payload", dict, True)

        scoring_url = (
            self._client.service_instance._href_definitions.get_deployment_href(
                deployment_id
            )
            + "/ai_service_stream"
        )

        with self._client.httpx_client.stream(
            url=scoring_url,
            json=ai_service_payload,
            headers=self._client._get_headers(),
            params=self._client._params(skip_for_create=True, skip_userfs=True),
            method="POST",
        ) as resp:
            if resp.status_code == 200:
                for chunk in resp.iter_lines():
                    field_name, _, response = chunk.partition(":")
                    if field_name == "data":
                        yield response
            else:
                raise ApiRequestFailure(f"Failure during AI Service run steam", resp)


### Definition of Runtime Context


class RuntimeContext:
    """
    Class included to keep the interface compatible with the Deployment's RuntimeContext
    used in AIServices implementation.

    :param api_client: initialized APIClient object with a set project ID or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient

    :param request_payload_json: Request payload for testing of generate/ generate_stream call of AI Service.
    :type request_payload_json: dict, optional

    ``
    RuntimeContext`` initialized for testing purposes before deployment:

    .. code-block:: python

        context = RuntimeContext(api_client=client, request_payload_json={"field": "value"})

    Examples of ``RuntimeContext`` usage within AI Service source code:


    .. code-block:: python

        def deployable_ai_service(context, **custom):
            task_token = context.generate_token()

            def generate(context) -> dict:
                user_token = context.get_token()
                headers = context.get_headers()
                json_body = context.get_json()
                ...
                return {"body": json_body}

            return generate

        generate = deployable_ai_service(context)
        generate_output = generate(context)  # returns {"body": {"field": "value"}}


    Change the JSON body in ``RuntimeContext``:

    .. code-block:: python

        context.request_payload_json = {"field2": "value2"}

        generate = deployable_ai_service(context)
        generate_output = generate(context)  # returns {"body": {"field2": "value2"}}
    """

    def __init__(self, api_client: APIClient, request_payload_json: dict | None = None):
        self._api_client = api_client
        self.request_payload_json = request_payload_json

    @property
    def request_payload_json(self) -> dict | None:
        return self._request_payload_json

    @request_payload_json.setter
    def request_payload_json(self, value: dict) -> None:
        try:
            json_value = json.loads(json.dumps(value))
        except TypeError as e:
            raise InvalidValue("request_payload_json", reason=str(e))

        self._request_payload_json = json_value

    def get_token(self) -> str:
        """Return user token."""
        return self.generate_token()

    def generate_token(self) -> str:
        """Return refreshed token."""
        return self._api_client._get_icptoken()

    def get_headers(self) -> dict:
        """Return headers with refreshed token"""
        return self._api_client._get_headers()

    def get_json(self) -> dict | None:
        """Get payload JSON send in body of API request to the generate or generate_stream method in deployed AIService.
        For testing purposes the payload JSON need to be set in RuntimeContext initialization
        or later as request_payload_json property.
        """
        return self.request_payload_json
