#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

"""
.. module:: APIClient
   :platform: Unix, Windows
   :synopsis: IBM watsonx.ai API Client.

.. moduleauthor:: IBM
"""

import copy
import logging
import os
from warnings import warn
from typing import Any, cast, TypeAlias
import json
import base64
import httpx

import ibm_watsonx_ai.utils
from ibm_watsonx_ai.projects import Projects
from ibm_watsonx_ai.trashed_assets import TrashedAssets
from ibm_watsonx_ai._wrappers.requests import (
    _get_httpx_client,
    _httpx_transport_params,
    _get_async_client,
)
from ibm_watsonx_ai.utils.auth import get_auth_method
from ibm_watsonx_ai.utils import get_user_agent_header
from ibm_watsonx_ai.utils.utils import (
    _APIClientSession,
    HttpClientConfig,
    DEFAULT_HTTP_CLIENT_CONFIG,
)
from ibm_watsonx_ai.Set import Set
from ibm_watsonx_ai.assets import Assets
from ibm_watsonx_ai.connections import Connections
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.deployments import Deployments
from ibm_watsonx_ai.experiments import Experiments
from ibm_watsonx_ai.export_assets import Export
from ibm_watsonx_ai.factsheets import Factsheets
from ibm_watsonx_ai.foundation_models_manager import FoundationModelsManager
from ibm_watsonx_ai.functions import Functions
from ibm_watsonx_ai.ai_services import AIServices
from ibm_watsonx_ai.hw_spec import HwSpec
from ibm_watsonx_ai.import_assets import Import
from ibm_watsonx_ai.service_instance import ServiceInstance
from ibm_watsonx_ai.messages.messages import Messages
from ibm_watsonx_ai.model_definition import ModelDefinition
from ibm_watsonx_ai.models import Models
from ibm_watsonx_ai.parameter_sets import ParameterSets
from ibm_watsonx_ai.pipelines import Pipelines
from ibm_watsonx_ai.pkg_extn import PkgExtn
from ibm_watsonx_ai.spaces import Spaces
from ibm_watsonx_ai.remote_training_system import RemoteTrainingSystem
from ibm_watsonx_ai.repository import Repository
from ibm_watsonx_ai.script import Script
from ibm_watsonx_ai.shiny import Shiny
from ibm_watsonx_ai.sw_spec import SwSpec
from ibm_watsonx_ai.task_credentials import TaskCredentials
from ibm_watsonx_ai.training import Training
from ibm_watsonx_ai.utils import CPDVersion
from ibm_watsonx_ai.volumes import Volume
from ibm_watsonx_ai.wml_client_error import NoWMLCredentialsProvided
from ibm_watsonx_ai.wml_client_error import WMLClientError

# requests module or requests.Session
RequestsLikeType: TypeAlias = Any


class APIClient:
    """The main class of ibm_watsonx_ai. The very heart of the module. APIClient contains objects that manage the service reasources.

    To explore how to use APIClient, refer to:

    - :ref:`Setup<setup>` - to check correct initialization of APIClient for a specific environment.
    - :ref:`Core<core>` - to explore core properties of an APIClient object.

    :param url: URL of the service
    :type url: str

    :param credentials: credentials used to connect with the service
    :type credentials: Credentials

    :param project_id: ID of the project that is used
    :type project_id: str, optional

    :param space_id: ID of deployment space that is used
    :type space_id: str, optional

    :param verify: certificate verification flag, deprecated, use Credentials(verify=...) to set `verify`
    :type verify: bool, optional

    :param httpx_client: A customizable `httpx.Client` for ModelInference, Embeddings and methods related to the deployments management and scoring.
        The `httpx.Client` is used to improve performance across deployments, foundation models, and embeddings. This parameter accepts two types of input:

        - A direct instance of `httpx.Client()`
        - A set of parameters provided via the `HttpClientConfig` class

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.utils.utils import HttpClientConfig

            limits=httpx.Limits(
                max_connections=5
            )
            timeout = httpx.Timeout(7)
            http_config = HttpClientConfig(timeout=timeout, limits=limits)

        If not provided, a default instance of `httpx.Client` is created.

        .. note::
            If you need to adjust timeouts or limits, using ``HttpClientConfig`` is the recommended approach.
            When the ``proxies`` parameter is provided in credentials, ``httpx.Client`` will use these proxies.
            However, if you want to create a separate ``httpx.Client``, all parameters must be provided by the user.
    :type httpx_client: httpx.Client, HttpClientConfig, optional

    :param async_httpx_client: A customizable `httpx.AsyncClient` for ModelInference. The `httpx.AsyncClient` is used to improve performance of foundation models inference. This parameter accepts two types of input:

        - A direct instance of `httpx.AsyncClient`
        - A set of parameters provided via the `HttpClientConfig` class

        **Example:**

        .. code-block:: python

            from ibm_watsonx_ai.utils.utils import HttpClientConfig

            limits=httpx.Limits(
                max_connections=5
            )
            timeout = httpx.Timeout(7)
            http_config = HttpClientConfig(timeout=timeout, limits=limits)

        If not provided, a default instance of `httpx.AsyncClient` is created.

        .. note::
            If you need to adjust timeouts or limits, using ``HttpClientConfig`` is the recommended approach.
            When the ``proxies`` parameter is provided in credentials, ``httpx.Client`` will use these proxies.
            However, if you want to create a separate ``httpx.Client``, all parameters must be provided by the user.

    :type async_httpx_client: httpx.AsyncClient, HttpClientConfig, optional

    **Example:**

    .. code-block:: python

        from ibm_watsonx_ai import APIClient, Credentials

        credentials = Credentials(
            url = "<url>",
            api_key = IAM_API_KEY
        )

        client = APIClient(credentials, space_id="<space_id>")

        client.models.list()
        client.deployments.get_details()

        client.set.default_project("<project_id>")

        ...

    """

    version: str | None = None
    _internal: bool = False

    def __init__(
        self,
        credentials: Credentials | dict[str, str] | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: str | bool | None = None,
        httpx_client: httpx.Client | HttpClientConfig = DEFAULT_HTTP_CLIENT_CONFIG,
        async_httpx_client: (
            httpx.AsyncClient | HttpClientConfig
        ) = DEFAULT_HTTP_CLIENT_CONFIG,
        **kwargs: Any,
    ) -> None:
        if (wml_credentials := kwargs.get("wml_credentials")) is not None:
            wml_credentials_parameter_deprecated_warning = (
                "`wml_credentials` parameter is deprecated, please use `credentials`"
            )
            warn(wml_credentials_parameter_deprecated_warning, category=DeprecationWarning)  # fmt: skip
            if not credentials:
                credentials = wml_credentials
        if wml_credentials is None and credentials is None:
            raise TypeError("APIClient() missing 1 required argument: 'credentials'")

        self._logger = logging.getLogger(__name__)

        wml_full_version = ""

        if verify is not None:
            verify_parameter_deprecated_warning = (
                "`verify` parameter is deprecated. "
                "Use `ibm_watsonx_ai.Credentials` for passing `verify` parameter."
            )
            warn(verify_parameter_deprecated_warning, category=DeprecationWarning)

        if isinstance(credentials, dict):
            credentials_parameter_as_dict_deprecated_warning = (
                "`credentials` parameter as dict is deprecated. "
                "Use `ibm_watsonx_ai.Credentials` for passing parameters."
            )
            warn(credentials_parameter_as_dict_deprecated_warning, category=DeprecationWarning)  # fmt: skip
            credentials = Credentials.from_dict(credentials, _verify=verify)

        if project_id is not None and space_id is not None:
            raise WMLClientError(
                "`project_id` parameter and `space_id` parameter cannot be set at the same time."
            )

        credentials._set_env_vars_from_credentials()

        if isinstance(credentials.verify, str):
            credentials.verify = True

        from ibm_watsonx_ai._wrappers import requests

        if credentials.proxies is not None:
            requests.additional_settings["proxies"] = credentials.proxies
        elif requests.additional_settings.get("proxies") is not None:
            del requests.additional_settings["proxies"]

        # At this stage `credentials` has type Dict[str, str]
        credentials = cast(Credentials, credentials)
        self.credentials = copy.deepcopy(credentials)
        self.default_space_id = None
        self.default_project_id = None
        self.project_type = None
        self.CLOUD_PLATFORM_SPACES = False
        self.PLATFORM_URL = None
        self.version_param = self._get_api_version_param()
        self.ICP_PLATFORM_SPACES = False  # This will be applicable for 3.5 and later and specific to convergence functionalities
        self.CPD_version = CPDVersion()
        self._iam_id = None
        self._spec_ids_per_state: dict = {}
        self.generate_ux_tag = True
        self.WCA: bool = False
        self._user_headers: dict | None = None  # Used in set_headers() method
        self.__session = None

        self.PLATFORM_URLS_MAP = {
            # Dallas
            "https://us-south.ml.cloud.ibm.com": "https://api.dataplatform.cloud.ibm.com",
            "https://private.us-south.ml.cloud.ibm.com": "https://private.api.dataplatform.cloud.ibm.com",
            # Frankfurt
            "https://eu-de.ml.cloud.ibm.com": "https://api.eu-de.dataplatform.cloud.ibm.com",
            "https://private.eu-de.ml.cloud.ibm.com": "https://private.api.eu-de.dataplatform.cloud.ibm.com",
            # London
            "https://eu-gb.ml.cloud.ibm.com": "https://api.eu-gb.dataplatform.cloud.ibm.com",
            "https://private.eu-gb.ml.cloud.ibm.com": "https://private.api.eu-gb.dataplatform.cloud.ibm.com",
            # Tokio
            "https://jp-tok.ml.cloud.ibm.com": "https://api.jp-tok.dataplatform.cloud.ibm.com",
            "https://private.jp-tok.ml.cloud.ibm.com": "https://private.api.jp-tok.dataplatform.cloud.ibm.com",
            # Sydney
            "https://au-syd.ml.cloud.ibm.com": "https://api.au-syd.dai.cloud.ibm.com",
            "https://private.au-syd.ml.cloud.ibm.com": "https://private.api.au-syd.dai.cloud.ibm.com",
            # Toronto
            "https://ca-tor.ml.cloud.ibm.com": "https://api.ca-tor.dai.cloud.ibm.com",
            "https://private.ca-tor.ml.cloud.ibm.com": "https://private.api.ca-tor.dai.cloud.ibm.com",
            # YPCR
            "https://yp-cr.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://private.yp-cr.ml.cloud.ibm.com": "https://private.api.dataplatform.test.cloud.ibm.com",
            # MCSP QA
            "https://wxai-qa.ml.cloud.ibm.com": "https://api.dai.test.cloud.ibm.com",
            "https://private.wxai-qa.ml.cloud.ibm.com": "https://private.api.dai.test.cloud.ibm.com",
            # YPQA
            "https://yp-qa.ml.cloud.ibm.com": "https://api.dataplatform.test.cloud.ibm.com",
            "https://private.yp-qa.ml.cloud.ibm.com": "https://private.api.dataplatform.test.cloud.ibm.com",
            # MCSP DEV
            "https://wml-mcsp-dev.ml.test.cloud.ibm.com": "https://api.dai.dev.cloud.ibm.com",
            "https://private.wml-mcsp-dev.ml.test.cloud.ibm.com": "https://private.api.dai.dev.cloud.ibm.com",
            # FVT
            "https://wml-fvt.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://private.wml-fvt.ml.test.cloud.ibm.com": "https://private.api.dataplatform.dev.cloud.ibm.com",
            # YS1Prod
            "https://us-south.ml.test.cloud.ibm.com": "https://api.dataplatform.dev.cloud.ibm.com",
            "https://private.us-south.ml.test.cloud.ibm.com": "https://private.api.dataplatform.dev.cloud.ibm.com",
        }

        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

        self.__predefined_instance_type_list = ["icp", "openshift"]
        if credentials is None:
            raise NoWMLCredentialsProvided()
        if self.credentials.url is None:
            raise WMLClientError(Messages.get_message(message_id="url_not_provided"))
        if not self.credentials.url.startswith("https://"):
            raise WMLClientError(Messages.get_message(message_id="invalid_url"))
        if self.credentials.url[-1] == "/":
            self.credentials.url = self.credentials.url.rstrip("/")
        with self._session:
            if self.credentials.instance_id is None:
                self.CLOUD_PLATFORM_SPACES = True
                self.ICP_PLATFORM_SPACES = False

                if self._internal:
                    self.PLATFORM_URL = self.credentials.url

                else:
                    if self.credentials.platform_url:
                        if not self.credentials.platform_url.startswith("https://"):
                            raise WMLClientError(
                                Messages.get_message(message_id="invalid_platform_url")
                            )
                        self.PLATFORM_URL = self.credentials.platform_url
                    elif self.credentials.url in self.PLATFORM_URLS_MAP.keys():
                        self.PLATFORM_URL = self.PLATFORM_URLS_MAP[self.credentials.url]
                    else:
                        raise WMLClientError(
                            Messages.get_message(
                                message_id="invalid_cloud_scenario_url"
                            )
                        )

                if not self._is_IAM():
                    raise WMLClientError(
                        Messages.get_message(message_id="apikey_not_provided")
                    )
            else:
                if (
                    "icp" == self.credentials.instance_id.lower()
                    or "openshift" == self.credentials.instance_id.lower()
                ):
                    if (
                        self.credentials.url in self.PLATFORM_URLS_MAP.keys()
                        or self.credentials.url in self.PLATFORM_URLS_MAP.values()
                    ):
                        raise WMLClientError(
                            Messages.get_message(message_id="invalid_cloud_url")
                        )

                    self.ICP_PLATFORM_SPACES = True
                    os.environ["DEPLOYMENT_PLATFORM"] = "private"

                    # Validate the cpd version:
                    response_get_wml_services = self._session.get(
                        f"{self.credentials.url}/ml/wml_services/version",
                        headers={"User-Agent": get_user_agent_header()},
                    )
                    if (
                        response_get_wml_services.status_code != 200
                    ):  # retry with endpoint for cpd 4.8 and higher
                        response_get_wml_services = self._session.get(
                            f"{self.credentials.url}/ml/wml_services/v2/version",
                            headers={"User-Agent": get_user_agent_header()},
                        )

                    if response_get_wml_services.status_code == 200:
                        wml_full_version = response_get_wml_services.json().get(
                            "version", ""
                        )
                        if wml_full_version:
                            wml_version = ".".join(wml_full_version.split(".")[:2])
                            if self.credentials.version is None:
                                self.credentials.version = wml_version
                            elif self.credentials.version != wml_version:
                                cpd_version_mismatch_warning = (
                                    f"The provided version: {self.credentials.version} "
                                    f"is different from the current CP4D version: {wml_version}. "
                                    f"Correct the credentials with proper CP4D version number."
                                )
                                warn(cpd_version_mismatch_warning)

                            if (
                                self.credentials.version
                                not in CPDVersion.supported_version_list
                            ):
                                raise WMLClientError(
                                    Messages.get_message(
                                        self.credentials.version,
                                        self.version,
                                        message_id="invalid_version_from_automated_check",
                                    )
                                )
                    else:
                        self._logger.debug(
                            f"GET /ml/wml_services/version failed with status code: {response_get_wml_services.status_code}."
                        )

                    # Condition for CAMS related changes to take effect (Might change)
                    if self.credentials.version is None:
                        raise WMLClientError(
                            Messages.get_message(
                                CPDVersion.supported_version_list,
                                message_id="version_not_provided",
                            )
                        )

                    if (
                        self.credentials.version.lower()
                        in CPDVersion.supported_version_list
                    ):
                        self.CPD_version.cpd_version = self.credentials.version.lower()
                        os.environ["DEPLOYMENT_PRIVATE"] = "icp4d"

                        if self.credentials.bedrock_url is None and self.CPD_version:
                            if self.CPD_version < 4.7:
                                bedrock_prefix = "https://cp-console"
                            else:
                                namespace_from_url = "-".join(
                                    self.credentials.url.split(".")[0].split("-")[1:]
                                )
                                route = (
                                    "cpd" if self.CPD_version >= 5.1 else "cp-console"
                                )
                                bedrock_prefix = f"https://{route}-{namespace_from_url}"
                            self.credentials.bedrock_url = ".".join(
                                [bedrock_prefix] + self.credentials.url.split(".")[1:]
                            )
                            self._is_bedrock_url_autogenerated = True

                    else:
                        self.ICP_PLATFORM_SPACES = False
                        raise WMLClientError(
                            Messages.get_message(
                                ", ".join(CPDVersion.supported_version_list),
                                message_id="invalid_version",
                            )
                        )

                else:
                    if self.credentials.url in self.PLATFORM_URLS_MAP:
                        raise WMLClientError(
                            Messages.get_message(
                                message_id="instance_id_in_cloud_scenario"
                            )
                        )
                    else:
                        raise WMLClientError(
                            Messages.get_message(message_id="invalid_instance_id")
                        )

            if (
                self.credentials.instance_id is not None
                and (
                    self.credentials.instance_id.lower()
                    not in self.__predefined_instance_type_list
                )
                and self.credentials.version is not None
            ):
                raise WMLClientError(
                    Messages.get_message(message_id="provided_credentials_are_invalid")
                )
            self._use_fm_ga_api = self.CLOUD_PLATFORM_SPACES or (
                self._check_if_fm_ga_api_available()
                if self.CPD_version <= 4.8
                else True
            )

            self._use_pta_ga_api = self.CLOUD_PLATFORM_SPACES or (
                self.CPD_version >= 5.0
            )

            self._auth_method = get_auth_method(self)
            self._auth_method.get_token()

            # For cloud, service_instance.details will be set during space creation( if instance is associated ) or
            # while patching a space with an instance

            self.service_instance: ServiceInstance = ServiceInstance(self)
            self.volumes = Volume(self)
            if self._use_fm_ga_api:
                self.foundation_models = FoundationModelsManager(self)

            if self.ICP_PLATFORM_SPACES:
                self.service_instance._refresh_details = True

            self.set = Set(self)

            if project_id:
                self.set.default_project(project_id)  # recognizes project type
            elif space_id:
                self.set.default_space(space_id)

            self.spaces = Spaces(self)
            self.projects = Projects(self)

            self.export_assets = Export(self)
            self.import_assets = Import(self)

            if self.ICP_PLATFORM_SPACES:
                self.shiny = Shiny(self)
                self.trashed_assets = TrashedAssets(self)

            self.script = Script(self)
            self.model_definitions = ModelDefinition(self)

            self.package_extensions = PkgExtn(self)
            self.software_specifications = SwSpec(self)

            self.hardware_specifications = HwSpec(self)

            self.connections = Connections(self)
            self.training: Training = Training(self)

            self.data_assets = Assets(self)

            self.deployments = Deployments(self)

            if self.CLOUD_PLATFORM_SPACES:
                self.factsheets = Factsheets(self)
                self.task_credentials = TaskCredentials(self)

            if self.CPD_version < 5.1 or wml_full_version == "5.1.0":
                pass  # AI services available only on CLOUD and CPD 5.1.1 or higher
            else:
                self.__ai_services = AIServices(self)

            self.remote_training_systems = RemoteTrainingSystem(self)

            self.repository = Repository(self)
            self._models = Models(self)

            self.pipelines = Pipelines(self)
            self.experiments = Experiments(self)
            self._functions = Functions(self)

            self.parameter_sets = ParameterSets(self)
            self._logger.info(
                Messages.get_message(message_id="client_successfully_initialized")
            )

            if isinstance(httpx_client, HttpClientConfig):
                httpx_client = _get_httpx_client(
                    transport_params=_httpx_transport_params(
                        self, limits=httpx_client.limits
                    ),
                    timeout=httpx_client.timeout,
                )

            self._httpx_client = httpx_client

            if isinstance(async_httpx_client, HttpClientConfig):
                async_httpx_client = _get_async_client(
                    transport_params=_httpx_transport_params(
                        self, limits=async_httpx_client.limits
                    ),
                    timeout=async_httpx_client.timeout,
                )

            self._async_httpx_client = async_httpx_client

    @property
    def wml_credentials(self) -> dict[str, str]:
        wml_credentials_attribute_deprecated = (
            "`wml_credentials` attribute is deprecated, "
            "please use `client.credentials` instead"
        )
        warn(wml_credentials_attribute_deprecated, DeprecationWarning)
        return self.credentials.to_dict()

    @wml_credentials.setter
    def wml_credentials(self, value: dict[str, str]) -> None:
        wml_credentials_attribute_deprecated = (
            "`wml_credentials` attribute is deprecated, "
            "please use `client.credentials` instead"
        )
        warn(wml_credentials_attribute_deprecated, DeprecationWarning)
        self.credentials = Credentials.from_dict(value)

    @property
    def wml_token(self) -> str | None:
        wml_token_attribute_deprecated = (
            "`wml_token` attribute is deprecated, please use `client.token` instead"
        )
        warn(wml_token_attribute_deprecated, DeprecationWarning)
        return self.token

    @wml_token.setter
    def wml_token(self, value: str) -> None:
        wml_token_attribute_deprecated = (
            "`wml_token` attribute is deprecated, please use `client.token` instead"
        )
        warn(wml_token_attribute_deprecated, DeprecationWarning)
        self.token = value

    @property
    def token(self) -> str:
        return self._auth_method.get_token()

    @token.setter
    def token(self, value: str) -> None:
        self._auth_method._token = value

    @property
    def _session(self) -> RequestsLikeType:
        if self.__session is None:
            self.__session = _APIClientSession(self)
        return self.__session

    @_session.setter
    def _session(self, value: RequestsLikeType):
        self.__session = value

    @property
    def _ai_services(self) -> AIServices:
        if self.CLOUD_PLATFORM_SPACES or (
            self.CPD_version >= 5.1 and self._is_ai_services_endpoint_available()
        ):
            return self.__ai_services
        else:
            raise WMLClientError(
                error_msg="AI service is unsupported for this release."
            )

    @property
    def proceed(self) -> bool:
        from ibm_watsonx_ai.utils.auth import TokenAuth

        warn(
            (
                "`APIClient.proceed` is deprecated and will be removed in future. To use `proceed` scenario, "
                "pass `token` into credentials without `apikey` or `password`, or use `APIClient.set_token` function."
            ),
            category=DeprecationWarning,
        )
        return isinstance(self._auth_method, TokenAuth)

    @property
    def httpx_client(self) -> httpx.Client:
        return self._httpx_client

    @httpx_client.setter
    def httpx_client(self, value: httpx.Client):
        if self._httpx_client:
            self._httpx_client.close()
        self._httpx_client = value

    @property
    def async_httpx_client(self) -> httpx.AsyncClient:
        return self._async_httpx_client

    @async_httpx_client.setter
    def async_httpx_client(self, value: httpx.AsyncClient):
        if self._async_httpx_client:
            self._async_httpx_client.aclose()
        self._async_httpx_client = value

    @staticmethod
    def _get_api_version_param() -> str:
        try:
            file_name = "API_VERSION_PARAM"
            path = os.path.dirname(ibm_watsonx_ai.utils.__file__)
            with open(os.path.join(path, file_name)) as file:
                return file.read().strip()
        except Exception:
            return "2021-06-21"

    def _check_if_either_is_set(self) -> None:
        if self.default_space_id is None and self.default_project_id is None:
            raise WMLClientError(
                Messages.get_message(
                    message_id="it_is_mandatory_to_set_the_space_project_id"
                )
            )

    def _check_if_space_is_set(self) -> None:
        if self.default_space_id is None:
            raise WMLClientError(
                Messages.get_message(message_id="it_is_mandatory_to_set_the_space_id")
            )

    def _params(
        self,
        skip_space_project_chk: bool = False,
        skip_for_create: bool = False,
        skip_userfs: bool = False,
    ) -> dict:
        params = {}
        params.update({"version": self.version_param})
        if not skip_for_create:
            if self.default_space_id is not None:
                params.update({"space_id": self.default_space_id})
            elif self.default_project_id is not None:
                params.update({"project_id": self.default_project_id})
            else:
                # For system software/hardware specs
                if skip_space_project_chk is False:
                    raise WMLClientError(
                        Messages.get_message(
                            message_id="it_is_mandatory_to_set_the_space_project_id"
                        )
                    )

        if (
            self.default_project_id
            and self.project_type == "local_git_storage"
            and not skip_userfs
        ):
            params.update({"userfs": "true"})
            if self._iam_id:
                params.update({"iam_id": str(self._iam_id)})

        if (
            not self.default_project_id
            or self.project_type != "local_git_storage"
            or skip_userfs
        ) and "userfs" in params:
            del params["userfs"]

        return params

    def _get_headers(
        self,
        content_type: str = "application/json",
        no_content_type: bool = False,
        zen: bool = False,
        projects_token: bool = False,
    ) -> dict:

        headers = {}

        if not no_content_type:
            headers["Content-Type"] = content_type

        token_to_use = (
            self.credentials.projects_token
            if projects_token and self.credentials.projects_token is not None
            else self.token
        )

        if len(token_to_use.split(".")) == 1:
            headers["Authorization"] = "Basic " + token_to_use
        else:
            headers["Authorization"] = "Bearer " + token_to_use

        if not zen:
            headers["User-Agent"] = get_user_agent_header()

        if not self.generate_ux_tag:
            headers.update({"X-WX-UX": "true"})
            self.generate_ux_tag = True

        if self.WCA:
            headers.update({"IBM-WATSONXAI-CONSUMER": "wca"})

        if (env_variable := os.environ.get("IBM_SDK_API_CLIENT_HEADERS")) is not None:
            headers = headers | json.loads(
                base64.b64decode(env_variable).decode("utf-8")
            )
        if self._user_headers:
            headers = headers | self._user_headers

        return headers

    def get_headers(
        self,
        content_type: str | None = "application/json",
        include_user_agent: bool = False,
    ) -> dict:
        """Get HTTP headers used during requests.

        :param content_type: value for `Content-Type` header, defaults to `application/json`
        :type name: str, optional

        :param include_user_agent: whether the result should include `User-Agent` header, defaults to `False`
        :type name: bool, optional

        :return: headers used during requests
        :rtype: dict
        """

        return self._get_headers(
            content_type=content_type or "",
            no_content_type=content_type is None,
            zen=not include_user_agent,
        )

    def set_token(self, token: str) -> None:
        """
        Method which allows refresh/set new User Authorization Token.

        .. note::
            Using this function will cause that token will not be automatically refreshed anymore, if `password` or `apikey` were passed.
            The user needs to take care of token refresh using `set_token` function from that point in time until they finish using the client instance.

        :param token: User Authorization Token
        :type token: str

        **Examples**

        .. code-block:: python

            client.set_token("<USER AUTHORIZATION TOKEN>")

        """
        self.credentials.token = token
        from ibm_watsonx_ai.utils.auth import TokenAuth

        if isinstance(self._auth_method, TokenAuth):
            self._auth_method.set_token(token)
        else:
            # the auth method type was changed to TokenAuth
            authentication_method_changed_warning = (
                "Authentication method changed to TokenAuth. "
                "The token will not be automatically refreshed from this point of time. "
                "Use `APIClient.set_token` function to manually update token."
            )

            warn(authentication_method_changed_warning)

            self._auth_method = TokenAuth(
                token, on_token_set=self.repository._refresh_repo_client
            )

    def set_headers(self, headers: dict) -> None:
        """
        Method which allows refresh/set new User Request Headers.

        :param headers: User Request Headers
        :type headers: dict

        **Examples**

        .. code-block:: python

            headers = {
                'Authorization': 'Bearer <USER AUTHORIZATION TOKEN>',
                'User-Agent': 'ibm-watsonx-ai/1.0.1 (lang=python; arch=x86_64; os=darwin; python.version=3.10.13)',
                'Content-Type': 'application/json'
            }

            client.set_headers(headers)

        """
        self._user_headers = headers

    def _get_icptoken(self) -> str:
        return self.token

    def _is_default_space_set(self) -> bool:
        if self.default_space_id is not None:
            return True
        return False

    def _is_IAM(self) -> bool:
        if self.credentials.api_key is not None:
            if self.credentials.api_key != "":
                return True
            else:
                raise WMLClientError(
                    Messages.get_message(message_id="apikey_value_cannot_be_empty")
                )
        elif self.credentials.token is not None:
            if self.credentials.token != "":
                return True
            else:
                raise WMLClientError(
                    Messages.get_message(message_id="token_value_cannot_be_empty")
                )
        else:
            return False

    def _check_if_fm_ga_api_available(self) -> bool:
        response_ga_api = self._session.get(
            url="{}/ml/v1/foundation_model_specs?limit={}".format(
                self.credentials.url, "1"
            ),
            params={"version": self.version_param},
        )
        return response_ga_api.status_code == 200

    def _is_ai_services_endpoint_available(self) -> bool:
        try:
            url = self.service_instance._href_definitions.get_ai_services_href()

            response_ai_services_api = self._session.get(
                url=f"{url}?limit=1",
                params=self._params(),
                headers=self._get_headers(),
            )
            return response_ai_services_api.status_code != 404
        except:
            return False
