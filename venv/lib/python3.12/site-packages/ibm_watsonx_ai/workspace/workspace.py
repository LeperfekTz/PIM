#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from copy import copy
from warnings import warn

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import MissingArgument
from ibm_watsonx_ai.wml_client_error import WMLClientError
from ..credentials import Credentials
from ..utils.autoai.enums import TShirtSize
from ..utils.autoai.errors import TShirtSizeNotSupported, SetIDFailed
from ..utils.autoai.utils import is_ipython


class WorkSpace:
    """WorkSpace class for Service authentication and project/space manipulation."""

    def __init__(
        self,
        credentials: Credentials = None,
        project_id: str = None,
        space_id: str = None,
        verify=None,
        **kwargs,
    ) -> None:
        """
        :param credentials: credentials to Service instance
        :type credentials: dict

        :param project_id: ID of the Watson Studio project
        :type project_id: str, optional

        :param space_id: ID of the Watson Studio Space
        :type space_id: str, optional

        **Example:**

        .. code-block: python

            from ibm_watsonx_ai.workspace import WorkSpace
            from ibm_watsonx_ai import Credentials

            ws = WorkSpace(
                credentials=Credentials(
                    api_key= IAM_API_KEY,
                    iam_serviceid_crn = "...",
                    instance_id = "...",
                    url = "https://us-south.ml.cloud.ibm.com"
                ),
                project_id="...",
                space_id="...")
        """
        # note: backward compatibility
        if (wml_credentials := kwargs.get("wml_credentials")) is not None:
            credentials = wml_credentials
            wml_credentials_deprecated_warning = (
                "`wml_credentials` is deprecated and will be removed in future. "
                "Instead, please use `credentials`."
            )
            warn(wml_credentials_deprecated_warning, category=DeprecationWarning)
        if wml_credentials is None and credentials is None:
            raise WMLClientError("No `credentials` provided")
        # --- end note
        self.credentials = copy(credentials)
        self.project_id = project_id
        self.space_id = space_id

        self.api_client = APIClient(credentials=self.credentials, verify=verify)

        if credentials.instance_id is None:
            if self.space_id is not None:
                self.api_client.set.default_space(self.space_id)

            elif self.project_id is not None:
                self.api_client.set.default_project(self.project_id)

            else:
                raise SetIDFailed(
                    f"project_id and space_id",
                    reason=f"project_id and space_id cannot be None at the same time.",
                )

        elif credentials.instance_id and credentials.instance_id.lower() in (
            "icp",
            "openshift",
        ):
            if self.project_id is None and self.space_id is None:
                raise MissingArgument(
                    "project_id",
                    reason="These credentials are from CP4D environment, "
                    'please specify "project_id"',
                )
            else:
                if self.project_id is not None:
                    outcome = self.api_client.set.default_project(self.project_id)

                else:
                    outcome = self.api_client.set.default_space(self.space_id)

                if outcome == "FAILURE":
                    raise SetIDFailed(
                        f'{"project_id" if self.project_id is not None else "space_id"}',
                        reason=f'This {"project_id" if self.project_id is not None else "space_id"}: '
                        f"{self.project_id if self.project_id is not None else self.space_id} "
                        f"cannot be found in current environment.",
                    )

    def __str__(self):
        return f"credentials: {self.credentials} project_id: {self.project_id} space_id = {self.space_id}"

    def __repr__(self):
        return self.__str__()

    @property
    def wml_credentials(self):
        wml_credentials_deprecated_warning = (
            "`wml_credentials` is deprecated and will be removed in future. "
            "Instead, please use `credentials`."
        )
        warn(wml_credentials_deprecated_warning, category=DeprecationWarning)
        return self.credentials.to_dict()

    @wml_credentials.setter
    def wml_credentials(self, var):
        wml_credentials_deprecated_warning = (
            "`wml_credentials` is deprecated and will be removed in future. "
            "Instead, please use `credentials`."
        )
        warn(wml_credentials_deprecated_warning, category=DeprecationWarning)
        self.credentials = Credentials.from_dict(var)

    @property
    def wml_client(self):
        wml_client_deprecated_warning = (
            "`wml_client` is deprecated and will be removed in future. "
            "Instead, please use `api_client`."
        )
        warn(wml_client_deprecated_warning, category=DeprecationWarning)
        return self.api_client

    @wml_client.setter
    def wml_client(self, var):
        wml_client_deprecated_warning = (
            "`wml_client` is deprecated and will be removed in future. "
            "Instead, please use `api_client`."
        )
        warn(wml_client_deprecated_warning, category=DeprecationWarning)
        self.api_client = var

    def restrict_pod_size(self, t_shirt_size: "TShirtSize") -> "TShirtSize":
        """Check t_shirt_size for AutoAI POD. Restrict sizes per environment.

        :param t_shirt_size: TShirt size to be validated and restricted
        :type t_shirt_size: TShirtSize

        :return: validated and restricted TShirt size
        :rtype: TShirtSize
        """
        # note: for testing purposes
        if self.credentials.__dict__.get("development", False):
            return t_shirt_size
        # --- end note

        default_cloud = TShirtSize.L
        default_cp4d = TShirtSize.M
        default_server = TShirtSize.M

        supported_cp4d = (TShirtSize.S, TShirtSize.M, TShirtSize.L, TShirtSize.XL)
        supported_server = (TShirtSize.S, TShirtSize.M, TShirtSize.L)

        # note: check CP4D and Server pod sizes
        if self.api_client.ICP_PLATFORM_SPACES:
            if t_shirt_size not in supported_cp4d:
                t_shirt_size_not_supported_warning = (
                    f'This t-shirt-size: "{t_shirt_size}" is not supported in CP4D. '
                    f"Supported sizes: {supported_cp4d} "
                    f"Continuing work with default size {default_cp4d}"
                )

                warn(t_shirt_size_not_supported_warning)
                if is_ipython():
                    print(t_shirt_size_not_supported_warning)

                return default_cp4d

            else:
                return t_shirt_size

        else:
            # note: allow every size in test envs
            if self.credentials.url and "test" in self.credentials.url:
                return t_shirt_size

            else:
                # note: raise an error for cloud if pod size is different, other just return
                if t_shirt_size != default_cloud:
                    raise TShirtSizeNotSupported(
                        t_shirt_size,
                        reason=f"This t-shirt size is not supported. Please use the default one: {default_cloud}",
                    )

                else:
                    return default_cloud
                # --- end note
