#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from abc import ABC
import json
import base64
from dataclasses import dataclass
from datetime import timedelta, datetime
from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.href_definitions import HrefDefinitions
from ibm_watsonx_ai.wml_resource import WMLResource

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class BaseAuth(ABC):
    """Base class for any authentication method used in the APIClient."""

    _token: str | None = None

    def get_token(self) -> str:
        """Returns the token.

        :returns: token to be used with service
        :rtype: str
        """
        raise NotImplementedError()


@dataclass
class TokenInfo:
    """Data class in which information about token and the token are returned to be put into auth method structures.
    `expiration_datetime` should be only set when token expiration information cannot be extracted from the token,
    and this is when token is not JWT token. Otherwise, `expiration_datetime` should be set to None.

    :param token: token to be used with service
    :type token: str

    :param expiration_datetime: datetime of token expiration, if the token is not JWT, otherwise should be set to None
    :type expiration_datetime: datetime or None, optional
    """

    token: str
    expiration_datetime: datetime | None = None


class RefreshableTokenAuth(BaseAuth, ABC):
    """Abstract base class of all authentication methods which are using token generation and refresh.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing

    :param expiration_timedelta: minimal time to token expiration, when the token refresh should be triggered
    :type expiration_timedelta: timedelta
    """

    _hardcoded_expiration_datetime: datetime | None = None

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None,
        on_token_refresh: Callable[[], None] | None,
        expiration_timedelta: timedelta,
    ) -> None:
        self._session = api_client._session
        self._credentials = api_client.credentials
        self._href_definitions = HrefDefinitions(
            api_client,
            api_client.CLOUD_PLATFORM_SPACES,
            api_client.PLATFORM_URL,
            api_client.ICP_PLATFORM_SPACES,
        )
        self._on_token_creation = on_token_creation
        self._on_token_refresh = on_token_refresh
        self._expiration_timedelta = expiration_timedelta

    def get_token(self) -> str:
        """Returns the token. If the token will be under minimal expiration timedelta, it will be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        if self._token is None:
            self._save_token_data(self._generate_token())
            if self._on_token_creation:
                self._on_token_creation()
            return self._token

        if self._is_refresh_needed():
            self._save_token_data(self._refresh_token())
            if self._on_token_refresh:
                self._on_token_refresh()

        return self._token

    def _generate_token(self) -> TokenInfo:
        """Generate token from scratch using user provided credentials.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        raise NotImplementedError()

    def _refresh_token(
        self,
    ) -> TokenInfo:
        """Refresh token.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        # if not provided implementation, refresh is handled as generation from creds
        return self._generate_token()

    def _is_refresh_needed(self) -> bool:
        """Check if the time of expiration is below minimal expiration timedelta.

        :returns: result of check
        :rtype: bool
        """
        if exp_datetime := self._get_expiration_datetime():
            return exp_datetime - self._expiration_timedelta < datetime.now()
        else:
            return True

    def _get_expiration_datetime(self) -> datetime:
        """Return expiration datetime. Implementation for JWT token.

        :returns: datetime of token expiration
        :rtype: datetime
        """
        if self._hardcoded_expiration_datetime is not None:
            return self._hardcoded_expiration_datetime

        token_parts = self._token.split(".")
        token_padded = token_parts[1] + "==="
        try:
            token_info = json.loads(
                base64.b64decode(token_padded).decode("utf-8", errors="ignore")
            )
        except ValueError:
            # If there is a problem with decoding (e.g. special char in token), add altchars
            token_info = json.loads(
                base64.b64decode(token_padded, altchars="_-").decode(
                    "utf-8", errors="ignore"
                )
            )
        token_expire = token_info.get("exp")

        return datetime.fromtimestamp(token_expire)

    def _save_token_data(self, token_info: TokenInfo) -> None:
        """Write data from TokenInfo into authentication method fields for its mechanism to work properly.

        :param token_info: data of token returned after generation or refresh of token
        :type token_info: TokenInfo
        """
        self._token = token_info.token
        self._hardcoded_expiration_datetime = token_info.expiration_datetime


class TokenAuth(BaseAuth):
    """Basic authetication method, the object is keeping existing token and return it when asked.
    Token cannot be refreshed.

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing

    :param token: token to be used with service
    :type token: str
    """

    def __init__(
        self, token: str, on_token_set: Callable[[], None] | None = None
    ) -> None:
        BaseAuth.__init__(self)
        WMLResource._validate_type(token, "token", str, mandatory=True)

        self._token = token
        self._on_token_set = on_token_set
        if self._on_token_set:
            self._on_token_set()

    def get_token(self) -> str:
        """Returns the token. The token will not be refreshed.

        :returns: token to be used with service
        :rtype: str
        """
        return self._token

    def set_token(self, token: str) -> None:
        """Set new token.

        :param token: token to be used with service
        :type token: str
        """
        self._token = token
        if self._on_token_set:
            self._on_token_set()


def get_auth_method(
    api_client: APIClient,
    on_token_set: Callable[[], None] | None = None,
    on_token_creation: Callable[[], None] | None = None,
    on_token_refresh: Callable[[], None] | None = None,
) -> BaseAuth:
    """
    Return authentication method using values from API client.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_set: callback which allows to notify about token set
    :type on_token_set: function which takes no params and returns nothing, optional

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional

    :returns: authentication method object
    :rtype: BaseAuth
    """
    creds = api_client.credentials

    if creds.token and not (creds._is_env_token and (creds.api_key or creds.password)):
        # situation one of these:
        # - there is token passed by user (and may be password or apikey)
        # - there is token from env and no additional password or api_key in the credentials
        return TokenAuth(creds.token, on_token_set=on_token_set)
    else:
        if getattr(creds, "token_function", False):
            from ibm_watsonx_ai.utils.auth.jwt_token_function_auth import (
                JWTTokenFunctionAuth,
            )

            return JWTTokenFunctionAuth(
                api_client,
                on_token_creation=on_token_creation,
                on_token_refresh=on_token_refresh,
            )
        elif api_client.CLOUD_PLATFORM_SPACES:
            from ibm_watsonx_ai.utils.auth.iam_auth import IAMTokenAuth

            return IAMTokenAuth(
                api_client,
                on_token_creation=on_token_creation,
                on_token_refresh=on_token_refresh,
            )
        else:
            from ibm_watsonx_ai.utils.auth.icp_auth import ICPAuth

            return ICPAuth(
                api_client,
                on_token_creation=on_token_creation,
                on_token_refresh=on_token_refresh,
            )
