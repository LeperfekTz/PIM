#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Callable

from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.utils.auth.base_auth import RefreshableTokenAuth, TokenInfo

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class IAMTokenAuth(RefreshableTokenAuth):
    """IAM token authentication method class.

    :param api_client: initialized APIClient object with set project or space ID
    :type api_client: APIClient

    :param on_token_creation: callback which allows to notify about token creation
    :type on_token_creation: function which takes no params and returns nothing, optional

    :param on_token_refresh: callback which allows to notify about token refresh
    :type on_token_refresh: function which takes no params and returns nothing, optional
    """

    def __init__(
        self,
        api_client: APIClient,
        on_token_creation: Callable[[], None] | None = None,
        on_token_refresh: Callable[[], None] | None = None,
    ) -> None:
        RefreshableTokenAuth.__init__(
            self, api_client, on_token_creation, on_token_refresh, timedelta(minutes=15)
        )

        if not api_client._is_IAM():
            raise WMLClientError(
                "api_key for IAM token is not provided in credentials for the client."
            )

        self._save_token_data(self._generate_token())
        if self._on_token_creation:
            self._on_token_creation()

        # update of minimal expiration datetime based on token expiration datetime
        delta = self._get_expiration_datetime() - datetime.now()
        if delta < self._expiration_timedelta:
            self._expiration_timedelta = (
                delta - timedelta(minutes=1) if delta > timedelta(minutes=1) else delta
            )

    def _generate_token(self) -> TokenInfo:
        """Generate token using IAM authentication.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic Yng6Yng=",
        }

        mystr = "apikey=" + self._href_definitions.get_iam_token_api()
        response = self._session.post(
            self._href_definitions.get_iam_token_url(), data=mystr, headers=headers
        )
        if response.status_code == 200:
            return TokenInfo(response.json().get("access_token"))
        else:
            raise WMLClientError("Error getting IAM Token.", response)
