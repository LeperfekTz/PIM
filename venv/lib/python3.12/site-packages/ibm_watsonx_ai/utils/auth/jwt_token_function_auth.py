#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from datetime import timedelta
from typing import Callable, TYPE_CHECKING

from ibm_watsonx_ai.utils.auth.base_auth import RefreshableTokenAuth, TokenInfo
from ibm_watsonx_ai.wml_client_error import WMLClientError

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


class JWTTokenFunctionAuth(RefreshableTokenAuth):
    """Token function authentication method class.

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
            self, api_client, on_token_creation, on_token_refresh, timedelta(minutes=50)
        )

        if (
            not hasattr(self._credentials, "token_function")
            or not self._credentials.token_function
        ):
            raise WMLClientError(
                'Error getting token with token function: "token_function" is mandatory element in credentials.'
            )

    def _generate_token(self) -> TokenInfo:
        """Generate token using token_function provided by user.

        :returns: token info to be used by auth method
        :rtype: TokenInfo
        """
        result = self._credentials.token_function(self._session)

        if isinstance(result, str):
            return TokenInfo(result)
        elif isinstance(result, TokenInfo):
            return result
        else:
            raise WMLClientError(
                "Value returned from `token_function` can be only string containing token, or TokenInfo object."
            )
