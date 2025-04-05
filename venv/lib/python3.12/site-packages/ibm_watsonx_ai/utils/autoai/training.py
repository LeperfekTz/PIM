#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
__all__ = ["is_run_id_exists"]

from typing import Dict, Optional
from warnings import warn

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.wml_client_error import ApiRequestFailure


def is_run_id_exists(
    credentials: Dict, run_id: str, space_id: Optional[str] = None, **kwargs
) -> bool:
    """Check if specified run_id exists for API client initialized with passed credentials.

    :param credentials: Service Instance credentials
    :type credentials: dict

    :param run_id: training run id of AutoAI experiment
    :type run_id: str

    :param space_id: optional space id
    :type space_id: str, optional

    """
    # note: backward compatibility
    if (wml_credentials := kwargs.get("wml_credentials")) is not None:
        if credentials is None:
            credentials = wml_credentials

        wml_credentials_deprecated_warning = (
            "`wml_credentials` is deprecated and will be removed in future. "
            "Instead, please use `credentials`."
        )
        warn(wml_credentials_deprecated_warning, category=DeprecationWarning)

    # --- end note
    client = APIClient(credentials)

    if space_id is not None:
        client.set.default_space(space_id)

    try:
        client.training.get_details(run_id, _internal=True)

    except ApiRequestFailure as e:
        if "Status code: 404" in str(e):
            return False

        else:
            raise e

    return True
