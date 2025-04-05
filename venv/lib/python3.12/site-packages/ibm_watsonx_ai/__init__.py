#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from importlib.metadata import version

package_name = __name__.replace("_", "-")
__version__ = version(package_name)

from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.client import APIClient
from ibm_watsonx_ai.utils.enums import AssetDuplicateAction

APIClient.version = __version__

__all__ = ["APIClient", "AssetDuplicateAction", "Credentials", "package_name"]
