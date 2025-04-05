#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from enum import Enum


class RShinyAuthenticationValues(Enum):
    """Allowable values of R_Shiny authentication."""

    ANYONE_WITH_URL = "anyone_with_url"
    ANY_VALID_USER = "any_valid_user"
    MEMBERS_OF_DEPLOYMENT_SPACE = "members_of_deployment_space"


class AssetDuplicateAction(Enum):
    """Allowed values for `duplicate_action` parameter.

    Available values:
     - IGNORE - the call will ignore the duplicate and create a new asset
     - REJECT - the call will fail and no asset will be created
     - UPDATE - the best matched duplicate will be updated with the incoming changes according to the predefined rules
     - REPLACE - the best matched duplicate will be overwritten with the input values according to the predefined rules
    """

    IGNORE = "IGNORE"
    REJECT = "REJECT"
    UPDATE = "UPDATE"
    REPLACE = "REPLACE"
