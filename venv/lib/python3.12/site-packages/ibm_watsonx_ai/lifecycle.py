#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from enum import Enum


class SpecStates(Enum):
    AVAILABLE = "available"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    CONSTRICTED = "constricted"
    RETIRED = "retired"
    WITHDRAWN = "withdrawn"

    UNSUPPORTED = "unsupported"
    CREATE_UNSUPPORTED = "create-unsupported"
