#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2021，2022.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import logging


logger = logging.getLogger(__name__)


class DataHandler:

    raise Exception("This data handler import path is deprecated. "
                    "Please do \"from ibm_watsonx_ai.federated_learning.data_handler "
                    "import DataHandler\" instead ")

    def __init__(self, **kwargs):
        pass
