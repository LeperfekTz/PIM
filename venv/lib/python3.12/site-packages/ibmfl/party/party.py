#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


import logging

logger = logging.getLogger(__name__)


class Party:

    def __init__(self, **kwargs):
        from ibm_watsonx_ai.party_wrapper import Party
        self.party = Party(**kwargs)

    def start(self):
        self.party.start()
