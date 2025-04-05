#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import sys


class PythonVersion(object):
    @staticmethod
    def significant():
        return "{}.{}".format(sys.version_info[0], sys.version_info[1])
