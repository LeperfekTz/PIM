#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

class MetaPropMissingError(Exception):
    pass


class UnsupportedTFSerializationFormat(Exception):
    pass


class UnmatchedKerasVersion(Exception):
    pass

class InvalidCaffeModelArchive(Exception):
    pass
