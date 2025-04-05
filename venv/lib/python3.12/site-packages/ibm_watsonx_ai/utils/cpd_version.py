#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import cast

from ibm_watsonx_ai.wml_client_error import WMLClientError
from ibm_watsonx_ai.messages.messages import Messages


class CPDVersion:
    """Storage for cpd version. Comparison operators are
    overloaded to allow comparison with numeric values.

    Class attribute:
    :param supported_version_list: List of supported CPD versions.
    :type supported_version_list: list

    Attribute:
    :param cpd_version: Store CPD version.
    :type cpd_version: Optional[str], optional

    .. code-block:: python

        from ibm_watsonx_ai.utils import CPDVersion

        version = CPDVersion()

        if not version:
            print("CPD version is None")

        version.cpd_version = '4.5'

        if version > 4:
            print("version greater than 4.0")

    """

    supported_version_list = ["4.0", "4.5", "4.6", "4.7", "4.8", "5.0", "5.1"]

    def __init__(self, version: str | None = None):
        self.cpd_version = version

    def __str__(self) -> str:
        version = self.__cpd_version  # type: ignore[has-type]
        return f"CPD version {version}" if version is not None else ""

    @property
    def cpd_version(self) -> float:
        """Attribute that stores cpd version. Before the value is set,
        validation is performed against a supported versions.
        """
        return self.__cpd_version  # type: ignore[has-type]

    @cpd_version.setter
    def cpd_version(self, value: float | str) -> None:
        if value is None:
            self.__cpd_version = value
        elif str(value) in CPDVersion.supported_version_list:
            self.__cpd_version = str(value)  # type: ignore[has-type]
        else:
            raise WMLClientError(
                Messages.get_message(
                    ", ".join(CPDVersion.supported_version_list),
                    message_id="invalid_version",
                )
            )

    @cpd_version.getter
    def cpd_version(self) -> float | None:
        value = self.__cpd_version  # type: ignore[has-type]
        if value is None:
            return None
        else:
            dot_index = value.find(".")
            value = value[: dot_index + 1] + value[dot_index + 1 :].replace(".", "")
            return float(value)

    def __bool__(self) -> bool:
        return bool(self.cpd_version)

    def __eq__(self, value: float) -> bool:  # type: ignore[override]
        return self.cpd_version == value

    def __ne__(self, value: float) -> bool:  # type: ignore[override]
        return not self.__eq__(value)

    def __lt__(self, value: float) -> bool:
        return bool(self) and self.cpd_version.__lt__(value)

    def __le__(self, value: float) -> bool:
        return bool(self) and (self.__lt__(value) or self.__eq__(value))

    def __gt__(self, value: float) -> bool:
        return bool(self) and not self.__le__(value)

    def __ge__(self, value: float) -> bool:
        return bool(self) and not self.__lt__(value)
