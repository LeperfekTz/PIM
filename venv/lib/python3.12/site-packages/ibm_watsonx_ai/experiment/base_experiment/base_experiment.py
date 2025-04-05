#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod

__all__ = ["BaseExperiment"]


class BaseExperiment(ABC):
    """Base abstract class for Experiment."""

    @abstractmethod
    def runs(self, *, filter: str):
        """Get the historical runs but with Pipeline name filter.

        :param filter: Pipeline name to filter the historical runs
        :type filter: str
        """
        pass
