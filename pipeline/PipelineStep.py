from abc import ABC, abstractmethod

from pipeline.IndexPositions import IndexPositions


class PipelineStep(ABC, IndexPositions):

    @abstractmethod
    def perform(self, data):
        pass

    @property
    def name(self) -> str:
        return type(self).__name__
