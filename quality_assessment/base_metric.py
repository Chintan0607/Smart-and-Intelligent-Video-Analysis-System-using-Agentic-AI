from abc import ABC, abstractmethod


class BaseQualityMetric(ABC):

    @abstractmethod
    def evaluate(self, image):
        pass