from abc import ABC, abstractmethod


class BaseFrameExtractor(ABC):
    """
    Abstract Base Class for all frame extractors.
    """

    def __init__(self, video_path, output_folder):
        self.video_path = video_path
        self.output_folder = output_folder

    @abstractmethod
    def extract(self):
        """
        Every child class must implement this method.
        """
        pass