from abc import ABC, abstractmethod


class Model(ABC):
    """This class represents a generic model with different method that every subclass model
        is likely to use and reimplement

    Args:
        ABC (ABC): Abstract class
    """

    def __init__(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _preprocessing(self):
        pass

    @abstractmethod
    def _postprocessing(self):
        pass
