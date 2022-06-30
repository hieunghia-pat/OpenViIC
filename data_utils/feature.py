from typing import Any

class Feature(object):
    def __init__(self, features: dict):
        self.__dict__ = features

    def data(self):
        return self.__dict__.items()

    def set(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, __name: Any) -> Any:
        return None