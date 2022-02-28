import numpy as np
import bottleneck as bn


class Tsd:

    __attributes__ = ["_data", "_dims", "_domain"]

    def __init__(self, data=None, dims=None, domain=None, unit=None) -> None:
        if data is None or len(data) == 0:
            # to allow the creation of empty Iv
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return
        else:
            ...
            # check the types of the inputs and check size to init correctly

    @property
    def data(self):
        return self._data

    @property
    def dims(self):
        return self._dims

    @property
    def domain(self):
        return self._domain

    @property
    def isempty(self):
        if self._data is None:
            return True
        else:
            return False

    @property
    def n_samples(self):
        if self.isempty:
            return 0
        else:
            return self._data.shape[-1]

    def __len__(self):
        return self.n_samples

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.isempty:
            return f"empty {cls_name}"
        else:
            return f"< {cls_name} object of size {self._data.shape} >"

