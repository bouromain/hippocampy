import numpy as np
import bottleneck as bn
from numpy.core.fromnumeric import squeeze

from hippocampy.utils.gen_utils import start_stop


class Iv:

    __attributes__ = ["_data", "_domain"]

    def __init__(self, data=None, domain=None, unit=None) -> None:

        if data is None or len(data) == 0:
            # to allow the creation of empty Iv
            print("create empty")
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return
        else:

            if isinstance(data, type(self)):
                # create from the same type
                data = data.data

            elif isinstance(data, (list, tuple, np.ndarray)):
                data = np.squeeze(data)

                # check we have a even number of elements
                if data.size % 2 != 0:
                    raise ValueError(
                        "Data should have the same number of starts and stops"
                    )

                if data.ndim == 1:
                    data = data.ravel()

                elif data.ndim == 2:
                    if not any(data.shape == 2):
                        raise ValueError(
                            "Data should have the same number of starts and stops"
                        )
                    if data.shape[1] != 2:
                        data = data.T

                else:
                    raise ValueError("Data can not be more than 2D")

            self._data = data
            self._domain = domain
            self._unit = unit

    @property
    def data(self):
        return self._data

    @property
    def domain(self):
        if self._domain is None:
            self._domain = type(self)([-np.inf, np.inf])
            return self._domain

    @domain.setter
    def domain(self, vals):
        # TODO allow the domain to be discontinuous notably when we input lists
        if isinstance(vals, type(self)):
            self._domain = vals
        elif isinstance(self, (tuple, list)):
            self._domain = type(self)([vals[0], vals[0]])

    @property
    def n_intevals(self):
        "number of intervals"
        if self.isempty:
            return None
        else:
            return len(self._data[:, 0])

    @property
    def isempty(self):
        if self._data is None:
            return True
        else:
            return False

    @property
    def starts(self):
        if self.isempty:
            return None
        else:
            return self.data[:, 0]

    @property
    def stops(self):
        if self.isempty:
            return None
        else:
            return self.data[:, 1]

    @property
    def max(self):
        return bn.nanmax(self.stops)

    @property
    def min(self):
        return bn.nanmin(self.starts)

    @property
    def centers(self):
        return bn.nanmean(self.data, axis=1)

    @property
    def lengths(self):
        return self.data[:, 1] - self.data[:, 0]

    @property
    def issorted(self):
        if self.isempty:
            return True
        else:
            return all(np.diff(self.starts) >= 0)

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, int):
            return type(self)([self.data[idx, :]])
        elif isinstance(idx, (list, tuple, np.array)):
            try:
                idx = np.squeeze(idx)
                return type(self)([self.data[idx, :]])
            except IndexError:
                raise IndexError(f"{self.__class__.__name__} index out of range")
            except Exception:
                raise TypeError(f"Unsuported indexing type {type(idx)}")
        else:
            raise TypeError(f"Unsuported indexing type {type(idx)}")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index
        if index > self.n_intevals - 1:
            raise StopIteration

        self._index += 1
        return type(self)(self.data[index, :])

    def __and__(self, other):
        ...
        # https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals

    def __contains__(self, other):
        """
        Contains is defined for Iv, arrays/list and values
        """

        if isinstance(other, type(self)):
            is_in = np.logical_and(
                other.starts >= self.starts, other.stops <= self.stops
            )
            return is_in

        elif isinstance(other, (np.ndarray, list)):
            return type(self)(other) in self

        elif isinstance(other, (int, float)):
            is_in = np.logical_and(other >= self.starts, other <= self.stops)
            return is_in

        else:
            raise TypeError

    def __len__(self):
        return self.n_intevals

    def __bool__(self):
        return not self.isempty

    def __nonzero__(self):
        return not self.isempty

    def __eq__(self, other):
        return (
            (self.starts == other.starts).all()
            and (self.stops == other.stops).all()
            and (self.domain == other.domain).all()
        )

    def __ne__(self, other):
        return not (self == other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.starts > other
        elif isinstance(other, type(self)):
            return self.starts > other.max

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.starts < other
        elif isinstance(other, type(self)):
            return self.starts < other.min

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.starts >= other
        elif isinstance(other, type(self)):
            return self.starts >= other.max

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.starts <= other
        elif isinstance(other, type(self)):
            return self.starts <= other.min

    def __invert__(self):
        # invert the interval using the domain
        pass

    def __or__(self):
        # calculate union
        pass

    def __rshift__(self, other):
        if not self.isempty:
            if isinstance(other, (int, float)):
                self.data = self.data + other
            else:
                raise TypeError

    def __lshift__(self, other):
        if not self.isempty:
            if isinstance(other, (int, float)):
                self.data = self.data - other
            else:
                raise TypeError

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.isempty:
            return f"empty {cls_name}"
        elif self.n_intevals == 1:
            return f"< {cls_name} object with 1 interval >"
        else:
            return f"< {cls_name} object with {self.n_intevals} intervals >"

    def _sort(self):
        """Sort interval data with start time"""
        idx = np.argsort(self.starts)
        self._data = self.data[idx, :]

    def from_bool(self, bool_vec):
        bool_vec = np.squeeze(np.array(bool_vec, dtype=bool))

        if not bool_vec.ndim == 1:
            raise ValueError("Boolean input shoudl be a vector")
        else:
            starts, stops = start_stop(bool_vec)
            return type(self)([starts, stops])

    # others