from hippocampy.matrix_utils import label
import numpy as np
import bottleneck as bn

from hippocampy.utils.gen_utils import start_stop

"""
Interval objects

References
----------
https://github.com/kvesteri/intervals/blob/master/intervals/interval.py
https://github.com/nelpy/nelpy/blob/43d07f3652324f8b89348a21fde04019164ab536/nelpy/core/_intervalarray.py

TODO correct the domain eg intesect the too domains too
for now it creates an infinite recursion as a domain is an Iv 

"""


def coerce_to_interval(func):
    def wrapper(self, arg):
        if (
            isinstance(arg, list)
            or isinstance(arg, tuple)
            or isinstance(arg, np.ndarray)
            or isinstance(arg, type(self))
        ):
            try:
                if arg is not None:
                    arg = type(self)(arg)
                return func(self, arg)
            except:
                raise TypeError(
                    f" Function {func.__name__} not implemented for type {type(arg).__name__}"
                )
        try:
            arg = type(self)(arg)
        except (ValueError, TypeError, OverflowError):
            pass
        return func(self, arg)

    return wrapper


class Iv:

    __attributes__ = ["_data", "_domain"]

    def __init__(self, data=None, domain=None, unit=None) -> None:

        if data is None or len(data) == 0:
            # to allow the creation of empty Iv
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return
        else:

            if isinstance(data, type(self)):
                # create from the same type
                data = data.data
                domain = data.domain
                unit = data.unit

            elif isinstance(data, (list, tuple, np.ndarray)):
                data = np.squeeze(data)

                # check we have a even number of elements
                if data.size % 2 != 0:
                    raise ValueError(
                        "Data should have the same number of starts and stops"
                    )

                if data.ndim == 1:
                    data = np.atleast_2d(data).reshape(-1, 2)

                elif data.ndim == 2:
                    if 2 not in data.shape:
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
    def n_intervals(self):
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
        if self.n_intervals == 1:
            return True
        else:
            return all(np.diff(self.starts) >= 0) and all(np.diff(self.stops) >= 0)

    @property
    def ismerged(self):
        """(bool) No overlapping intervals exist."""
        if self.isempty:
            return True
        if self.n_intervals == 1:
            return True
        if not self.issorted:
            self._sort()
        if not all(np.diff(self.stops) >= 0):
            return False

        return (self.starts[1:] > self.stops[:-1]).all()

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, int):
            return type(self)([self.data[idx, :]])
        elif isinstance(idx, (list, tuple, np.ndarray)):
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
        if index > self.n_intervals - 1:
            raise StopIteration

        self._index += 1
        return type(self)(self.data[index, :])

    def __and__(self, other):
        return self.intersect(other)

    @coerce_to_interval
    def intersect(self, other):

        """
        perform the intersection of an set of interval with a
        single or an other set of intervals


        """
        # https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals

        new_starts = []
        new_stops = []

        for tmp_other in other:
            is_intersect = np.logical_and(
                self.starts <= tmp_other.stops, self.stops >= tmp_other.starts
            )
            # now we found the intersecting interval we can either have
            # no intersection, a single of multiple ones
            if is_intersect.any():
                for tmp_self in self[is_intersect]:
                    new_starts.append(max([tmp_self.starts, tmp_other.starts]))
                    new_stops.append(min([tmp_self.stops, tmp_other.stops]))
        new_starts = np.asarray(new_starts)
        new_stops = np.asarray(new_stops)

        return type(self)(np.hstack((new_starts, new_stops)))

    @coerce_to_interval
    def __or__(self, other):
        """Define union between intervals
        we will do that by joining the two sets of interval and by
        merging them (eg: joining overlapping intervals)
        """
        new_self = self.append(other)
        new_self._sort()

        return new_self.merge()

    def union(self, other):
        return self | other

    @coerce_to_interval
    def append(self, other):

        if self.isempty:
            return other
        if other.isempty:
            return self

        return type(self)(np.vstack((self.data, other.data)))

    @coerce_to_interval
    def __rshift__(self, other):
        return self.append(other)

    @coerce_to_interval
    def __lshift__(self, other):
        return self.append(other)

    def __invert__(self):
        # invert the interval using the domain
        pass

    def __contains__(self, other):
        """
        Contains is defined for Iv, arrays/list and values
        """
        return (self.contain(other)).any()

    @coerce_to_interval
    def contain(self, other):
        """
        return boolean showing which interval contains
        the defined Iv, arrays/list or values
        """
        if isinstance(other, (int, float)):
            is_in = np.logical_and(other >= self.starts, other <= self.stops)
            return is_in

        else:
            is_in = np.zeros((1, self.n_intervals), dtype=bool)

            for tmp_out in other:
                tmp_in = np.logical_and(
                    tmp_out.starts >= self.starts, tmp_out.stops <= self.stops
                )
                is_in = np.logical_or(is_in, tmp_in)
            return is_in

    def __len__(self):
        return self.n_intervals

    def __bool__(self):
        return not self.isempty

    def __nonzero__(self):
        return not self.isempty

    @coerce_to_interval
    def __eq__(self, other):
        return (
            (self.starts == other.starts).all()
            and (self.stops == other.stops).all()
            and (self.domain == other.domain).all()
        )

    def __ne__(self, other):
        return not (self == other)

    @coerce_to_interval
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return self.starts > other
        else:
            return self.starts > other.max

    @coerce_to_interval
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return self.starts < other
        else:
            return self.starts < other.min

    @coerce_to_interval
    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return self.starts >= other
        else:
            return self.starts >= other.max

    @coerce_to_interval
    def __le__(self, other):
        if isinstance(other, (int, float)):
            return self.starts <= other
        else:
            return self.starts <= other.min

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.isempty:
            return f"empty {cls_name}"
        elif self.n_intervals == 1:
            return f"< {cls_name} object with 1 interval >"
        else:
            return f"< {cls_name} object with {self.n_intervals} intervals >"

    def _sort(self):
        """Sort interval data with start time"""
        idx = np.argsort(self.starts)
        self._data = self.data[idx, :]

    def from_bool(self, bool_vec):
        bool_vec = np.squeeze(np.array(bool_vec, dtype=bool))

        if not bool_vec.ndim == 1:
            raise ValueError("Boolean input should be a vector")
        else:
            starts, stops = start_stop(bool_vec)
            starts = np.nonzero(starts)[0]
            stops = np.nonzero(stops)[0]
            self._data = np.vstack((starts, stops)).T
            self._domain = [0, len(starts)]

    def merge(self, *, gap=0.0, overlap=0.0, max_len=np.Inf):
        ...

        if gap < 0:
            raise ValueError(f"Gap should not be negative")
        if overlap < 0:
            raise ValueError(f"Gap should not be negative")
        if max_len < 0:
            raise ValueError(f"Gap should not be negative")

        if self.isempty:
            return self

        # if alredy merged return self

        if self.ismerged and gap == 0:
            return self

        new_starts = []
        new_stops = []

        to_merge = self.starts[1:] - self.stops[:-1] > -overlap
        # add an element at the begining

        # a = np.array([0,0,0,1,1,0,1,0,1,1],dtype=bool)

        # o,_ = start_stop(a)
        # o1,_= start_stop(~a)
        # oo = np.logical_or(o,o1)
        # i = np.nonzero(oo)[0]
        # i = np.insert(i,0,0)
        # i = np.append(i,len(a)+1)
        # print(i-0.5)
        # r = np.arange(len(a))
        # print(r)
        # np.digitize(r,i)

        for it_merge in np.unique(to_merge[to_merge != 0]):
            tmp_start = min(self[to_merge == it_merge].starts)
            tmp_stop = max(self[to_merge == it_merge].stops)

            if tmp_stop - tmp_start < max_len:
                # only merge if resulting length does not exceed
                # a given maximum length
                new_starts.append(tmp_start)
                new_stops.append(tmp_stop)

        new_starts = np.asarray(new_starts)
        new_stops = np.asarray(new_stops)

        return type(self)(np.hstack((new_starts, new_stops)))
