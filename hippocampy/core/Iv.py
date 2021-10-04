import numpy as np
import bottleneck as bn

from hippocampy.utils.gen_utils import start_stop

"""
Interval objects

References
----------
https://github.com/kvesteri/intervals/blob/master/intervals/interval.py
https://github.com/nelpy/nelpy/blob/43d07f3652324f8b89348a21fde04019164ab536/nelpy/core/_intervalarray.py

TODO check the intersect function and make a exclude function (eg exclude an IV from an other) 

"""


def coerce_to_interval(func):
    def wrapper(self, arg):
        if (
            isinstance(arg, list)
            or isinstance(arg, tuple)
            or isinstance(arg, np.ndarray)
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
                _data = data
                data = _data.data
                domain = _data.domain
                unit = _data._unit

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
        else:
            self._domain = type(self)(self._domain)
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
        return (self.starts - self.stops) / 2

    @property
    def lengths(self):
        return self.stops - self.starts

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

    @property
    def degenerate(self):
        """ an interval is considered degenerate if it is empty"""
        return self.starts == self.stops

    def __getitem__(self, idx):
        if self.isempty:
            return self

        if isinstance(idx, int):
            return type(self)([self.data[idx, :]])
        if isinstance(idx, slice):
            return type(self)([self.data[idx, :]])

        elif isinstance(idx, (list, tuple, np.ndarray)):
            try:
                idx = np.squeeze(idx)
                return type(self)([self.data[idx, :]])
            except IndexError:
                raise IndexError(f"{self.__class__.__name__} index out of range")
            except Exception:
                raise TypeError(f"Unsupported indexing type {type(idx)}")
        else:
            raise TypeError(f"Unsupported indexing type {type(idx)}")

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
        
        https://scicomp.stackexchange.com/questions/26258/the-easiest-way-to-find-intersection-of-two-intervals
        """
        new_starts, new_stops = self._intersect(other)
        domain_new_starts, domain_new_stops = self.domain._intersect(other.domain)

        # remove potential interval duplicates
        data_out = np.unique(np.hstack((new_starts, new_stops)), axis=0)
        out = type(self)(data=data_out, domain=[domain_new_starts, domain_new_stops])
        # remove empty intervals
        m = out.degenerate

        return out[~m]

    def _intersect(self, other):
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

        return np.asarray(new_starts), np.asarray(new_stops)

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
            and (self.domain.data == other.domain.data).all()
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
        """ 
        create a Iv from a boolean vector

        the domain will be affected to 0 and len(bool_vector)
        
        """
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
        """ merge overlaping intervals"""

        if gap < 0:
            raise ValueError(f"Gap should not be negative")
        if overlap < 0:
            raise ValueError(f"Overlap should not be negative")
        if max_len < 0:
            raise ValueError(f"Maximum length should not be negative")

        if self.isempty:
            return self

        # if already merged return self

        if self.ismerged and gap == 0:
            return self

        if not self.issorted:
            self._sort()

        # remove included intervals
        m = self.included(self)
        self = self[~m]

        # we can now start the merge
        new_starts = []
        new_stops = []

        prev_start = self[0].starts
        prev_stop = self[0].stops

        for it, tmp_iv in enumerate(self[1:]):
            new_start = min(tmp_iv.starts, prev_start)
            new_stop = max(tmp_iv.stops, prev_stop)

            # we want to merge if:
            is_overlap = (prev_stop + gap) - tmp_iv.starts >= overlap
            is_too_long = new_stop - new_start > max_len

            if is_overlap and (not is_too_long):
                # but also if it does not merge with the next interval
                if it < len(self) - 2:
                    overlap_next = tmp_iv.stops - self[it + 2].starts >= overlap
                else:
                    overlap_next = False

                if overlap_next:
                    prev_start = new_start
                    prev_stop = new_stop
                else:
                    new_starts.append(new_start)
                    new_stops.append(new_stop)
                    prev_start = new_start
                    prev_stop = new_stop

            else:
                # if no overlap, keep the interval
                new_starts.append(prev_start)
                new_stops.append(prev_stop)
                prev_start = tmp_iv.starts
                prev_stop = tmp_iv.stops

        self._data = np.hstack((np.asarray(new_starts), np.asarray(new_stops)))

        return self

    @coerce_to_interval
    def included(self, other):
        """ return a mask of interval in "other" which are completely included 
        in the set of intervals"""
        mask = np.empty((len(other), 1), dtype=bool)
        for i, tmp_it in enumerate(other):
            mask[i] = np.logical_and(
                tmp_it.starts > self.starts, tmp_it.stops < self.stops
            ).any()
        return np.squeeze(mask)

