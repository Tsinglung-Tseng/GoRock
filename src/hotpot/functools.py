import re
import numpy as np
import pandas as pd
import functools
import tensorflow as tf


class FuncDataFrame:
    def __init__(self, df):
        self.df = df

    def where(self, **kwargs):
        if len(kwargs) != 1:
            raise ValueError("where clause support one condition at once!")
        for key, value in kwargs.items():
            return FuncDataFrame(self.df[self.df[key] == value])

    def filter(self, key_list):
        return FuncDataFrame(self.df[key_list])


class FuncList:
    def __init__(self, ldata):
        self.ldata = ldata

    def to_list(self):
        return list(self.ldata)

    def map(self, func):
        return FuncList(map(func, self.ldata))

    def filter(self, func):
        return FuncList(filter(func, self.ldata))

    def split_by_regex(self, re_string):
        return FuncList(re.split(re_string, self.ldata))

    def flat(self):
        """
        Flat a nested list.
        """
        buf = []

        def flat_list(arg):
            def _flat_list(arg):
                def _on_iterable(arg):
                    _flat_list(arg)

                def _on_single(arg):
                    buf.append(arg)

                if maybe_type(arg, iter) is None:
                    _on_single(arg)
                else:
                    for i in arg:
                        _on_iterable(i)
                return buf

            return _flat_list(arg)

        for arg in self.ldata:
            flat_list(arg)

        return buf


class FuncArray:
    def __init__(self, array):
        self.array = array

    @staticmethod
    def from_pd_series(series):
        return FuncArray(np.array(series.apply(lambda i: np.array(i)).to_list()))

    def to_tensor(self):
        return tf.convert_to_tensor(self.to_numpy())

    def to_numpy(self):
        if isinstance(self.array, pd.Series):
            return np.array(self.array.to_list())
        if isinstance(self.array, np.ndarray):
            return self.array
        if isinstance(self.array, tf.Tensor):
            return self.array.numpy()
        if isinstance(self.array, pd.DataFrame):
            return np.array(self.array)

    def to_list(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.array.shape

    def rollaxis(self, axis, start=0):
        return FuncArray(np.rollaxis(self.to_numpy(), axis, start))

    def expand_dims(self, axis):
        return FuncArray(FuncArray(tf.expand_dims(self.to_tensor(), axis)).to_numpy())

    def concatenate_with(self, other, axis):
        return FuncArray(np.concatenate([self.array, other.array], axis))

    def shrink(self, axis):
        def _index_list_by_list(l, ind):
            return [l[i] for i in ind]

        def _list_exclude_list(l_1, l_2):
            return [i for i in list(range(l_1)) if i not in l_2]

        def _insert_to_list(l: "target list", i: "position to insert", n: "value"):
            left_half = l[:i]
            right_half = l[i:]
            return [*left_half, n, *right_half]

        self_shape = self.shape
        self_dim = len(self_shape)
        dim_after_shrink = self_dim - len(axis)
        to_be_shrinked = _index_list_by_list(self_shape, axis)

        shrinked_size = functools.reduce(lambda x, y: x * y, to_be_shrinked)
        result_dim = _insert_to_list(
            _index_list_by_list(self_shape, _list_exclude_list(self_dim, axis)),
            axis[0],
            shrinked_size,
        )
        return FuncArray(self.to_numpy().reshape(result_dim))
