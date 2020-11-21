import re


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
