from collections import abc


class TFWrapperCrasher:
    def __init__(self, obj):
        self.obj = obj

    @staticmethod
    def is_simple(obj):
        _primitive = (int, float, str, bool)
        if type(obj) in _primitive:
            return True
        return False

    @staticmethod
    def is_list(obj):
        if isinstance(obj, abc.Iterable) and not isinstance(obj, str):
            return True
        return False

    @staticmethod
    def is_mapping(obj):
        if isinstance(obj, abc.Mapping):
            return True
        return False

    def __call__(self):
        def _rebuild_list(l):
            _buf = []
            for i in l:
                if TFWrapperCrasher.is_simple(i):
                    _buf.append(i)
                elif TFWrapperCrasher.is_mapping(i):
                    _buf.append(_rebuild_mapping(i))
                elif TFWrapperCrasher.is_list(i):
                    _buf.append(_rebuild_list(i))
            return _buf

        def _rebuild_mapping(m):
            _buf = {}
            for k, v in m.items():
                if TFWrapperCrasher.is_simple(v):
                    _buf[k] = v
                elif TFWrapperCrasher.is_mapping(v):
                    _buf[k] = _rebuild_mapping(v)
                elif TFWrapperCrasher.is_list(v):
                    _buf[k] = _rebuild_list(v)
            return _buf

        return _rebuild_mapping(self.obj)
