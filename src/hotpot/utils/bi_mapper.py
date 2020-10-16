from collections import abc

from ..registry import Registry


class ConfigBiMapping:
    __primitive = (bool, int, str, float)

    load_mapping = {key.__name__: key for key in Registry.bi_mapping_items}
    dump_mapping = {key: key.__name__ for key in Registry.bi_mapping_items}

    @staticmethod
    def load(obj):
        result = {}
        for k, v in obj.items():
            if not isinstance(v, abc.Mapping):
                if v in ConfigBiMapping.load_mapping.keys():
                    result[k] = ConfigBiMapping.load_mapping[v]()
                else:
                    result[k] = v
            else:
                result[k] = ConfigBiMapping.load(v)
        return result

    @staticmethod
    def dump(a_dict):
        def _is_primitive(var):
            return type(var) in ConfigBiMapping.__primitive

        result = {}
        for k, v in a_dict.items():
            if not isinstance(v, abc.Mapping):
                if _is_primitive(v):
                    result[k] = v
                else:
                    result[k] = v.__class__.__name__
            else:
                result[k] = ConfigBiMapping.dump(v)

        return result
