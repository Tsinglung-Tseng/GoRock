class FuncList:
    def __init__(self, ldata):
        self.ldata = ldata

    def to_list(self):
        return list(self.ldata)

    def map(self, func):
        return FuncList(map(func, self.ldata))

    def filter(self, func):
        return FuncList(filter(func, self.ldata))


class MAC:
    """
    >>> MAC(geometry_mac).to_pair()
    >>> [('/gate/world/daughters/name', 'OpticalSystem'),
         ('/gate/world/daughters/insert', 'box'),
         ...
    """
    def __init__(self, raw_mac):
        self.raw_mac = raw_mac

    def to_json(self):
        raise NotImplemented

    def to_pair(self):
        return (
            FuncList(self.raw_mac.split('\n'))
                .filter(lambda i: len(i) != 0)
                .filter(lambda i: i[0] == '/')
                .map(lambda i: i.split())
                .map(lambda i: (i[0], ' '.join(i[1:])))
                .to_list()
        )

    def build(self):
        #TODO: replace exec with real mac
        return

