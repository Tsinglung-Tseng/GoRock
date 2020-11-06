import re


class FuncList:
    def __init__(self, L):
        self.L = L
    
    def __repr__(self):
        return str(self.L)
        
    def to_list(self):
        return list(self.L)

    def map(self, func):
        return FuncList(list(map(func, self.L)))

    def filter(self, func):
        return FuncList(filter(func, self.L))

class FuncJson:
    def __init__(self, J):
        self.J = J
        
    def __repr__(self):
        return str(self.J)
    
    def __getitem__(self, idx):
        return FuncJson(self.J[idx])
    
    def keys(self):
        return self.J.keys()
    
    def values(self):
        return self.J.values()

    def items(self):
        return self.J.items()
    
    def get_by_regex(self, r):
        return FuncJson({
            k: v
            for k, v in self.J.items()
            if re.match(r, k) is not None
        })
    
    def apply_on_values(self, func):
        return FuncList(list(map(func, self.values())))
