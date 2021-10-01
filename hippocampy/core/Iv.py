import numpy as np
import bottleneck as bn

class Iv:

    __attributes__ = ["_data","_domain"]
    def __init__(self, data=None,domain=None,unit=None) -> None:
        
        if data is None or len(data)==0:
            # to allow the creation of empty Iv
            print("create empty")
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return
        else:
            data = np.squeeze(data)
            print(data)

    def __repr__(self) -> str:
        pass
    
    @property
    def domain(self):
        if self._domain is None:
            self._domain = type(self)([-np.inf, np.inf])
            return self._domain
    
    @domain.setter
    def domain(self, vals):
        # TODO allow the domain to be discontinuous notabli when we input lists
        if isinstance(vals,type(self)):
            self._domain = vals
        elif isinstance(self,(tuple,list)):
            self._domain=type(self)([vals[0], vals[0]])