import pickle
import warnings

class SessionState:
    
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)
        
    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(f)
        
    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            d = pickle.load(f)
        return d