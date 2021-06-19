# Traits for traces

###################
#### BASE TYPE ####
###################

class Trace:
    
    def __init__(self):
        pass
    
    def __repr__(self):
        return "Trace"


#######################
#### FEEDING TYPES ####
#######################

class SlitFedTrace(Trace):

    def __repr__(self):
        return "Slit Fed Trace"

class FiberFedTrace(Trace):

    def __repr__(self):
        return "Fiber Fed Trace"

######################
#### OBJECT TYPES ####
######################

class PointSourceTrace(Trace):
    
    def __repr__(self):
        return "Stellar Trace"

class SkyTrace(Trace):
    
    def __repr__(self):
        return "Stellar Trace"

class SparseLampEmissionTrace(Trace):
    
    def __repr__(self):
        return "Spare Lamp Emission Trace"