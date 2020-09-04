import nengo
from nengo.processes import Piecewise

min_r = -1
max_r = 1

def scale_down(x,min_r, max_r):
    return ((max_r-min_r) * (x - min_r) / (max_r - min_r)) + min_r
def scale_up_deg(x):
    return ((max_r-min_r) * (x + 1) / 2) +  min_r
    
model = nengo.Network()
with model:
    # representing an integrator of acceleration
    integrator = nengo.Ensemble(n_neurons=100, dimensions=2)
    # representing the current location in x,y,z coordinates
    current_location = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Create a piecewise step function for input
    stim  = nengo.Node(Piecewise({0: scale_down(0,min_r,max_r), 2: scale_down(0.5,min_r,max_r), 4: scale_down(1,min_r,max_r), 6: scale_down(-0.5,min_r,max_r), 8: scale_down(-1,min_r,max_r)}))
    
    # Connect the population to itself using a long time constant (tau) 
    # for stability
    tau = 1
    nengo.Connection(integrator, integrator, synapse=tau)

    # Connect the input using the same time constant as on the recurrent
    # connection to make it more ideal
    nengo.Connection(stim, integrator[0], transform=tau, synapse=tau)
    
    # x[1] is the current state of the integrator,
    # x[0] is the new input acceleration
    def calculate_xyz(x):
        return scale_up_deg((x[1])) + ((x[0]))
        # return ((x[1])) + ((x[0]))
        
    # Connect the integrator to the output ensamble to point the current 
    # location based on the integration
    nengo.Connection(integrator, current_location, transform=tau, synapse=tau, function=calculate_xyz)