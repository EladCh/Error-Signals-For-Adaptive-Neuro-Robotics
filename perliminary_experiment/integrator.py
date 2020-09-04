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
    ################################
    ######         x          ######
    ################################
    # representing an integrator of acceleration
    integrator_x = nengo.Ensemble(n_neurons=100, dimensions=2)
    # representing the current location in x,y,z coordinates
    current_location_x = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Create a piecewise step function for input
    stim_x  = nengo.Node(Piecewise({0: scale_down(0,min_r,max_r), 2: scale_down(0.5,min_r,max_r), 4: scale_down(1,min_r,max_r), 6: scale_down(-0.5,min_r,max_r), 8: scale_down(-1,min_r,max_r)}))
    
    # Connect the population to itself using a long time constant (tau) 
    # for stability
    tau = 1
    nengo.Connection(integrator_x, integrator_x, synapse=tau)

    # Connect the input using the same time constant as on the recurrent
    # connection to make it more ideal
    nengo.Connection(stim_x, integrator_x[0], transform=tau, synapse=tau)
    
    # x[1] is the current state of the integrator,
    # x[0] is the new input acceleration
    def calculate_xyz(x):
        return scale_up_deg((x[1])) + ((x[0]))
        # return ((x[1])) + ((x[0]))
        
    # Connect the integrator to the output ensamble to point the current 
    # location based on the integration
    nengo.Connection(integrator_x, current_location_x, transform=tau, synapse=tau, function=calculate_xyz)
    
    ################################
    ######         y          ######
    ################################
    # representing an integrator of acceleration
    integrator_y = nengo.Ensemble(n_neurons=100, dimensions=2)
    # representing the current location in x,y,z coordinates
    current_location_y = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Create a piecewise step function for input
    stim_y  = nengo.Node(Piecewise({0: scale_down(0,min_r,max_r), 1: scale_down(1,min_r,max_r), 3: scale_down(0.5,min_r,max_r), 5: scale_down(-0.5,min_r,max_r), 7: scale_down(-1,min_r,max_r)}))
    
    # Connect the population to itself using a long time constant (tau) 
    # for stability
    nengo.Connection(integrator_y, integrator_y, synapse=tau)

    # Connect the input using the same time constant as on the recurrent
    # connection to make it more ideal
    nengo.Connection(stim_y, integrator_y[0], transform=tau, synapse=tau)
        
    # Connect the integrator to the output ensamble to point the current 
    # location based on the integration
    nengo.Connection(integrator_y, current_location_y, transform=tau, synapse=tau, function=calculate_xyz)
    
    ################################
    ######         z          ######
    ################################
    # representing an integrator of acceleration
    integrator_z = nengo.Ensemble(n_neurons=100, dimensions=2)
    # representing the current location in x,y,z coordinates
    current_location_z = nengo.Ensemble(n_neurons=100, dimensions=1)

    # Create a piecewise step function for input
    stim_z  = nengo.Node(Piecewise({0: scale_down(0,min_r,max_r), 2: scale_down(0.5,min_r,max_r), 4: scale_down(1,min_r,max_r), 6: scale_down(-0.5,min_r,max_r), 8: scale_down(-1,min_r,max_r)}))
    
    # Connect the population to itself using a long time constant (tau) 
    # for stability
    nengo.Connection(integrator_z, integrator_z, synapse=tau)

    # Connect the input using the same time constant as on the recurrent
    # connection to make it more ideal
    nengo.Connection(stim_z, integrator_z[0], transform=tau, synapse=tau)
        
    # Connect the integrator to the output ensamble to point the current 
    # location based on the integration
    nengo.Connection(integrator_z, current_location_z, transform=tau, synapse=tau, function=calculate_xyz)
    
    # --------------------------------------------------------------------------
    # connect all the current coordinates to coherent ansamble to simplify view
    current_location = nengo.Ensemble(n_neurons=100, dimensions=3)
    nengo.Connection(current_location_x, current_location[0])
    nengo.Connection(current_location_y, current_location[1])
    nengo.Connection(current_location_z, current_location[2])

    
    