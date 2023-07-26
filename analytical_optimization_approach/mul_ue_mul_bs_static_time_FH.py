import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import pickle

### SCENARIO:
## We have multiple Base Stations (BSs) and each with multiple Resource Blocks (RBs).
## We also have multiple User Equipment (UEs) which observe certain theoretically
## maximum SINR when connected to a particular BS at a particular time.
## OBJECTIVE: Maximize the total bitrate of the system summed across all UEs and summed
## from time '1 --> T' (where BitRate is log(1+SINR)).
## CONSTRAINTS: Each UE can connect only to a single BS at a particular time 't'
## (no restriction on the number of RBs allocated). Each RB can be allocated to a single UE
## at a particular time 't'.
## NOTE: For our simulation, we have randomly sampled the noise from probability distribution
## apriori and make it available to the controller, effectively making it a open-loop optimal control. 


def buildModel(bs=2, ue=2, max_rbs=5, min_rbs=2, bs_height=50,bs_pos=[], bs_power=[]
               ue_pos=[], discountng=1, horizon=20, fading_mean=1,noise_mean=0,
               noise_variance=1, seed=0):

    # Check that the number of lists of positions given match the number of UEs
    assert len(ue_pos) == ue, "Number of UEs doesn't match the length of UE positions list given !!!"
    # Check if the number of UE positions given for each UE, matches the number of time steps
    for u in range(ue):
        assert len(ue_pos[u]) == steps, "Number of steps doesn't match the length of UE positions given for " + str(u) + " !!!"
        
        
    #initialize GEKKO model
    m = GEKKO()
    # APOPT option is selected
    m.options.solver = 1

    # Set Seed for reproducibility
    np.random.seed(seed)

    # For each BS randomly sample the number of RBs (maybe set deterministically)
    bs_rbs = {}
    for b in range(bs):
        bs_rbs[b] = np.random.randint(low=min_rbs,high=max_rbs)

    # Calculate SINR for each UE for each position at each time instant in the horizon (based on d^(-4) law)
    sinr = {}
    ue_pos = np.array(ue_pos)
    for t in range(horizon):
        # (BS,UE) dimensinal matrix to store SINR data for each instant
        sinr_t = np.zeros((bs,ue))
        for b in range(bs):

            # Calculate distance between a particular BS and all the UEs
            ue_pos_tb = ue_pos[:,t,:] - bs_pos[b]
            ue_distance_tb = np.sum(ue_pos_tb*ue_pos_tb, axis=1)**0.5

            # Sample Fading and Noise
            fading_tb = np.random.exponential((bs,ue), scale=fading_mean)
            noise_tb = np.random.normal((bs,ue), loc=noise_mean, scale=noise_variance)
            noise_power_tb = noise*noise
            obs_power_tb = (bs_power[b]*fading_tb)/(ue_distance_tb)**4

            # Calculate SINR (S/(I+N)) as observed by each UE from b'th BS at time 't'
            sinr_t[b,:] = (obs_power_tb)/(np.sum(obs_power_tb) - obs_power_tb + noise_power_tb)
            
        sinr[t] = sinr_t
            

    # Decision variables for Link and Allocation Matrices
    # The Link Matrix decided to which particular BS a UE is connected (if connected)
    # The Allocation Matrix decides the resources allocated to a particular UE from a
    # particular BS (NOTE: It is entirely possible to allocate RBs to a UE but not be linked
    # in which case, effectively a UE receives no bitrate from a BS).# Decision variables for Link and Allocation Matrices
    # The Link Matrix decided to which particular BS a UE is connected (if connected)
    # The Allocation Matrix decides the resources allocated to a particular UE from a
    # particular BS (NOTE: It is entirely possible to allocate RBs to a UE but not be linked
    # in which case, effectively a UE receives no bitrate from a BS).
    """Dimensions make the positional significance of a variable self explanatory"""
    lv = m.Array(m.Var,(horizon, bs, ue),lb=0,ub=1, integer=True) # Link Matrix
    
    av = {} # Allocation Matrix dictionary for each BS
    for b in range(bs):
        av[b] = m.Array(m.Var,(horizon, bs_rbs[b], ue),lb=0,ub=1, integer=True)


    # CONSTRAINTS:
    # Ensure each UE connected to a single BS at particular time 't':
    for t in range(horizon):
        for u in range(ue):
            lv_sum = m.Var(value=0, name='Link Sum (time,u) ' + str((t,u)))
            m.Equation(lv_sum == sum(lv[t, 0:bs, u]))
            m.Equation(lv_sum <= 1)

    # Ensure each RB is connected to a single UE:
    for t in range(horizon):
        for b in range(bs):
            for r in range(bs_rbs[i]):
                av_sum = m.Var(value=0, name='Alloc Sum (time,b,u)' + str((t,b,r)))
                m.Equation(av_sum == sum(av[b][t, r, 0:ue]))
                m.Equation(av_sum <= 1)



    # OBJECTIVE:
    # The entries of the matrix denote the utility/objective attained
    # by a particular (bs,ue) pair for a given allocation at a particular time 't'
    obj_u = m.Array(m.Var,(horizon, bs, ue))
    for t in range(horizon):
        for u in range(ue):
            for b in range(0,bs):
                obj_u[t,b,u] = lv[b][u]*sum(av[b][:,u])*sinr[t][b][u]

    for u in range(ue):
        m.Maximize(m.log(1+sum(obj_u[:,u])))


    #Set global options
    m.options.IMODE = 3 #steady state optimization

    print(m.path)
    #Solve simulation
    m.solve()

 
