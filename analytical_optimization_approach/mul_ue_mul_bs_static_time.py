import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import pickle

### SCENARIO:
## We have multiple Base Stations (BSs) and each with multiple Resource Blocks (RBs).
## We also have multiple User Equipment (UEs) which observe certain theoretically
## maximum SINR when connected to a particular BS.
## OBJECTIVE: Maximize the total bitrate of the system summed across all UEs
## where BitRate is log(1+SINR).
## CONSTRAINTS: Each UE can connect only to a single BS (no restriction on the number 
## of RBs allocated). Each RB can be allocated to a single UE.
## NOTE: We randomly sampled the SIR from a probability distribution for our simulation.


"""Build model function"""

def buildModel(bs=2, ue=2, max_rbs=5, min_rbs=2, seed=0):
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

    # Random SINR being sampled
    sir = np.random.random((bs,ue))

    # Decision variables for Link and Allocation Matrices
    # The Link Matrix decided to which particular BS a UE is connected (if connected)
    # The Allocation Matrix decides the resources allocated to a particular UE from a
    # particular BS (NOTE: It is entirely possible to allocate RBs to a UE but not be linked
    # in which case, effectively, a UE receives no bitrate from a BS).
    """Dimensions make the positional significance of a variable self-explanatory"""
    lv = m.Array(m.Var,(bs, ue),lb=0,ub=1, integer=True) # Link Matrix
    
    av = {} # Allocation Matrix dictionary for each BS
    for b in range(bs):
        av[b] = m.Array(m.Var,(bs_rbs[b], ue),lb=0,ub=1, integer=True)


    # CONSTRAINTS:
    # Ensure each UE connected to a single BS:
    for u in range(ue):
        lv_sum = m.Var(value=0, name='Link Sum ' + str(u))
        m.Equation(lv_sum == sum(lv[0:bs, u]))
        m.Equation(lv_sum <= 1)

    # Ensure each RB is connected to a single UE:
    for b in range(bs):
        for r in range(bs_rbs[b]):
            av_sum = m.Var(value=0, name='Alloc Sum ' + str((b,r)))
            m.Equation(av_sum == sum(av[b][r, 0:ue]))
            m.Equation(av_sum <= 1)



    # OBJECTIVE:
    # The entries of the matrix denote the utility/objective attained
    # by a particular (bs,ue) pair for a given allocation
    obj_u = m.Array(m.Var,(bs, ue))
    for u in range(ue):
        for b in range(0,bs):
            obj_u[b,u] = lv[b][u]*sum(av[b][:,u])*sir[b][u]

    for u in range(ue):
        m.Maximize(m.log(1+sum(obj_u[:,u])))


    #Set global options
    m.options.IMODE = 3 #steady state optimization

    print(m.path)
    #Solve simulation
    m.solve()

    for i in range(ue):
        print(np.log(1+sum(obj_u[:,i]).value))
    print('Link Matrix')
    print(lv)
    print('Allocation Matrices')
    print(av)
    print('SIR Matrix')
    print(sir)



# Run different (bs,ue) scenarios to understand the empirical time complexity
max_bs = 15
min_bs = 5
bs_step = 5

max_ue = 15
min_ue = 5
ue_step = 5


data = {}
for b in range(min_bs, max_bs, bs_step):
    for u in range(min_ue, max_ue, ue_step):
        start = timer()
        buildModel(bs=b, ue=u, max_rbs=5, min_rbs=2, seed=b*u)
        end = timer()
        print("(BS,UE): " + str((b,u)))
        print("Time: " + str(timedelta(seconds=end-start)))
        data[(b,u)] = timedelta(seconds=end-start)


with open('data.p', 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('data.p', 'rb') as fp:
    data = pickle.load(fp)
