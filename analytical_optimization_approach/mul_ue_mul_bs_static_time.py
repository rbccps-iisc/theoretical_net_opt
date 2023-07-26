import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import pickle

### SCENARIO:
## We have multiple Base Stations (BSs) and each with multiple Resource Blocks (RBs).
## We also have multiple User Equipments (UEs) which observe certain theoretically
## maximum SINR when connected to a particular BS.
## OBJECTIVE: Maximize the total bitrate of the system summed across all UEs
## where BitRate is log(1+SINR).
## CONSTRAINTS: Each UE can connect only to a single BS (no restriction on the number 
## of RBs allocated). Each RB can be allocated to a dingle UE.
## NOTE: For our simulation we have randomly sampled the bitrate from  probability distribution.


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
    for i in range(bs):
        bs_rbs[i] = np.random.randint(low=min_rbs,high=max_rbs)

    # Random BitRates being sampled
    br = np.random.random((bs,ue))

    # Decision variables for Link and Allocation Matrices
    # The Link Matrix decided to which particular BS a UE is connected to (if connected)
    # The Allocation Matrix decides the resources allocated to a particular UE from a
    # particular BS (NOTE: It is entirely possible to allocate RBs to a UE but not be linked
    # in which case effectively a UE receives no bitrate from a BS).
    """Dimensions make the positional significance of a variable self explanatory"""
    lv = m.Array(m.Var,(bs, ue),lb=0,ub=1, integer=True) # Link Matrix
    
    av = {} # Allocation Matrix dictionary for each BS
    for i in range(bs):
        av[i] = m.Array(m.Var,(bs_rbs[i], ue),lb=0,ub=1, integer=True)


    # CONSTRAINTS:
    # Ensure each UE connected to a single BS:
    for i in range(ue):
        lv_sum = m.Var(value=0, name='Link Sum ' + str(i))
        m.Equation(lv_sum == sum(lv[0:bs, i]))
        m.Equation(lv_sum <= 1)

    # Ensure each RB is connected to a single UE:
    for i in range(bs):
        for j in range(bs_rbs[i]):
            av_sum = m.Var(value=0, name='Alloc Sum ' + str((i,j)))
            m.Equation(av_sum == sum(av[i][j, 0:ue]))
            m.Equation(av_sum <= 1)



    # OBJECTIVE:
    # The entries of the matrix denote the utility/objective attained
    # by a barticular (bs,ue) pair for a given allocation
    obj_u = m.Array(m.Var,(bs, ue))
    for i in range(ue):
        for j in range(0,bs):
            obj_u[j,i] = lv[j][i]*sum(av[j][:,i])*br[j][i]

    for i in range(ue):
        m.Maximize(m.log(1+sum(obj_u[:,i])))


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
    print('SINR Matrix')
    print(br)



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
