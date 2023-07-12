import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta
import pickle


# Build model function

def buildModel(bs=2, ue=2, max_rbs=5, min_rbs=2):
    #initialize GEKKO model
    m = GEKKO()
    # APOPT option is selected
    m.options.solver = 1


    # Seed
    np.random.seed(0)

    bs_rbs = {}
    for i in range(bs):
        bs_rbs[i] = np.random.randint(low=min_rbs,high=max_rbs)

    # Random BitRates
    br = np.random.random((bs,ue))

    # Link and Allocation Matrices
    lv = m.Array(m.Var,(bs, ue),lb=0,ub=1, integer=True)
    av = {}
    for i in range(bs):
        av[i] = m.Array(m.Var,(bs_rbs[i], ue),lb=0,ub=1, integer=True)




    ## Constraints 
    for i in range(ue):
        lv_sum = m.Var(value=0, name='Link Sum ' + str(i))
        m.Equation(lv_sum == sum(lv[0:bs, i]))
        m.Equation(lv_sum <= 1)


    for i in range(bs):
        for j in range(bs_rbs[i]):
            av_sum = m.Var(value=0, name='Alloc Sum ' + str((i,j)))
            m.Equation(av_sum == sum(av[i][j, 0:ue]))
            m.Equation(av_sum <= 1)



    # Objective function
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
    print('Link')
    print(lv)
    print('Allocation')
    print(av)
    print('Bit Rate')
    print(br)



# Run different (bs,ue) scenarios to understand the empirical time complexity 
data = {}
for b in range(5,21,5):
    for u in range(5,40):
        print(b,u)
        start = timer()
        buildModel(bs=b, ue=u, max_rbs=5, min_rbs=2)
        end = timer()
        print(timedelta(seconds=end-start))
        data[(b,u)] = timedelta(seconds=end-start)

with open('data.p', 'wb') as fp:
    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('data.p', 'rb') as fp:
    data = pickle.load(fp)
