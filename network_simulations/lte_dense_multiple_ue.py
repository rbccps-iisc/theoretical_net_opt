from operator import itemgetter
from collections import Counter
import numpy as np
import scipy.stats as ss
import random
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box



"""Generates a random number of points for a poisson process with expectation lam*totalArea"""
### type(lam) == float
### type(totalArea) == float

def generatePoints(lam, totalArea):
    return ss.poisson(totalArea*lam).rvs()
    

"""Generates the PPP populating the world. Insert your own model of PPP if required"""
### type(dimensions) == np.array, list, tuple

def generatePPP(num_points=10, dimensions=None, layout='rect'):
    if layout == 'rect':
        assert len(dimensions) == 2
        x = dimensions[0]*ss.uniform.rvs(0,1,((num_points,1)))
        y = dimensions[1]*ss.uniform.rvs(0,1,((num_points,1)))
        points = np.hstack((x,y))

    elif layout == 'circle':
        assert len(dimensions) == 1
        r = dimensions[0]*ss.uniform.rvs(0,1,((num_points,1)))
        theta = 2*np.pi*ss.uniform.rvs(0,1,((num_points,1)))
        points = np.hstack((r*np.cos(theta),r*np.sin(theta)))
    
    return points

##### END OF FUNCTIONS FOR GENERATING BS POSITIONS #####



    

"""Custom functions for random vehicle movement"""

# Random 2D Walk Model
def radomWayPointMovement(maxSpeed, pos, stepSize, dimensions, movement):
    speed = 2*maxSpeed*np.random.uniform(low=0, high=1)
    angle = 2*np.pi*np.random.uniform(low=0, high=1)
    pos = pos + stepSize*np.array([speed*np.cos(angle), speed*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0]*1000)
    pos[1] = np.clip(pos[1:], 0, dimensions[1]*1000)
    return {'pos':pos,'speed':speed,'angle':angle}


# Quasi Random 2D walk model
# UNNECESSARY KEYWORDS
def quasiRandomWayPointMovement(maxSpeed, maxSpeedChange, speed, maxAngleChange, angle, pos,
                                stepSize, dimensions, movement, trajectory, bsConnection, rbs):
    speed = speed + np.random.uniform(low=-maxSpeedChange, high=maxSpeedChange)
    angle = angle + np.random.uniform(low=-maxAngleChange, high=maxAngleChange)
    pos = pos + stepSize*np.array([speed*np.cos(angle), speed*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0])
    pos[1] = np.clip(pos[1:], 0, dimensions[1])
    return {'pos':pos,'speed':speed,'angle':angle}

# Custom 2D movement
def customMovement(trajectory, steps):
    pass

##### END OF FUNCTIONS FOR GENERATING UE MOBILITY MODELS #####





"""Custom functions for channel model"""
### type(power) == float
### type(positions) == np.array
### type(bsHeight) == float
### type(channelLossModel) == string


def standardChannelLossModel(power, distance, channelLossModel,
                             noiseParams=(0,0), fadingParams=(0,),
                             noiseType='gaussian', fadingType='exponential', samples=5):

    ## Generate fading
    if fadingType == 'exponential':
        fading = np.mean(np.random.exponential(scale=fadingParams[0], size=(len(distance), samples)), axis=1)

    ## Generate noise
    if noiseType == 'gaussian':
        noise = np.mean(np.random.normal(loc=noiseParams[0], scale=noiseParams[1], size=(len(distance), samples)), axis=1)

    ## Based on the channle model choose the attenuation w.r.t actual distance
    if channelLossModel == 'friisPropLossModel':
        return power/distance**2, noise, power*fading/distance**2 + noise

    elif channelLossModel == 'squaredFriisPropLossModel':
        return power/distance**4, noise, power*fading/distance**4 + noise
    

""" More complicated channel model maybe created here"""
       

##### END OF FUNCTIONS FOR CHANNEL MODELS #####






"""Class which takes the different attributes of a BS and initializes a BS which will calculate different
   quantities like bitrate and signal strength based on these parameters"""
class BaseStation(object):
    def __init__(self, **params):
        self.bs_config = params

    ## Updates the UEs connected to a BS. 'connections' contain a list of UEs along with their index, position,
    ## and the number of RBs they hold from the particular BS.
    def updateConnections(self, connections):
        self.connections = connections


    ## Function which takes in the channel or the environment associated with the BS and calculates quantities
    ## like signal strength and bitrate of the user.
    def generateObservations(self, ueList, environment):

        ## Iterate over all UEs and get their position
        uePositions = np.array([])
        for ue in ueList:
            uePositions = np.append(uePositions, ue.getUeStatus()['pos'])
        uePositions = np.reshape(uePositions, (-1,2))

        ## Calculate the distance from the BS
        distance = uePositions - np.array(self.bs_config['pos'])
        distance = (np.sum(distance**2, axis=1) + self.bs_config['height']**2)**0.5

        #print(self.bs_config['channelLossModel'])

        ### BAD WAY !!! channel loss doesn't modularize/generalize
        powerIdeal, noise, powerRecvd = standardChannelLossModel(self.bs_config['power'], distance, self.bs_config['channelLossModel'],
                                           noiseParams=environment['noiseParams'], fadingParams=environment['fadingParams'], 
                                           noiseType=environment['noiseType'], fadingType=environment['fadingType'], samples=3)
        return powerIdeal, noise, powerRecvd
    

    ### GETTER FUNCTIONS ###
    ## Getter function for parameters of a Base Station (fixed)
    def getParameters(self):
        return dict(self.bs_config)

    ## Getter function for current connections of a Base Station
    def getConnections(self):
        return np.copy(self.connections)


"""Returns a dictionary of BS objects with different parameters"""
#### PARAMETERS ####

### type(positions) == 'random' or np.array/tuple/list  (2D data structure containing the positions)
### type(randomParameters) == dictionary (parameters for a PPP and clustered PPP model if earlier paramater is random)
### type(bsHeight) == np.array/tuple/list  (1D data structure containing the heights of BS)
### type(bsPower) == np.array/tuple/list  (1D data structure containing the powers of BS)
### type(resourceBlocks) == np.array/tuple/list  (1D data structure containing the heights of BS)
### type(channelLossModel) == *function (address of the function used to calculate signal attenuation or basically channel model)
### type(resourceBlocks) == np.array/tuple/list  (1D data structure containing the heights of BS)


def createBSList(positions,bsHeight, bsPower, resourceBlocks, 
                 multiplexing, channelLossModel):
    
    ## Check consistency in the length of the data structures
    assert len(positions) == len(bsHeight) == len(bsPower) == len(resourceBlocks) == len(multiplexing) == len(channelLossModel),\
    'Inconsistency in length of Data Structures containing Base Station Parameters!'
    
    bsList = {}
    for i in range(len(positions)):
        params = {'pos':positions[i], 'height':bsHeight[i], 'power':bsPower[i], 'rbs':resourceBlocks[i],
                  'channelModel':channelLossModel[i], 'multiplexing':multiplexing[i]}
        bsList[i] = BaseStation(**params)
    return bsList


##### END OF FUNCTIONS FOR BASE STATION CREATION #####







"""Class which takes the different attributes of a UE and initializes a UE with arguments like unique id,
   position, movement attributes (velocity, movement pattern function), signal strength, BS to which it is connected"""
class UserEquipment(object):
    def __init__(self, **params):
        self.ue_config = params

    ## Updates the UEs connection to a BS.
    def updateConnections(self, bsConnection, rbs):
        self.ue_config['bsConnection'] = bsConnection
        self.ue_config['rbs'] = rbs


    ## Function which finds out the next position when the UE moves.
    def generateMovement(self):
        ueMovementStatus = self.ue_config['movement'](**self.ue_config)

        # Update the position, velocity and angle of UE to be used in the next time step
        self.ue_config['pos'] = ueMovementStatus['pos']
        self.ue_config['speed'] = ueMovementStatus['speed']
        self.ue_config['angle'] = ueMovementStatus['angle']
        

    ### GETTER FUNCTIONS ###
    def getUeStatus(self):
        return dict(self.ue_config)




"""Returns a dictionary of BS objects with different parameters"""
#### PARAMETERS ####

### type(positions) == 'random' or np.array/tuple/list  (2D data structure containing the positions)
### type(randomParameters) == dictionary (parameters for a PPP and clustered PPP model if earlier paramater is random)
### type(bsHeight) == np.array/tuple/list  (1D data structure containing the heights of BS)
### type(bsPower) == np.array/tuple/list  (1D data structure containing the powers of BS)
### type(resourceBlocks) == np.array/tuple/list  (1D data structure containing the heights of BS)
### type(channelLossModel) == *function (address of the function used to calculate signal attenuation or basically channel model)
### type(resourceBlocks) == np.array/tuple/list  (1D data structure containing the heights of BS)


def createUEList(positions, velocity, angle, bsConnections, rbs):
    
    ## Check consistency in the length of the data structures
    assert len(positions) == len(velocity) == len(angle) == len(bsConnections) == len(rbs),\
    'Inconsistency in length of Data Structures containing User Equipment Parameters!'
    
    ueList = {}
    for i in range(len(positions)):
        params = {'pos':positions[i], 'vel':bsHeight[i], 'angle':bsPower[i], 'bsConnection':bsConnection[i], 'rbs':rbs[i]}
        ueList[i] = UserEquipment(**params)
    return ueList








"""Class which creates the gym environment for LTE"""
#### PARAMETERS ####

### type(bsList) == dictionary of objects of type BaseStation
### type(ueList) == dictionary of objects of type UserEquipment
class LteEnv(Env):
    def __init__(self, bsList, ueList, steps=100000, history=100):

        self.bsList = bsList
        self.ueList = ueList
        self.steps = steps
        self.history = history ## Maintains history of observations uptill self.history steps

        # Set the counter of number of steps to decide when to terminate
        self.steps_done = 0
 
        obs_length = len(bsList)*len(ueList)
        action_length = len(bsList)*len(ueList)
        
        # Define a 2-D observation space
        self.observation_space = Box(low = np.zeros((len(bsList), len(ueList))), high = np.inf*np.ones((len(bsList), len(ueList))),
                                     shape=(len(bsList), len(ueList)), dtype = np.float64)
            
        # Define the action space ranging from 0 to the observation length
        self.action_space = Discrete(len(bsList)*len(ueList))

        # Create an initial observation and return it to 'reset()'. Will maintain
        # 100 past observations
        self.observations = {'powerIdeal':[], 'noise':[],'powerRecvd':[]}

        """self.obs_config = {'pathloss':params['pathloss'], 'fading':params['fading'], 'bs_positions':self.bs_positions,
                           'bs_height':params['bs_height'],'tier1bsindex':len(self.sparse_bs_points), 'tier2bsindex':len(self.city_bs_points),
                           'tier1bspower':params['sparse_bs_power'],'tier2bspower':params['city_bs_power'],
                           'ue':params['ue']}"""
      
    

    def reset(self, environment, stepSize=0.01, seed=5):      
        # Set the seed for the Gym Simulation for sampling noise and fading random variables
        np.random.seed(seed)
        # Set the time granularity for samplimg
        self.stepSize = stepSize
        print("Current environment seed: " + str(seed))
        self.seed = seed

        # The environment variables like the noise mean and variance, fading mean.
        self.environment = environment

        powerIdealList = []
        noiseList = []
        powerRecvdList = []
    
        for bs in self.bsList:
            powerIdeal, noise, powerRecvd = bs.generateObservations(self.ueList, self.environment)
        
            # Append the observations from Base Station sequentially
            powerIdealList.append(powerIdeal)
            noiseList.append(noise)
            powerRecvdList.append(powerRecvd)

        # Append the first observation from all Base Stations (maintains a local history)
        self.observations['powerIdeal'].append(np.array(powerIdealList))
        self.observations['noise'].append(np.array(noiseList))
        self.observations['powerRecvd'].append(np.array(powerRecvdList))
        #print(self.observations['powerRecvd'])
        return np.array(self.observations['powerRecvd'][-1])



    def step(self, action=None):
        self.steps_done += 1

        ## The simulation has reached the maximum number of steps
        if self.steps_done >= self.steps:
            return np.zeros((len(bsList), len(ueList))), 0, True, {}
        
        reward = 0
        # Calculate rewards and take a UE step
        for ueIndex in range(len(self.ueList)):
            ue = ueList[ueIndex]
            bsIndex = ue.getUeStatus()['bsConnection']
            rbs = ue.getUeStatus()['rbs']
            N = self.observations['noise'][-1][bsIndex,ueIndex]
            P = self.observations['powerRecvd'][-1][bsIndex,ueIndex] - N  
            I = np.sum((self.observations['powerRecvd'][-1] - self.observations['noise'][-1])[:, ueIndex])
            sinr = P/(N+I)
            reward += self.stepSize*rbs*np.log(1+sinr)/np.log(2)
            
            
            #print(ue.ue_config['pos'], ue.ue_config['speed'], ue.ue_config['angle'])
            ue.generateMovement()
            #print(ue.ue_config['pos'], ue.ue_config['speed'], ue.ue_config['angle'])
            #print()

        
        if action is not None and np.argmax(action) != 0:
            ## An array of positive integers where >0 indicates a connection to a BS and the
            ## number indicates the number of resource block allocated. We first reshape the
            ## 1D array of length=numBs*numUE to a 2D array of dimensions=(numBS,numUE)
            action = np.reshape(action, (len(self.bsList), len(self.ueList)))
            
            ### Update the connections based on the action given
            connections = np.array(np.nonzero(action)).T
            for c in connections:
                rbs = action[c[0]][c[1]]
                self.ueList[c[1]].updateConnections(bsConnection=c[0], rbs=rbs)              
        else:
            print('Invalid Action !!! Continuing with earlier connection.')
            


        powerIdealList = []
        noiseList = []
        powerRecvdList = []
    
        for bs in self.bsList:
            powerIdeal, noise, powerRecvd = bs.generateObservations(ueList, self.environment)
        
            # Append the observations from Base Station sequentially
            powerIdealList.append(powerIdeal)
            noiseList.append(noise)
            powerRecvdList.append(powerRecvd)

        # Append the first observation from all Base Stations (maintains a local history)
        self.observations['powerIdeal'].append(np.array(powerIdealList))
        self.observations['noise'].append(np.array(noiseList))
        self.observations['powerRecvd'].append(np.array(powerRecvdList))

        ## Truncate the history as per given length
        self.observations['powerIdeal'] = self.observations['powerIdeal'][-self.history:]
        self.observations['noise'] = self.observations['noise'][-self.history:]
        self.observations['powerRecvd'] = self.observations['powerRecvd'][-self.history:]
        
        """
        ## Need to think about the reward function
        reward = self.getThroughput()

        # If Handoff occurs then subtract the cost of handoff from the reward
        if len(self.observations['connection']) >= 2:
            self.handovers += int(bool(self.observations['connection'][-1] - self.observations['connection'][-2]))
            reward = reward - self.config['handover_cost']*int(bool(self.observations['connection'][-1] - self.observations['connection'][-2]))"""
        
        return np.copy(self.observations['powerRecvd'][-1]), reward, False, {}




    ## Functions to access information for debugging
    def _getObservation(self):
        return np.copy(self.observations)








#### CREATE A LIST OF BASE STATIONS:
### Consider a 3 tier network of Base Stations


### Set numpy seed
np.random.seed(19)


### Set the granularity of time (in seconds)
stepSize = 0.01

### Total dimension of the map (Dimensions are in metres)

totalDimensions = [100,100]
totalArea = totalDimensions[0]*totalDimensions[1]

### Total dimension of the city (Dimensions are in kilometres)
### Can also be made random as per a certain mean

cityDimensions = [20,20]
cityArea = cityDimensions[0]*cityDimensions[1]


### Parameters dictating density of Base Stations per square kilometre
### type(lambda_sparse) == float (indicates the density of some central tier 1 BS in general most powerful and tall)
### type(lambda_city) == float (indicates the density of centres of cities around which BSs are populated)
### type(lambda_2) == float (indicates the density of tier 2 BS aroud centres of cities as populated randomly by lambda_city)
### type(lambda_3) == float (indicates the density of tier 3 BS aroud centres of cities as populated randomly by lambda_city)
### Further tiers of BS and there random placement may be done!

lambda_sparse = 0.01
lambda_city = 0.002
lambda_2 = 0.01
lambda_3 = 0.01


### Parameters describing the power, height, resourceBlocks, multiplexing and channel loss model for a particular tier of BS
### (maybe extended to varitions within the same tier of BS)

bs_height_sparse = 100   # (in metres)
bs_height_tier1 = 50     # (in metres)
bs_height_tier2 = 25     # (in metres)

bs_power_sparse = 20
bs_power_tier1 = 10
bs_power_tier2 = 5

bs_rbs_sparse = 20
bs_rbs_tier1 = 10
bs_rbs_tier2 = 5

bs_multiplexing_sparse = 'ofdm'
bs_multiplexing_tier1 = 'ofdm'
bs_multiplexing_tier2 = 'ofdm'


bs_channelLossModel_sparse = 'squaredFriisPropLossModel'
bs_channelLossModel_tier1 = 'squaredFriisPropLossModel'
bs_channelLossModel_tier2 = 'squaredFriisPropLossModel'



### Dictionary to store Base Station information
bs_dict = {}

s=0

num_sparse_bs = generatePoints(lambda_sparse, totalArea)
num_city = generatePoints(lambda_city, totalArea)

s += num_sparse_bs


bs_sparse_positions = generatePPP(num_points=num_sparse_bs, dimensions=totalDimensions, layout='rect')
bs_params_sparse = {'power':bs_power_sparse, 'height':bs_height_sparse, 'rbs':bs_rbs_sparse, 'multiplexing':bs_multiplexing_sparse,
                    'channelLossModel':bs_channelLossModel_sparse, 'type':'sparse'}

### Create a list of BaseStation objects
bsList = [BaseStation(**{**bs_params_sparse,'pos':key}) for key in bs_sparse_positions]


### Generate the centres of the cities
city_pos = generatePPP(num_points=num_city, dimensions=totalDimensions, layout='rect')


bs_params_tier1 = {'power':bs_power_tier1, 'height':bs_height_tier1, 'rbs':bs_rbs_tier1, 'multiplexing':bs_multiplexing_tier1,
                    'channelLossModel':bs_channelLossModel_tier1, 'type':'tier1'}
bs_params_tier2 = {'power':bs_power_tier2, 'height':bs_height_tier2, 'rbs':bs_rbs_tier2, 'multiplexing':bs_multiplexing_tier2,
                    'channelLossModel':bs_channelLossModel_tier2, 'type':'tier2'}

for i in range(num_city):
    num_2 = generatePoints(lambda_2, cityArea)
    s+=num_2

    ## Check whether proper eliminations is happening
    #print(num_2)
    city_tier1_bs_positions = generatePPP(num_points=num_2, dimensions=cityDimensions, layout='rect')
    city_tier1_bs_positions = city_pos[i] + city_tier1_bs_positions

    ### Remove points outside the given dimensions
    #print((city_tier1_bs_positions <= np.array(totalDimensions)).all(axis=1))
    city_tier1_bs_positions = city_tier1_bs_positions[(city_tier1_bs_positions <= np.array(totalDimensions)).all(axis=1)]
    city_tier1_bs_positions = city_tier1_bs_positions[(city_tier1_bs_positions >= np.array([0,0])).all(axis=1)]
    bsList += [BaseStation(**{**bs_params_tier1,'pos':key}) for key in city_tier1_bs_positions]


    
for i in range(num_city):
    num_3 = generatePoints(lambda_3, cityArea)
    s+=num_3
    city_tier2_bs_positions = generatePPP(num_points=num_3, dimensions=cityDimensions, layout='rect')
    city_tier2_bs_positions = city_pos[i] + city_tier2_bs_positions
    
    ### Remove points outside the given dimensions
    city_tier2_bs_positions = city_tier2_bs_positions[(city_tier2_bs_positions <= np.array(totalDimensions)).all(axis=1)]
    city_tier2_bs_positions = city_tier2_bs_positions[(city_tier2_bs_positions >= np.array([0,0])).all(axis=1)]
    bsList += [BaseStation(**{**bs_params_tier2,'pos':key}) for key in city_tier2_bs_positions]


#### Plot the Base Stations:

sparse_bs = []
tier1_bs = []
tier2_bs = []

for bs in bsList:
    params = bs.getParameters()
    if params['type'] == 'sparse':
        sparse_bs += [list(params['pos'])]
    elif params['type'] == 'tier1':
        tier1_bs += [list(params['pos'])]
    elif params['type'] == 'tier2':
        tier2_bs += [list(params['pos'])]


### Plot the Base Stations for visualization

"""
plt.scatter(np.array(sparse_bs)[:,0], np.array(sparse_bs)[:,1], color='red', marker='*', s=100, label='sparse')
plt.scatter(np.array(tier1_bs)[:,0], np.array(tier1_bs)[:,1], color='green', marker='*', s=100, label='tier1')
plt.scatter(np.array(tier2_bs)[:,0], np.array(tier2_bs)[:,1], color='blue', marker='*', s=100, label='tier2')

plt.legend(loc="upper left")

plt.xlabel('kilometres')
plt.ylabel('kilometres')
plt.show()"""

### Shave off out of bounds base stations

"""
l = len(bs_pos)
bsHeight = l*[1]
bsPower = l*[100]
resourceBlocks = l*[5] 
multiplexing = l*['ofdm']
channelLossModel = l*['squaredFriisPropLossModel']


### Another way to create the same thing using helper function:
### bsList = createBSList(bs_pos, bsHeight, bsPower, resourceBlocks, 
###                      multiplexing, channelLossModel)


"""

lambda_ue = 0.02
num_ue = generatePoints(lambda_ue, totalArea)
ue_positions = generatePPP(num_points=num_ue, dimensions=totalDimensions, layout='rect')

maxSpeed = 25
maxSpeedChange = 5
maxAngleChange = 0.1

### Create a random list of average velocities
initialSpeed = np.random.uniform(low=-20, high=20, size=num_ue)
### Create a random list of initial angles
initialAngle = np.random.uniform(low=0, high=2*np.pi, size=num_ue)

ue_params = {'maxSpeed':maxSpeed, 'maxSpeedChange':maxSpeedChange, 'maxAngleChange':maxAngleChange,
             'stepSize':stepSize, 'dimensions':totalDimensions,
             'movement':quasiRandomWayPointMovement, 'trajectory':None}

### Create a list of UserEquipment objects
ueList = [UserEquipment(**{**ue_params,'pos':ue_positions[i], 'speed':initialSpeed[i], 'angle':initialAngle[i]}) for i in range(num_ue)]

### Set BS connections randomly with only 1 resource block alloted
for ue in ueList:
    bsConnection = random.randrange(len(bsList))
    ue.updateConnections(bsConnection=bsConnection, rbs=1)

environment = {'noiseParams':(0,0), 'fadingParams':(1,), 'noiseType':'gaussian', 'fadingType':'exponential'}

print(s)
print(num_ue)

env = LteEnv(bsList, ueList)
obs = env.reset(environment)
obs, reward, _, _ = env.step()
print(reward)
obs, reward, _, _ = env.step()
print(reward)



### Connect to nearest Base Stations (depends on the order of UEs and BSs)


#bsList = createBSList(positions, randomnessParameters, bsHeight, bsPower, resourceBlocks, 
#             multiplexing, channelLossModel)





