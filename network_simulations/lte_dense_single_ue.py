import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box

# For performance optimization
from timeit import default_timer as timer
from datetime import timedelta



"""Generates the PPP/PCP populating the world. Insert your own model of PPP/PCP if required"""
## Takes in the number of points to be placed in an area
## Returns the position of points
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



        
""" Function which takes the pathloss exponent, fading, BS positions, height, type of BS and the position
    of the user and returns the signal powers received from each BS by the user"""
def generateObservations(**params):
    """pathloss, fading, bs_positions, bs_height, tier1bsindex, tier2bsindex,
                         tier1bspower, tier2bspower, ue_initial_pos, ue_final_pos,
                         ue_velocity, ue_angle):"""
    observations = np.array([])

    ## Power Transmitted by BS
    power_transmit = np.concatenate((params['tier1bspower']*np.ones(params['tier1bsindex']),
                                   params['tier2bspower']*np.ones(params['tier2bsindex'])))
    ## Power Received by BS
    power_recv = power_transmit / (np.sum((params['bs_positions']-params['ue_final_pos'])**2, axis=1)\
                                   + params['bs_height']**2)**(params['pathloss']/2)
    ## Random Gaussian noise with scale and mean from the power received. Better absolute scale and mean?
    noise = generateNoise(mean=np.max(power_recv/100), n=params['tier1bsindex']+params['tier2bsindex'], samples=3)
    ## Exponential Fading
    fading_obs = generateFading(fading_type=params['fading'][0], params=params['fading'][1:],
                                n=params['tier1bsindex']+params['tier2bsindex'], samples=3)
    
        
    return power_recv, noise, fading_obs



 
"""Insert custom fading modules to suit needs"""
def generateFading(fading_type, params, n, samples):
    if fading_type == 'exponential':
        assert len(params) >= 1
        #return np.sum(np.random.exponential(scale=1/params[0], size=(samples,n)), axis=0)
        return np.mean(np.ones((samples,n)), axis=0)


"""Generate Gaussian noise around a certain mean, the choices are arbitrary"""

def generateNoise(mean, n, samples):
    #return np.sum(np.random.normal(loc=mean, scale=mean, size=(samples,n)), axis=0)
    return np.mean(np.random.normal(loc=0, scale=0, size=(samples,n)), axis=0)
    
    

"""Custom functions for random vehicle movement"""

## Random 2D Walk Model
def radomWayPointMovement(vel_high, vel_low, pos, step, dimensions):
    vel = (vel_high-vel_low)*np.random.uniform(low=0, high=1)
    angle = 2*np.pi*np.random.uniform(low=0, high=1)
    pos = pos + step*np.array([vel*np.cos(angle), vel*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0]*1000)
    pos[1] = np.clip(pos[1:], 0, dimensions[1]*1000)
    return pos


## Quasi Random 2D walk model
## The model randomly changes the previous velocity and direction within a certain range of values
def quasiRandomWayPointMovement(vel_change, vel, angle_change, angle, pos, step, dimensions):
    vel = vel + np.random.uniform(low=-vel_change, high=vel_change)
    angle = angle + np.random.uniform(low=-angle_change, high=angle_change)
    pos = pos + step*np.array([vel*np.cos(angle), vel*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0]*1000)
    pos[1] = np.clip(pos[1:], 0, dimensions[1]*1000)
    return pos,vel,angle




    

"""Class which creates the gym environment for LTE"""
class LteEnv(Env):
    def __init__(self, **params):

        # Store variable values for reset to create the same environment (possibly with a different seed)
        assert len(params['dimensions']) == 2
        # Create dictionaries to maintain initial configuration parameters for the environment and the enviroment instantiated
        self.config = params


        # Set the seed for generating the Base Stations
        np.random.seed(params['bs_seed'])
        print('BS Seed: ' + str(params['bs_seed']))
        print('UE Seed: ' + str(params['ue_seed']))

        # Set the counter of number of steps to decide when to terminate
        self.steps_done = 0
        
        self.ue_trajectory = params['ue_trajectory']
        self.ue_index = 0

        # Maintain the number of handovers
        self.handovers = 0

        # Lists to store traces/data for plotting and results
        self.ue_trace = []
        self.ue_trace.append(params['ue_initial_pos'])
        self.power_theoretical_trace = []
        self.max_power = []
        self.max_throughput = []
        self.best_action = []


        ## Pre-Generated Scenarios
        if params['scenario'] == 'custom':
            obs_length = len(params['bs_positions'])
            
        ## Generate the positions as numpy arrays
        if params['scenario'] == 'random':
            # Dictionary to store base station details for plotting
            self.bs_dict = {}
            
            totalArea = params['dimensions'][0]*params['dimensions'][1]
            num_ppp1_sparse = ss.poisson(totalArea*params['lambda_sparse']).rvs()
            num_ppp2_centred = ss.poisson(totalArea*params['lambda_centres']).rvs()

            ## Generate the coordinates for the sparse base stations (converted to metres)
            self.sparse_bs_points = 1000*generatePPP(num_points=num_ppp1_sparse, dimensions=params['dimensions'], layout='rect')
            self.bs_dict['sparse_bs'] = self.sparse_bs_points

            ## Generate the coordinates for the city base stations (converted to metres)
            ## Generate city centres around which a PCP will be generated
            centred_points = generatePPP(num_points=num_ppp2_centred, dimensions=params['dimensions'], layout='rect')
            city_bs_points=np.array([])
            ## Mean dimensions of the city 
            for city in range(num_ppp2_centred):
                if params['city_scale_distribution'] == 'deterministic':
                    city_radius = params['city_scale']
                elif params['city_scale_distribution'] == 'normal':
                    city_radius = np.random.normal(params['city_scale'])
                    
                totalArea = np.pi*city_radius**2
                num_bs = ss.poisson(totalArea*params['lambda_dense']).rvs()
                city_points = 1000*(centred_points[city] + \
                              generatePPP(num_points=num_bs, dimensions=[city_radius], layout='circle'))

                ## Remove points outside our rectangular window
                city_points = city_points[(city_points <= np.array(params['dimensions'])*1000).all(axis=1)]
                city_points = city_points[(city_points >= np.array([0,0])).all(axis=1)]
                city_bs_points = np.append(city_bs_points, city_points)
                self.bs_dict[city] = city_points
            self.city_bs_points = np.reshape(city_bs_points, (-1,2))
            obs_length = len(self.sparse_bs_points) + len(self.city_bs_points)

            
        
        # Define a 2-D observation space (for the gym environment)
        self.observation_space = Box(low = np.zeros(2*obs_length), 
                                     high = np.maximum(params['sparse_bs_power'], params['city_bs_power'])*np.ones(2*obs_length),
                                     dtype = np.float64)
    
        
        # Define the action space ranging from 0 to the observation length (for the gym environment)
        self.action_space = Discrete(obs_length,)



        # Create an initial observation and return it to 'reset()'. Will maintain
        # 100 past observations
        self.observations = {'power':[], 'connection':[], 'noise':[], 'fading':[]}
        
        self.bs_positions = np.vstack((self.sparse_bs_points, self.city_bs_points))
        self.obs_config = {'pathloss':params['pathloss'], 'fading':params['fading'], 'bs_positions':self.bs_positions,
                           'bs_height':params['bs_height'],'tier1bsindex':len(self.sparse_bs_points), 'tier2bsindex':len(self.city_bs_points),
                           'tier1bspower':params['sparse_bs_power'],'tier2bspower':params['city_bs_power'],
                           'ue_initial_pos':np.copy(params['ue_initial_pos']),'ue_final_pos':np.copy(params['ue_initial_pos']),
                           'ue_velocity':params['ue_initial_vel'], 'ue_angle':params['ue_initial_angle']}

        
        
        

        


    """ Function to simulate/update the movement of the user based on a given determinsitic trajectory or a random
        waypoint model"""
    def ueMovementStep(self):
        self.obs_config['ue_initial_pos'] = np.copy(self.obs_config['ue_final_pos'])
        
        if self.config['ue_movement'] == 'RWP':
            self.obs_config['ue_final_pos'] = radomWayPointMovement(vel_high=self.config['vel_bound'], vel_low=0, pos=self.obs_config['ue_final_pos'],
                                                                    step=self.config['sampling_rate'],dimensions=self.config['dimensions'])

        elif self.config['ue_movement'] == 'QRWP':
            self.obs_config['ue_final_pos'],vel,angle = quasiRandomWayPointMovement(vel_change=self.config['max_vel_change'], vel=self.obs_config['ue_velocity'],
                                                                          angle_change=self.config['max_angle_change'] , angle=self.obs_config['ue_angle'],
                                                                          pos=self.obs_config['ue_final_pos'], step=self.config['sampling_rate'],
                                                                          dimensions=self.config['dimensions'])
            self.obs_config['ue_velocity'] = vel
            self.obs_config['ue_angle'] = angle
            ## Hitting a wall condition, then reverse direction
            if self.obs_config['ue_final_pos'][0]==self.obs_config['ue_initial_pos'][0] or self.obs_config['ue_final_pos'][1]== self.obs_config['ue_initial_pos'][1]:
                self.obs_config['ue_velocity'] = -self.obs_config['ue_velocity']
            
  
        # Need to ensure the end condition is enforced
        elif self.obs_config['ue_movement'] == 'custom':
            self.ue_index += 1
            self.obs_config['ue_final_pos'] = np.array(self.obs_config['ue_trajectory'][self.ue_index], dtype=np.float64)

        # Store the trace/position for ploting
        self.ue_trace.append(self.obs_config['ue_final_pos'])
                       



    """Function to calculate the theoretical throughput through numerical integration.
       The integration takes place through the sampling period and the fading is pertrubed by a small
       amount by following normal distribution during the period. Additionally a Gaussian noise is added."""
    def getThroughput(self):

        ## Unit of rate is time here, instead of Hz
        samples = self.config['sampling_rate']/self.config['integration_sampling_rate']
        
        ## the distance between the two points divided by certain granularity (uncomment when updation of theoretical power is
        ## required even for integration sampling, computational overhead)
        ## diff = (self.obs_config['ue_final_pos'] - self.obs_config['ue_initial_pos'])/sampling_rate
        ## pos = np.copy(self.obs_config['ue_initial_pos'])
        ## obs_copy = self.obs_config.copy()
        ## print(self.obs_config['ue_final_pos'], self.obs_config['ue_initial_pos'])
        
        
        power = np.copy(self.observations['power'][-1])
        noise = np.copy(self.observations['noise'][-1])
        fading = np.copy(self.observations['fading'][-1])
        connection = np.copy(self.observations['connection'][-1])
        throughput = 0
        best_throughput = 0

        self.best_action.append(np.argmax(power))
        
        for s in range(int(samples)):
            if self.config['noiseless']:
                noise_random = np.random.normal(noise, 0)
                fading_random = np.random.normal(fading, 0)
            else:
                noise_random = np.random.normal(noise, np.absolute(noise)/10)
                fading_random = np.random.normal(fading, np.absolute(fading)/100)
                
            faded_signal = fading_random*power
            #FORMULA = log(1+signal_strength/(noise+interfering_signals))
            sinr = (faded_signal[connection]) / (np.sum(faded_signal) + np.sum(noise_random) - faded_signal[connection])
            if 1+sinr > 0:
                throughput += self.config['integration_sampling_rate']*np.log(1+sinr)/np.log(2)

            
            sinr_best = (faded_signal[self.best_action[-1]]) / (np.sum(faded_signal) + np.sum(noise_random) - faded_signal[self.best_action[-1]])
            if 1+sinr_best > 0:
                best_throughput += self.config['integration_sampling_rate']*np.log(1+sinr_best)/np.log(2)

        self.max_throughput.append(best_throughput*self.config['bandwidth'])
            

            # TO be used when the theoretical power needs to be updated as per integration
            # sampling rate (heavy computation)
            # obs_copy['ue_final_pos'] = pos + diff
            # obs = generateObservations(**obs_copy)
        #print(throughput*self.config['bandwidth'])
        return throughput*self.config['bandwidth']
                                   

    # Functions to access information for debugging
    def _getObservation(self):
        return np.copy(self.observations)

    def _getUePosition(self):
        return self.obs_config['ue_final_pos']

    def _getSeed(self):
        return self.config['bs_seed'], self.config['ue_seed']

    def _getBSDetails(self):
        if self.config['scenario'] == 'custom':
            return self.config['bs_positions']
        elif self.config['scenario'] == 'random':
            return self.bs_dict

    def _getHandoverDetails(self):
        print('Handovers: ' + str(self.handovers), 'Handover Cost: ' + str(self.config['handover_cost']))
        print()
        return self.handovers, self.config['handover_cost']
    
    def _getBestThroughput(self):
        return self.max_throughput

    def _getBestActions(self):
        return self.best_action
    
    
    def step(self, action=None):
        self.steps_done += 1
        if self.steps_done >= self.config['steps']:
            return np.zeros(2*len(self.observations['power'][-1])), 0, True, {}

        if action is not None and np.argmax(action) != 0:
            ## An array of all 0's except for the index of BS to which the UE wants to connect   
            self.observations['connection'].append(np.argmax(action))
        else:
            ## By deafult connect to the best signal strength
            idx = np.argmax(self.observations['power'][-1]*self.observations['fading'][-1] \
                            + self.observations['noise'][-1])
            self.observations['connection'].append(idx)
        
        # Take a UE step
        self.ueMovementStep()

        # Update the new SINR observations
        power, noise, fading = generateObservations(**self.obs_config)
        self.observations['power'].append(power)
        self.observations['noise'].append(noise)
        self.observations['fading'].append(fading)



        # Maintain past history only for a fixed number of time steps to maintain low memory usage
        if len(self.observations['power']) > self.config['sinr_history']:
            self.observations['power'] = self.observations['power'][1:]
            self.observations['noise'] = self.observations['noise'][1:]
            self.observations['fading'] = self.observations['fading'][1:]
            self.observations['connection'] = self.observations['connection'][1:]
        
        reward = self.getThroughput()

        # If Handoff occurs then subtract the cost of handoff from the reward
        if len(self.observations['connection']) >= 2:
            self.handovers += int(bool(self.observations['connection'][-1] - self.observations['connection'][-2]))
            reward = reward - self.config['handover_cost']*int(bool(self.observations['connection'][-1] - self.observations['connection'][-2]))
        
        return np.copy(np.append(power*fading+noise, action)), reward, False, {}


        
    ## bs_seed: Changes Base Station layout
    ## ue_seed: Changes Random path followed by UE
    def reset(self, bs_seed=5, ue_seed=6):
        self.config['bs_seed'] = bs_seed
        self.config['ue_seed'] = ue_seed
        self.__init__(**self.config)
        np.random.seed(self.config['ue_seed'])

        
        power, noise, fading = generateObservations(**self.obs_config)
        
        # Append the first observation
        self.observations['power'].append(power)
        self.observations['noise'].append(noise)
        self.observations['fading'].append(fading)

        self.max_power.append(np.argmax(power))
        
        
        return np.copy(np.append(power*fading+noise, np.zeros(len(power))))
        
        
    


config = {'scenario':'random', 'bs_positions':None, 'lambda_sparse':1,
          'lambda_centres':0.2, 'lambda_dense':4, 'dimensions':[10,10], 'integration_sampling_rate':0.01,
          'handover_cost':0, 'bandwidth':10,
          'city_dimensions':'random', 'city_scale':0.25, 'city_scale_distribution':'normal',
          'bs_height':50, 'sparse_bs_power':20, 'city_bs_power':20, 'pathloss':4, 'fading':['exponential',1],
          'noiseless':True,
          'ue_movement':'QRWP', 'vel_bound':150, 'ue_initial_pos':(500,500), 'ue_initial_vel':20,
          'ue_initial_angle':1, 'max_vel_change':2, 'max_angle_change':0.09, 'ue_trajectory':None, 
          'bs_seed':1374, 'ue_seed':137,'sampling_rate':0.1, 'steps':100000, 'generate_plot':True, 'generate_trace':True,
          'sinr_history':100}

env = LteEnv(**config)
bs_seed= 589
ue_seed = 6
env.reset(bs_seed=bs_seed, ue_seed=ue_seed)
bs = env._getBSDetails()

"""
# Implement count for number of handover decisions
# Config file
# Plotting


## Test scenario of multiple UE paths and a single BS environment
ue_seed = [6] #,100,200,300]
bs_seed = [589] #,45,56,7,9,98,34,55,66]
ue_pos = {}
fig = plt.figure()
ax = fig.add_subplot(111)

start = timer()
s=0
for b in bs_seed: 
    for seed in ue_seed:
        env.reset(bs_seed=b, ue_seed=seed)
        ue_pos[seed] = []
        ue_pos[seed].append(env._getUePosition())

        for _ in range(10000):
            env.step()
            ue_pos[seed].append(env._getUePosition())

        ue_pos[seed] = np.array(ue_pos[seed])
        ax.plot(ue_pos[seed][:,0], ue_pos[seed][:,1])



    bs = env._getBSDetails()
    # Plot BASE STATIONS
    # Sparse Base Stations are in Green, * marker
    # City Base Stations are in Red, . marker



    for i in bs.keys():
        s += len(bs[i])
        if isinstance(i, int):
            ax.scatter(bs[i][:,0], bs[i][:,1], color='red', marker='.', s=100)
        else:
            ax.scatter(bs[i][:,0], bs[i][:,1], color='green', marker='*', s=100)

    plt.show()
end = timer()
            
print(end-start)
print(s/len(bs_seed))"""

