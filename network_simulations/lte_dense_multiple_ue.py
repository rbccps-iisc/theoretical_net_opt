import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box


"""Generates the PPP populating the world. Insert your own model of PPP if required"""
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

    

"""Custom functions for random vehicle movement"""

# Random 2D Walk Model
def radomWayPointMovement(vel_high, vel_low, pos, step, dimensions):
    vel = (vel_high-vel_low)*np.random.uniform(low=0, high=1)
    angle = 2*np.pi*np.random.uniform(low=0, high=1)
    pos = pos + step*np.array([vel*np.cos(angle), vel*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0]*1000)
    pos[1] = np.clip(pos[1:], 0, dimensions[1]*1000)
    return pos


# Quasi Random 2D walk model
def quasiRandomWayPointMovement(vel_change, vel, angle_change, angle, pos, step, dimensions):
    vel = vel + np.random.uniform(low=-vel_change, high=vel_change)
    angle = angle + np.random.uniform(low=-angle_change, high=angle_change)
    pos = pos + step*np.array([vel*np.cos(angle), vel*np.sin(angle)])
    pos[0] = np.clip(pos[:1], 0, dimensions[0]*1000)
    pos[1] = np.clip(pos[1:], 0, dimensions[1]*1000)
    return pos,vel,angle



"""Returns a dictionary of BS with different positions and the other attributes being same"""
def createBSDictionary(start, bstype, positions, capacity, power):
    bs = {}
    for i in range(len(positions)):
        bs[positions[i]] = {}
        bs[positions[i]]['type'] = bstype
        bs[positions[i]]['capacity'] = capacity
        bs[positions[i]]['power'] = power
        bs[positions[i]]['connections']
    return bs






"""Class which takes the different attributes of a BS and distance to a UE in account
   and calculates the signal strength received by a UE"""
class BsUeBitRate(object):
    def __init__(self, **params):
        self.br_config = params


    """ Function which takes the pathloss exponent, fading, BS positions, height, type of BS and the position
    of the user and returns the signal powers received from each BS by the user"""
    def generateObservations(self):
    """pathloss, fading, bs_positions, bs_height, tier1bsindex, tier2bsindex,
                         tier1bspower, tier2bspower, ue_initial_pos, ue_final_pos,
                         ue_velocity, ue_angle"""
        observations = np.array([])
    
        power_transmit = np.concatenate((self.br_config['tier1bspower']*np.ones(self.br_config['tier1bsindex']),
                                   self.br_config['tier2bspower']*np.ones(self.br_config['tier2bsindex'])))
        power_recv = power_transmit / (np.sum((self.br_config['bs_positions']-self.br_config['ue_final_pos'])**2, axis=1)\
                                   + self.br_config['bs_height']**2)**(self.br_config['pathloss']/2)
    
        noise = generateNoise(mean=np.max(power_recv/10), n=self.br_config['tier1bsindex']+self.br_config['tier2bsindex'], samples=3)
        fading_obs = generateFading(fading_type=self.br_config['fading'][0], params=self.br_config['fading'][1:],
                                n=self.br_config['tier1bsindex']+self.br_config['tier2bsindex'], samples=3)
    
        obs = noise + fading_obs*actual_power
    
        return power_recv, noise, fading_obs


 
    """Insert custom fading modules to suit needs"""
    def generateFading(fading_type, params, n, samples):
        if fading_type == 'exponential':
            assert len(params) >= 1
            return np.sum(np.random.exponential(scale=1/params[0], size=(samples,n)), axis=0)


    """Generate Gaussian noise around a certain mean, the choices are arbitrary"""
    def generateNoise(mean, n, samples):
        return np.sum(np.random.normal(loc=mean, scale=mean, size=(samples,n)), axis=0)
    


    def standardBR(self):
        












"""Class which creates the gym environment for LTE"""
class LteEnv(Env):
    def __init__(self, **params):

        # Store variable values for reset to create the same environment (possibly with a different seed)
        assert len(params['dimensions']) == 2
        # Create dictionaries to maintain initial configuration parameters for the environment and the enviroment instantiated
        self.config = params


        # Set the seed
        np.random.seed(params['seed'])
        print(params['seed'])

        # Set the counter of number of steps to decide when to terminate
        self.steps_done = 0
        
        self.ue_trajectory = params['ue_trajectory']
        self.ue_index = 0
        

        # Lists to store traces/data for plotting and results
        self.ue_trace = []
        self.ue_trace.append(params['ue_initial_pos'])
        self.power_theoretical_trace = []




        # Index UEs in a dictionary.
        # Each index contains a dictionary with 'type', 'position', 'capacity', 'power',
        # 'ue_connection_list'
        self.ue = {}
        for i in params['ue']:
            if params['ue'][i]['position'] == 'random':
                initial_position = np.array(np.random.uniform(low=0, high=params['dimensions'][0]*1000),
                                            np.random.uniform(low=0, high=params['dimensions'][1]*1000))             
            else:
                if params['ue'][i]['mobility'] != 'RWP' and params['ue'][i]['mobility'] != 'QRWP':
                    assert len(params['ue'][i]['mobility']) >= params['steps'], "Length of trajectory is less than the no. of simulation steps !!!"
                initial_position = params['ue'][i]['initial_pos']
                
            self.ue[i] = {}                        
            self.ue[i]['mobility'] = params['ue'][i]['mobility']
            self.ue[i]['initial_pos'] = initial_position
            self.ue[i]['initial_vel'] = params['ue'][i]['initial_vel']
            self.ue[i]['upper_vel'] = params['ue'][i]['vel_bound']
            self.ue[i]['initial_angle'] = params['ue'][i]['initial_angle']
            self.ue[i]['max_vel_change'] = params['ue'][i]['max_vel_change']
            self.ue[i]['max_angle_change'] = params['ue'][i]['max_angle_change']                   

        



        
        


        # Generate the positions of BSs as numpy arrays and then index it in a dictionary
        # Dictionary to contain all the BSs indexed with numbers
        # Each index contains a dictionary with 'type', 'position', 'capacity', 'power',
        # 'ue_connection_list'
        self.bs = {}
        if params['bs_scenario'] == 'custom':
            obs_length = len(params['bs_positions'])
            self.bs = createBSDictionary(start=0, bstype='sparse', positions=params['bs_positions'],
                                         capacity=params['sparse_capacity'], power=params['sparse_bs_power'])
            
        
        if params['bs_scenario'] == 'random':
            # Dictionary to store base station details for plotting
            self.bs_dict = {}
            
            totalArea = params['dimensions'][0]*params['dimensions'][1]
            num_ppp1_sparse = ss.poisson(totalArea*params['lambda_sparse']).rvs()
            num_ppp2_centred = ss.poisson(totalArea*params['lambda_centres']).rvs()

            # Generate the coordinates for the sparse base stations (converted to metres)
            self.sparse_bs_points = 1000*generatePPP(num_points=num_ppp1_sparse, dimensions=params['dimensions'], layout='rect')

            self.bs = createBSDictionary(bstype='sparse', positions=self.sparse_bs_points,
                                         capacity=params['sparse_capacity'], power=params['sparse_bs_power'])
            
            
            #self.bs_dict['sparse_bs'] = self.sparse_bs_points

            # Generate the coordinates for the city base stations
            centred_points = generatePPP(num_points=num_ppp2_centred, dimensions=params['dimensions'], layout='rect')
            city_bs_points=np.array([])
            for city in range(num_ppp2_centred):
                if params['city_scale_distribution'] == 'deterministic':
                    city_radius = params['city_scale']
                elif params['city_scale_distribution'] == 'normal':
                    city_radius = np.random.normal(params['city_scale'])
                    
                totalArea = np.pi*city_radius**2
                num_bs = ss.poisson(totalArea*params['lambda_dense']).rvs()
                city_points = 1000*(centred_points[city] + \
                              generatePPP(num_points=num_bs, dimensions=[city_radius], layout='circle'))
                city_points = city_points[(city_points <= np.array(params['dimensions'])*1000).all(axis=1)]
                city_points = city_points[(city_points >= np.array([0,0])).all(axis=1)]
                
                self.bs = {**self.bs, **createBSDictionary(bstype='sparse', positions=city_points,
                                                           capacity=params['city_capacity'], power=params['city_bs_power'])}
                
                #city_bs_points = np.append(city_bs_points, city_points)
                #self.bs_dict[city] = city_points
                #self.
            #self.city_bs_points = np.reshape(city_bs_points, (-1,2))
            #obs_length = len(self.bs)
            


        obs_length = len(self.ue)*len(self.bs)
        action_length = len(self.ue)*len(self.bs)
        
        # Define a 2-D observation space
        self.observation_space = Box(low = np.zeros(2*obs_length), 
                                     high = np.maximum(params['sparse_bs_power'], params['city_bs_power'])*np.ones(2*obs_length),
                                     dtype = np.float64)
    
        
        # Define the action space ranging from 0 to the observation length
        self.action_space = Discrete(obs_length,)



        # Create an initial observation and return it to 'reset()'. Will maintain
        # 100 past observations
        self.observations = {'power':[], 'connection':[], 'noise':[], 'fading':[]}
        #print(len(self.city_bs_points))
        
        self.bs_positions = np.vstack((self.sparse_bs_points, self.city_bs_points))
        #print(self.bs_positions[0:5])
        #print(len(self.bs_positions))
        self.obs_config = {'pathloss':params['pathloss'], 'fading':params['fading'], 'bs_positions':self.bs_positions,
                           'bs_height':params['bs_height'],'tier1bsindex':len(self.sparse_bs_points), 'tier2bsindex':len(self.city_bs_points),
                           'tier1bspower':params['sparse_bs_power'],'tier2bspower':params['city_bs_power'],
                           'ue':params['ue']}

        
        


    def reset(self, seed=5):
        self.config['seed'] = seed
        self.__init__(**self.config)
        power, noise, fading = generateObservations(**self.obs_config)
        
        # Append the first observation
        self.observations['power'].append(power)
        # print(power[0:5])
        self.observations['noise'].append(noise)
        self.observations['fading'].append(fading)
        
        return np.copy(np.append(power*fading+noise, np.zeros(len(power))))


