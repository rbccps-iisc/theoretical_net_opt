"""CODE TAKEN FROM"""

## https://mengxinji.github.io/Blog/2019-04-08/Actor-Critic/

## We have modified the code from the above source to suit our requirements


import gym, os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


import numpy as np
import random
import numpy_indexed as npi
from lte_dense_single_ue import env as env
from matplotlib import pyplot as plt
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Remove redundant depreceation warnings
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.int` is a deprecated alias')






## Defining the Neural network for the Actor i.e., the NN of the agent which
## outputs action probabilities.
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, self.action_size)

    def forward(self, state):
        state = state.float()
        state = state.to(device)
        
        output = F.normalize(state, p=2.0, dim=0)
        output = F.leaky_relu(self.linear1(output), negative_slope=0.01)
        output = F.leaky_relu(self.linear2(output), negative_slope=0.01)
        output = F.leaky_relu(self.linear3(output), negative_slope=0.01)
        output = self.linear4(output)
        test = F.softmax(output.detach(), dim=-1)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution, test
    

## Defining the Neural network for the Critic i.e., the NN of the agent which
## critics the actor.
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, state):
        state = state.float()
        state = state.to(device)
        output = F.normalize(state, p=2.0, dim=0)
        output = F.leaky_relu(self.linear1(output), negative_slope=0.01)
        output = F.leaky_relu(self.linear2(output), negative_slope=0.01)
        output = F.leaky_relu(self.linear3(output), negative_slope=0.01)
        value = self.linear4(output)
        return value



## Function which modifes the state to select the top 'n' candidates based on a certain
## decision history. This is to ensure the action space remains constant and not too large.

## current_candidates: The current top 'n' BS. Initially all set to -1
## past_state_history: List of signal strengths recorded for 'total_hist_length' of steps.
## averaging: This variable indicates the number of past signals which we average over to
##            obtain the decision variable e.g. if averaging=3, we average over the
##            recent 3 observed signal strengths and return them as a decision variable.
## dec_hist_length: This variable indicates the number of past states one wants to include
##                  as dcision variables. This means we will average over those number of past 
##                  states contiguously.

def state_modifier(current_candidates, past_state_history, total_hist_length, averaging,
                   dec_hist_length):
    state = []
    # Reshape to 2D array where each row is a past observation, last row -> latest observation
    past_state_history = np.array(past_state_history).reshape((total_hist_length,-1))
    
    mean = np.mean(past_state_history, axis=0)

    # Best candidates over the past 100 observations based on the mean
    
    #indices = np.argpartition(mean, -len(current_candidates))[-len(current_candidates):]
    indices = (-mean).argsort()[:len(current_candidates)]
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    
    new_candidates = np.array(list(set(indices) - set(current_candidates)))
    relegated_candidates = np.array(list(set(current_candidates) - set(indices)))

    # 'Set' data structure ensures only one instance of data is present, initially
    # all values of candidate BS are set to -1. 
    if len(relegated_candidates) == 1 and relegated_candidates[0] == -1:
        for i in range(dec_hist_length):
            mean = np.mean(past_state_history[-1-i:-1-i-averaging:-1, new_candidates], axis=0)
            state.append(mean)
        state = np.array(state).flatten()
        return new_candidates, torch.tensor(state)
        
        

    # Replace the relegated candidates with new candidates
    replacement_positions = npi.indices(current_candidates, relegated_candidates)
    current_candidates[replacement_positions] = new_candidates

    for i in range(dec_hist_length):
        mean = np.mean(past_state_history[-1-i:-1-i-averaging:-1,current_candidates], axis=0)

        state.append(mean)
    state = np.array(state).flatten()
    return current_candidates, torch.tensor(state)




TOTAL_HIST_LENGTH = 100  # The n-maximum BS signals among the average of the past 100 samples will be considered for handover
                         # corresponds to 'total_hist_length' parameter in the state_modifier() function
DEC_HIST_LENGTH = 5      # The past number of observations you want as decision variables
                         # corresponds to 'dec_hist_length' parameter in the state_modifier() function
AVERAGING = 2            # Averaging observations over past history.
                         # corresponds to 'averaging' parameter in the state_modifier() function
CAND_BS = 10             # The number of BS candidates one wants to consider



input_length = DEC_HIST_LENGTH*CAND_BS
output_length = CAND_BS


## Computes returns for the actions with TD(0) bootstrapping
## with values being provided by the critic network
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


## Learning rate
lr = 0.001

def trainIters(actor, critic, n_iters, horizon, bs_seed, ue_seed):
    optimizerA = optim.RMSprop(actor.parameters(), lr=lr)
    optimizerC = optim.RMSprop(critic.parameters(), lr=lr)

    """
    handoffs = []       # Maintain the number of handoffs
    actions_taken = []   # Maintain the actions taken for the final iteration
    max_actions_taken = [] # Theoretically best possible action
    every_ep_rewards = [] # Maintain the average rewards 
    max_reward = []  # Maintain the  max rewards
    sig_strength = []
    max_sig_strength = []
    throughput = []
    max_throughput = []"""

    records = {'handoffs':[], 'actions_taken':[], 'max_actions_taken':None, 'every_ep_rewards':[],
               'sig_power':None, 'max_sig_power':None, 'throughput':[], 'max_throughput':None}
    
    for itr in range(n_iters):
        log_probs = []
        values = []
        rewards = []
        masks = []
        ep_rewards = []
        entropy = 0
        state=env.reset(bs_seed=bs_seed, ue_seed=ue_seed)
        past_state_history = []

        # Modify the state 
        len_state = len(state)
        # Maintain past state history which is to be used for modifying the state. Initially we start by replicating
        # the initial state multiple times. Gradually it gets replaced
        past_state_history += TOTAL_HIST_LENGTH*list(state[:len_state//2])

        # Current top candidates which can be selected and their past averaged history 
        current_candidates, next_modified_state = state_modifier(current_candidates=-1+np.zeros(CAND_BS, dtype=np.int64),
                                                past_state_history=past_state_history, total_hist_length=TOTAL_HIST_LENGTH,
                                                averaging=AVERAGING, dec_hist_length=DEC_HIST_LENGTH)
        ##########

        next_modified_state = next_modified_state.to(device)
        

        for i in range(horizon):
            
            dist, test = actor(next_modified_state)
            value = critic(next_modified_state)
            action = dist.sample()
            
            action_actual = np.zeros(len_state)
            action_actual[current_candidates[action]] = 1

            
            next_state, reward, done, _ = env.step(torch.tensor(action_actual).cpu().numpy())
            ep_rewards.append(reward)

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            past_state_history = past_state_history[len_state//2:] + list(next_state[:len_state//2])
            current_candidates, next_modified_state = state_modifier(current_candidates=current_candidates,
                                                past_state_history=past_state_history, total_hist_length=TOTAL_HIST_LENGTH,
                                                averaging=AVERAGING, dec_hist_length=DEC_HIST_LENGTH)

            if itr == n_iters-1:
                records['actions_taken'].append(current_candidates[action])
                records['throughput'].append(reward)
                

            if done:
                #print('Iteration: {}, Score: {}'.format(iter, i))
                episode_durations.append(i + 1)
                plot_durations() 
                break

        print('Iteration: ' + str(itr))
        print('Handovers and Rewards')
        records['handoffs'].append(env._getHandoverDetails())
        print(np.mean(ep_rewards))
        print()

        records['every_ep_rewards'].append(np.mean(ep_rewards))

        if itr == n_iters-1:
            records['max_actions_taken'] = env._getBestAction()
            records['max_throughput'] = env._getBestThroughput()
            records['max_sig_power'] = env._getMaxPower()
            records['sig_power'] = env._getSelectedPower()

            
            
        next_modified_state = torch.tensor(next_modified_state).to(device)
        next_value = critic(next_modified_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() 
        critic_loss = advantage.pow(2).mean() 


        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=50.0, norm_type=2)
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=50.0, norm_type=2)

        print('Critic Loss: '+str(critic_loss)) 
        print('Actor Loss: '+str(actor_loss))


        optimizerA.step()
        optimizerC.step()

    return records


trials = 20
trial_records = {}

for t in range(6,trials):
    print('Trial: ' + str(t))
    trial_records[t] = {'npseed':t*7, 'torchseed':t*17, 'randomseed':t*23, 'records':None}
    torch.manual_seed(t*17)
    random.seed(t*23)
    np.random.seed(t*7)
    actor = Actor(input_length, output_length).to(device)
    critic = Critic(input_length, output_length).to(device)
    records = trainIters(actor, critic, n_iters=500, horizon=10000,bs_seed=589, ue_seed=6)
    trial_records[t]['records'] = records

    
with open('exp3.pickle', 'wb') as f:
    pickle.dump(trial_records, f)



    
