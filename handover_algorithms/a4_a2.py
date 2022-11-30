import numpy as np

# Sources:https://telecompedia.net/a4-inter-frequency-handover-in-lte/
#        :https://www.researchgate.net/publication/338342350_Performance_Evaluation_of_A2-A4-RSRQ_and_A3-RSRP_Handover_Algorithms_in_LTE_Network

# Algorithm Description

# Takes in a 'lower threshold'. If the current connection has a signal strength below
# 'lower threshold' for time given by 'time to trigger' it causes and event A2. If the
# event A2 has occured and after which the connection manages to exceed lower threshold
# for time to trigger then A2 is reverted and we again begin from the start (Called A1 event)
# If A1 has not occured and a signal strength from a different base station (BS) manages to 
# exceed the upper threshold for time to trigger then the handover is done to that BS (Event A4). 
# If thereare multiple BSs which satisfy the critaeria, the one with the highest average signal 
# strength is chosen.



class a4_a2(object):
    def __init__(self, lower_threshold, upper_threshold, time_to_trigger, step):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.history_length = time_to_trigger / steps    # the history of connection strengths that need to be maintained
        self.history = []
        self.a2 = 0                                      # flag for occurence of event A2
        self.bs_connected = None                         # BS to which the UE is currently connected

    def next_action(self, obs):
        self.history.append(obs)

        # Initially when starting out connect to the highest signal strength
        if self.bs_connected == None:
            self.bs_connected = np.argmax(obs)
            return np.eye(len(obs))[np.argmax(obs)]

        # When A2 event hasn't occured continously check whether it can occur or not
        elif len(self.history) > self.history_length and self.a2 == 0:
            self.history = self.history[1:]
            signal_strength = np.squeeze(np.array(self.history))
            status = np.mean(signal_strength[:,self.bs_connected] >= self.lower_threshold)
            if status == 1:
                self.a2 = 1

        # Event A2 has occured, now check for the occurence of A1 or A4    
        elif len(self.history) > self.history_length and self.a2 == 1:
            candidate_bs = []
            self.history = self.history[1:]
            signal_strength = np.squeeze(np.array(self.history))

            # Check for event A1
            status = np.mean(signal_strength[:,self.bs_connected] >= self.lower_threshold)
            if status == 1:
                self.a2 = 0

            # Check for event A4
            candidate_bs = []
            for i in range(signal_strength.shape[1]):
                if i != self.bs_connected:
                    status = np.mean(signal_strength[:,i] >= self.upper_threshold)
                    if status == 1:
                        candidate_bs.append(i)

            # Find the BS with the maximum avergae signal strength
            if len(candidate_bs):
                candidate_strength = np.take(signal_strength, candidate_bs, axis=1)
                candidate_strength = np.sum(candidate_strength, axis=0)
                candidate = np.argmax(candidate_strength)
                self.a2 = 0
                self.bs_connected = candidate

            return np.eye(len(obs))[self.bs_connected]
                


    def reset(self, lower_threshold, upper_threshold, time_to_trigger, step):
        self.__init__(lower_threshold, upper_threshold, time_to_trigger, step)


#a4a2 = a4_a2()
        
