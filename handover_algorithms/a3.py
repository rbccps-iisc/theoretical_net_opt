import numpy as np

# Sources:https://telecompedia.net/a3-intra-frequency-handover-in-lte/
#        :https://www.researchgate.net/publication/338342350_Performance_Evaluation_of_A2-A4-RSRQ_and_A3-RSRP_Handover_Algorithms_in_LTE_Network

# Algorithm Description:

#


class a3(object):
    def __init__(self, threshold, time_to_trigger, step):
        self.threshold = threshold
        self.history_length = time_to_trigger / steps    # the history of connection strengths that need to be maintained
        self.history = []
        self.trigger = 0                                 # flag for occurence of threshold crossing event
        self.bs_connected = None                         # BS to which the UE is currently connected


    def next_action(self, obs):
        self.history.append(obs)

        # Initially when starting out connect to the highest signal strength
        if self.bs_connected == None:
            self.bs_connected = np.argmax(obs)
            return np.eye(len(obs))[np.argmax(obs)]


        # When trigger event hasn't occured continously check whether it can occur or not
        elif len(self.history) > self.history_length:
            self.history = self.history[1:]
            signal_strength = np.squeeze(np.array(self.history))

            candidate_bs = []
            for i in range(signal_strength.shape[1]):
                if i != self.bs_connected:
                    status = np.mean([signal_strength[:,i]-signal_strength[:,self.bs_connected]] >= self.threshold)
                    if status == 1:
                        candidate_bs.append(i)

            # Find the BS with the maximum avergae signal strength
            if len(candidate_bs):
                candidate_strength = np.take(signal_strength, candidate_bs, axis=1)
                candidate_strength = np.sum(candidate_strength, axis=0)
                candidate = np.argmax(candidate_strength)
                self.bs_connected = candidate
 

        # None of the above event is satisfied then maintain the connection
        return np.eye(len(obs))[self.bs_connected]
