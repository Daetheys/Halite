import tensorflow as tf
import numpy as np

class Bot:
    def __init__(self):
        pass
    def compute_actions_proba(self,observation):
        raise NotImplementedError
    def sample_actions_indexs(self,actions_proba,return_proba=False):
        (sh_proba,sy_proba) = actions_proba
        sh_action = tf.random.categorical(tf.math.log([sh_proba]),1)[0,0]
        sh_action = tf.random.categorical(tf.math.log([sh_proba]),1)[0,0]
        if return_proba:
            probas_sh = tf.concat([sh_proba[:,a] for a in sh_action],axis=1)
            probas_sy = tf.concat([sy_proba[:,a] for a in sy_action],axis=1)
            probas = (probas_sh,probas_sy)
            return action,probas
        return action
    def sample_actions(self,actions_proba,return_proba=False):
        (sh_proba,sy_proba) = self.sample_actions_indexs
        return (sh_proba,-sy_proba-1)

    def compute_actions_indexs(self,observation):
        proba = self.compute_actions_proba(observation)
        actions_indexs = self.sample_actions_indexs(proba)
        return actions_indexs

    def compute_actions(self,observation):
        proba = self.compute_actions_proba(observation)
        actions = self.sample_actions(proba)
        return actions

class RandomBot(Bot):
    def __init__(self):
        pass
    def compute_actions_proba(self,observation):
        (sh_obs,sy_obs) = observation
        #Uniform proba for each move
        sh_proba = tf.ones((sh_obs.shape[0],))/sh_obs.shape[0]
        sy_proba = tf.ones((sy_obs.shape[0],))/sy_obs.shape[0]
        return (sh_proba,sy_proba)

class LearnerBot(Bot):
    def __init__(self,model):
        self.model = model
        self.reset()
    def reset(self):
        self.obs_batch = []
        self.action_batch = []
    def loss(self):
        raise NotImplementedError
    def compute_actions_proba(self,observation):
        self.obs_batch.append(observation)
        (sh_obs,sy_obs) = observation
        #Send requests to the VBM
        sh_proba = self.model.request(sh_obs,0)
        sy_proba = self.model.request(sy_obs,1)
        #Returns references to results (before flush)
        return (sh_proba,sy_proba)
    def sample_actions_indexs(self,actions_proba,return_proba=False):
        if return_proba:
            action,_ = super().sample_actions_indexs(actions_proba,return_proba=return_proba)
        else:
            action = super().sample_actions_indexs(actions_proba,return_proba=return_proba)
        self.action_batch.append(action)
        return action

class RLBot(LearnerBot):
    def __init__(self,vbm):
        super().__init__(vbm)
        self.explo_rate = 0.05

    def loss_precompute(self):
        self.probas_wrapper = []
        for o in self.obs_batch:
            proba = self.compute_actions_proba(o)
            self.probas_wrapper.append(proba)
    
    def loss_compute(self,reward):
        l = 0.
        for t,p in self.probas_wrapper:
            l -= reward*tf.math.log(p())
        return l

    def sample_actions_indexs(self,actions_proba,return_proba=False):
        if np.random.random() < self.explo_rate:
            #Explore -> sample according to uniform proba
            (sh_actions_proba,sy_actions_proba) = actions_proba
            sh_uniform_proba = tf.ones(sh_actions_proba.shape)/sh_actions_proba.shape[0]
            sy_uniform_proba = tf.ones(sy_actions_proba.shape)/sy_actions_proba.shape[0]
            uniform_proba = (sh_uniform_proba,sy_uniform_proba)
            return super().sample_actions_indexs(uniform_proba,return_proba=return_proba)
        else:
            #Exploit
            return super().sample_actions_indexs(actions_proba,return_proba=return_proba)

    def compute_actions_indexs(self,observation):
        """ This function flushes the model """
        proba = self.compute_actions_proba(observation)
        self.vbm.flush()
        action = self.sample_actions(proba)
        return action