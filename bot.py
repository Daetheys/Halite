import tensorflow as tf
import numpy as np

EPS = 10**-100

class Bot:
    def __init__(self):
        pass

    def reset(self):
        pass

    def compute_actions_proba(self,observation):
        raise NotImplementedError

    def sample_actions_indexs(self,actions_proba,return_proba=False):
        (sh_probas,sy_probas) = (actions_proba[0](),actions_proba[1]())
        sh_actions = tf.random.categorical(tf.math.log(sh_probas),1)[:,0]
        sy_actions = tf.random.categorical(tf.math.log(sy_probas),1)[:,0]
        actions = (sh_actions,sy_actions)
        if return_proba:
            lprobas_sh = [sh_probas[i,a] for i,a in enumerate(sh_actions)]
            actions_probas_sh = tf.convert_to_tensor(lprobas_sh)
            lprobas_sy = [sy_probas[i,a] for i,a in enumerate(sy_actions)]
            actions_probas_sy = tf.convert_to_tensor(lprobas_sy)
            actions_probas = (actions_probas_sh,actions_probas_sy)
            return actions,actions_probas
        return actions

    def sample_actions(self,actions_proba):
        (sh_proba,sy_proba) = self.sample_actions_indexs(actions_proba,return_proba=False)
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
        self.explo_rate = 0 

    def compute_actions_proba(self,observation):
        """ Observation is a (sh_input,sy_input) object with both array of shape Nx21x21x9.
         0 : halite array // 1 : player ship position array // 2 : player ship halite array //
          3 : player shipyard array // 4 : ennemy ship array // 5 ennemy ship halite array //
          6 ennemy shipyard array // 7 : ship to action // 8 : shipyard to action """
        (sh_obs,sy_obs) = observation
        #Uniform proba for each move
        sh_proba = lambda : tf.ones((sh_obs.shape[0],6))/6
        sy_proba = lambda : tf.ones((sy_obs.shape[0],2))/2
        """ You need to return a N_shx6 array for ships with proba for each action for each ship 
        and a N_syx2 for shipyard with proba for each action for each shipyard """
        return (sh_proba,sy_proba)

def AdvancedCollectBot(Bot):
    def __init__(self):
        targets = 0

    def compute_actions_proba(self,observation):
        ships = 0

class LearnerBot(Bot):
    def __init__(self,model):
        self.model = model
        self.reset()
        self.reset_batch()

    def reset_batch(self):
        self.obs_batch = []
        self.action_batch = []

    def add_batch(self,o,batch):
        #print(len(batch))
        if len(batch) > 30:
            del batch[0]
        batch.append(o)

    def reset_model(self):
        self.model.reset()

    def reset(self):
        super().reset()
        self.reset_model()

    def loss(self):
        raise NotImplementedError

    def compute_actions_proba(self,observation,store_batch=True):
        if store_batch:
            self.add_batch(observation,self.obs_batch)
             #self.obs_batch.append(observation)
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
        self.add_batch(action,self.action_batch)
        #self.action_batch.append(action)
        return action

class RLBot(LearnerBot):
    def __init__(self,vbm):
        super().__init__(vbm)
        self.explo_rate = 0.2
        self.gamma = 0.99

    def loss_precompute(self):
        self.probas_wrapper = []
        for o in self.obs_batch:
            proba = self.compute_actions_proba(o,store_batch=False)
            self.probas_wrapper.append(proba)
    
    def loss_compute(self,reward):
        l = 0.
        length = len(self.probas_wrapper)
        for t,p in enumerate(self.probas_wrapper):
            #Get proba for each possible move for each ship.shipyard
            (sh_probas,sy_probas) = (p[0](),p[1]())
            #(sh_probas,sy_probas) = (sh_probas*(1-EPS)+EPS,sy_probas*(1-EPS)+EPS)
            #Get chosen actions
            (sh_actions,sy_actions) = self.action_batch[t]
            if t %5 == 0:
                print(tf.reduce_min(sh_probas).numpy(),tf.reduce_max(sh_probas).numpy(),tf.reduce_min(sy_probas).numpy(),tf.reduce_max(sy_probas).numpy())
            #Compute log proba of chosen actions
            log_proba_played = 0.
            for i,sh_action in enumerate(sh_actions):
                action_proba = sh_probas[i,sh_action]
                if action_proba > 0:
                    log_proba_played += tf.math.log(action_proba)
                else:
                    log_proba_played += tf.math.log(action_proba+EPS)
            for i,sy_action in enumerate(sy_actions):
                action_proba = sy_probas[i,sy_action]
                if action_proba > 0:
                    log_proba_played += tf.math.log(action_proba)
                else:
                    log_proba_played += tf.math.log(action_proba+EPS)
            #Add with reward and gamma factor
            reinforce_loss = -reward*(self.gamma**(length-t))*log_proba_played
            #Add crossentropy term
            entropy_loss = -tf.reduce_sum(tf.math.log(sh_probas)) - tf.reduce_sum(tf.math.log(sy_probas))
            entropy_loss *= 80
            #print("entropy",entropy_loss)
            #print("reinforce",reinforce_loss)
            #Add to loss
            l += reinforce_loss + entropy_loss
        return l

    def sample_actions_indexs(self,actions_proba,return_proba=False):
        if np.random.random() < self.explo_rate:
            #Explore -> sample according to uniform proba
            (sh_actions_proba,sy_actions_proba) = (actions_proba[0](),actions_proba[1]())
            sh_uniform_proba = lambda : tf.ones(sh_actions_proba.shape)/sh_actions_proba.shape[0]
            sy_uniform_proba = lambda : tf.ones(sy_actions_proba.shape)/sy_actions_proba.shape[0]
            uniform_proba = (sh_uniform_proba,sy_uniform_proba)
            return super().sample_actions_indexs(uniform_proba,return_proba=return_proba)
        else:
            #Exploit -> sample according to given actions_proba vector
            return super().sample_actions_indexs(actions_proba,return_proba=return_proba)

    def compute_actions_indexs(self,observation):
        """ This function flushes the model """
        proba = self.compute_actions_proba(observation)
        self.vbm.flush()
        action = self.sample_actions(proba)
        return action