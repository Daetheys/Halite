import tensorflow as tf
import numpy as np

EPS = 10**-3

class Bot:
    def __init__(self,id):
        self.id = id

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
    def __init__(self,id):
        super().__init__(id)
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

class ShyBot(Bot):
    def __init__(self,id):
        super().__init__(id)
        self.final_score = 0

    def ship_to_halite(self,sh_obs):
        def position(X):
            if np.sum(X) == 0:
                return None
            return np.unravel_index(X.argmax(),X.shape)
        def distance(a,b):
            xa,ya = a
            xb,yb = b
            return abs(xa-xb) + abs(ya-yb)

        no_turn = sh_obs[0,0,9]
        sy_pos = position(sh_obs[:,:,3])
        sh_pos = position(sh_obs[:,:,7])
        if sy_pos is None:
            return np.array([0,0,0,0,0.5,0.5]),sh_obs[...,:7]
        d = distance(sy_pos,sh_pos)
        spendable = 10 - no_turn
        if spendable < d:
            return np.array([0,0,0,0,0,1]),sh_obs[...,:7]
        targets = sh_obs[:,:,0].copy()
        for x in range(9):
            for y in range(9):
                d1 = distance((x,y),sh_pos) + distance((x,y),sy_pos)
                targets[x,y] = max(spendable-1-d1,0) * targets[x,y]
        targets = (1 - sh_obs[:,:,1] - sh_obs[:,:,3]+ sh_obs[:,:,7]) * targets
        if spendable <= d:
            target = sy_pos
        else:
            target = position(targets)
            if target is None:
                target = sy_pos
        has_people = sh_obs[...,1] + sh_obs[...,4] + sh_obs[...,6]
        if sy_pos == sh_pos:
            x,y = sh_pos
            if has_people[x-1][y] == 0:
                target = x-1,y
            elif has_people[x+1][y] == 0:
                target = x+1,y
            elif has_people[x][y-1] == 0:
                target = x,y-1
            elif has_people[x][y+1] == 0:
                target = x,y+1
        '''print('{} | {},{} → {},{} → {} {} , {} + {} = {}'.format(\
                int(sh_obs[sh_pos[0],sh_pos[1],2]), \
                sh_pos[0],sh_pos[1], \
                target[0],target[1], \
                sy_pos[0],sy_pos[1], \
                distance(sh_pos,target), distance(target,sy_pos), distance(sh_pos,target)+distance(sy_pos,target)))'''
        world = sh_obs[...,:7]
        world[...,1] -= sh_obs[...,7]
        if sh_pos[0] > target[0] and (has_people[sh_pos[0]-1,sh_pos[1]] == 0 or (target[0] == sy_pos[0] and target[1] == sy_pos[1])):
            world[sh_pos[0]-1,sh_pos[1]] = 1
            return np.array([0,0,0,1,0,0]),world
        elif sh_pos[0] < target[0] and (has_people[sh_pos[0]+1,sh_pos[1]] == 0 or (target[0] == sy_pos[0] and target[1] == sy_pos[1])):
            world[sh_pos[0]+1,sh_pos[1]] = 1
            return np.array([0,1,0,0,0,0]),world
        elif sh_pos[1] > target[1] and (has_people[sh_pos[0],sh_pos[1]-1] == 0 or (target[0] == sy_pos[0] and target[1] == sy_pos[1])):
            world[sh_pos[0],sh_pos[1]-1] = 1
            return np.array([1,0,0,0,0,0]),world
        elif sh_pos[1] < target[1] and (has_people[sh_pos[0],sh_pos[1]+1] == 0 or (target[0] == sy_pos[0] and target[1] == sy_pos[1])):
            world[sh_pos[0]-1,sh_pos[1]+1] = 1
            return np.array([0,0,1,0,0,0]),world
        world[...,1] += sh_obs[...,7]
        return np.array([0,0,0,0,0,1]), world


    def compute_actions_proba(self,observation):
        sh_obs,sy_obs = observation
        nh,ny = sh_obs.shape[0],sy_obs.shape[0]
        if nh == 0 and ny == 0:
            return (lambda: np.zeros((0,6)), lambda: np.zeros((0,6)))
        no_turn = sy_obs[0][0,0,9] if nh == 0 else sh_obs[0][0,0,9]
        #print('################ {} ################'.format(int(no_turn)))
        #print(nh,ny)
        if no_turn <= 0 :
            return (lambda: np.array([0,0,0,0,1,0]).astype(float).reshape(1,6),\
                    lambda : np.zeros((0,2)).astype(float))
        else:
            sh_actions, sy_actions = np.zeros((0,6)), np.zeros((0,2))
            world = None if nh == 0 else sh_obs[0,:,:,0:7]
            for i in range(nh):
                obs = np.concatenate((world,sh_obs[i,:,:,7:]), axis=2)
                ship_action,world = self.ship_to_halite(obs)
                sh_actions = np.vstack([sh_actions, ship_action])
            for i in range(ny):
                if nh <= 2:
                    sy_actions = np.vstack([sy_actions, np.array([1,0])])
                else :
                    sy_actions = np.vstack([sy_actions, np.array([0,1])])
            return (lambda :sh_actions,lambda :sy_actions)


class FighterBot(Bot):
    def __init__(self,id):
        super().__init__(id)
        self.final_score = 0

    def ship_to_halite(self,sh_obs):
        def position(X):
            if np.sum(X) == 0:
                return None
            return np.unravel_index(X.argmax(),X.shape)
        def distance(a,b):
            xa,ya = a
            xb,yb = b
            return abs(xa-xb) + abs(ya-yb)

        no_turn = sh_obs[0,0,9]
        sh_pos = position(sh_obs[:,:,7])
        spendable = 10 - no_turn
        targets = sh_obs[:,:,5] + sh_obs[:,:,6]
        for x in range(9):
            for y in range(9):
                d1 = distance((x,y),sh_pos)
                targets[x,y] = max(spendable-d1,0) * targets[x,y]
        targets = (1 - sh_obs[:,:,1] - sh_obs[:,:,3]+ sh_obs[:,:,7]) * targets
        target = position(targets)
        if target is None:
            target = sh_pos
        has_people = sh_obs[...,1]
        '''print('{} | {},{} → {},{} , {}'.format(\
                int(sh_obs[sh_pos[0],sh_pos[1],2]), \
                sh_pos[0],sh_pos[1], \
                target[0],target[1], \
                distance(sh_pos,target)))'''
        world = sh_obs[...,:7]
        world[...,1] -= sh_obs[...,7]
        if sh_pos[0] > target[0] and has_people[sh_pos[0]-1,sh_pos[1]] == 0:
            world[sh_pos[0]-1,sh_pos[1]] = 1
            return np.array([0,0,0,1,0,0]),world
        elif sh_pos[0] < target[0] and has_people[sh_pos[0]+1,sh_pos[1]] == 0:
            world[sh_pos[0]+1,sh_pos[1]] = 1
            return np.array([0,1,0,0,0,0]),world
        elif sh_pos[1] > target[1] and has_people[sh_pos[0],sh_pos[1]-1] == 0:
            world[sh_pos[0],sh_pos[1]-1] = 1
            return np.array([1,0,0,0,0,0]),world
        elif sh_pos[1] < target[1] and has_people[sh_pos[0],sh_pos[1]+1] == 0:
            world[sh_pos[0]-1,sh_pos[1]+1] = 1
            return np.array([0,0,1,0,0,0]),world
        world[...,1] += sh_obs[...,7]
        return np.array([0,0,0,0,0,1]), world


    def compute_actions_proba(self,observation):
        sh_obs,sy_obs = observation
        nh,ny = sh_obs.shape[0],sy_obs.shape[0]
        if nh == 0 and ny == 0:
            return (lambda: np.zeros((0,6)), lambda: np.zeros((0,6)))
        no_turn = sy_obs[0][0,0,9] if nh == 0 else sh_obs[0][0,0,9]
        #print('################ {} ################'.format(int(no_turn)))
        #print(nh,ny)
        if no_turn <= 0 :
            return (lambda: np.array([0,0,0,0,1,0]).astype(float).reshape(1,6),\
                    lambda : np.zeros((0,2)).astype(float))
        else:
            sh_actions, sy_actions = np.zeros((0,6)), np.zeros((0,2))
            world = None if nh == 0 else sh_obs[0,:,:,0:7]
            for i in range(nh):
                obs = np.concatenate((world,sh_obs[i,:,:,7:]), axis=2)
                ship_action,world = self.ship_to_halite(obs)
                sh_actions = np.vstack([sh_actions, ship_action])
            for i in range(ny):
                if nh <= 20:
                    sy_actions = np.vstack([sy_actions, np.array([1,0])])
                else :
                    sy_actions = np.vstack([sy_actions, np.array([0,1])])
            return (lambda :sh_actions,lambda :sy_actions)


def AdvancedCollectBot(Bot):
    def __init__(self):
        targets = 0

    def compute_actions_proba(self,observation):
        ships = 0

class LearnerBot(Bot):
    def __init__(self,model,id):
        super().__init__(id,)
        self.model = model
        self.batch_size = 5
        self.reset()
        self.reset_batch()

    def reset_batch(self):
        self.obs_batch = [[] for i in range(self.batch_size)]
        self.action_batch = [[] for i in range(self.batch_size)]
        self.batch_index = 0

    def next_batch(self):
        #Go to next batch
        self.batch_index = (self.batch_index+1)%self.batch_size
        #Free the batch
        self.obs_batch[self.batch_index] = []
        self.action_batch[self.batch_index] = []

    def add_batch(self,o,batch):
        assert len(batch) == self.batch_size
        batch[self.batch_index].append(o)
        assert len(batch[self.batch_index])<=10

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

    def sample_actions_indexs(self,actions_proba,return_proba=False,store_batch=True):
        if return_proba:
            action,_ = super().sample_actions_indexs(actions_proba,return_proba=return_proba)
        else:
            action = super().sample_actions_indexs(actions_proba,return_proba=return_proba)
        if store_batch:
            self.add_batch(action,self.action_batch)
        #self.action_batch.append(action)
        return action

class RLBot(LearnerBot):
    def __init__(self,vbm,id):
        super().__init__(vbm,id)
        self.explo_rate = 1
        self.gamma = 1
        
        self.reward_batch = [0 for i in range(self.batch_size)]
        self.index_reward = 0

    def add_reward(self,reward):
        self.reward_batch[self.index_reward]= reward
        self.index_reward = (self.index_reward+1)%self.batch_size

    def loss_precompute(self):
        self.probas_wrapper = [[] for i in range(self.batch_size)]
        for i,bo in enumerate(self.obs_batch):
            for o in bo:
                proba = self.compute_actions_proba(o,store_batch=False)
                self.probas_wrapper[i].append(proba)
    
    def loss_compute(self):
        l = 0.
        length = len(self.probas_wrapper)
        for b,bpw in enumerate(self.probas_wrapper):
            for t,p in enumerate(bpw):
                #Get proba for each possible move for each ship.shipyard
                (sh_probas,sy_probas) = (p[0](),p[1]())
                sh_probas = sh_probas*(1-EPS)+EPS
                sy_probas = sy_probas*(1-EPS)+EPS
                #Get chosen actions
                (sh_actions,sy_actions) = self.action_batch[b][t]
                #if t==0 and self.id==0:
                #    print(self.obs_batch[0][0][0][0,0,0,0])
                #    print(self.action_batch[0][0][0][0])
                #    print(self.reward_batch)
                if t %5 == 0 and self.id==0:
                    sh_min = tf.reduce_min(sh_probas).numpy()
                    sh_min_idx = np.where(sh_probas==sh_min)[1]
                    sh_max = tf.reduce_max(sh_probas).numpy()
                    sh_max_idx = np.where(sh_probas==sh_max)[1]
                    sy_min = tf.reduce_min(sy_probas).numpy()
                    sy_min_idx = np.where(sy_probas==sy_min)[1]
                    sy_max = tf.reduce_max(sy_probas).numpy()
                    sy_max_idx = np.where(sy_probas==sy_max)[1]
                    print(sh_min,sh_min_idx,sh_max,sh_max_idx,sy_min,sy_min_idx,sy_max,sy_max_idx)
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
                reinforce_loss = -self.reward_batch[b]*(self.gamma**(length-t))*log_proba_played
                #Add crossentropy term
                entropy_loss = -tf.reduce_sum(tf.math.log(sh_probas)) - tf.reduce_sum(tf.math.log(sy_probas))
                entropy_loss *= 0.02
                #if t%5==0 and self.id==0:
                #    print('r',reinforce_loss.numpy())
                #    print('e',entropy_loss.numpy())
                #print("entropy",entropy_loss)
                #print("reinforce",reinforce_loss)
                #Lasso loss
                #lasso_loss = tf.reduce_mean([tf.reduce_mean(tf.math.abs(w)) for w in self.model.parameters()])
                #Add to loss
                l += reinforce_loss + entropy_loss #+ lasso_loss
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
class ExploratorRLBot(RLBot):
    def __init__(self,vbm,id):
        super().__init__(vbm,id)
    def loss_compute(self):
        l = 0.
        length = len(self.probas_wrapper)
        for b,bpw in enumerate(self.probas_wrapper):
            for t,p in enumerate(bpw):
                #Get proba for each possible move for each ship.shipyard
                (sh_probas,sy_probas) = (p[0](),p[1]())
                sh_probas = sh_probas*(1-EPS)+EPS
                sy_probas = sy_probas*(1-EPS)+EPS
                #Get chosen actions
                (sh_actions,sy_actions) = self.action_batch[b][t]
                #if t==0 and self.id==0:
                #    print(self.obs_batch[0][0][0][0,0,0,0])
                #    print(self.action_batch[0][0][0][0])
                #    print(self.reward_batch)
                if t %5 == 0 and self.id==0:
                    sh_min = tf.reduce_min(sh_probas).numpy()
                    sh_min_idx = np.where(sh_probas==sh_min)[1]
                    sh_max = tf.reduce_max(sh_probas).numpy()
                    sh_max_idx = np.where(sh_probas==sh_max)[1]
                    sy_min = tf.reduce_min(sy_probas).numpy()
                    sy_min_idx = np.where(sy_probas==sy_min)[1]
                    sy_max = tf.reduce_max(sy_probas).numpy()
                    sy_max_idx = np.where(sy_probas==sy_max)[1]
                    print(sh_min,sh_min_idx,sh_max,sh_max_idx,sy_min,sy_min_idx,sy_max,sy_max_idx)
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
                reinforce_loss = 0
                if self.reward_batch[b]>0:
                    reinforce_loss += -self.reward_batch[b]*(self.gamma**(length-t))*log_proba_played
                reinforce_loss *= 100
                #Add crossentropy term
                entropy_loss = -tf.reduce_sum(tf.math.log(sh_probas)) - tf.reduce_sum(tf.math.log(sy_probas))
                entropy_loss *= 0.02
                #if t%5==0 and self.id==0:
                #    print('r',reinforce_loss.numpy())
                #    print('e',entropy_loss.numpy())
                #print("entropy",entropy_loss)
                #print("reinforce",reinforce_loss)
                #Lasso loss
                #lasso_loss = tf.reduce_mean([tf.reduce_mean(tf.math.abs(w)) for w in self.model.parameters()])
                #Add to loss
                l += reinforce_loss + entropy_loss #+ lasso_loss
        return l
