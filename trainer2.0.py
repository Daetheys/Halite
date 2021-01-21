from engine.game import Game
from tools import get_nn_input
from engine.tools import *
from tools import *
import numpy as np
import tensorflow as tf
import time
np.set_printoptions(precision=15)

EPS = 10**-30

class Env:
    def step(self,action):
        raise NotImplementedError

    def reset(self):
        pass

class VecEnv(Env):
    def step(self,actions):
        raise NotImplementedError

    def stepi(self,action,i):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def reseti(self,i):
        raise NotImplementedError

class HaliteTrainer:
    def __init__(self,vbm,batch_size=5):
        self.batch_size = batch_size
        self.vbm = vbm

        self.explo_rate = 0.05

        self.gamma = 0.99
        
        self.reset()

    def save(self):
        self.vbm.save()
        
    def reset(self):
        #Reset games
        self.games = [Game(None,None) for _ in range(self.batch_size)]
        self.game_length = self.games[0].length
        
        self.reward_batch = np.full((self.game_length,self.batch_size),None) #Need to keep all of them to compute outcome

        self.vbm.reset()

    def save(self,prefix):
        self.vbm.save(prefix)

    def learn(self,nb,prefix):
        for _ in range(nb):
            print(_)
            for i in range(self.game_length):
                self.step()
            self.fit()
            self.reset()
            self.save(prefix)

    def step(self):
        
        t = self.games[0].nb_step
        
        probas = [[None for __ in range(self.batch_size)] for _ in range(2)]
        
        for i,g in enumerate(self.games):
            #Input for neural network for sh : ships | sy : shipyards | p0 : player0 | p1 : player1

            inp_sh_p0 , inp_sy_p0 , inp_sh_p1 , inp_sy_p1 = get_nn_input(g)

            #Send requests and store outputs
            probas[0][i] = g.players[0].agent.compute_actions_proba((inp_sh_p0,inp_sy_p0))
            probas[1][i] = g.players[1].agent.compute_actions_proba((inp_sh_p1,inp_sy_p1))

        #Flush
        self.vbm.flush()

        #Apply actions
        for i,g in enumerate(self.games):
            
            (sh_actions_index0,sy_actions_index0) = g.players[0].agent.sample_actions(probas[0][i])
            (sh_actions_index1,sy_actions_index1) = g.players[1].agent.sample_actions(probas[1][i])

            #Step and compute reward
            reward = g._step(actions[0],actions[1])
            #Store reward
            self.reward_batch[t,i] = reward

        if t == self.game_length-1:
            halite0 = [self.games[i].players[0].halite for i in range(len(self.games))]
            halite1 = [self.games[i].players[1].halite for i in range(len(self.games))]
            print("mean halite p0",np.mean(halite0),np.std(halite0))
            print("mean halite p1",np.mean(halite1),np.std(halite1))

        self.vbm.reset()

    def loss(self):
        self.vbm.reset()
        for g in self.games:
            g.players[0].loss_precompute()
        self.vbm.flush()
        l = tf.reduce_mean([g.players[0].loss_compute() for g in self.games])
        
        print("loss : ",l.numpy(),np.unique(self.reward_batch[-1],return_counts=True))
        return out
            
    def fit(self,nb_epochs=5):
        self.process_batch()
        opt = tf.keras.optimizers.Adam(10**-4)
        for _ in range(nb_epochs):
            opt.minimize(self.loss,self.vbm.parameters())
