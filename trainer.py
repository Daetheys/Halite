from engine.game import Game
from tools import get_nn_input
from engine.tools import *
import numpy as np
import tensorflow as tf
import time
np.set_printoptions(precision=15)

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
        self.reset()
        
    def reset(self):
        self.games = [Game(None,None) for _ in range(self.batch_size)]
        self.obs_batch =    [[[None for _ in range(4)] for __ in range(self.batch_size)] for ___ in range(400)]
        self.proba_batch =  [[[[] for _ in range(4)] for __ in range(self.batch_size)] for ___ in range(400)]
        self.action_batch = [[[[] for _ in range(4)] for __ in range(self.batch_size)] for ___ in range(400)]
        self.reward_batch = np.zeros((400,self.batch_size))

        
    def step(self):

        ti = time.perf_counter()
        
        t = self.games[0].nb_step
        
        out_sh = [[None for __ in range(self.batch_size)] for _ in range(2)]
        out_sy = [[None for __ in range(self.batch_size)] for _ in range(2)]
        
        for i,g in enumerate(self.games):
            #Input for neural network for sh : ships | sy : shipyards | p0 : player0 | p1 : player1

            inp_sh_p0 , inp_sy_p0 , inp_sh_p1 , inp_sy_p1 = get_nn_input(g)
            
            #Send requests and store outputs
            out_sh[0][i] = self.vbm.request(inp_sh_p0,0)
            out_sy[0][i] = self.vbm.request(inp_sy_p0,1)
            out_sh[1][i] = self.vbm.request(inp_sh_p1,0)
            out_sy[1][i] = self.vbm.request(inp_sy_p1,1)

            self.obs_batch[t][i] = [inp_sh_p0,inp_sy_p0,inp_sh_p1,inp_sy_p1]

        print('-',time.perf_counter()-ti)
        ti = time.perf_counter()
            
        #Flush
        self.vbm.flush()

        print('--',time.perf_counter()-ti)
        ti = time.perf_counter()

        #Apply actions
        for i,g in enumerate(self.games):
            
            actions = [actions_dict() for _ in range(2)]

            for k in range(2):
                #Ships
                ship_proba = out_sh[k][i]()
                for j,sh_action_proba in enumerate(ship_proba):
                    proba = sh_action_proba
                    proba = proba/proba.sum()
                    action = np.random.choice(ship_actions,p=proba)
                    actions[k][action].append(g.players[k].ships[j])

                    self.proba_batch[t][i][k*2].append(proba)
                    self.action_batch[t][i][k*2].append(action)

                #Shipyard
                for j,sy_action_proba in enumerate(out_sy[k][i]()):
                    proba = sy_action_proba
                    proba = proba/proba.sum()
                    action = np.random.choice(shipyard_actions,p=proba)
                    actions[k][action].append(g.players[k].shipyards[j])

                    self.proba_batch[t][i][k*2+1].append(proba)
                    self.action_batch[t][i][k*2+1].append(action)
            
            reward = g._step(actions[0],actions[1])
            self.reward_batch[t,i] = reward

        print('---',time.perf_counter()-ti)

    def loss(self,ind_p):
        reward = self.reward_batch.copy()
        if ind_p:
            reward *= -1
        
        p = np.ones((400,self.batch_size))
        for t in range(400):
            for i in range(len(self.proba_batch[t])):
                for j in range(len(self.proba_batch[t][i][ind_p*2])):
                    p[t,i] *= self.proba_batch[t][i][ind_p*2][j][self.action_batch[t][i][ind_p*2][j]]
                for j in range(len(self.proba_batch[t][i][ind_p*2+1])):
                    p[t,i] *= self.proba_batch[t][i][ind_p*2+1][j][self.action_batch[t][i][ind_p*2+1][j]]
                    
        gamma = 0.9
        g = np.zeros((400,self.batch_size))
        g[400-1] = reward[400-1]
        for t in range(400-1):
            g[400-t-2] = g[400-t-1]*gamma+reward[400-t-2]

        return -np.sum(np.log(p)*g)
            
    def fit(self,nb_epochs=50):
        opt = tf.keras.optimizers.Adam()
        for _ in range(nb_epochs):
            loss = lambda : loss(0)+loss(1)
            opt.minimize(loss,self.vbm.parameters)
