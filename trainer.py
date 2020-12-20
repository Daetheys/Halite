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
        self.games = [Game(None,None) for _ in range(batch_size)]
        self.vbm = vbm
        self.obs_batch = []
        
    def step(self):
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

        #Flush
        self.vbm.flush()

        #Apply actions
        for i,g in enumerate(self.games):
            
            actions = [actions_dict() for _ in range(2)]

            for k in range(2):
                #Ships
                for j,sh_action_proba in enumerate(out_sh[k][i]()):
                    proba = sh_action_proba
                    proba = proba/proba.sum()
                    action = np.random.choice(ship_actions,p=proba)
                    actions[k][action].append(g.players[k].ships[j])

                #Shipyard
                for j,sy_action_proba in enumerate(out_sy[k][i]()):
                    proba = sy_action_proba
                    proba = proba/proba.sum()
                    action = np.random.choice(shipyard_actions,p=proba)
                    actions[k][action].append(g.players[k].shipyards[j])
            
            g._step(actions[0],actions[1])
    
    def fit(self):
        pass
