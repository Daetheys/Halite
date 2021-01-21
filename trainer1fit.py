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

        self.t_ratio = 5/100
        self.explo_rate = 0.05

        self.gamma = 0.99
        
        self.reset()

    def save(self,prefix):
        self.vbm.save(prefix)
        
    def reset(self):
        #Reset games
        self.games = [Game(None,None) for _ in range(self.batch_size)]
        self.game_length = self.games[0].length

        self.loss_batch = [[0. for i in range(self.batch_size)] for k in range(2)]
        
        self.vbm.reset()

    def learn(self,nb,prefix):
        for _ in range(nb):
            print(_)
            with tf.GradientTape() as tape:
                for i in range(self.game_length):
                    #print(i)
                    self.step()
                #print("workers finished")
                loss = tf.reduce_mean(self.loss_batch[0])
                print("loss",loss.numpy())
                grad = tape.gradient(loss,self.vbm.parameters())
                #print("grad ok")
            opt = tf.keras.optimizers.Adam(10**-2)
            opt.apply_gradients(zip(grad,self.vbm.parameters()))
            #print("optimized")
            self.reset()
            self.save(prefix)

    def step(self):
        
        t = self.games[0].nb_step
        length = self.game_length
        
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
        rewards = np.zeros(self.batch_size)
        for i,g in enumerate(self.games):
            
            actions = [actions_dict() for _ in range(2)]

            for k in range(2): #Each player
                #Ships
                ship_proba = out_sh[k][i]()
                explo_rate = self.explo_rate
                if k == 1:
                    explo_rate = 1
                for j,sh_action_proba in enumerate(ship_proba):
                    proba = sh_action_proba
                    if np.random.random() < explo_rate:
                        #Explore
                        action = tf.random.categorical(tf.math.log(tf.ones((1,6))),1)[0,0]
                    else:
                        #Exploit
                        action = (tf.random.categorical(tf.math.log([proba]),1)[0,0])
                    actions[k][action.numpy()].append(g.players[k].ships[j])
                    if proba[action] > 0:
                        self.loss_batch[k][i] -= self.gamma**(length-t)*tf.math.log(proba[action])
                        if tf.math.is_inf(tf.math.log(proba[action])):
                            print('sh',k,i,self.loss_batch[k][i],proba[action])
                            assert False

                #Shipyard
                for j,sy_action_proba in enumerate(out_sy[k][i]()):
                    proba = sy_action_proba
                    if np.random.random() < explo_rate:
                        #Explore
                        action = -tf.random.categorical(tf.math.log(tf.ones((1,2))),1)[0,0]-1
                    else:
                        #Exploit
                        action = -tf.random.categorical(tf.math.log([proba]),1)[0,0] -1
                    actions[k][action.numpy()].append(g.players[k].shipyards[j])
                    #Add action to batch
                    if proba[action] > 0:
                        self.loss_batch[k][i] -= self.gamma**(length-t)*tf.math.log(proba[action])
                        if tf.math.is_inf(tf.math.log(proba[action])):
                            print('sh',k,i,self.loss_batch[k][i],proba[action])
                            assert False

            #Step and compute reward
            rewards[i] = g._step(actions[0],actions[1])
        #print(t,self.game_length)
        if t == self.game_length-1:
            #print("rewards : ",rewards)
            for i in range(self.batch_size):
                self.loss_batch[0][i] *= rewards[i]
                self.loss_batch[1][i] *= -rewards[i]

            halite0 = [self.games[i].players[0].halite for i in range(len(self.games))]
            halite1 = [self.games[i].players[1].halite for i in range(len(self.games))]
            print("mean halite p0",np.mean(halite0),np.std(halite0),np.unique(rewards,return_counts=True))
            print("mean halite p1",np.mean(halite1),np.std(halite1))

        self.vbm.reset()
            
    def fit(self,nb_epochs=5):
        self.process_batch()
        opt = tf.keras.optimizers.Adam()
        for _ in range(nb_epochs):
            opt.minimize(self.loss,self.vbm.parameters())
