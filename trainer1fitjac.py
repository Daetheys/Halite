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

    def save(self):
        self.vbm.save()
        
    def reset(self):
        #Reset games
        self.games = [Game(None,None) for _ in range(self.batch_size)]
        self.game_length = self.games[0].length
        
        self.gradient_batch = [tf.zeros((2,self.batch_size,*w.shape),dtype=tf.float32) for w in self.vbm.parameters()]
        self.vbm.reset()

    def step(self):
        
        with tf.GradientTape() as tape:

            #tape.watch(self.vbm.parameters())
            

            t = self.games[0].nb_step
            length = self.games[0].length

            to_grad = [[tf.convert_to_tensor(0.,dtype=tf.float32) for j in range(self.batch_size)] for i in range(2)]
            #to_grad = tf.Variable(tf.zeros((2,self.batch_size),dtype=tf.float64))
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
            rewards = np.zeros((self.batch_size,))
            for i,g in enumerate(self.games):
                
                actions = [actions_dict() for _ in range(2)]

                for k in range(2): #Each player
                    #Ships
                    ship_proba = out_sh[k][i]()
                    
                    for j,sh_action_proba in enumerate(ship_proba):
                        proba = sh_action_proba
                        if np.random.random() < self.explo_rate:
                            #Explore
                            action = tf.random.categorical(tf.math.log(tf.ones((1,6))),1)[0,0]
                        else:
                            #Exploit
                            action = (tf.random.categorical(tf.math.log([proba]),1)[0,0])

                        actions[k][action.numpy()].append(g.players[k].ships[j])
                        #Update gradient for selected actions
                        to_grad[k][i] += (self.gamma**(length-t)) * tf.math.log(sh_action_proba[action])
                        #to_grad[k,i].assign(to_grad[k,i]+(self.gamma**(length-t)) * tf.math.log(sh_action_proba[action]))

                    #Shipyard
                    for j,sy_action_proba in enumerate(out_sy[k][i]()):
                        proba = sy_action_proba
                        if np.random.random() < self.explo_rate:
                            #Explore
                            action = -tf.random.categorical(tf.math.log(tf.ones((1,2))),1)[0,0]-1
                        else:
                            #Exploit
                            action = -tf.random.categorical(tf.math.log([proba]),1)[0,0] -1
                        actions[k][action.numpy()].append(g.players[k].shipyards[j])
                        #Update gradient for selected actions
                        to_grad[k][i] += (self.gamma**(length-t)) * tf.math.log(sy_action_proba[action])
                        #to_grad[k,i].assign(to_grad[k,i]+(self.gamma**(length-t)) * tf.math.log(sy_action_proba[action]))
                #Step the game and get rewards (are 0 except at the end of the game : +/- 1)
                rewards[i] = g._step(actions[0],actions[1])
            
            #Convert to_grad to tensor for jacobian computation
            to_grad2 = tf.convert_to_tensor(to_grad)

            #Drop data to decrease jacobian computation time
            indx_drop = np.random.choice(self.batch_size,1,replace=False) #Only keep data from 10 games uniformely chosen
            indx_mask = np.zeros((self.batch_size,))
            indx_mask[indx_drop] = 1
            indx_mask = np.asarray(indx_mask,dtype=np.bool)
            to_grad_drop = tf.boolean_mask(to_grad2,indx_mask,axis=1)

        #Jacobian computation
        v_drop = tape.jacobian(to_grad_drop,self.vbm.parameters())

        v = [tf.Variable(tf.zeros((2,self.batch_size,*w.shape),dtype=tf.float32)) for w in self.vbm.parameters()]
        for w in range(len(self.vbm.parameters())):
            if not(v_drop[w] is None):
                for e,i in enumerate(indx_drop):
                    v[w][:,i].assign(v_drop[w][:,e])

        #Add jacobians to free tensorflow graph (and avoid memory issues)
        for w in range(len(self.vbm.parameters())):
            self.gradient_batch[w] += v[w]

        #End of the game : update gradient batch with the reward
        #print(t,self.game_length-1)
        if t == self.game_length-1:
            for w in range(len(self.vbm.parameters())):
                for k in range(2):
                    player = 1
                    if k == 1:
                        player = -1
                    for i in range(len(self.batch_size)):
                        self.gradient_batch[w][k][i] *= player * rewards[i]
            #Mean games
            for w in range(len(self.vbm.parameters())):
                self.gradient_batch[w] = tf.reduce_mean(self.gradient_batch[w],axis=1)

            halite0 = [self.games[i].players[0].halite for i in range(len(self.games))]
            halite1 = [self.games[i].players[1].halite for i in range(len(self.games))]
            print("mean halite p0",np.mean(halite0),np.std(halite0))
            print("mean halite p1",np.mean(halite1),np.std(halite1))

        self.vbm.reset()
            
    def fit(self,nb_epochs=5):
        opt = tf.keras.optimizers.Adam()
        for _ in range(nb_epochs):
            opt.apply_gradients(zip(self.gradient_batch[0],self.vbm.parameters()))
