from engine.game import Game
from tools import get_nn_input
from engine.tools import *
from tools import *
import numpy as np
import tensorflow as tf
import time
np.set_printoptions(precision=15)

EPS = 10**-5

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

        self.t_ratio = 3/100
        self.explo_rate = 0.05
        
        self.reset()
        
    def reset(self):
        #Reset games
        self.games = [Game(None,None) for _ in range(self.batch_size)]
        self.game_length = self.games[0].length
        
        #Compute the ratio of steps that will be kept
        ratio = self.t_ratio
        minibatch_size = int(self.game_length*ratio)
        self.minibatch_size = minibatch_size

        #Select timesteps to avoid having too big batch (vary between games of the same batch)
        self.t_batch_list = [np.sort(np.random.choice(self.game_length,minibatch_size,replace=False)) for _ in range(self.batch_size)]
        self.t_batch = [{} for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for j,t in enumerate(self.t_batch_list[i]):
                self.t_batch[i][t] = j
        
        #Intialize data batches
        # - Minimized batches
        self.obs_batch = np.full((minibatch_size,self.batch_size,4),None,dtype=object)
        self.action_batch = np.full((minibatch_size,self.batch_size,4),None,dtype=object)
        # - Whole batches
        self.reward_batch = np.full((self.game_length,self.batch_size),None) #Need to keep all of them to compute outcome


        
    def step(self):
        
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

            #Uniformly select steps (to limit the size of the batch otherwise it cannot fit in memory)
            try:
                t_ind = self.t_batch[i][t]
                self.obs_batch[t_ind,i,0] = inp_sh_p0
                self.obs_batch[t_ind,i,1] = inp_sy_p0
                self.obs_batch[t_ind,i,2] = inp_sh_p1
                self.obs_batch[t_ind,i,3] = inp_sy_p1
            except KeyError:
                pass

        #Flush
        self.vbm.flush()

        #Apply actions
        for i,g in enumerate(self.games):
            
            actions = [actions_dict() for _ in range(2)]

            for k in range(2):
                #Ships
                ship_proba = out_sh[k][i]()
                
                for j,sh_action_proba in enumerate(ship_proba):
                    proba = sh_action_proba
                    if np.random.random() < self.explo_rate:
                        action = tf.random.categorical(tf.math.log(tf.ones((1,6))),1)[0,0]
                    else:
                        action = (tf.random.categorical(tf.math.log([proba]),1)[0,0])
                    actions[k][action.numpy()].append(g.players[k].ships[j])
                    try:
                        t_ind = self.t_batch[i][t]
                        if j == 0:
                            self.action_batch[t_ind,i,k*2] = [action]
                        else:
                            self.action_batch[t_ind,i,k*2].append(action)
                    except KeyError:
                        pass

                #Shipyard
                for j,sy_action_proba in enumerate(out_sy[k][i]()):
                    proba = sy_action_proba
                    if np.random.random() < self.explo_rate:
                        action = -tf.random.categorical(tf.math.log(tf.ones((1,2))),1)[0,0]-1
                    else:
                        action = -tf.random.categorical(tf.math.log([proba]),1)[0,0] -1
                    actions[k][action.numpy()].append(g.players[k].shipyards[j])
                    try:
                        t_ind = self.t_batch[i][t]
                        if j == 0:
                            self.action_batch[t_ind,i,k*2+1] = [action]
                        else:
                            self.action_batch[t_ind,i,k*2+1].append(action)
                    except KeyError:
                        pass

            #Step and compute reward
            reward = g._step(actions[0],actions[1])
            #Store reward
            self.reward_batch[t,i] = reward

        self.vbm.reset()

    def process_batch(self):
        #Preprocess inputs
        self.vbm.reset()
        self.lambda_proba_move = [[[None,None] for _ in range(self.batch_size)] for __ in range(self.minibatch_size)]
        for t in range(self.minibatch_size):
            for i in range(self.batch_size):
                for k in range(2):
                    sh_proba = self.vbm.request(self.obs_batch[t,i,2*k],0)
                    sy_proba = self.vbm.request(self.obs_batch[t,i,2*k+1],1)
                    def wrap():
                        #Closure
                        t2 = t
                        i2 = i
                        k2 = k
                        sh_proba2 = sh_proba
                        sy_proba2 = sy_proba
                        def prob():
                            shp = tf.ones((1,),dtype=tf.float64)
                            if not(self.action_batch[t2,i2,2*k2] is None):
                                sh_action_indexs = tf.convert_to_tensor(self.action_batch[t2,i2,2*k2]).numpy()
                                indices = tf.range(len(sh_action_indexs),dtype=tf.int64)
                                X,Y = tf.meshgrid(indices,sh_action_indexs)
                                indexes = tf.stack([X,Y],axis=-1)[0]%6
                                shp = sh_proba2()+EPS
                                shp = shp/tf.reduce_sum(shp)
                                #print("sh",t2,i2,k2,sh_proba2(),shp,sh_action_indexs,indexes)
                                    
                                shp = tf.gather_nd(shp,indexes)
                            syp = tf.ones((1,),dtype=tf.float64)
                            if not(self.action_batch[t2,i2,2*k2+1] is None):
                                sy_action_indexs = tf.convert_to_tensor(self.action_batch[t2,i2,2*k2+1]).numpy()
                                indices = tf.range(len(sy_action_indexs),dtype=tf.int64)
                                X,Y = tf.meshgrid(indices,sy_action_indexs)
                                indexes = tf.stack([X,Y],axis=-1)[0]%2
                                syp = sy_proba2()+EPS
                                syp = syp/tf.reduce_sum(syp)
                                #print("sy",t2,i2,k2,sy_proba2(),syp,sy_action_indexs,indexes)
                                syp = tf.gather_nd(syp,indexes)
                            out = tf.reduce_prod(shp)*tf.reduce_prod(syp)
                            return out
                        return prob
                    self.lambda_proba_move[t][i][k] = wrap()
        
        #Compute outcome
        reward = self.reward_batch.copy()
        gamma = 0.99
        self.gt = np.zeros((self.game_length,self.batch_size),dtype=np.float64)
        self.gt[self.game_length-1] = reward[self.game_length-1]
        for t in range(self.game_length-1):
            self.gt[self.game_length-t-2] = self.gt[self.game_length-t-1]*gamma+reward[self.game_length-t-2]

        self.reduced_gt = np.zeros((self.minibatch_size,self.batch_size))
        for i in range(self.batch_size):
            for t_ind,t in enumerate(self.t_batch_list[i]):
                self.reduced_gt[t_ind][i] = self.gt[t,i]

    def loss(self):
        self.vbm.flush()
        proba_move = [[ [None,None]  for _ in range(len(self.lambda_proba_move[i]))] for i in range(len(self.lambda_proba_move))]
        for t in range(self.minibatch_size):
            for i in range(self.batch_size):
                for k in range(2):
                    proba_move[t][i][k] = self.lambda_proba_move[t][i][k]()
        proba_move = tf.convert_to_tensor(proba_move)
        #find_nan(proba_move)
        #find_nan(self.reduced_gt)
        loss0 = -tf.math.reduce_sum(tf.math.log(proba_move[:,:,0])*self.reduced_gt)
        loss1 = -tf.math.reduce_sum(tf.math.log(proba_move[:,:,1])*(-self.reduced_gt))
        out = loss0#+loss1
        print("loss : ",loss0.numpy(),loss1.numpy(),np.sum(self.reward_batch))
        return out
            
    def fit(self,nb_epochs=5):
        self.process_batch()
        opt = tf.keras.optimizers.Adam()
        for _ in range(nb_epochs):
            opt.minimize(self.loss,self.vbm.parameters())
