from engine.game import Game
from tools import get_nn_input
from engine.tools import *
from tools import *
import numpy as np
import tensorflow as tf
import time
from bot import LearnerBot
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
    def __init__(self,bot1,bot2,vbm,batch_size=5):
        self.batch_size = batch_size
        #self.bots = [bot1,bot2]
        self.vbm = vbm

        self.explo_rate = 0.05

        self.gamma = 0.99

        self.bots = [(bot1(2*i),bot2(2*i+1)) for i in range(self.batch_size)]
        
        self.reset()

        self.halite_list = []

    def save(self):
        self.vbm.save()
        
    def reset(self):
        #Reset games
        
        self.games = [Game(*self.bots[i]) for i in range(self.batch_size)]
        self.game_length = self.games[0].length
        
        self.vbm.reset()

        self.rewards = np.zeros(self.batch_size)

    def save(self,prefix):
        self.vbm.save(prefix)

    def learn(self,nb,prefix):
        for n in range(nb):
            print(n)
            for i in range(self.game_length):
                self.step()
            self.fit()
            for g in self.games:
                g.players[0].agent.explo_rate = 1/(n+1)
                g.players[1].agent.explo_rate = 1/(n+1)
            self.reset()
            self.save(prefix)
        print(self.halite_list)

    def step(self):
        
        t = self.games[0].nb_step
        
        probas = [[None for __ in range(self.batch_size)] for _ in range(2)]

        #Precompute actions
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
            actions_list0 = g.players[0].agent.sample_actions(probas[0][i])
            actions0 = compute_action_dict(actions_list0,g.players[0])
            actions_list1 = g.players[1].agent.sample_actions(probas[1][i])
            actions1 = compute_action_dict(actions_list1,g.players[1])

            #Step and compute reward
            reward = g._step(actions0,actions1)
            #Store reward
            self.rewards[i] = reward
        self.rewards = (self.rewards-self.rewards.mean())/self.rewards.std()
        for i,g in enumerate(self.games):
            g.players[0].agent.add_reward(self.rewards[i])
            if isinstance(g.players[1].agent,LearnerBot):
                g.players[1].agent.add_reward(-self.rewards[i])

        #End game
        if t == self.game_length-1:
            halite0 = [self.games[i].players[0].halite for i in range(len(self.games))]
            halite1 = [self.games[i].players[1].halite for i in range(len(self.games))]
            print("mean halite p0",np.mean(halite0),np.std(halite0),np.max(halite0),np.min(halite0))
            print("mean halite p1",np.mean(halite1),np.std(halite1),np.max(halite1),np.min(halite1))
            self.halite_list.append(np.mean(halite0))
            #Normalize rewards
            

        self.vbm.reset()

    def loss(self):
        self.vbm.reset()
        for g in self.games:
            g.players[0].agent.loss_precompute()
            if isinstance(g.players[1].agent,LearnerBot):
                g.players[1].agent.loss_precompute()
        self.vbm.flush()
        l = tf.reduce_mean([g.players[0].agent.loss_compute(self.rewards[i]) for i,g in enumerate(self.games)])
        if isinstance(self.bots[0][1],LearnerBot):
            l += tf.reduce_mean([g.players[1].agent.loss_compute(-self.rewards[i]) for i,g in enumerate(self.games)])
        print("-----------------loss : ",l.numpy())
        assert not(tf.math.is_nan(l))
        return l
            
    def fit(self,nb_epochs=7):
        opt = tf.keras.optimizers.Adam(10**-3) #Reset Adam momentums between fits
        for _ in range(nb_epochs):
            #with tf.GradientTape() as tape:
            #    loss = self.loss()
            #    grad = tape.gradient(loss,self.vbm.parameters())
            #    grad_clipped = [tf.clip_by_value(g,-10**-2,10**-2) for g in grad]
            #opt.apply_gradients(zip(grad_clipped,self.vbm.parameters()))
            opt.minimize(self.loss,self.vbm.parameters())
