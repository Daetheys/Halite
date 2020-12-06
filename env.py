from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

class Env:
    def __init__(self):
        pass
    def step(self,actions):
        pass
    def reset(self):
        pass

class HaliteEnv(Env):
    """ Halite Gym Env """
    def __init__(self,nb_agents): #nb players in the game
        self.nb_agents = nb_agents
        
    def step(self,actions):
        for (s,a) in actions:
            s.next_action = a
        self.board.next()
        return (self.board.observation,self.board.configuration)

    def reset(self):
        env = make("halite",debug="True")
        self.board = Board(env.reset(self.nb_agents)[0].observation,env.configuration)
        return (self.board.observation,self.board.configuration)

class HaliteVecEnv(Env):
    """ Vectorized Halite Gym Env """
    def __init__(self,nb_agents,size):
        self.envs = [Env(nb_agents) for i in range(size)]

    def step(self,lactions):
        lobs = []
        for i,actions in enumerate(lactions):
            obs = self.envs[i].step(actions)
            lobs.append(obs)
        return lobs

    def reset(self):
        lobs = []
        for e in self.envs:
            obs = e.reset()
            lobs.append(obs)
        return lobs
            
            
