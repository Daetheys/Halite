from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

class Env:
    def __init__(self):
        pass
    def step(self,actions):
        pass
    def reset(self):
        pass

CONFIG = {"actTimeout":2,
          "agentExec":"PROCESS",
          "agentTimeout":10,
          "collectRate":0.25,
          "convertCost":500,
          "episodeSteps":400,
          "halite":24000,
          "moveCost":0,
          "regenRate":0.02,
          "runTimeout":600,
          "size":21,
          "spawnCost":500}

class HaliteEnv(Env):
    """ Halite Gym Env """
    def __init__(self,nb_agents): #nb players in the game
        self.nb_agents = nb_agents
        self.kenv = make("halite",debug="True")
        self.reset()
        
    def step(self,actions):
        for (s,a) in actions:
            s.next_action = a
        self.board.next()
        return (self.board.observation,self.board.configuration)

    def reset(self):
        out = self.kenv.reset(self.nb_agents)[0]["observation"]
        return out
        
class HaliteVecEnv(Env):
    """ Vectorized Halite Gym Env """
    def __init__(self,nb_agents,size):
        self.envs = [HaliteEnv(nb_agents) for i in range(size)]
        self.config = self.envs[0].kenv.configuration

    def step(self,lactions):
        lobs = []
        for i,actions in enumerate(lactions):
            obs = self.envs[i].step(actions)
            lobs.append(obs)
        return lobs

    def stepi(self,action,i):
        return self.envs[i].step(action)

    def reseti(self,i):
        obs = self.envs[i].reset()
        return obs

    def reset(self):
        lobs = []
        for i in range(len(self.envs)):
            obs = self.reseti(i)
            lobs.append(obs)
        return lobs
            
            
