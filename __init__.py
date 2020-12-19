from engine.game import Game

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

class HaliteEnv:
    def __init__(self,agent1,agent2):
        self.game = Game(agent1,agent2)
        
    def step(self,actions):
        self.game.step()
