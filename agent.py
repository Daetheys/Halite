from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
from tools import *

class Agent:
    def __init__(self):
        pass

    def get_actions(self,obs):
        pass

class QAgent(Agent):
    """  Agent with my idea for a Q learning policy """
    def __init__(self,model):
        self.model = model

    def step(self,obs,ship,actions, ACTIONS, get_actions):
        '''Choose a random action from a set for the ship according to the Q
        value distribution.
        '''
        #Obs is of shape [batch_size,*inp_shape]
        (obs,config) = obs
        board = Board(obs,config)
        actions = get_actions(board.current_player,ship,config) # Get possible shipyard actions
        actions_size = len(actions)
        Q = np.zeros((actions_size,))
        new_obs = [None]*actions_size
        for i,a in enumerate(actions):
            obs = simulate(board,ship,a) # Simulate move
            inps = get_input(obs,config) # Get inputs for nn
            q = self.model(inps) # Get v value
            Q[i] = q
            new_obs[i] = obs
        Q /= np.sum(Q) # Turn Q into a probability vector and sample next action from it
        action = np.random.choice(range(actions_size),p=Q)
        board = Board(new_obs[action],config) # Update board
        return (new_obs[action],config), actions[action]

    def conversion_step(self,obs,ship,actions):
        return self.step(obs,ship,actions,get_shipyard_actions)

    def conversion_step(self,obs,ship,actions)
        return self.step(obs,ship,actions,get_ship_actions)

    def get_actions(self,obs):
        actions = []
        for ship in player.ships: #1. SPAWNING
            obs, action = self.spawning_step(obs, ship, actions)
            actions.append(action)
        for ship in player.ships: #2. Conversion and 3. Movement
            obs, action = self.conversion_step(obs, ship, actions)
            actions.append(action)
        return actions
