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
        #Obs is of shape [batch_size,*inp_shape]
        action_size = len(ACTIONS.keys())
        (obs,config) = obs
        board = Board(obs,config)
        actions = get_actions(player,ship,config) # Get possible shipyard actions
        Q = np.zeros((action_size,))
        new_obs = [None]*action_size
        for a in actions:
            obs = simulate(board,ship,a) # Simulate move
            inps = get_input(obs,config) # Get inputs for nn
            q = self.model(inps) # Get v value
            Q[a] = q
            new_obs[a] = obs
        Q /= np.sum(Q) # Turn Q into a probability vector and sample next action from it
        action = np.random.choice(range(action_size),p=Q)
        board = Board(new_obs[action],config) # Update board
        return (new_obs[action],config), ACTIONS[action]

    def conversion_step(self,obs,ship,actions):
        return self.step(obs,ship,actions,SHIPYARD_ACTIONS,get_shipyard_actions)

    def conversion_step(self,obs,ship,actions)
        return self.step(obs,ship,actions,SHIP_ACTIONS,get_ship_actions)

    def get_actions(self,obs):
        actions = []
        #Compute Q values
        player = board.current_player
        #1. SPAWNING
        for ship in player.ships:
            obs, action = self.spawning_step(obs, ship, actions)
            actions.append(action)
        #2. Conversion and 3. Movement
        for ship in player.ships:
            obs, action = self.conversion_step(obs, ship, actions)
            actions.append(action)
        return actions
