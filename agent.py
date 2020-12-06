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

    def get_actions(self,obs):
        #Obs is of shape [batch_size,*inp_shape]
        (obs,config) = obs
        action = []
        board = Board(obs,config)
        #Compute Q values
        player = board.current_player
        #1. SPAWNING
        for sy in player.ships:
            #Get actions for the ship
            actions = get_shipyard_actions(player,sy,config)
            Q = np.zeros((2,))
            lobs = [None]*2
            for a in actions:
                #Simulate move
                obs = simulate(board,sy,a)
                #Get inputs for nn
                inps = get_input(obs,config)
                #get v value
                q = self.model(inps)
                #store v value and observation
                Q[a] = q
                lobs[a] = obs
            #Compute proba off of v values
            Q /= np.sum(Q)
            #Sample next action
            action = np.random.choice(range(2),p=Q)
            #Get new board with results of the action
            board = Board(lobs[action],config)
            #Set action to the ship
            actions.append((sy,SHIPYARD_ACTIONS[action]))
        #2. Conversion and 3. Movement
        for s in player.ships:
            actions = get_ship_actions(player,s,config)
            Q = np.zeros((6,))
            lobs = [None]*6
            for a in actions:
                obs = simulate(board,s,a)
                inps = get_input(obs,config)
                q = self.model(inps)
                Q[a] = q
                lobs[a] = obs
            Q /= np.sum(Q)
            action = np.random.choice(range(6),p=Q)
            board = Board(lobs[action],config)
            actions.append((sy,SHIPYARD_ACTIONS[action]))
        return actions
