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
    def __init__(self,model,env):
        self.model = model
        self.batch_size = 100
        self.env = env

        self.steps = [0]*self.batch_size
        self.ship_ids = [0]*self.batch_size
        self.shipyard_ids = [0]*self.batch_size

        self.obs = self.env.reset()
        self.config = self.env.config

    def step(self,obs,boards,ships):
        '''Choose a random action from a set for the ship according to the Q
        value distribution.
        '''
        assert len(obs) == self.batch_size
        assert len(ships) == self.batch_size
        #Obs is of shape [batch_size,*inp_shape]
        obs = obs
        #boards = [Board(o,self.config) for o in obs]
        lobs = [[]]
        inps = []
        for i in range(self.batch_size):
            lobs.append([])
            action_space = SHIP_ACTIONS if isinstance(ships[i],Ship) else SHIPYARD_ACTIONS
            for a in action_space:
                obs = simulate(boards[i],ships[i],a) # Simulate move
                lobs[i].append(obs)
                inp = get_input(obs,self.config) # Get inputs for nn
                inps.append(inp)
        inps = np.array(inps)
        print("computing Q",inps.shape)
        Q = self.model(inps)
        print("--",Q.shape)
        print("computed Q")
        Q2 = []
        buf = 0
        for i in range(self.batch_size):
            Q2.append(Q[buf:buf+len(lobs[i]),0].numpy().tolist())
            buf += len(lobs[i])
        Q = [Q[i]/np.sum(Q[i]) for i in range(self.batch_size)]
        actions = [np.random.choice(range(len(Q[i])),p=Q[i]) for i in range(self.batch_size)]
        obs = [lobs[i][actions[i]] for i in range(self.batch_size)]
        action_tuple = [(ships[i],actions[i]) for i in range(self.batch_size)]
        return obs, action_tuple

    def minimal_step(self):
        obs = self.obs
        ship_array = []
        boards = []
        for i in range(self.batch_size):
            while True: #Will only break when a new ship/shipyard is added to the ship_array list. If their isn't any ship nor shipyard available for a player this might loop infinitely (assert is here for that)
                board = Board(obs[i],self.config)
                boards.append(board)
                shipyards = board.current_player.shipyards
                ships = board.current_player.ships
                assert len(shipyards) + len(ships) >= 1
                if not(self.steps[i]): #Shipyard
                    if self.shipyard_ids[i] >= len(shipyards):
                        self.steps[i] = 1
                    else:
                        ship_array.append(shipyards[self.shipyard_ids[i]])
                        break
                if self.steps[i]:
                    if self.ship_ids[i] >= len(ships):
                        self.steps[i] = 0
                        self.shipyard_ids = 0
                        self.ship_ids = 0
                        #All ships have been computed -> step the env to prepare for next move
                        obs,_,_,_ = self.env.step(self.actions[i])
                        self.obs[i] = obs
                    else:
                        ship_array.append(ships[self.ship_ids[i]])
                        break

        #Compute actions
        obs,self.actions = self.step(obs,boards,ship_array)
        
