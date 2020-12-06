from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import time
env = make("halite",debug=True)

def get_model():
    def wing():
        inp = Input(shape=(21,21,1))
        x = inp
        nb = 5
        x = Conv2D(nb,(3,3),padding='valid')(x)
        x = Conv2D(nb,(3,3),padding='valid')(x)
        return inp,x
    nb_wings = 5
    lwings = [wing() for i in range(nb_wings)]
    inps = [inp for (inp,x) in lwings]
    x = sum([x for (inp,x) in lwings])

    nb = 10
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)

    nb = 20
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(4,4),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)

    x = Reshape((nb,))(x)
    x = Dense(1)(x)

    m = tf.keras.Model(inputs=inps,outputs=x)
    m.compile(loss="mse",optimizer="Adam")
    #m.build((1,21,21,1))
    #m.summary()
    return m

def ship_array(l):
    arr = np.zeros((1,21,21,1))-1
    for s in l:
        x = s.position.x
        y = s.position.y
        arr[0,x,y,0] = s.halite
    return arr

def shipyard_array(l):
    arr = np.zeros((1,21,21,1))
    for s in l:
        x = s.position.x
        y = s.position.y
        arr[0,x,y,0] = 1
    return arr

SHIP_ACTIONS = [ShipAction.WEST,ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,None,ShipAction.CONVERT]
def get_ship_actions(p,s,config):
    #0:left 1:top 2:right 3:bottom 4:collect 5:yard
    actions = [0,1,2,3,4,5]
    if s.position.x == 0:
        actions.remove(0)
    if s.position.x == 20:
        actions.remove(2)
    if s.position.y == 0:
        actions.remove(1)
    if s.position.y == 20:
        actions.remove(3)
    if p.halite < config["convertCost"]:
        actions.remove(6)
    return actions

SHIPYARD_ACTIONS = [None,ShipyardAction.SPAWN]
def get_shipyard_actions(p,s,config):
    #0:pass 1:spawn
    actions = [0,1]
    if p.halite > config["spawnCost"]:
        actions.remove(1)
    return actions

def get_input(obs,config):
    board = Board(obs,config)
    ships = []
    shipyards = []
    players = [board.current_player]+board.opponents
    for p in players:
        ships.append(ship_array(p.ships))
        shipyards.append(shipyard_array(p.shipyards))
    halite = np.reshape(obs["halite"],(1,21,21,1))
    inps = ships + shipyards + [halite]
    return inps

def simulate(board,s,a):
    if isinstance(s,Ship):
        s.next_action = SHIP_ACTIONS[a]
        return board.next().observation
    if isinstance(s,Shipyard):
        s.next_action = SHIPYARD_ACTIONS[a]
        return board.next().observation

model = get_model()
def agent(obs,config):
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
            q = model(inps)
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
        sy.next_action = SHIPYARD_ACTIONS[action]
    #2. Conversion and 3. Movement
    for s in player.ships:
        actions = get_ship_actions(player,s,config)
        Q = np.zeros((6,))
        lobs = [None]*6
        for a in actions:
            obs = simulate(board,s,a)
            inps = get_input(obs,config)
            q = model(inps)
            Q[a] = q
            lobs[a] = obs
        Q /= np.sum(Q)
        action = np.random.choice(range(6),p=Q)
        board = Board(lobs[action],config)
        s.next_action = SHIP_ACTIONS[action]
    return player.next_actions
    
t = time.clock()
env.run([agent,agent])
print(time.clock()-t)
d = env.render(mode="html",width=800,height=600)
with open("render.html","w") as f:
    f.write(d)
plt.show()
