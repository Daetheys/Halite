from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np

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
