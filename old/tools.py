from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *
import numpy as np

def ship_array(l):
    arr = np.zeros((21,21,1))-1
    for s in l:
        x = s.position.x
        y = s.position.y
        arr[x,y,0] = s.halite
    return arr

def shipyard_array(l):
    arr = np.zeros((21,21,1))
    for s in l:
        x = s.position.x
        y = s.position.y
        arr[x,y,0] = 1
    return arr

SHIP_ACTIONS = [ShipAction.WEST,ShipAction.NORTH,ShipAction.EAST,ShipAction.SOUTH,None,ShipAction.CONVERT]
def get_ship_actions(p,config):
    #0:left 1:top 2:right 3:bottom 4:collect 5:yard
    if p.halite < config["convertCost"]:
        return SHIP_ACTIONS[:-1]
    return SHIP_ACTIONS

SHIPYARD_ACTIONS = [None,ShipyardAction.SPAWN]
def get_shipyard_actions(p,config):
    #0:pass 1:spawn
    if p.halite < config["spawnCost"]:
        return SHIPYARD_ACTIONS[:-1]
    return SHIPYARD_ACTIONS

def get_input(obs,config):
    board = Board(obs,config)
    ships = []
    shipyards = []
    players = [board.current_player]+board.opponents
    for p in players:
        ships.append(ship_array(p.ships))
        shipyards.append(shipyard_array(p.shipyards))
    halite = np.reshape(obs["halite"],(21,21,1))
    ships = np.array(ships)
    shipyards = np.array(shipyards)
    inps = np.concatenate([ships,shipyards,[halite]],axis=0)
    return inps

def simulate(board,s,a):
    if isinstance(s,Ship):
        s.next_action = a
        return board.next().observation
    if isinstance(s,Shipyard):
        s.next_action = a
        return board.next().observation
