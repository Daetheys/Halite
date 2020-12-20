from engine.player import ShipMove,ShipyardMove
import copy

_action_dict = {ShipyardMove.HOLD:[],
                ShipyardMove.SPAWN:[],
                ShipMove.LEFT:[],
                ShipMove.DOWN:[],
                ShipMove.RIGHT:[],
                ShipMove.UP:[],
                ShipMove.CONVERT:[],
                ShipMove.COLLECT:[]}

def actions_dict():
    return copy.deepcopy(_action_dict)

ship_actions = [ShipMove.LEFT,ShipMove.DOWN,ShipMove.RIGHT,ShipMove.UP,ShipMove.CONVERT,ShipMove.COLLECT]

ship_move_actions = [ShipMove.LEFT,ShipMove.DOWN,ShipMove.RIGHT,ShipMove.UP]

shipyard_actions = [ShipyardMove.HOLD,ShipyardMove.SPAWN]
