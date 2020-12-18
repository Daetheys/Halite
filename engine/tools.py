from player import ShipMove,ShipyardMove
_action_dict = {ShipyardMove.HOLD:[],
            ShipyardMove.SPAWN:[],
            ShipMove.MOVE:[],
            ShipMove.CONVERT:[],
            ShipMove.COLLECT:[]}
def actions_dict():
    return _action_dict.copy()
