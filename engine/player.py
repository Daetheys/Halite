class Player:
    def __init__(self,index,agent,root):
        self.root = root

        self.index = index
        self.agent = agent
        
        self.ships = []
        self.ship_array = [[[] for _ in range(21)] for _ in range(21)]

        self.shipyards = []
        self.shipyard_array = [[0 for _ in range(21)] for _ in range(21)]

        self.halite = 5000

        init_ship = Ship(index*10+5,10,self)
        self.add_ship(init_ship)

    def compute_actions(self):
        return self.agent(self)
        
    def add_ship(self,ship):
        assert ship.root == self
        self.ships.append(ship)
        self.ship_array[ship.y][ship.x].append(ship)
        
    def remove_ship(self,ship):
        try:
            self.ships.remove(ship)
        except ValueError:
            pass
        self.ship_array[ship.y][ship.x].remove(ship)

    def add_shipyard(self,shipyard):
        self.shipyards.append(shipyard)
        self.shipyard_array[shipyard.y][shipyard.x] = shipyard
        
    def remove_shipyard(self,shipyard):
        try:
            self.shipyards.remove(shipyard)
        except ValueError:
            pass
        self.shipyard_array[shipyard.y][shipyard.x] = 0

class Shipyard:
    def __init__(self,x,y,root):
        self.root = root
        self.x = x
        self.y = y
        self.alive = True

    def spawn(self):
        if self.root.halite >= 500:
            self.root.halite -= 500
            ship = Ship(self.x,self.y,self.root)
            self.root.add_ship(ship)

    def remove(self):
        self.alive = False
        self.root.remove_shipyard(self)

class Ship:
    def __init__(self,x,y,root):
        self.root = root
        self.alive = True
        self.x = x
        self.y = y
        self.halite = 0

    def remove(self):
        self.alive = False
        self.root.remove_ship(self)

    def deposit(self):
        self.root.halite += self.halite
        self.halite = 0
        
    def move(self,m):
        self.root.ship_array[self.y][self.x].remove(self)
        if m == ShipMove.LEFT:
            self.x = (self.x-1)%21
        elif m == ShipMove.DOWN:
            self.y = (self.y-1)%21
        elif m == ShipMove.RIGHT:
            self.x = (self.x+1)%21
        elif m == ShipMove.UP:
            self.y = (self.y-1)%21
        else:
            print(m,m.__class__)
            assert False #unknown move
        self.root.ship_array[self.y][self.x].append(self)
            
    def convert(self):
        shipyard_b = self.root.root.players[0].shipyard_array[self.y][self.x] or self.root.root.players[1].shipyard_array[self.y][self.x]
        if self.root.halite+self.halite >= 500 and not(shipyard_b):
            #Pay halite and add remaining positive / negative halite to the player
            self.halite -= 500
            self.root.halite += self.halite
            #Remove ship
            self.remove()
            #Add shipyard
            shipyard = Shipyard(self.x,self.y,self.root)
            self.root.add_shipyard(shipyard)
            #Destroy underneath halite
            self.root.root.set_halite(self.x,self.y,0)

    def collect(self):
        #Get harvested value of halite
        v = self.root.root.get_halite(self.x,self.y)
        #Remove it from halite array
        self.root.root.set_halite(self.x,self.y,v*0.75)
        #Add it to ship halite
        self.halite += v*0.25
            

from enum import IntEnum
class ShipMove(IntEnum):
    MOVE = -1
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    CONVERT = 4
    COLLECT = 5

class ShipyardMove(IntEnum):
    HOLD = 0
    SPAWN = 1
