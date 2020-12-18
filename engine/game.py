import numpy as np
from player import Player
from tools import *
import time

class Game:
    def __init__(self,agent1,agent2):
        self.init_halite()
        self.players = [Player(0,agent1,self),Player(1,agent2,self)]
        self.current_player = 0
        self.nb_step = 0
        
    def init_halite(self):
        self.halite = np.zeros((21,21))
        self.halite_mult = 1
    
    def get_halite(self,x,y):
        return self.halite[y,x]*self.halite_mult

    def set_halite(self,x,y,v):
        self.halite[y,x] = v/self.halite_mult

    def step(self):
        self._step(self.players[0].compute_actions(),self.players[1].compute_actions())

    def _step(self,actions1,actions2):
        #Step 1 - Spawn
        for shipyard in actions1[ShipyardMove.SPAWN]+actions2[ShipyardMove.SPAWN]:
            shipyard.spawn()
        #Step 2 - Conversion
        for ship in actions1[ShipMove.CONVERT]+actions2[ShipMove.CONVERT]:
            ship.convert()
        #Step 3 - Movement
        for (ship,m) in actions1[ShipMove.MOVE]+actions2[ShipMove.MOVE]:
            ship.move(m)
        #Step 4 - Ship collisions
        self.ship_collisions()
        #Step 5 - Shipyard collisions
        self.shipyard_collisions()
        #Step 6 - Halite deposit
        self.halite_deposit()
        #Step 7 - Collect
        for ship in actions1[ShipMove.COLLECT]+actions2[ShipMove.COLLECT]:
            if ship.alive:
                ship.collect()
        #Step 8 - Halite Regeneration
        self.halite_regeneration()
        #Step 9 - End
        self.nb_step += 1

    def halite_regeneration(self):
        self.halite_mult *= 1.02

    def halite_deposit(self):
        for sy in self.players[0].shipyards + self.players[1].shipyards:
            s = sy.root.ship_array[sy.y][sy.x]
            try:
                s[0].deposit()
            except IndexError:
                pass

    def ship_collisions(self):
        j = 0
        while True:
            try:
                s = (self.players[0].ships + self.players[1].ships)[j]
            except IndexError:
                break
            ships = self.players[0].ship_array[s.y][s.x] + self.players[1].ship_array[s.y][s.x]
            if len(ships) == 1:
                pass
            else:
                halite = [s2.halite for s2 in ships]
                indexs = np.argsort(halite)
                if halite[indexs[0]] == halite[indexs[1]]:
                    h = 0
                    for s2 in ships:
                        h += s2.halite
                        s2.remove()
                    print(h,self.get_halite(s.x,s.y))
                    self.set_halite(s.x,s.y,self.get_halite(s.x,s.y)+h)
                else:
                    for i in indexs[1:]:
                        ships[i].remove()
                        ships[indexs[0]].halite += ships[i].halite
            j += 1

    def shipyard_collisions(self):
        for k in range(2):
            for sy in self.players[k].shipyards:
                if self.players[1-k].ship_array[sy.y][sy.x] != []:
                    sy.remove()
                                                        
                    
