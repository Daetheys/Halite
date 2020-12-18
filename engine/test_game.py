from game import Game
from player import Player,Ship,Shipyard,ShipMove,ShipyardMove
from tools import *

def wait_agent(p):
    return actions_dict()

def test_init():
    game = Game(wait_agent,wait_agent)
    
    players = game.players
    
    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array

    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array

    assert len(players) == 2
    assert len(ships0) == 1
    assert len(shipyards0) == 0
    assert len(ships1) == 1
    assert len(shipyards1) == 0

    s0 = ships0[0]
    assert s0.x == 5 and s0.y == 10
    assert ship_array0[10][5] == [s0]

    s1 = ships1[0]
    assert s1.x == 15 and s1.y == 10
    assert ship_array1[10][15] == [s1]
    
def test_step():
    game = Game(wait_agent,wait_agent)
    game.step()

    players = game.players
    
    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array

    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array

    assert len(players) == 2
    assert len(ships0) == 1
    assert len(shipyards0) == 0
    assert len(ships1) == 1
    assert len(shipyards1) == 0

    s0 = ships0[0]
    assert s0.x == 5 and s0.y == 10
    assert ship_array0[10][5] == [s0]

    s1 = ships1[0]
    assert s1.x == 15 and s1.y == 10
    assert ship_array1[10][15] == [s1]

def test_move():
    game = Game(wait_agent,wait_agent)
    s = game.players[0].ships[0]
    
    for i in range(5):
        s.move(ShipMove.LEFT)

        players = game.players

        ships0 = players[0].ships
        ship_array0 = players[0].ship_array
        shipyards0 = players[0].shipyards
        shipyard_array0 = players[0].shipyard_array

        ships1 = players[1].ships
        ship_array1 = players[1].ship_array
        shipyards1 = players[1].shipyards
        shipyard_array1 = players[1].shipyard_array

        assert len(players) == 2
        assert len(ships0) == 1
        assert len(shipyards0) == 0
        assert len(ships1) == 1
        assert len(shipyards1) == 0

        s0 = ships0[0]
        assert s0.x == 5-i-1 and s0.y == 10
        assert ship_array0[10][5-i-1] == [s0]
        assert ship_array0[10][5] == []

        s1 = ships1[0]
        assert s1.x == 15 and s1.y == 10
        assert ship_array1[10][15] == [s1]

    s.move(ShipMove.LEFT)

    players = game.players

    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array
    
    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array
    
    assert len(players) == 2
    assert len(ships0) == 1
    assert len(shipyards0) == 0
    assert len(ships1) == 1
    assert len(shipyards1) == 0
    
    s0 = ships0[0]
    assert s0.x == 20 and s0.y == 10
    assert ship_array0[10][20] == [s0]
    assert ship_array0[10][5] == []
    
    s1 = ships1[0]
    assert s1.x == 15 and s1.y == 10
    assert ship_array1[10][15] == [s1]

def test_add_ship():
    game = Game(wait_agent,wait_agent)
    s = Ship(0,0,game.players[0])

    game.players[0].add_ship(s)

    assert len(game.players[0].ships) == 2
    assert game.players[0].ship_array[0][0] != []
    
def test_remove_ship():
    game = Game(wait_agent,wait_agent)
    s = game.players[0].ships[0]

    s.remove()

    assert len(game.players[0].ships) == 0
    assert game.players[0].ship_array[10][5] == []

def test_convert1():
    game = Game(wait_agent,wait_agent)
    s = game.players[0].ships[0]
    
    assert game.players[0].halite == 5000
    assert s.halite == 0
    
    s.convert()

    assert game.players[0].halite == 4500
    
    players = game.players

    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array
    
    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array

    assert len(ships0) == 0
    assert len(ships1) == 1

    assert ship_array0[10][5] == []
    assert ship_array1[10][15] != []

    assert len(shipyards0) == 1
    assert len(shipyards1) == 0

    assert shipyard_array0[10][5] != 0
    assert shipyard_array1[10][15] == 0

def test_convert2():
    game = Game(wait_agent,wait_agent)
    s = game.players[0].ships[0]
    s.halite = 200
    
    assert game.players[0].halite == 5000
    
    s.convert()

    assert game.players[0].halite == 4700
    
    players = game.players

    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array
    
    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array

    assert len(ships0) == 0
    assert len(ships1) == 1

    assert ship_array0[10][5] == []
    assert ship_array1[10][15] != []

    assert len(shipyards0) == 1
    assert len(shipyards1) == 0

    assert shipyard_array0[10][5] != 0
    assert shipyard_array1[10][15] == 0

def test_convert3():
    game = Game(wait_agent,wait_agent)
    s = game.players[0].ships[0]

    game.players[0].halite = 499
    
    assert s.halite == 0
    
    s.convert()

    assert game.players[0].halite == 499
    
    players = game.players

    ships0 = players[0].ships
    ship_array0 = players[0].ship_array
    shipyards0 = players[0].shipyards
    shipyard_array0 = players[0].shipyard_array
    
    ships1 = players[1].ships
    ship_array1 = players[1].ship_array
    shipyards1 = players[1].shipyards
    shipyard_array1 = players[1].shipyard_array

    assert len(ships0) == 1
    assert len(ships1) == 1

    assert ship_array0[10][5] != []
    assert ship_array1[10][15] != []

    assert len(shipyards0) == 0
    assert len(shipyards1) == 0

    assert shipyard_array0[10][5] == 0
    assert shipyard_array1[10][15] == 0

def test_collect():
    game = Game(wait_agent,wait_agent)

    s = game.players[0].ships[0]

    game.halite[10,5] = 40

    assert game.players[0].halite == 5000

    s.collect()

    assert game.get_halite(5,10) == 30
    assert s.halite == 10
    assert game.players[0].halite == 5000

def test_deposit():
    game = Game(wait_agent,wait_agent)

    s = game.players[0].ships[0]

    s.halite = 50

    sy = Shipyard(5,10,game.players[0])
    game.players[0].add_shipyard(sy)

    assert game.players[0].halite == 5000

    s.deposit()

    assert s.halite == 0
    assert game.players[0].halite == 5050

def test_ship_collision1():

    game = Game(wait_agent,wait_agent)
    s1 = Ship(10,11,game.players[0])
    s1.halite = 10
    s2 = Ship(10,11,game.players[0])
    s2.halite = 20
    game.players[0].add_ship(s1)
    game.players[0].add_ship(s2)

    assert len(game.players[0].ships) == 3
    assert len(game.players[0].ship_array[11][10]) == 2

    game.ship_collisions()

    assert len(game.players[0].ships) == 2
    assert len(game.players[0].ship_array[11][10]) == 1
    assert game.players[0].ship_array[11][10][0] == s1
    assert s1.halite == 30

def test_ship_collision2():

    game = Game(wait_agent,wait_agent)
    s1 = Ship(10,11,game.players[0])
    s1.halite = 10
    s2 = Ship(10,11,game.players[1])
    s2.halite = 20
    game.players[0].add_ship(s1)
    game.players[1].add_ship(s2)

    assert len(game.players[0].ships) == 2
    assert len(game.players[0].ship_array[11][10]) == 1

    assert len(game.players[1].ships) == 2
    assert len(game.players[1].ship_array[11][10]) == 1

    game.ship_collisions()

    assert len(game.players[0].ships) == 2
    assert len(game.players[0].ship_array[11][10]) == 1
    assert game.players[0].ship_array[11][10][0] == s1

    assert len(game.players[1].ships) == 1
    assert len(game.players[1].ship_array[11][10]) == 0

    assert s1.halite == 30

def test_ship_collision3():

    game = Game(wait_agent,wait_agent)
    s1 = Ship(10,11,game.players[0])
    s1.halite = 10
    s2 = Ship(10,11,game.players[0])
    s2.halite = 10
    game.players[0].add_ship(s1)
    game.players[0].add_ship(s2)

    assert len(game.players[0].ships) == 3
    assert len(game.players[0].ship_array[11][10]) == 2

    assert game.get_halite(s1.x,s1.y) == 0

    game.ship_collisions()

    assert len(game.players[0].ships) == 1
    assert len(game.players[0].ship_array[11][10]) == 0

    assert game.get_halite(s1.x,s1.y) == 20

def test_ship_collision4():

    game = Game(wait_agent,wait_agent)
    s1 = Ship(10,11,game.players[0])
    s1.halite = 10
    s2 = Ship(10,11,game.players[1])
    s2.halite = 10
    game.players[0].add_ship(s1)
    game.players[1].add_ship(s2)

    assert len(game.players[0].ships) == 2
    assert len(game.players[0].ship_array[11][10]) == 1

    assert len(game.players[1].ships) == 2
    assert len(game.players[1].ship_array[11][10]) == 1

    assert game.get_halite(s1.x,s1.y) == 0
    
    game.ship_collisions()

    assert len(game.players[0].ships) == 1
    assert len(game.players[0].ship_array[11][10]) == 0

    assert len(game.players[1].ships) == 1
    assert len(game.players[1].ship_array[11][10]) == 0

    assert game.get_halite(s1.x,s1.y) == 20

def test_shipyard_collision1():
    game = Game(wait_agent,wait_agent)

    sy = Shipyard(15,10,game.players[0])
    game.players[0].add_shipyard(sy)

    game.shipyard_collisions()

    assert len(game.players[0].shipyards) == 0
    assert game.players[0].shipyard_array[10][15] == 0

def test_shipyard_collision2():
    game = Game(wait_agent,wait_agent)

    sy = Shipyard(5,10,game.players[1])
    game.players[1].add_shipyard(sy)

    game.shipyard_collisions()

    assert len(game.players[1].shipyards) == 0
    assert game.players[1].shipyard_array[10][5] == 0

def test_halite_regeneration():
    game = Game(wait_agent,wait_agent)

    game.set_halite(0,0,100)
    assert game.get_halite(0,0) == 100

    game.halite_regeneration()
    
    assert game.get_halite(0,0) == 102
    assert game.get_halite(0,1) == 0

def test__step():
    game = Game(wait_agent,wait_agent)
    
    p0 = game.players[0]
    p1 = game.players[1]
    
    s1 = Ship(15,15,p0) #collect
    game.set_halite(15,15,1)
    s1.halite = 10
    p0.add_ship(s1)
    s2 = Ship(16,15,p1) #move left
    s2.halite = 10
    p1.add_ship(s2)

    s3 = Ship(9,7,p0) #move up
    p0.add_ship(s3)

    s4 = Ship(10,11,p1) #collect
    game.set_halite(10,11,100)
    p1.add_ship(s4)

    s5 = Ship(16,9,p0) #move right and deposit
    s5.halite = 20
    p0.add_ship(s5)

    s6 = Ship(7,7,p1) #convert
    game.set_halite(7,7,20)
    s6.halite = 700
    p1.add_ship(s6)

    s7 = Ship(7,8,p0) #Move up
    p0.add_ship(s7)

    sy1 = Shipyard(17,9,p0)
    p0.add_shipyard(sy1)

    sy2 = Shipyard(4,5,p1) #spawn
    p1.add_shipyard(sy2)

    actions0 = actions_dict()
    actions0[ShipMove.MOVE] = [(s3,ShipMove.UP),(s5,ShipMove.RIGHT),(s7,ShipMove.UP)]
    actions0[ShipMove.COLLECT] = [s1]

    actions1 = actions_dict()
    actions1[ShipyardMove.SPAWN] = [sy2]
    actions1[ShipMove.CONVERT] = [s6]
    actions1[ShipMove.MOVE] = [(s2,ShipMove.LEFT)]
    actions1[ShipMove.COLLECT] = [s4]
    
    game._step(actions0,actions1)

    assert p0.ship_array[15][15] == []
    assert p1.ship_array[15][15] == []
    assert game.get_halite(15,15) == 21*1.02

    assert p0.ship_array[7][9] == []
    assert p0.ship_array[6][9] == [s3]

    assert s4.halite == 25
    assert game.get_halite(10,11) == 75*1.02

    assert s5.halite == 0
    assert p0.halite == 5020

    assert p1.halite == 5000 - 500 + 200
    assert p1.shipyard_array[7][7] == 0
    assert p0.ship_array[7][7] == [s7]
    assert game.get_halite(7,7) == 0

    assert p1.ship_array[5][4] != []
