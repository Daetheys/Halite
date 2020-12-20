import numpy as np
from engine.player import ShipMove,ShipyardMove
from engine.game import Game

def get_nn_input(g):
    inp = np.zeros((21,21,7))
    halite = g.get_full_halite().reshape((21,21))
    inp[:,:,0] = halite
    for sh in g.players[0].ships:
        inp[sh.y,sh.x,1] = 1
        inp[sh.y,sh.x,2] = sh.halite
    for sy in g.players[0].shipyards:
        inp[sy.y,sy.x,3] = 1
    for sh in g.players[1].ships:
        inp[sh.y,sh.x,4] = 1
        inp[sh.y,sh.x,5] = sh.halite
    for sy in g.players[1].shipyards:
        inp[sy.y,sy.x,6] = 1
        
    inp.reshape((1,21,21,7))
    
    inp_sh0 = np.zeros((len(g.players[0].ships),21,21,7))
    inp_sy0 = np.zeros((len(g.players[0].shipyards),21,21,7))
    inp_sh1 = np.zeros((len(g.players[1].ships),21,21,7))
    inp_sy1 = np.zeros((len(g.players[1].shipyards),21,21,7))
    
    for i,s in enumerate(g.players[0].ships):
        inp_sh0[i] = inp.copy()
        inp_sh0[i,s.y,s.x,1] = -1
    for i,s in enumerate(g.players[0].shipyards):
        inp_sy0[i] = inp.copy()
        inp_sy0[i,s.y,s.x,3] = -1
        
    inp = inp[:,:,[0,4,5,6,1,2,3]]
    for i,s in enumerate(g.players[1].ships):
        inp_sh1[i] = inp.copy()
        inp_sh1[i,s.y,s.x,1] = -1
    for i,s in enumerate(g.players[1].shipyards):
        inp_sy1[i] = inp.copy()
        inp_sy1[i,s.y,s.x,3] = -1
    return inp_sh0,inp_sy0,inp_sh1,inp_sy1

def choice(a,p):
    assert a.shape[1] == p.shape[1]
    b = np.zeros((a.shape[0],))
    for i in range(a.shape[0]):
        b[i] = np.random.choice(a,1,p=p[i])

