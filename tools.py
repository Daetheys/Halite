import numpy as np
import copy
from engine.player import ShipMove,ShipyardMove
from engine.game import Game
from engine.tools import actions_dict


def get_nn_input(g):
    inp = np.zeros((21,21,10))
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
        
    inp.reshape((1,21,21,10))
    
    inp_sh0 = np.zeros((len(g.players[0].ships),21,21,10),dtype=np.float32)
    inp_sy0 = np.zeros((len(g.players[0].shipyards),21,21,10),dtype=np.float32)
    inp_sh1 = np.zeros((len(g.players[1].ships),21,21,10),dtype=np.float32)
    inp_sy1 = np.zeros((len(g.players[1].shipyards),21,21,10),dtype=np.float32)
    
    for i,s in enumerate(g.players[0].ships):
        inp_sh0[i] = inp.copy()
        inp_sh0[i,s.y,s.x,7] = 1*(g.nb_step+1)
    for i,s in enumerate(g.players[0].shipyards):
        inp_sy0[i] = inp.copy()
        inp_sy0[i,s.y,s.x,8] = 1*(g.nb_step+1)
        
    inp = inp[:,:,[0,4,5,6,1,2,3,7,8,9]]
    for i,s in enumerate(g.players[1].ships):
        inp_sh1[i] = inp.copy()
        inp_sh1[i,s.y,s.x,7] = 1
    for i,s in enumerate(g.players[1].shipyards):
        inp_sy1[i] = inp.copy()
        inp_sy1[i,s.y,s.x,8] = 1

    inp_sh0[:,0,0,9] = g.nb_step
    inp_sy0[:,0,0,9] = g.nb_step
    inp_sh1[:,0,0,9] = g.nb_step
    inp_sy1[:,0,0,9] = g.nb_step

    inp_sh0[:,1,0,9] = g.players[0].halite
    inp_sy0[:,1,0,9] = g.players[0].halite
    inp_sh1[:,1,0,9] = g.players[1].halite
    inp_sy1[:,1,0,9] = g.players[1].halite

    inp_sh0[:,0,1,9] = g.players[0].halite
    inp_sy0[:,0,1,9] = g.players[0].halite
    inp_sh1[:,0,1,9] = g.players[1].halite
    inp_sy1[:,0,1,9] = g.players[1].halite

    return inp_sh0,inp_sy0,inp_sh1,inp_sy1

def choice(a,p):
    assert a.shape[1] == p.shape[1]
    b = np.zeros((a.shape[0],))
    for i in range(a.shape[0]):
        b[i] = np.random.choice(a,1,p=p[i])

def find_nan(a,block=True):
    print("find nan",a.shape)
    vmax = a.shape
    v = [0]*len(vmax)
    f = False
    while True:
        val = a
        for k in range(len(v)):
            val = val[v[k]]
        if np.isnan(val):
            print("-nan-",v)
            f = True
        v[0] += 1
        for k in range(len(v)):
            if v[k]>=vmax[k]:
                v[k] = 0
                if k+1>=len(vmax):
                    return 0
                v[k+1] += 1
            
def compute_action_dict(actions_list,p):
    d = actions_dict()
    (sh_actions,sy_actions) = actions_list
    for i,sh in enumerate(p.ships):
        d[sh_actions[i].numpy()].append(sh)
    for i,sy in enumerate(p.shipyards):
        d[sy_actions[i].numpy()].append(sy)
    return d