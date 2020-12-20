from engine.game import *
from engine.tools import *
import time

def agent(p):
    return actions_dict()

g = Game(agent,agent)
t = time.perf_counter()
for i in range(400*1000):
    g.step()
print(time.perf_counter()-t)
