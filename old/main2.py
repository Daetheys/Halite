from agent import *
from env import *
from model import *
import time

model = wings_model()
env = HaliteVecEnv(2,100)
agent = QAgent(model,env)
for i in range(20):
    print("i",i)
    t = time.perf_counter()
    agent.minimal_step()
    print(time.perf_counter()-t)
