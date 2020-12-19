from agent import *
from env import *
from trainer import *
from model import *

model = wings_model()
server_model = ServerModel(model)

def agent_creator():
    agent = QAgent(server_model)
    return agent

def env_creator(nb_players=2):
    env = HaliteEnv(nb_players)
    return env

trainer = Trainer(env_creator,agent_creator,100) #100 (env,agents)
trainer.compute_batch(100) #Launches the game for 100 steps
