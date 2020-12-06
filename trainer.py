import threading

class Trainer:
    """" Training system with multiple agents and environments in parallel """
    def __init__(self,env_creator,agent_creator,nb_agents):
        self.envs = [env_creator() for i in range(nb_agents)]
        self.agents = [agent_creator() for i in range(nb_agents)]
        self.batch = [None for i in range(nb_agents)]
        self.nb_agents = nb_agents

    def compute_batch(self,size):
        """ Launches threads for each (agent,env) """
        ths = []
        for i in range(self.nb_agents):
            th = threading.Thread(target=self.compute_batch_agent,args=(i,size))
            ths.append(th)
        for th in ths:
            th.start()
        for th in ths:
            th.join()

    def compute_batch_agent(self,index,size):
        """ Computes a batch for a specific (agent,env) """
        batch = []
        obs = self.envs[index].reset()
        for i in range(size):
            actions = self.agents[index].get_actions(obs)
            newobs,reward,done,_ = self.envs[index].step(actions)
            batch.append( (obs,actions,reward) )
            obs = newobs
        self.batch[index] = batch
