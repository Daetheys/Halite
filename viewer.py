from engine.game import Game
from tools import get_nn_input
import matplotlib.pyplot as plt
from tools import compute_action_dict
import tensorflow as tf

class Viewer:
    def __init__(self,bot1,bot2,vbm):
        self.game = Game(bot1,bot2)
        self.vbm = vbm
    
    def step(self):
        inp_sh_p0 , inp_sy_p0 , inp_sh_p1 , inp_sy_p1 = get_nn_input(self.game)

        proba0 = self.game.players[0].agent.compute_actions_proba((inp_sh_p0 , inp_sy_p0))
        proba1 = self.game.players[1].agent.compute_actions_proba((inp_sh_p1 , inp_sy_p1))

        self.vbm.flush()

        actions_list0 = self.game.players[0].agent.sample_actions(proba0)
        actions0 = compute_action_dict(actions_list0,self.game.players[0])
        actions_list1 = self.game.players[1].agent.sample_actions(proba1)
        actions1 = compute_action_dict(actions_list1,self.game.players[1])

        self.game._step(actions0,actions1)

    def view(self):
        inp_sh_p0,inp_sy_p0,_,_ = get_nn_input(self.game)
        inp = tf.concat([inp_sh_p0,inp_sy_p0],axis=0)[0]
        for i in range(7):
            plt.subplot(self.game.length+1,7,self.game.nb_step*7+i+1)
            if i == 0:
                plt.text(-70, 2,str(self.game.players[0].halite)[:4]+"-"+str(self.game.players[1].halite)[:4], fontsize = 10, color = 'black')#, backgroundcolor = 'white')
            plt.imshow(inp[:,:,i])
            plt.xticks([])
            plt.yticks([])

    def play(self):
        self.view()
        for i in range(self.game.length):
            self.step()
            self.view()
        plt.show()