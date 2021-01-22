from engine.game import Game
from tools import get_nn_input
import matplotlib.pyplot as plt

class Viewer:
    def __init__(self,bot1,bot2,vbm):
        self.game = Game(bot1,bot2)
        self.vbm = vbm
    
    def step(self):
        inp_sh_p0 , inp_sy_p0 , inp_sh_p1 , inp_sy_p1 = get_nn_input(self.game)

        proba0 = self.game.players[0].agent.compute_actions_proba((inp_sh_p0 , inp_sy_p0))
        proba1 = self.game.players[1].agent.compute_actions_proba((inp_sh_p1 , inp_sy_p1))

        self.vbm.flush()

        actions0 = self.game.players[0].agent.sample_actions(proba0)
        actions1 = self.game.players[1].agent.sample_actions(proba1)

        self.game._step(actions0,actions1)

    def view(self):
        inp_sh_p0,_,_,_ = get_nn_input(self.game)
        inp = inp_sh_p0[0]
        for i in range(7):
            plt.subplot(1,7,i+1)
            plt.imshow(inp[:,:,i])
        plt.show()

    def play(self):
        self.view()
        for i in range(self.game.length):
            self.step()
            self.view()