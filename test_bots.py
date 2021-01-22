from bot import *
import tensorflow as tf
import numpy as np
def test_bot():
    bot = Bot()
    sh_proba = lambda : tf.ones((10,6))/6
    sy_proba = lambda : tf.ones((3,2))/2
    actions,p = bot.sample_actions_indexs((sh_proba,sy_proba),return_proba=True)

    assert len(actions)==2
    assert len(p)==2
    assert np.all(np.isclose(p[0].numpy(),np.ones(10)/6))
    assert np.all(np.isclose(p[1].numpy(),np.ones(3)/2))
    assert np.all(0<=actions[0])
    assert np.all(actions[0]<=5)
    assert np.all(0<=actions[1])
    assert np.all(actions[1]<=1)

if __name__ == '__main__':
    test_bot()