from trainer2 import HaliteTrainer
from model import VBM
from nn import wings_model,parallel_model
import time
import tensorflow as tf
import os
from bot import *
tf.keras.backend.set_floatx('float64')
#tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

mz,msh,msy = parallel_model()#wings_model()
vbm = VBM(mz,msh,msy)

bot1 = lambda : RLBot(vbm)
bot2 = lambda : RLBot(vbm)#RandomBot()#

trainer = HaliteTrainer(bot1,bot2,vbm,batch_size=100)
trainer.learn(1000,'tiny2')
"""
for j in range(50):
    print("----",j)
    for i in range(trainer.game_length):
        print(i,end="\r")
        trainer.step()
    trainer.fit()
    trainer.reset()
    trainer.save()
"""
