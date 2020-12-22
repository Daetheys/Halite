from trainer import HaliteTrainer
from model import VBM
from nn import wings_model
import time
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

mz,msh,msy = wings_model()
vbm = VBM(mz,msh,msy)
trainer = HaliteTrainer(vbm,batch_size=100)
for j in range(20):
    print("----",j)
    for i in range(trainer.game_length):
        print(i,end="\r")
        trainer.step()
    trainer.fit()
    trainer.reset()
