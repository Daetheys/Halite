from viewer import *
import tensorflow as tf
from model import VBM
from bot import *

mz = tf.keras.models.load_model("Models\\tiny-modelz.m")
m1 = tf.keras.models.load_model("Models\\tiny-model1.m")
m2 = tf.keras.models.load_model("Models\\tiny-model2.m")

vbm = VBM(mz,m1,m2)
class EmptyVBM:
    def flush(self):
        pass
    def reset(self):
        pass

bot1 = RLBot(vbm,0)
bot2 = RandomBot(1)

viewer = Viewer(bot1,bot2,vbm)
viewer.play()