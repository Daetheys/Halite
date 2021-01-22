from viewer import *
#import tensorflow as tf
#from nn import *
#from model import VBM
from bot import *

#mz = tf.keras.models.load_model("Models\\tiny2-modelz.m")
#m1 = tf.keras.models.load_model("Models\\tiny2-model1.m")
#m2 = tf.keras.models.load_model("Models\\tiny2-model2.m")

#vbm = VBM(mz,m1,m2)
class EmptyVBM:
    def flush(self):
        pass
    def reset(self):
        pass

bot1 = RandomBot()
bot2 = RandomBot()

viewer = Viewer(bot1,bot2,EmptyVBM())
viewer.play()