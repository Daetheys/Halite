import tensorflow as tf
from tensorflow.keras.layers import *
import threading
import time
import numpy as np
import sys

def wings_model():
    def wing():
        inp = Input(shape=(21,21,1))
        x = inp
        nb = 5
        x = Conv2D(nb,(3,3),padding='valid')(x)
        x = Conv2D(nb,(3,3),padding='valid')(x)
        return inp,x
    nb_wings = 5
    lwings = [wing() for i in range(nb_wings)]
    inps = [inp for (inp,x) in lwings]
    x = sum([x for (inp,x) in lwings])

    nb = 10
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)

    nb = 20
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(4,4),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)

    x = Reshape((nb,))(x)
    x = Dense(1)(x)

    m = tf.keras.Model(inputs=inps,outputs=x)
    m.compile(loss="mse",optimizer="Adam")
    #m.build((1,21,21,1))
    #m.summary()
    return m

class ServerModel:
    def __init__(self,model):
        super().__init__()
        self.daemon = True
        self.model = model
        self.inp = []
        self.request_lock = threading.RLock()
        self.waiting_locks = []

        self.length_max = 15
        self.length_treshold = 10

        self.out_reset()

    def __call__(self,inp):
        return self.request(inp)
        
    def out_reset(self):
        self.out = (np.zeros(self.length_max),threading.Event())

    def request(self,inp):
        #Wait for its turn to put in the stack
        with self.request_lock:
            ind = len(self.inp)
            self.inp.append(inp)
            out = self.out
            self.check_compute()
        #Wait for the results to be ready
        time.sleep(0.01)
        out[1].wait()
        return out[0][ind]

    def check_compute(self):
        if len(self.inp) >=  self.length_treshold:
            inp = np.array(self.inp)
            self.inp = []
            out = self.model(inp)
            for i,o in enumerate(out):
                self.out[0][i] = o
            self.out[1].set()
            self.out_reset()
