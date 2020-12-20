import tensorflow as tf
from tensorflow.keras.layers import *
import threading
import time
import numpy as np
import sys

def wings_model():
    """Generate Keras model for wing.

    wings are :
        - halite ressources
        - friendly ship
        - ennemy ship
        - halite loaded in friendly ships
        - halite loaded in ennemy ships
    """

    #Z Model
    def wing(inp):
        #inp = Input(shape=(21,21,1))
        x = inp
        nb = 5
        x = Conv2D(nb,(3,3),padding='valid')(x)
        x = Conv2D(nb,(3,3),padding='valid')(x)
        x = tf.activation.relu(x)
        return x
    nb_wings = 5
    inp = Input(shape=(nb_wings,21,21,1))
    lwings = [wing(inp[:,i]) for i in range(nb_wings)]
    x = tf.reduce_sum([x for x in lwings],axis=0)

    nb = 10
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)
    x = tf.activation.relu(x)

    nb = 20
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(4,4),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)
    x = tf.activation.relu(x)
    x = Reshape((nb,))(x)

    mz = tf.keras.Model(inputs=inp,outputs=x)

    #Sh Model
    inpsh = Input(shape=(nb,))
    x = inpsh
    x = Dense(50,activation='relu')
    x = Dense(6,activation='softmax')
    msh = tf.keras.Model(inputs=inpsh,outputs=x)

    #Sy Model
    inpsy = Input(shape=(nb,))
    x = inpsh
    x = Dense(50,activation='relu')
    x = Dense(2,activation='softmax')
    msy = tf.keras.Model(inputs=inpsy,outputs=x)
    
    return mz,msh,msy

class ServerModel:
    """A server to buffer input up to a certain number and feed them to a
    model.
    """
    def __init__(self,model, daemonize=True, length_treshold=10, length_max=15):
        # super().__init__() # There is no inheritance, no need to call super
        self.daemon = daemonize
        self.model = model
        self.inp = []
        self.request_lock = threading.RLock()
        self.waiting_locks = []

        self.length_max = length_max
        self.length_treshold = length_treshold

        self.out_reset()

    def __call__(self,inp):
        """Shortcut for request, which adds a data on the input buffer."""
        return self.request(inp)

    def out_reset(self):
        """Reset the output buffer."""
        self.out = (np.zeros(self.length_max),threading.Event())

    def request(self,inp):
        """Adds a data on the input buffer, and check for threshold."""
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
        """If the number of data supplied is higher than the server threshhold,
        asks the server to flush the input buffer.
        """
        if len(self.inp) >=  self.length_treshold:
            self.flush()

    def flush(self):
        """Forces the server to compute the model output on the input it was
        previously fed, regardless of number of data supplied.
        """
        inp = np.array(self.inp)
        self.inp = []
        out = self.model(inp)
        for i,o in enumerate(out):
            self.out[0][i] = o
        self.out[1].set()
        self.out_reset()
