import tensorflow as tf
from tensorflow.keras.layers import *

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
    inp = Input(shape=(21,21,7))
    x = inp

    nb = 5
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = tf.nn.relu(x)
    
    nb = 10
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)
    x = tf.nn.relu(x)

    nb = 20
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(4,4),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)
    x = tf.nn.relu(x)
    x = Flatten()(x)

    mz = tf.keras.Model(inputs=inp,outputs=x)

    #Sh Model
    inpsh = Input(shape=(nb,))
    x = inpsh
    x = Dense(50,activation='relu')(x)
    x = Dense(6,activation='softmax')(x)
    msh = tf.keras.Model(inputs=inpsh,outputs=x)

    #Sy Model
    inpsy = Input(shape=(nb,))
    x = inpsy
    x = Dense(50,activation='relu')(x)
    x = Dense(2,activation='softmax')(x)
    msy = tf.keras.Model(inputs=inpsy,outputs=x)
    
    return mz,msh,msy
