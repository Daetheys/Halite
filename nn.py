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
    inp = Input(shape=(21,21,9))
    x = inp

    nb = 25
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = tf.nn.relu(x)
    
    nb = 40
    x = Conv2D(nb,(2,2),padding='valid')(x)
    x = Conv2D(nb,(3,3),padding='valid')(x)
    x = MaxPool2D((2,2),padding='valid')(x)
    x = tf.nn.relu(x)

    nb = 150
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

def parallel_conv(x,nb_channels,type=None,activation=None):
    lx = []
    for k in range(len(x)):
        if type == 'conv':
            xk = Conv2D(nb_channels,(3,3),padding='same',activation=activation)(x[k])
        elif type == 'convtrans':
            xk = Conv2DTranspose(nb_channels,(3,3),padding='same',activation=activation)(x[k])
        else:
            assert False #Unknown type for parallel conv
        lx.append(xk)
    #out = tf.concat(lx,axis=-1)
    out = lx
    return out

def parallel_max_pool(x):
    lx = []
    for k in range(len(x)):

        lx.append(MaxPool2D((2,2))(x[k]))
    return lx

def parallel_model():
    activation = 'relu'
    inp = Input(shape=(21,21,9))
    x = inp
    xa = Conv2D(8,(3,3),padding='same')(x)
    x1b = parallel_conv([x[:,:,:,i:i+1] for i in range(x.shape[-1])],2,'conv')
    x = tf.concat([xa,*x1b],axis=-1)
    x = MaxPool2D((2,2))(x)
    x1b = parallel_max_pool(x1b)
    xa = Conv2D(8,(3,3),padding='same',activation=activation)(x)
    x11b = parallel_conv(x1b,2,'conv',activation=activation)
    x = tf.concat([xa,*x11b],axis=-1)
    x = MaxPool2D((2,2))(x)
    #x11b = parallel_max_pool(x11b)
    """
    xa = Conv2D(16,(3,3),padding='same')(x)
    x2b = parallel_conv(x11b,4,'conv')
    x = tf.concat([xa,*x2b],axis=-1)
    xa = Conv2D(16,(3,3),padding='same',activation=activation)(x)
    x22b = parallel_conv(x2b,4,'conv',activation=activation)
    x = tf.concat([xa,*x22b],axis=-1)
    x = MaxPool2D((2,2))(x)
    x22b = parallel_max_pool(x22b)
    
    xa = Conv2D(32,(3,3),padding='same')(x)
    x3b = parallel_conv(x22b,8,'conv')
    x = tf.concat([xa,*x3b],axis=-1)
    xa = Conv2D(32,(3,3),padding='same',activation=activation)(x)
    x33b = parallel_conv(x3b,8,'conv',activation=activation)
    x = tf.concat([xa,*x33b],axis=-1)
    x = MaxPool2D((2,2))(x)
    x33b = parallel_max_pool(x33b)
    """
    x = Flatten()(x)
    x0 = x

    mz = tf.keras.Model(inputs=inp,outputs=x)
    for w in mz.weights:
        w.assign(w/10)

    #Sh Model
    inpsh = Input(shape=x0.shape[1:])
    x = inpsh
    x = Dense(32,activation=activation)(x)
    x = Dense(6,activation='softmax')(x)
    msh = tf.keras.Model(inputs=inpsh,outputs=x)
    for w in msh.weights:
        w.assign(w/10)

    #Sy Model
    inpsy = Input(shape=x0.shape[1:])
    x = inpsy
    x = Dense(32,activation=activation)(x)
    x = Dense(2,activation='softmax')(x)
    msy = tf.keras.Model(inputs=inpsy,outputs=x)
    for w in msy.weights:
        w.assign(w/10)

    #msh.summary()
    #assert False

    return mz,msh,msy

def dense_model():
    activation = 'relu'
    inp = Input(shape=(9,9,10))
    x = inp
    x = Flatten()(x)
    x = Dense(64,activation=activation)(x)
    mz = tf.keras.Model(inputs=inp,outputs=x)
    for w in mz.weights:
        w.assign(w/100)

    x0 = x

    inpsh = Input(shape=x0.shape[1:])
    x = inpsh
    x = Dense(64,activation=activation)(x)
    x = Dense(6,activation='softmax')(x)
    msh = tf.keras.Model(inputs=inpsh,outputs=x)
    for w in msh.weights:
        w.assign(w/100)

    inpsy = Input(shape=x0.shape[1:])
    x = inpsy
    x = Dense(64,activation=activation)(x)
    x = Dense(2,activation='softmax')(x)
    msy = tf.keras.Model(inputs=inpsy,outputs=x)
    for w in msy.weights:
        w.assign(w/100)

    return mz,msh,msy