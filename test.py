import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

def get_model():
    inp = Input(shape=(20,20,7))
    x = inp
    x = Conv2D(25,(2,2),padding='same')(x)
    x = Conv2D(25,(2,2),padding='same')(x)
    #x = MaxPool2D((2,2))(x)
    x1 = tf.nn.relu(x)

    x = Conv2D(50,(2,2),padding='same')(x1)
    x = Conv2D(50,(2,2),padding='same')(x)
    #x = MaxPool2D((2,2))(x)
    x2 = tf.nn.relu(x)

    x = Conv2D(100,(2,2),padding='same')(x2)
    x = Conv2D(100,(2,2),padding='same')(x)
    #x = MaxPool2D((2,2))(x)
    x3 = tf.nn.relu(x)

    #x = Flatten()(x2)
    #x = Dense(20,activation='relu')(x)
    #x = Dense(20*20*10,activation='relu')(x)

    #x = Reshape((20,20,1))(x)
    #x = x + x2
    x = Conv2DTranspose(100,(2,2),padding='same')(x3)
    x = Conv2DTranspose(100,(2,2),padding='same')(x)
    x = tf.nn.relu(x) + x3

    x = Conv2DTranspose(50,(2,2),padding='same')(x)
    x = Conv2DTranspose(50,(2,2),padding='same')(x)
    x = tf.nn.relu(x) + x2

    x = Conv2DTranspose(25,(2,2),padding='same')(x)
    x = Conv2DTranspose(25,(2,2),padding='same')(x)
    x = tf.nn.relu(x) + x1

    x = Conv2DTranspose(7,(2,2),padding='same')(x)
    x = Conv2DTranspose(7,(2,2),padding='same')(x)
    x = tf.concat([tf.nn.relu(x[:,:,:,:1]),tf.nn.sigmoid(x[:,:,:,1:])],axis=-1)
    #x = tf.math.exp(x)

    m = tf.keras.Model(inputs=inp,outputs=x)
    def loss(y_true,y_pred):
        eps = 10**-90
        #Crossentropy
        cross = 0
        #Halite
        #cross = tf.reduce_mean(((y_pred[:,:,:,0]-y_true[:,:,:,0])/24000+)**2)
        cross = tf.reduce_mean(tf.math.log(tf.math.abs(y_pred[:,:,:,0]-y_true[:,:,:,0])+eps))
        #Ships and shipyards
        for i in [1,3,4,6]:
            y_predi = y_pred[:,:,:,i]
            y_truei = y_true[:,:,:,i]
            cross_t1 = tf.math.log(y_predi**y_truei)
            cross_t2 = tf.math.log((1-y_predi)**(1-y_truei))
            cross += -tf.reduce_mean( cross_t1 + cross_t2 )
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min(y_predi),tf.reduce_max(y_predi)])
        #MSE
        #mse = tf.reduce_mean((y_true-y_pred)**2)

        #Print
        #cross = tf.compat.v1.Print(cross,[cross],"loss")
        
        out = cross#+mse
        return out
    m.compile(loss=loss,optimizer="Adam",metrics="accuracy")
    return m

def get_data(nb):
    datainn = np.zeros((nb,20,20,7))
    dataout = np.zeros((nb,20,20,7))
    #Ships
    for i in range(nb):
        #Halite
        datainn[i,:,:,0] = np.random.uniform(0,1000,(20,20))
        dataout[i,:,:,0] = datainn[i,:,:,0]
        #Ships player
        ship_halite = np.random.uniform(0,1000,(20,20))
        for _ in range(5):
            x = np.random.randint(0,20)
            y = np.random.randint(0,20)
            datainn[i,y,x,1] = 1
            halite = ship_halite[y,x]
            datainn[i,y,x,2] = halite
            if np.random.random() > 0.5:
                x += np.random.randint(-1,2)
                x = x%20
            else:
                y += np.random.randint(-1,2)
                y = y%20
            dataout[i,y,x,1] = 1
            dataout[i,y,x,2] = halite
        #Shipyard player
        for _ in range(2):
            x = np.random.randint(0,20)
            y = np.random.randint(0,20)
            datainn[i,y,x,3] = 1
            dataout[i,y,x,3] = 1
        #Ships opponent
        ship_halite = np.random.uniform(0,1000,(20,20))
        for _ in range(5):
            x = np.random.randint(0,20)
            y = np.random.randint(0,20)
            datainn[i,y,x,4] = 1
            halite = ship_halite[y,x]
            datainn[i,y,x,5] = halite
            if np.random.random() > 0.5:
                x += np.random.randint(-1,2)
                x = x%20
            else:
                y += np.random.randint(-1,2)
                y = y%20
            dataout[i,y,x,4] = 1
            dataout[i,y,x,5] = halite
        #Shipyard opponent
        for _ in range(2):
            x = np.random.randint(0,20)
            y = np.random.randint(0,20)
            datainn[i,y,x,6] = 1
            dataout[i,y,x,6] = 1
    return datainn,dataout


model = get_model()
model.summary()
[model.weights[i].assign(model.weights[i]/1000) for i in range(len(model.weights))]

datainn,dataout = get_data(10000)
model.fit(datainn,dataout,epochs=3)

evalinn,evalout = get_data(10000)
model.evaluate(evalinn,evalout)

for i in range(0,7):
    plt.subplot(3,7,i+1)
    plt.imshow(evalinn[0,:,:,i])

for i in range(0,7):
    plt.subplot(3,7,i+7+1)
    plt.imshow(evalout[0,:,:,i])

out = model(evalinn[[0]])
for i in range(0,7):
    plt.subplot(3,7,i+2*7+1)
    plt.imshow(out[0,:,:,i])
    
plt.show()
