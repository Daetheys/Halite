import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

def get_model():
    inp = Input(shape=(20,20,7))
    x = inp
    xa = Conv2D(10,(3,3),padding='same')(x)
    x1b = parallel_conv([x[:,:,:,i:i+1] for i in range(x.shape[-1])],4,'conv')
    x = tf.concat([xa,*x1b],axis=-1)
    xa = Conv2D(10,(3,3),padding='same',activation='relu')(x)
    x11b = parallel_conv(x1b,4,'conv',activation='relu')
    x = tf.concat([xa,*x11b],axis=-1)
    x1 = x

    xa = Conv2D(25,(3,3),padding='same')(x)
    x2b = parallel_conv(x11b,4,'conv')
    x = tf.concat([xa,*x2b],axis=-1)
    xa = Conv2D(25,(3,3),padding='same',activation='relu')(x)
    x22b = parallel_conv(x2b,4,'conv',activation='relu')
    x = tf.concat([xa,*x22b],axis=-1)
    x2 = x
	
    #x = Conv2D(100,(3,3),padding='same')(x2)
    #x = Conv2D(100,(3,3),padding='same')(x)
    #x3 = tf.nn.relu(x)

    #x = Conv2DTranspose(100,(3,3),padding='same')(x3)
    #x = Conv2DTranspose(100,(3,3),padding='same')(x)
    #x = tf.nn.relu(x) + x3

    xa = Conv2DTranspose(25,(3,3),padding='same')(x)
    x3b = parallel_conv(x22b,4,'convtrans')
    x = tf.concat([xa,*x3b],axis=-1)
    xa = Conv2DTranspose(25,(3,3),padding='same',activation='relu')(x)
    x33b = parallel_conv(x3b,4,'convtrans',activation='relu')
    x = tf.concat([xa,*x33b,x2],axis=-1) #Res net structure

    xa = Conv2DTranspose(10,(3,3),padding='same')(x)
    x4b = parallel_conv(x33b,4,'convtrans')
    x = tf.concat([xa,*x4b],axis=-1)
    xa = Conv2DTranspose(10,(3,3),padding='same',activation='relu')(x)
    x44b = parallel_conv(x4b,4,'convtrans',activation='relu')
    x = tf.concat([xa,*x44b,x1],axis=-1) #Res net structure

    lx = []
    for i in range(7):
        xi = x
        #xi = Conv2DTranspose(10,(3,3),padding='same')(x)
        #xi = Conv2DTranspose(10,(3,3),padding='same')(xi)
        #xi = tf.nn.relu(xi)
        
        xi = Conv2DTranspose(1,(3,3),padding='same')(xi)
        xi = Conv2DTranspose(1,(3,3),padding='same')(xi)
        if i == 0:
            xi = tf.nn.elu(xi)+1
        elif i in [1,3,4,6]:
            xi = tf.nn.sigmoid(xi)
        else:
            xi = (tf.nn.elu(xi)+1)*tf.stop_gradient(lx[-1])
        lx.append(xi)
    x = tf.concat(lx,axis=-1)
    #x = Conv2DTranspose(7,(2,2),padding='same')(x)
    #x = Conv2DTranspose(7,(2,2),padding='same')(x)
    
    #x = tf.concat([tf.nn.relu(x[:,:,:,:1]),tf.nn.sigmoid(x[:,:,:,1:])],axis=-1)
    #x = tf.math.exp(x)

    m = tf.keras.Model(inputs=inp,outputs=x)
    def loss(y_true,y_pred):
        eps = 0
        #Crossentropy
        cross = 0
        #Halite
        cross = tf.reduce_mean(((y_pred[:,:,:,0]-y_true[:,:,:,0])/24000)**2)*100
        cross += tf.reduce_mean(((y_pred[:,:,:,2]-y_true[:,:,:,2])/24000)**2)
        cross += tf.reduce_mean(((y_pred[:,:,:,5]-y_true[:,:,:,5])/24000)**2)
        #cross = tf.reduce_mean(tf.math.log(tf.math.abs(y_pred[:,:,:,0]-y_true[:,:,:,0])+eps))
        #Ships and shipyards
        for i in [1,3,4,6]:
            y_predi = y_pred[:,:,:,i]
            y_truei = y_true[:,:,:,i]
            cross_t1 = tf.math.log(y_predi**y_truei)
            cross_t2 = tf.math.log((1-y_predi)**(1-y_truei))
            cross += -tf.reduce_mean( cross_t1 + cross_t2 )
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min(y_predi),tf.reduce_max(y_predi)],'ships')
        """
        for i in  [2,5]:
            y_predi = y_pred[:,:,:,i]
            y_truei = y_true[:,:,:,i]
            maxi = 2*y_truei+10
            y_truei_zero = tf.cast(y_truei==0,dtype=tf.float64)
            y_truei_nozero= y_truei_zero + y_truei
            y_predi_corr = -tf.math.log(1+tf.math.exp(tf.cast(maxi-y_predi<10,dtype=tf.float64)*(maxi-y_predi)))*tf.cast(maxi-y_predi<10,dtype=tf.float64) + maxi - tf.cast(maxi-y_predi>=10,dtype=tf.float64)*(maxi-y_predi)
            cross_t1 = tf.math.log((y_predi/y_truei_nozero)**(1-y_truei_zero))*y_truei
            cross_t2 = tf.math.log(((maxi-y_predi_corr)/(maxi-y_truei)))*(maxi-y_truei)
            cross += -tf.reduce_mean( cross_t1 + cross_t2 )
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min(y_predi),tf.reduce_max(y_predi),tf.reduce_min(y_truei),tf.reduce_max(y_truei)],'y')
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min(y_predi_corr),tf.reduce_max(y_predi_corr)],'corr')
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min(y_predi/y_truei_nozero),tf.reduce_max(y_predi/y_truei_nozero)],'hships')
            #cross = tf.compat.v1.Print(cross,[tf.reduce_min((maxi-y_predi_corr)/(maxi-y_truei)),tf.reduce_min((maxi-y_predi_corr)),tf.reduce_max((maxi-y_predi_corr)/(maxi-y_truei))],'hships2')
            #cross = tf.compat.v1.Print(cross,[-tf.reduce_mean(cross_t1),-tf.reduce_mean(cross_t2)],'cross')
        """
        #MSE
        #mse = tf.reduce_mean((y_true-y_pred)**2)

        #Print
        #cross = tf.compat.v1.Print(cross,[cross],"loss")
        
        out = cross#+mse
        return out
    opt = tf.keras.optimizers.Adam(10**-3)
    m.compile(loss=loss,optimizer=opt)
    return m

def parallel_conv(x,nb_channels,type=None,activation=None):
    lx = []
    for k in range(len(x)):
        if type == 'conv':
            xk = Conv2D(nb_channels,(3,3),padding='same',activation=activation)(x[k])
        elif type == 'convtrans':
            xk = Conv2DTranspose(nb_channels,(3,3),padding='same',activation=activation)(x[k])
        elif type == 'maxpool':
            xk = MaxPool2D((2,2))(x[k])
        elif type == 'upsample':
            xk = UpSamping2D((2,2))(x[k])
        else:
            assert False #Unknown type for parallel conv
        lx.append(xk)
    #out = tf.concat(lx,axis=-1)
    out = lx
    return out

def get_model2():
    activation = 'relu'

    inp = Input(shape=(21,21,9))
    x = inp
    xa = Conv2D(8,(3,3),padding='same')(x)
    x1b = parallel_conv([x[:,:,:,i:i+1] for i in range(x.shape[-1])],2,'conv')
    x = tf.concat([xa,*x1b],axis=-1)
    xa = Conv2D(8,(3,3),padding='same',activation=activation)(x)
    x11b = parallel_conv(x1b,2,'conv',activation=activation)
    x = tf.concat([xa,*x11b],axis=-1)
    x = MaxPool2D((2,2))(x)
    x11b = parallel_conv(x11b,None,'maxpool')
    x1 = x

    xa = Conv2D(16,(3,3),padding='same')(x)
    x2b = parallel_conv(x11b,4,'conv')
    x = tf.concat([xa,*x2b],axis=-1)
    xa = Conv2D(16,(3,3),padding='same',activation=activation)(x)
    x22b = parallel_conv(x2b,4,'conv',activation=activation)
    x = tf.concat([xa,*x22b],axis=-1)
    x = MaxPool2D((2,2))(x)
    x22b = parallel_conv(x22b,None,'maxpool')
    x2 = x

    xa = Conv2D(32,(3,3),padding='same')(x)
    x3b = parallel_conv(x22b,8,'conv')
    x = tf.concat([xa,*x3b],axis=-1)
    xa = Conv2D(32,(3,3),padding='same',activation=activation)(x)
    x33b = parallel_conv(x3b,8,'conv',activation=activation)
    x = tf.concat([xa,*x33b],axis=-1)
    x = MaxPool2D((2,2))(x)

    x = UpSampling2D((2,2))(x)
    xa = Conv2DTranspose(25,(3,3),padding='same')(x)
    x4b = parallel_conv(x33b,4,'convtrans')
    x = tf.concat([xa,*x4b],axis=-1)
    xa = Conv2DTranspose(25,(3,3),padding='same',activation='relu')(x)
    x44b = parallel_conv(x3b,4,'convtrans',activation='relu')
    x = tf.concat([xa,*x44b,x2,*x22b],axis=-1) #Res net structure

    x = UpSampling2D((2,2))(x)
    x44b = parallel_conv(x44b,None,'upsample')
    xa = Conv2DTranspose(10,(3,3),padding='same')(x)
    x5b = parallel_conv(x44b,4,'convtrans')
    x = tf.concat([xa,*x5b],axis=-1)
    xa = Conv2DTranspose(10,(3,3),padding='same',activation='relu')(x)
    x55b = parallel_conv(x5b,4,'convtrans',activation='relu')
    x = tf.concat([xa,*x55b,x1,*x11b],axis=-1) #Res net structure

    x = UpSampling2D((2,2))(x)
    x55b = parallel_conv(x44b,None,'upsample')
    xa = Conv2DTranspose(10,(3,3),padding='same')(x)
    x6b = parallel_conv(x55b,4,'convtrans')
    x = tf.concat([xa,*x6b],axis=-1)
    xa = Conv2DTranspose(10,(3,3),padding='same',activation='relu')(x)
    x66b = parallel_conv(x6b,4,'convtrans',activation='relu')

    lx = []
    for i in range(7):
        xi = x
        xi = Conv2DTranspose(1,(3,3),padding='same')(xi)
        xi = Conv2DTranspose(1,(3,3),padding='same')(xi)
        if i == 0:
            xi = tf.nn.elu(xi)+1
        elif i in [1,3,4,6]:
            xi = tf.nn.sigmoid(xi)
        else:
            xi = (tf.nn.elu(xi)+1)*tf.stop_gradient(lx[-1])
        lx.append(xi)
    x = tf.concat(lx,axis=-1)

    m = tf.keras.Model(inputs=inp,outputs=x)
    def loss(y_true,y_pred):
        eps = 0
        #Crossentropy
        cross = 0
        #Halite
        cross = tf.reduce_mean(((y_pred[:,:,:,0]-y_true[:,:,:,0])/24000)**2)*100
        cross += tf.reduce_mean(((y_pred[:,:,:,2]-y_true[:,:,:,2])/24000)**2)
        cross += tf.reduce_mean(((y_pred[:,:,:,5]-y_true[:,:,:,5])/24000)**2)
        #Ships and shipyards
        for i in [1,3,4,6]:
            y_predi = y_pred[:,:,:,i]
            y_truei = y_true[:,:,:,i]
            cross_t1 = tf.math.log(y_predi**y_truei)
            cross_t2 = tf.math.log((1-y_predi)**(1-y_truei))
            cross += -tf.reduce_mean( cross_t1 + cross_t2 )
        out = cross#+mse
        return out
    opt = tf.keras.optimizers.Adam(10**-3)
    m.compile(loss=loss,optimizer=opt)
    return m

from scipy.signal import convolve2d
def get_data(nb):
    datainn = np.zeros((nb,20,20,7))
    dataout = np.zeros((nb,20,20,7))
    #Ships
    for i in range(nb):
        #Halite
        datainn[i,:,:,0] = np.random.uniform(0,1000,(20,20))
        kernel = np.array([[-1,-1,-1],[-1,1,-1],[-1,-1,-1]])
        datainn[i,:,:,0] *= 24000/np.sum(datainn[i,:,:,0])
        for _ in range(2):
            datainn[i,:,:,0] = convolve2d(datainn[i,:,:,0],kernel,boundary='symm',mode='same')
        datainn[i,:,:,0] += 10
        datainn[i,:,:,0] *= 24000/np.sum(datainn[i,:,:,0])
        datainn[i,:,:,0] -= np.min(datainn[i,:,:,0])
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
model.fit(datainn,dataout,epochs=10)#,batch_size=100)

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
print(out[0,:,:,0])
plt.show()
