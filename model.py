import tensorflow as tf
import time
import numpy as np

MAXBATCHSIZE = 20000

class VecBranchModel:
    def __init__(self,modelz,model1,model2):
        self.modelz = modelz
        self.models = [model1,model2]

        self.input_shape = (None,21,21,10)#modelz.input_shape

        self.reset()

    def save(self,prefix):
        self.modelz.save("Models\\"+prefix+"-modelz.m")
        self.models[0].save("Models\\"+prefix+"-model1.m")
        self.models[1].save("Models\\"+prefix+"-model2.m")

    def request(self,inp,index):
        x = inp.shape[0]
        out_ind = len(self.inp[index])
        self.inp[index].append(inp)
        
        return (lambda : self.out[index][out_ind])

    def forward(self,inp0,inp1):
        inp = tf.concat([inp0,inp1],axis=0)
        if inp.shape[0] > 10000:
            print("big flush",inp.shape[0])
            mid = inp.shape[0]//2
            out0A,out1A = self.forward(inp0[:mid],inp1[:max(mid-len(inp0),0)])
            out0B,out1B = self.forward(inp0[mid:],inp1[max(mid-len(inp0),0):])
            return tf.concat([out0A,out0B],axis=0),tf.concat([out1A,out1B],axis=0)
        else:
            outz = self.modelz(inp)
            out0 = self.models[0](outz[:inp0.shape[0]])
            out1 = self.models[1](outz[inp0.shape[0]:inp0.shape[0]+inp1.shape[0]])
            return out0,out1

    def flush(self):
        inp0 = tf.concat(self.inp[0],axis=0)
        inp1 = tf.concat(self.inp[1],axis=0)
        out0,out1 = self.forward(inp0,inp1)
        #print("forwarded")

        self.out = [[],[]]
        offset = 0
        for i in range(len(self.inp[0])):
            x = len(self.inp[0][i])
            self.out[0].append(out0[offset:offset+x])
            offset += x
        offset = 0
        for i in range(len(self.inp[1])):
            x = len(self.inp[1][i])
            self.out[1].append(out1[offset:offset+x])
            offset += x

    def reset(self):
        inp_shape = (0,*self.input_shape[1:])
        self.inp = [[tf.zeros(inp_shape,dtype=tf.float64)],[tf.zeros(inp_shape,dtype=tf.float64)]]
        self.out = [[],[]]

    def parameters(self):
        out = self.modelz.variables + self.models[0].variables + self.models[1].variables
        return out
        
VBM = VecBranchModel
