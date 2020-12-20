import tensorflow as tf
import time
import numpy as np

MAXBATCHSIZE = 20000

class VecBranchModel:
    def __init__(self,modelz,model1,model2):
        self.modelz = modelz
        self.models = [model1,model2]

        inp_shape = (MAXBATCHSIZE,*modelz.input_shape[1:])
        self.inp = [np.zeros(inp_shape) for _ in range(2)]
        self.inp_index = [0,0]

        out_shape = [(MAXBATCHSIZE,*self.models[i].output_shape[1:]) for i in range(len(self.models))]
        
        self.out = [np.zeros(out_shape[i]) for i in range(len(self.models))]

    def request(self,inp,index):
        x = inp.shape[0]
        self.inp[index][self.inp_index[index]:self.inp_index[index]+x] = inp
        self.inp_index[index] += x
        start_index = self.inp_index[index]
        return (lambda : self.out[index][start_index-x:start_index])

    def flush(self):
        inps = [self.inp[i][:self.inp_index[i]] for i in range(2)]
        s = [len(inps[i]) for i in range(2)]
        inp = tf.concat(inps,axis=0)
        outz = self.modelz(inp)
        out0 = self.models[0](outz[:s[0]]).numpy()
        out1 = self.models[1](outz[s[0]:s[0]+s[1]]).numpy()
        self.out[0][:s[0]] = out0
        self.out[1][:s[1]] = out1

    def parameters(self):
        return self.modelz.variables + self.models[0].variables + self.models[1].variables
        
VBM = VecBranchModel
