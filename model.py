import tensorflow as tf

MAXBATCHSIZE = 10000
class VecModel:
    def __init__(self,model):
        self.model = model
        self.inp = tf.Variable((MAXBATCHSIZE,5,21,21,1))
        self.inp_index = 0
        self.out = tf.Variable((MAXBATCHSIZE,))

    def request(self,inp):
        # inp.shape = (x,5,21,21,1)
        x = inp.shape[0]
        self.inp[self.inp_index:self.inp_index+x].assign(inp)
        self.inp_index += x
        return self.out[self.inp_index-x:self.inp_index]

    def flush(self):
        out = self.model(self.inp[:self.inp_index])
        self.out[:self.inp_index].assign(out)

class VecBranchModel:
    def __init__(self,modelz,model1,model2):
        self.modelz = modelz
        self.models = [model1,model2]

        inp_shape = (MAXBATCHSIZE,*modelz.input_shape[1:])
        self.inp = [tf.Variable(tf.zeros(inp_shape),dtype=tf.float32) for _ in range(2)]
        self.inp_index = [0,0]

        out_shape = [(MAXBATCHSIZE,*self.models[i].output_shape[1:]) for i in range(len(self.models))]
        
        self.out = [tf.Variable(tf.zeros(out_shape[i])) for i in range(len(self.models))]

    def request(self,inp,index):
        x = inp.shape[0]
        self.inp[index][self.inp_index[index]:self.inp_index[index]+x].assign(inp)
        self.inp_index[index] += x
        start_index = self.inp_index[index]
        return (lambda : self.out[index][start_index-x:start_index])

    def flush(self):
        inps = [self.inp[i][:self.inp_index[i]] for i in range(2)]
        s = [len(inps[i]) for i in range(2)]
        inp = tf.concat(inps,axis=0)
        outz = self.modelz(inp)
        out0 = self.models[0](outz[:s[0]])
        out1 = self.models[1](outz[s[0]:s[0]+s[1]])
        self.out[0][:s[0]].assign(out0)
        self.out[1][:s[1]].assign(out1)
