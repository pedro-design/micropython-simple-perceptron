
import math
import random
import machine
import gc
gc.collect()
total = gc.mem_alloc() + gc.mem_free()

def view_arch(model):
    for i in range(len(model)):
        for x in range(len(model[i])):
            print("layer :",model[i][x].name,"in op :",i,"W :",model[i][x].weight)

def free_mem():
    print("------------------")
    print(" {} mb used of {}".format(gc.mem_alloc()/8000,total/8000))

#funciones matematicas
def dot(x, y):
    try:
        v =sum(x_i*y_i for x_i, y_i in zip(x, y))
    except:
        v = x*y
    return v
def sigmoid(x, Derivative=False):
   # print("@",x)
    if not Derivative:
        return 1 / (1 + math.exp (-x))
    else:
        out = 1 / (1 + math.exp (-x))
        return out * (1 - out)
def mse(p,t):
    return  ((p - t)**2)

def derivate(p,t):
    return -1*(2 * (p-t))



def nn_help():
    print("nn build help: 1x1 is linear ,2x1 is 2 inputs , 1 output, 1x2 is 1 input,2 outputs")
    print("how to apply gradients: for 1 neuron in 1 hiden layer is layer.apply_grad(expected or act of next neuron in nex layer,- the input of neuron)")
    

class  Dense_1x1():
    def __init__(self,name):
        self.lr=0.001
        self.name = name
        self.bias =0
        self.weight =random.getrandbits(4)/10
        self.prediction=0
    def predict(self,input1):
        self.prediction= (input1*self.weight)+self.bias
        return  self.prediction
    def apply_grad(self,inp,ou):
        self.prediction= inp*self.weight
        self.bias +=(ou-self.prediction)*(self.lr/100)
        self.weight+= (ou-self.prediction)*self.lr
        return  ou-self.prediction

class  Dense_1x2():
    def __init__(self,name):
        self.lr=0.01
        self.name = name
        self.bias =[0 ,0]
        self.weight =[random.getrandbits(4)/10,random.getrandbits(4)/10]
        self.prediction=[0,0]
    def predict(self,input1):
        self.prediction= [(input1*self.weight[0])+self.bias[0],(input1*self.weight[1])+self.bias[1]]
        return  self.prediction
    def apply_grad(self,inp,ou):
        self.prediction= [inp*self.weight[0],inp*self.weight[1]]
        self.bias[0] +=(ou[0]-self.prediction[0])*(self.lr/100)
        self.bias[1] +=(ou[1]-self.prediction[1])*(self.lr/100)
        self.weight[0]+= (ou[0]-self.prediction[0])*self.lr
        self.weight[1]+= (ou[1]-self.prediction[1])*self.lr
        return  list(set(ou)- set(self.prediction))

class  Dense_2x1():
    def __init__(self,name):
        self.lr=0.01
        self.name = name
        self.bias =[0,0]
        self.weight =[random.getrandbits(4)/10,random.getrandbits(4)/10]
        self.prediction=0
    def predict(self,input1):
        self.prediction= ((input1[0]*self.weight[0])+self.bias[0])+((input1[1]*self.weight[1])+self.bias[1])
        return  self.prediction
    def apply_grad(self,inp,ou):
        self.prediction= ((inp[0]*self.weight[0])+self.bias[0])+((inp[1]*self.weight[1])+self.bias[1])
        self.bias[0] +=(ou-self.prediction)*(self.lr/100)
        self.bias[1] +=(ou-self.prediction)*(self.lr/100)
        self.weight[0]+= (ou-self.prediction)*self.lr
        self.weight[1]+= (ou-self.prediction)*self.lr
        return  ou-self.prediction


#--------------------



