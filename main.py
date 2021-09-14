import math 
import random
import nn
#from timeit import timeit
from nn import neural_network, error,normalize,random_float
#import machine
#machine.freq(160000000)
#dummy wheater  dataset
dummy = [100,110,102]
DdatasetX = []
DdatasetY = [[100,86,110]]
nsamples = 90
for i in range(nsamples):
    if random.getrandbits(2) == 1:
        dummy=[dummy[0]+random.getrandbits(4)-random.getrandbits(4),dummy[1]+random.getrandbits(4)-random.getrandbits(4),dummy[2]+random.getrandbits(4)-random.getrandbits(4)]
    else:
        dummy=[dummy[0]-random.getrandbits(4)+random.getrandbits(4),dummy[1]-random.getrandbits(4)+random.getrandbits(4),dummy[2]-random.getrandbits(4)+random.getrandbits(4)]
    DdatasetX.append(dummy)
for i in range(1,len(DdatasetX)):
    DdatasetY.append(DdatasetX[i])
DdatasetY = normalize(DdatasetY)#normalize to 0-1 for prevent gradients exploding
DdatasetX = normalize(DdatasetX)
print('test dataset ',end='')# check if the dataset x and y have the same len
print(len(DdatasetX[3]),'H',len(DdatasetY[3]))
if len(DdatasetX) == len(DdatasetY):
    print("ok")
    test1 = True 
else :
    test1 = False
    print('bad dataset')

#create the nn :neural_network(number of layers,[layers ex: 3,3,3],lr,momemtum)
nn = neural_network(4,[6,6,6,3],0.15,0.01)#define neural net smaller leraning rate more iter will take to end
print(nn.foward(DdatasetX[1]))#get a prediction from a random init net
#for plotting:

#prrls = []#prediction list for plotting
#tmperrs = []#tmp error list
###################
end = False
#iterations = 100
c = 0
errls13  =nn.train(DdatasetX,DdatasetY,50,stop=0.01,lr_steps = 500)#usaje neural-net.train(X,Y,proggress print,stop value for loss,lr_reduce steps)

#nn.save("iris")

#-------evaluate-------
future = []
future1 = []
nn.save("tutorial1",save_train_val = True)
nn_loaded = neural_network(2,[2,2],0.015,0.0001)#build a holder nn
nn_loaded.load("tutorial1")#load rebuilds the nn
for i in range(10):#iterating over the dataset
    print("predicted: ",nn.foward(DdatasetX[i]),"real: ",DdatasetY[i])
print("future test:")
values=nn.foward(DdatasetX[29])

value = [values[0]-random_float()/100,values[1]-random_float()/100,values[2]-random_float()/100]
for i in range(len(DdatasetX)):
    values=nn.foward(value)
    values1 = nn_loaded.foward(value)
    print("predicted step {}: ".format(i),value)
    value1 = [values[0],values[1],values[2]]
    value = [values[0]+random_float(),values[1]+random_float(),values[2]+random_float()]#add noise
    future.append(value)
    future1.append(value1)
#print(nn.weights)
#print(nn.bias)
#print(nn.derivatesW)
#print(nn.momentum)
#---------------plot data-------------------
#optional section if not using in micropython
#import matplotlib.pyplot as plt
#plt.plot(errls13)
#plt.show()
#plt.plot(DdatasetY)
#plt.show()
#plt.plot(future)
#plt.plot(future1)
#plt.show()
#plt.plot(prrls)
#plt.show()


