import math 
import random
import nn
from timeit import timeit
from nn import neural_network, error,normalize
#import machine
#machine.freq(160000000)
#dummy wheater  dataset
dummy = [100,110]
DdatasetX = []
DdatasetY = [[100,110]]
nsamples = 300
for i in range(nsamples):
    if random.getrandbits(2) == 1:
        dummy=[dummy[0]+random.getrandbits(4)-random.getrandbits(4),dummy[1]+random.getrandbits(4)-random.getrandbits(4)]
    else:
        dummy=[dummy[0]-random.getrandbits(4)+random.getrandbits(4),dummy[1]-random.getrandbits(4)+random.getrandbits(4)]
    DdatasetX.append(dummy)
for i in range(1,len(DdatasetX)):
    DdatasetY.append(DdatasetX[i])
DdatasetY = normalize(DdatasetY)#normalize to 0-1 for prevent gradients exploding
DdatasetX = normalize(DdatasetY)
print('test dataset ',end='')# check if the dataset x and y have the same len
if len(DdatasetX) == len(DdatasetY):
    print("ok")
    test1 = True 
else :
    test1 = False
    print('bad dataset')

        
nn = neural_network(4,[2,2,2,2],0.00015,0.01)#define neural net smaller leraning rate more iter will take to end
print(nn.foward(DdatasetX[1]))#get a prediction from a random init net
#for plotting:

#prrls = []#prediction list for plotting
#tmperrs = []#tmp error list
###################
end = False
iterations = 100
c = 0
errls13  =nn.train(DdatasetX,DdatasetY,10,stop=0.01)#usaje neural-net.train(X,Y,proggress print,stop value for loss)

#nn.save("iris")

#-------evaluate-------
future = []


for i in range(10):#iterating over the dataset
    print("predicted: ",nn.foward(DdatasetX[i]),"real: ",DdatasetY[i])
print("future test:")
values=nn.foward(DdatasetX[29])
value = [values[0],values[1]]
for i in range(len(DdatasetX)):
    values=nn.foward(value)
    print("predicted step {}: ".format(i),value)
    value = [values[0]+random.getrandbits(1)-random.getrandbits(1),values[1]+random.getrandbits(1)-random.getrandbits(1)]#add noise
    future.append(value)
#---------------plot data-------------------
#optional section if not using in micropython
import matplotlib.pyplot as plt
plt.plot(errls13)
plt.show()
plt.plot(DdatasetY)
plt.show()
plt.plot(future)
plt.show()
#plt.plot(prrls)
#plt.show()


