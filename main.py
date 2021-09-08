import math 
import random
import nn
from nn import neural_network, error
dataset=[[5.1,3.5,1.4,0.2],#I. setosa
         [4.9,3.0,1.4,0.2],
         [5.1,3.5,1.4 ,0.2],
         [7.0,3.2,4.7,1.4],#I. versicolor
         [6.4,3.2,4.5,1.5],
         [6.9,3.1,4.9,1.5]]
# mini iris dataset 
trueval=[[0,1],#I. setosa
         [0,1],
         [0,1],
         [1,0],#I. versicolor
         [1,0],
         [1,0]]




print('test dataset ',end='')# check if the dataset x and y have the same len
if len(dataset) == len(trueval):
    print("ok")
    test1 = True 
else :
    test1 = False
    print('bad dataset')

        
nn = neural_network(8,[4,4,3,3,3,2,2,2],0.15)#define neural net 
print(nn.foward(dataset[1]))#get a prediction from a random init net
#for plotting:
errls13 = []#error list for plotting
#prrls = []#prediction list for plotting
#tmperrs = []#tmp error list
###################
end = False
iterations = 20
c = 0

for i in range(iterations):#train
    if test1 == True and end == False:
        if c == int(iterations /30):#simple progress bar
            print("-",end="")
            c = 0
        tmperrs1 = []#tmp error list
        for w in range(len(dataset)):#pass over the dataset
            nn.foward(dataset[w])#get predictions and update internal state
            tmperrs1.append(sum(error(trueval[w],nn.foward(dataset[w]))))#get error of the nn and add to its list
            nn.back(trueval[w])#back propagation
           # print("-sd",error(trueval[w],nn.foward(dataset[w])))
        c = c+1
       # print(sum(tmperrs))
       #debug data :
  #      errls13.append(sum(tmperrs1))#get the sum of temp error list and add it to the error list
        if sum(tmperrs1) > 90 or sum(tmperrs1) < 0.1  :#protection anti-nan
                print("train is failing")
                end = True
                break
            
        
print("#")
#-------evaluate-------
print("last error ",sum(tmperrs1))

for i in range(len(dataset)):#iterating over the dataset
    print("predicted: ",nn.foward(dataset[i]),"real: ",trueval[i])
    
#---------------plot data-------------------
#optional section if not using in micropython
# import matplotlib.pyplot as plt
#plt.plot(errls13)
#plt.show()
#plt.plot(prrls)
#plt.show()


