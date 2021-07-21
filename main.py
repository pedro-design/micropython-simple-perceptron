import nn
from nn import *
import math
#dummy dataset for this example
dataset = [[2.7810836,2.550537003],
    [1.465489372,2.362125076],
    [3.396561688,4.400293529],
    [1.38807019,1.850220317,
    [3.06407232,3.005305973],
    [7.627531214,2.759262235],
    [5.332441248,2.088626775],
    [6.922596716,1.77106367],
    [8.675418651,-0.242068655],
    [7.673756466,3.508563011]]
expexted=[[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]
x = 0
Dense1 = Dense_1x2("Input1")
hiden1=Dense_1x1("h1")
hiden2=Dense_1x1("h2")
Dense2 = Dense_2x1("Output1")

model = [[Dense1],[hiden1,hiden2],[Dense2]]
#----list model arch--------
view_arch(model)
a = 0
free_mem()#---get free ram 
print("training")#for training you need to build the graph for compute and predict the data then apply gradients
def model_foward(dataset,x):#u can change the name off this function its juts the model compute ops
        Dout1 = Dense1.predict(dataset[x][0])#input OPS   model:
        hiden1p=hiden1.predict(Dout1[0])#hiden op           . -: : :-.
        hiden2p= hiden2.predict(Dout1[1])#hiden op          1->2  1-1 2->1
        Dout2 = Dense2.predict([hiden1p,hiden2p])
        return Dout1,hiden1p,hiden2p,Dout2 #dont forget return the ops 
#machine.freq(160000000) #--- use this for change cpu frec of you mcu for fast training. Use the frecs of you board
for i in range(100):#iterations
    if a == 100/10:#simple progress bar
        a = 0
        print("-",end="")
    for x in range(len(dataset)):
        #Dout1 = Dense1.predict(dataset[x])
        Dout1,hiden1p,hiden2p,Dout2  = model_foward(dataset,x)#use the model compute function
        Dense1.apply_grad(dataset[x][0],[expexted[x][0]-hiden1p,expexted[x][0]-hiden2p])#input gradients
        hiden1.apply_grad(Dout1[0],expexted[x][0]-Dout2)#this grad is for out 1 of input neuron 
        hiden2.apply_grad(Dout1[1],expexted[x][0]-Dout2)#how works: its taking the out 1 of the input layer and adapting for the next layer 
        Dense2.apply_grad([hiden1p,hiden2p],expexted[x][0])#to use
     
    a = a+1
#machine.freq(80000000) #--- use this for change cpu frec of the mcu for low power use the frecs of you board
print("#")#train is over

for x in range(len(dataset)):
        Dout1,hiden1p,hiden2p,Dout2 = model_foward(dataset,x)#call function and set value ops
        print("prediction: ","input act:",[Dout1],"hiden acts:",[hiden1p,hiden2p],"out act :",[Dout2],"target : ",expexted[x][0],"mean squared error:",mse(Dout2,expexted[x][0]))
#view new weights in arch
view_arch(model)


