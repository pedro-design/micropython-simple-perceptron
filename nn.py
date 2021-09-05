import math 
import random
import gc
gc.collect()
total = gc.mem_alloc() + gc.mem_free()
def free_mem():
    print("------------------")
    print(" {} mb used of {}".format(gc.mem_alloc()/8000,total/8000))
         
dataset=[[5.1,3.5,1.4,0.2],#I. setosa
         [4.9,3.0,1.4,0.2],
         [5.1,3.5,1.4 ,0.2],
         [7.0,3.2,4.7,1.4],#I. versicolor
         [6.4,3.2,4.5,1.5],
         [6.9,3.1,4.9,1.5]]
# mini iris dataset 
trueval=[[0],#I. setosa
         [0],
         [0],
         [1],#I. versicolor
         [1],
         [1]]




print('test dataset ',end='')# check if the dataset x and y have the same len
if len(dataset) == len(trueval):
    print("ok")
    test1 = True 
else :
    test1 = False
    print('bad dataset')
#-------helper functions -----------
def initW(nb):# init random weight
    W = []
    for i in range(nb):
        W.append(random.getrandbits(4)/10)
    return W
    del W 
def dot(w1,w2):# dot product
    d = []
    try:
        for a in range(len(w1)):
            for b in range(len(w2)):
                d.append(w1[a]*w2[b])
    except:
        for a in range(len(w1)):
            d.append(w1[a]*w2[0])
    return sum(d)

def error(t,p):# square error
    er = []
    try:
        for i in range(len(t)):
            er.append((t[i]-p) ** 2)
        return sum(er)
    except:
        return (p-t) ** 2
    del er
def derivate(p,t):#derivate
    return 2 * (p-t)


#---------DEFINE NETWOTK GRAPH ----------
#model arch is this 4 inputs, 2 hiden nodes,2 hiden nodes, 1 out
class neural_network():
    def __init__ (self):
        self.N1 = initW(1)#neuron 1 ,layer 1 
        self.N2 = initW(1)#neuron 2 ,layer 1
        self.N3 = initW(1)#neuron 1 ,layer 2 
        self.N4 = initW(1)#neuron 2 ,layer 2
        self.Nout1 = initW(1)#neuron 1 ,layer 3
        self.lr = 0.01# learning rate if nan prediction, lower this value
    def foward(self,x):
        # connect layers using dot([prevlayer neurons act],current neuron) example:
        #self.neuron1act = dot([self.InputV[0],self.InputV[1]:: the first 2 inputs],self.N1::neuron 1 )
        #-------input layer -------- 
        self.InputV = x
        #------layer 1 -------------
        self.p1 = dot([self.InputV[0],self.InputV[1]],self.N1) #add the correct number of input neurons
        self.p2 = dot([self.InputV[2],self.InputV[3]],self.N2)
         #------layer 2 ------------
        self.p3 = dot([self.p1,self.p2],self.N3)
        self.p4 = dot([self.p1,self.p2],self.N4)

        #------out layer  -------------
        self.pout =dot([self.p3,self.p4],self.Nout1)
        return self.pout#get outs
    
    def back(self,y):
        self.de = (derivate(self.pout,y[0]) * self.lr) #get the common derivate
        self.Nout1 = [self.Nout1[0] - (self.de*self.pout*self.Nout1[0])] #set out neuron weight
        #---------------layer 2 ---------------------
        self.N3= [self.N3[0] - (self.de* self.p3 * self.N3[0])]#apply grads
        self.N3dev = (self.de* self.p3 * self.N3[0])#get individual derivates
        self.N4= [self.N4[0] - (self.de* self.p4 * self.N4[0])]#apply grads
        self.N4dev = (self.de* self.p4 * self.N4[0])#get individual derivates
        #---------------layer 1 ---------------------
        self.N1= [self.N1[0] - (self.N3dev* self.p1 * self.N1[0])]#apply grads
        self.N2= [self.N2[0] - (self.N4dev* self.p2 * self.N2[0])]#apply grads
#-----------------------------------------------------grads are calculated using the chain rule
        
nn = neural_network()#define neural net 
print(nn.foward(dataset[1]))#get a prediction from a random init net
errls1 = []#error list for plotting
prrls = []#prediction list for plotting
tmperrs = []#tmp error list
end = False
iterations = 100
c = 0
for i in range(iterations):#train
    if test1 == True and end == False:
        if c == int(iterations /30):#simple progress bar
            print("-",end = "")
            c = 0
        tmperrs = []#tmp error list
        for w in range(len(dataset)):#pass over the dataset
            prrls.append( nn.foward(dataset[w]))#get predictions and add to the list
            tmperrs.append( error(trueval[w],nn.foward(dataset[w])))#get error of the nn and add to its list
            nn.back(trueval[w])#back propagation
        c = c+1 
        if sum(tmperrs) > 6 :#protection anti-nan
                end = True
                break
            
        errls1.append(sum(tmperrs))#get the sum of temp error list and add it to the error list
print("#")
free_mem()
#-------evaluate-------
for i in range(len(dataset)):#iterating over the dataset
    print("predicted: ",nn.foward(dataset[i]),"real: ",trueval[i])
    
#---------------plot data-------------------
#optional section if not using in micropython
#import matplotlib.pyplot as plt

#print("last error ",sum(tmperrs))
#plt.plot(errls1)
#plt.show()
#plt.plot(prrls)
#plt.show()


