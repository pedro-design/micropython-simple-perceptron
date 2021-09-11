import math 
import random
#import pickle
#-------helper functions -----------
def normalize(V):
    res = []
    max_, min_ = max(V), min(V)
    print(max_,min_)
    max_, min_ = max(max_), min(min_)
    print(max_,min_)
    for i in V:
      #  print(i)
        
        res.append([(j - min_)/(max_ - min_) for j in i])
    return res
    
def initW(nb):# init random weight
    W = []
    for i in range(nb):
        W.append((random.getrandbits(8)/255)/2)
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
        try:
            return [(p[0]-t[0]) ** 2,(p[1]-t[1]) ** 2]
        except:
            return (p-t) ** 2
        
    del er
def derivate(p,t):#derivate
    return 2 * (p-t)


#---------DEFINE NETWOTK GRAPH ----------

class neural_network():
    def __init__ (self,layers,nodes,lr,m):
        self.weights = [[] for i in range(layers)]
        self.nl=layers
        self.layers = nodes
        self.bias =[[] for i in range(layers)]
        self.predictions = [[] for i in range(layers)]
        print("Dense nodes : ",nodes)
        self.predictions = [[] for i in range(self.nl)]
        self.derivatesW = [[0] for i in range(layers)]
        self.m = m
        self.momentum = [[0] for i in range(layers)]
        for i in range(len(self.weights)):
            self.weights[i] = [initW(1) for q in range(nodes[i])]
            self.predictions[i] = [0 for q in range(nodes[i])]
            self.derivatesW[i] =  [0 for q in range(nodes[i])]
            self.bias[i] = [0 for q in range(nodes[i])]
            self.momentum[i]= [0 for q in range(nodes[i])]
        self.lr = lr# learning rate
        self.weights[-1] = [initW(1) for q in range(nodes[-1])]
       # self.momentum = 0
        self.d = 0
    def train(self,X,Y,iterations,autorun = False,stop=1):
        self.c = 0
        self.end = 0
        self.counter = 0
        self.counter1 = 0
        self.errls13 = []#error list for plotting
        self.ploss= 1
        while(self.end == 0):
            self.tmperrs1 = []#tmp error list
            for w in range(len(X)):#pass over the dataset
              self.foward(Y[w])#get predictions and update internal state
              try:
                  self.tmperrs1.append(sum(error(Y[w],self.foward(X[w]))))#get error of the nn and add to its list
              except:
                  self.tmperrs1.append(error(Y[w],self.foward(X[w])))
              self.back(Y[w])#back propagation 
            if self.c == int(iterations):#simple progress bar
                    print("iter: {} loss: {} ".format(self.counter1,sum(self.tmperrs1)))
                    self.c = 0
            
            
            self.c = self.c+1
            if sum(self.tmperrs1) < stop:
                self.end = 1
                print("#")
            self.counter = self.counter +1
            self.counter1 = self.counter1 +1
            self.errls13.append(sum(self.tmperrs1))#get the sum of temp error list and add it to the error list
            if self.ploss == sum(self.tmperrs1):
                print("model not learning, end training")
                self.end = 1
            if self.counter > 2 :
                self.counter = 0
                self.ploss = sum(self.tmperrs1)
            if (sum(self.tmperrs1) > 100 and self.counter >10 )or sum(self.tmperrs1) < 0.0001  :#protection anti-nan
                    raise Exception('gradients Exploding, try reducing the layers or changing lr and momentum')
                    break
            
        
        print("#")
        print("last error: ",sum(self.tmperrs1))
        return self.errls13
        textfile.close()
    def foward(self,x):
        self.InputV = x
        if len(self.InputV) > len(self.weights[0]) :
            for i in range(len(self.weights[0]))  :
                self.predictions[0][i]=dot([self.InputV[i],self.InputV[i+1]],self.weights[0][i])+self.bias[0][i]
            if self.d == 0:
                self.d = 1
                print("###### WARNING ######")
                print("using minor number of neurons than the data feed, can cause undercoverage problems or slow learning")
                print("")
        elif len(self.InputV) < len(self.weights[0]) :
              raise Exception('use the same number of input neurons as the data feed')
        else:
            for i in range(len(self.InputV)):
                self.predictions[0][i]=dot([self.InputV[i]],self.weights[0][i])+self.bias[0][i]
        for q in range(len(self.weights)):#iterate over layers
            for x in range(len(self.weights[q])):
                    self.predictions[q][x]=dot([self.InputV[e] for e in range(len(self.weights[q]))],self.weights[q][x])+self.bias[q][x]
              
        if len(self.predictions[-1]) == 1 :
            return self.predictions[-1][0]#get outs
        else :
            return self.predictions[-1]
    
    def back(self,y):#back propagate error
        
        self.derivateR=[(derivate(self.predictions[-1][s],y[s]) * self.lr)for s in range(len(y))]
        self.momentum[-1]=[[self.derivateR[i]* self.m] for i in range(len(self.derivateR))]
        self.bias[-1] = self.derivateR
        self.derivatesW[-1] = [self.weights[-1][r][0] - (self.weights[-1][r][0]*self.predictions[-1][r]*self.derivateR[r])+self.momentum[r][0]  for r in range(len(y))]
        self.weights[-1]=[[self.derivatesW[-1][i]] for i in range(len(self.derivatesW[-1]))]
        for q in reversed(range(len(self.weights)-1)):  #iterate over layers and nodes
            for x in range(len(self.weights[q])):
                
                 if q == 0:                
                         for t in range(len(self.derivatesW[q+1])):
                             self.momentum[q][t] = self.derivatesW[q+1][t]* self.m
                             self.derivatesW[q][t] = self.weights[q][t][0] - (self.weights[q][t][0]*self.predictions[q][t]*self.derivatesW[q+1][t])+self.momentum[q][t] 
                             self.bias[q][t] = self.derivatesW[q+1][t]
                 if len(self.derivatesW[q+1]) ==1:
                           self.momentum[q][t] = self.derivatesW[q+1][0]* self.m
                           self.bias[q][x] = self.derivatesW[q+1][0]
                           self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][0])+self.momentum[q][0] 
                 else:
                     if len(self.derivatesW[q+1]) > len(self.derivatesW[q]):
                         self.tmp = 0
                         self.ctmp = 0
                         while(self.tmp != len(self.derivatesW[q+1])-1 or  self.ctmp != len(self.derivatesW[q]) ):
                              self.momentum[q][self.ctmp] = self.derivatesW[q+1][self.ctmp]* self.m
                              self.derivatesW[q][self.ctmp] = self.weights[q][self.ctmp][0] - (self.weights[q][self.ctmp][0]*self.predictions[q][self.ctmp]*self.derivatesW[q+1][self.tmp])+self.momentum[q][self.tmp] 
                              self.tmp = self.tmp +1
                              self.ctmp = self.ctmp+1
                              self.bias[q][x] = self.derivatesW[q+1][self.tmp]
                   #      print(">",self.tmp,len(self.derivatesW[q+1]) )
                         self.derivatesW[q][x] = sum([self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][self.tmp-1]),self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][self.tmp])])
                     elif len(self.derivatesW[q+1]) == len(self.derivatesW[q]) :
                           self.momentum[q][x] = self.derivatesW[q+1][x]* self.m
                           self.bias[q][x] = self.derivatesW[q+1][x]
                           self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][x])

                     else:
                    #     print("<",len(self.derivatesW[q+1]) / len(self.derivatesW[q]))
                         for t in range(len(self.derivatesW[q+1])):
                                    self.momentum[q][x] = self.derivatesW[q+1][x]* self.m
                                    self.bias[q][x] = self.derivatesW[q+1][t]
                                    self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][t])+self.momentum[q][t]
                 if math.isinf(self.derivatesW[q][x]) == True or math.isnan(self.derivatesW[q][x]) ==True  or  math.isnan(self.weights[q][x][0]) == True or math.isinf(self.derivatesW[q][x]) == True or math.isnan(self.derivatesW[q][x]) == True:

                    self.derivatesW[q][x] = 0.0001
                    self.weights[q][x] =initW(1)
                 else:
                     self.weights[q][x] = [self.derivatesW[q][x]]
                 
            

#-----------------------------------------------------grads are calculated using the chain rule