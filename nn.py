import math 
import random
#-------helper functions -----------
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
        self.bias =[[] for i in range(layers)]
        self.predictions = [[] for i in range(layers)]
        print("Dense nodes : ",nodes)
        self.predictions = [[] for i in range(self.nl)]
        self.derivatesW = [[0] for i in range(layers)]
        self.m = m
        for i in range(len(self.weights)):
            self.weights[i] = [initW(1) for q in range(nodes[i])]
            self.predictions[i] = [0 for q in range(nodes[i])]
            self.derivatesW[i] =  [0 for q in range(nodes[i])]
            self.bias[i] = [0 for q in range(nodes[i])]
        self.lr = lr# learning rate
        self.weights[-1] = [initW(1) for q in range(nodes[-1])]
        self.momentum = 0
        self.d = 0
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
        self.derivateR=[(derivate(self.predictions[-1][s],y[s]) * self.lr)+self.momentum for s in range(len(y))]
        self.momentum=sum(self.derivateR)* self.m
        self.bias[-1] = self.derivateR
        self.derivatesW[-1] = [self.weights[-1][r][0] - (self.weights[-1][r][0]*self.predictions[-1][r]*self.derivateR[r]) for r in range(len(y))]
        self.weights[-1]=[[self.derivatesW[-1][i]] for i in range(len(self.derivatesW[-1]))]
        for q in reversed(range(len(self.weights)-1)):  #iterate over layers and nodes
            for x in range(len(self.weights[q])):
                 if math.isinf(self.derivatesW[q][x]) == True:
                    self.derivatesW[q][x] = 0
                 if q == 0:                
                         for t in range(len(self.derivatesW[q+1])):
                             self.derivatesW[q][t] = self.weights[q][t][0] - (self.weights[q][t][0]*self.predictions[q][t]*self.derivatesW[q+1][t])
                             self.bias[q][t] = self.derivatesW[q+1][t]
                 if len(self.derivatesW[q+1]) ==1:
                           self.bias[q][x] = self.derivatesW[q+1][0]
                           self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][0])
                 else:
                     if len(self.derivatesW[q+1]) > len(self.derivatesW[q]):
                         self.tmp = 0
                         self.ctmp = 0
                         while(self.tmp != len(self.derivatesW[q+1])-1 or  self.ctmp != len(self.derivatesW[q]) ):
                              self.derivatesW[q][self.ctmp] = self.weights[q][self.ctmp][0] - (self.weights[q][self.ctmp][0]*self.predictions[q][self.ctmp]*self.derivatesW[q+1][self.tmp])
                              self.tmp = self.tmp +1
                              self.ctmp = self.ctmp+1
                              self.bias[q][x] = self.derivatesW[q+1][self.tmp]
                   #      print(">",self.tmp,len(self.derivatesW[q+1]) )
                         self.derivatesW[q][x] = sum([self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][self.tmp-1]),self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][self.tmp])])
                     elif len(self.derivatesW[q+1]) == len(self.derivatesW[q]) :
                           self.bias[q][x] = self.derivatesW[q+1][x]
                           self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][x])

                     else:
                    #     print("<",len(self.derivatesW[q+1]) / len(self.derivatesW[q]))
                         for t in range(len(self.derivatesW[q+1])):
                                    self.bias[q][x] = self.derivatesW[q+1][t]
                                    self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][t])
                 self.weights[q][x] = [self.derivatesW[q][x]]
            

#-----------------------------------------------------grads are calculated using the chain rule