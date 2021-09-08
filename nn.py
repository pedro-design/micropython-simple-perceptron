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
    def __init__ (self,layers,nodes,lr):
        self.weights = [[] for i in range(layers)]
        self.nl=layers
        self.predictions = [[] for i in range(layers)]
        print("Dense nodes : ",nodes)
        self.predictions = [[] for i in range(self.nl)]
        self.derivatesW = [[0] for i in range(layers)]
        for i in range(len(self.weights)):
            self.weights[i] = [initW(1) for q in range(nodes[i])]
            self.predictions[i] = [0 for q in range(nodes[i])]
            self.derivatesW[i] =  [0 for q in range(nodes[i])]
        self.lr = 0.05# learning rate
        self.weights[-1] = [initW(1) for q in range(nodes[-1])]
        self.momentum = 0
    def foward(self,x):
        self.InputV = x
        if len(self.InputV) > len(self.weights[0]) :
            for i in range(len(self.weights[0]))  :
                self.predictions[0][i]=dot([self.InputV[i],self.InputV[i+1]],self.weights[0][i])
            if self.d == 0:
                self.d = 1
                print("###### WARNING ######")
                print("using minor number of neurons than the data feed, can cause undercoverage problems or slow learning")
                print("")
        elif len(self.InputV) < len(self.weights[0]) :
              raise Exception('use the same number of input neurons as the data feed')
        else:
            for i in range(len(self.InputV)):
                self.predictions[0][i]=dot([self.InputV[i]],self.weights[0][i])
        for q in range(len(self.weights)):#iterate over layers
            for x in range(len(self.weights[q])):
                self.predictions[q][x]=dot([self.InputV[e] for e in range(len(self.weights[q]))],self.weights[q][x])
        if len(self.predictions[-1]) == 1 :
            return self.predictions[-1][0]#get outs
        else :
            return self.predictions[-1]
    
    def back(self,y):#back propagate error
        self.derivateR=[(derivate(self.predictions[-1][s],y[s]) * self.lr)+self.momentum for s in range(len(y))]
        self.momentum=sum(self.derivateR)* self.lr
        self.derivatesW[-1] = [self.weights[-1][r][0] - (self.weights[-1][r][0]*self.predictions[-1][r]*self.derivateR[r]) for r in range(len(y))]
        self.weights[-1]=[[self.derivatesW[-1][i]] for i in range(len(self.derivatesW[-1]))]
        for q in reversed(range(len(self.weights)-1)):  #iterate over layers and nodes
            for x in range(len(self.weights[q])):
                 if q == 0:                
                         for t in range(len(self.derivatesW[q+1])):
                             self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][t])      
                 if len(self.derivatesW[q+1]) ==1:
                           self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][0])
                 else:
                         for t in range(len(self.derivatesW[q+1])):
                                    self.derivatesW[q][x] = self.weights[q][x][0] - (self.weights[q][x][0]*self.predictions[q][x]*self.derivatesW[q+1][t])
                 self.weights[q][x] = [self.derivatesW[q][x]]
            

#-----------------------------------------------------grads are calculated using the chain rule