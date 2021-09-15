# micropython-simple-perceptron
simple lib for making perceptron in micropython
this lib includes the build blocks for creating neural nets using only micropython libs
its train funtion is currently SGD + Momentum

# to do next

- [X] create the lib and see good results
- [X] add squared loss
- [X] add linear act
- [X] add all the basic math functions
- [ ] add more losses
- [X] train function and momemtum
- [ ] add more activations functions
- [x] add a compiler for easy usage
- [x] added save and load

simple guide :
```python
import math 
import random
import nn
from nn import neural_network, error,normalize,random_float
#random_float produces a random float
print(random_float)
#normalize is a good tool for nn, its normalize data from 0 to 1
print(normalize([[0,1,2,3,4,5],[6,7,8,9,10]]))
nnEXAMPLE = neural_network(4,[6,6,6,3],0.15,0.01) #nlayers,[layers],lr,momentum(momentum is important to tune , if to high the model will not learn,if to low, it will get stuck at a local minima)
#if you whant to fit 1 value, use this:
nnEXAMPLE.foward(value_to_feed)
nnEXAMPLE.back(true_value)
#---------------------
nnEXAMPLE.train(X,Y,50,stop=0.01,lr_steps = 50)#x,y, callbacks , lr_Stop,lr_decay_steps
nnEXAMPLE.save("tutorial1",save_train_val = True)#its saves as json
nnEXAMPLE.foward(X[0])#predict values
nn_loaded = neural_network(2,[2,2],0.015,0.0001)#build a holder nn
nn_loaded.load("tutorial1")#load rebuilds the nn
nn_loaded.foward(X[0])#predict will be the same as the first one
```
if you want more layers at the input layers, the layers need to be the double of input data, or the number of values of the input data,if they are less layers,will cause undercoveraje

note : this lib uses a good amout of ram, it dont run on a esp8266 and its 86 kb of ram
