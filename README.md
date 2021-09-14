# micropython-simple-perceptron
simple lib for making perceptron in micropython
this lib includes the build blocks for creating neural nets using only micropython libs
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

nnEXAMPLE = neural_network(4,[6,6,6,3],0.15,0.01) #nlayers,[layers],lr,momentum
nnEXAMPLE.train(X,Y,50,stop=0.01,lr_steps = 50)#x,y, callbacks , lr_Stop,lr_decay_steps
nnEXAMPLE.save("tutorial1",save_train_val = True)
nnEXAMPLE.foward(X[0])#predict values
nn_loaded = neural_network(2,[2,2],0.015,0.0001)#build a holder nn
nn_loaded.load("tutorial1")#load rebuilds the nn
nn_loaded.foward(X[0])#predict will be the same as the firstone
```
