from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
def cross_entropy_error(y, t):
	# y为1维时，即
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
y=np.array([[0,1,0,0,0,0,0,0,0,0],[0,0,0.2,0.8,0,0,0,0,0,0]])
t_onehot=np.array([[0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0]])#one-hot
t = t_onehot.argmax(axis=1)#非one-hot
print('hello')