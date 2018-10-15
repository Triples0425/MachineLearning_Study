#import tensorflow as tf

#hello = tf.constant('Hello tenserflow')
#sess = tf.Session()
#print(sess.run(hello))

# 탭 말고, 스페이스 4번!!

import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pylab as plt
from MyCustom import *
from OptionalFile.mnist import load_mnist
from PIL import Image

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 형상을 원래 이미지의 크기로 변형
print(img.shape)  # (28, 28)

img_show(img)








