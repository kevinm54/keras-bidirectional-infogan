import os
import numpy as np

import keras
from keras.datasets import mnist

from bidirectional_infogan import BidirectionalInfoGAN

src_dir = os.path.dirname(__file__)
base_dir = os.path.join(src_dir, "../")
log_dir = os.path.join(base_dir, "logs/test/")
bigan = BidirectionalInfoGAN(load_dir=log_dir, model_suffix="final")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, axis=3)
X_train = X_train.astype('float32')/255.0
X_train = 2.0*X_train - 1.0 # Scale [-1,1] for tanh
X_test = np.expand_dims(X_test, axis=3)
X_test = X_test.astype('float32')/255.0
X_test = 2.0*X_test - 1.0

accuracy = bigan.evaluate(X_test, y_test, match='auto')

print("accuracy = ", accuracy)