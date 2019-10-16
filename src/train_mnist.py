import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import GaussianNoise, Concatenate
from keras.initializers import he_normal, truncated_normal
from keras.optimizers import Adam
from keras.datasets import mnist

from bidirectional_infogan import BidirectionalInfoGAN

np.random.seed(100319)

bigan = BidirectionalInfoGAN()

max_iters = bigan.params['max_iters']
batch_size = bigan.params['batch_size']

# Load data
(X_train, _), (X_test,_) = mnist.load_data()
X_train = np.expand_dims(X_train, axis=3)
X_train = X_train.astype('float32')/255.0
X_train = 2.0*X_train - 1.0 # Scale [-1,1] for tanh
X_test = np.expand_dims(X_test, axis=3)
X_test = X_test.astype('float32')/255.0
X_test = 2.0*X_test - 1.0


print("Shape of X_train: ", np.shape(X_train))
data_size = np.shape(X_train)[0]
X_test_batch = X_test[:1000] # No need to run all

# Start training
t0 = time.time()
for iteration in range(max_iters):
    idx = np.random.choice(range(data_size), batch_size, replace=False)
    X_batch = X_train[idx,:,:,:]
    z_batch = bigan.sample_z(batch_size)
    c_batch = bigan.sample_c(batch_size)
    
    lossD, lossGE = bigan.train(X_batch, z_batch, c_batch)
    
    if iteration % 10 == 0:
        print("Iteration: %d; D_loss: %f; G_loss: %f; t: %ds" %
            (iteration, lossD[0], lossGE[0], time.time()-t0))
    
    if iteration % 1000 == 0:
        # Visualize categorical variables
        num_samples = 10
        num_cats = 10
        fig, axs = plt.subplots(num_cats, num_samples)
        z_test = bigan.sample_z(num_samples)
        c_base = bigan.sample_c(num_samples)
        c_dim = bigan.params['c_dim']
        c_base = np.random.uniform(-1.0, 1.0, size=(num_samples, c_dim))
        c_base[:,:num_cats] = 0
        for catid in range(num_cats):
            c_test = c_base.copy()
            c_test[:,catid] = 1
            encoding = np.concatenate((z_test, c_test), axis=1)
            gen = bigan.G.predict(encoding)
            gen = 0.5*gen + 0.5 # scale back to [0,1]
            
            for sample in range(num_samples):
                axs[catid, sample].imshow(gen[sample,:,:,0])
        
        # Add some handy labels
        for rid in range(num_cats):
            row_title = "Cat" + str(rid)
            axs[rid,0].set_ylabel(row_title,
                rotation=0, size='large', horizontalalignment='right')
        for rid in range(num_cats):
            for cid in range(num_samples):
                axs[rid,cid].set_xticks([])
                axs[rid,cid].set_yticks([])
        
        # Save the figure
        filename = "gen_cat_iter%06d.png" % iteration
        fig.savefig(os.path.join(bigan.get_log_dir(), filename))
        plt.close(fig)
        
        # Sample some real images and plot with encodings
        encoding = bigan.E.predict(X_test_batch)
        c_encoding = encoding[:,bigan.params['z_dim']:]
        fig, axs = plt.subplots(num_cats, num_samples)
        for catid in range(num_cats):
            topN_idx = np.argsort(c_encoding, axis=0)[-num_samples:, catid]
            for n in range(num_samples):
                axs[catid, n].imshow(X_test_batch[topN_idx[n],:,:,0])
        for rid in range(num_cats):
            row_title = "Cat" + str(rid)
            axs[rid,0].set_ylabel(row_title,
                rotation=0, size='large', horizontalalignment='right')
        for rid in range(num_cats):
            for cid in range(num_samples):
                axs[rid,cid].set_xticks([])
                axs[rid,cid].set_yticks([])
        filename = "real_cat_iter%06d.png" % iteration
        fig.savefig(os.path.join(bigan.get_log_dir(), filename))
        plt.close(fig)
        
        # Test continuous variables
        num_cont = bigan.params['num_cont_vars']
        fig, axs = plt.subplots(num_cont*2, num_samples)
        for varid in range(num_cont):
            topN_idx = np.argsort(c_encoding, axis=0)[-num_samples:, varid+num_cats]
            for n in range(num_samples):
                axs[varid*2, n].imshow(X_test_batch[topN_idx[n],:,:,0])
            bottomN_idx = np.argsort(c_encoding, axis=0)[:num_samples, varid+num_cats]
            for n in range(num_samples):
                axs[varid*2+1, n].imshow(X_test_batch[bottomN_idx[n],:,:,0])
        for rid in range(num_cont):
            row_title = "+Var" + str(rid)
            axs[rid*2,0].set_ylabel(row_title,
                rotation=0, size='large', horizontalalignment='right')
            row_title = "-Var" + str(rid)
            axs[rid*2+1,0].set_ylabel(row_title,
                rotation=0, size='large', horizontalalignment='right')
        for rid in range(num_cont*2):
            for cid in range(num_samples):
                axs[rid,cid].set_xticks([])
                axs[rid,cid].set_yticks([])
        filename = "real_cont_iter%06d.png" % iteration
        fig.savefig(os.path.join(bigan.get_log_dir(), filename))
        plt.close(fig)
    
    if iteration % 10000 == 0:
        bigan.save("iter%d" % iteration)
bigan.save("final")