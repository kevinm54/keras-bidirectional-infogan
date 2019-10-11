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

from utils import *

np.random.seed(100319)

# Hyperparameters
BATCH_SIZE = 128                    # Number of images per batch
MAX_ITERS = 30000                   # Maximum number of iterations
IMG_WIDTH, IMG_HEIGHT = 28, 28      # Image dimensions
IMG_CHANNELS = 1                    # Number of image channels
LR_D = 1e-4                         # Learning rate for discriminator
LR_G = 5e-4                         # Learning rate for generator and encoder
LR_RED = 5                          # Learning rate decay factor
LD_D = 1e-3
LD_G = 1e-3
FINAL_LR_D = LR_D / LR_RED          # Final learning rate for discriminator
FINAL_LR_G = LR_G / LR_RED          # Final learning rate for generator and encoder
DECAY_STEPS_D = MAX_ITERS           # Number of steps over which the learning rate is decayed (D)
DECAY_STEPS_G = MAX_ITERS           # Number of steps over which the learning rate is decayed (G & E)
BETA1_D = 0.5                       # Beta1 value for Adam optimizer (D)
BETA1_G = 0.5                       # Beta1 value for Adam optimizer (G & E)

Z_DIM = 16                          # Dimensionality of the z vector (noise)
CONT_VARS = 2                       # Number of continuous variables
DISC_VARS = [10]                    # Number of categorical classes per categorical variable

NUM_DISC_VARS = 0
for cla in DISC_VARS:
    NUM_DISC_VARS += cla            # Total number of discrete variables
C_DIM = NUM_DISC_VARS + CONT_VARS   # Dimensionality of the c vector (target variables)

# Create paths and log parameter info
timestamp = time.strftime('%y-%m-%d-%H%M%S', time.localtime())
src_dir = os.path.dirname(__file__)
base_dir = os.path.join(src_dir, "../")
log_dir = os.path.join(base_dir, "logs/mnist_" + timestamp)
os.makedirs(log_dir, exist_ok=True)

# Activation functions for Generator, Discriminator, and Encoder
g_activation = 'elu'
e_activation = 'elu'
d_activation = 'elu'

# Placeholders for the input values
X = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
z = Input(shape=(Z_DIM,))
c = Input(shape=(C_DIM,))
phase = True

def conv2d_bn_act(
        inputs,
        filters,
        kernel_size,
        kernel_init,
        activation,
        strides,
        noise = False,
        noise_std = 0.5,
        padding = 'valid'):
    """
        Shortcut for a module of convolutional layer, batch normalization and
                possibly adding of Gaussian noise.
        :param inputs: input data
        :param filters: number of convolutional filters
        :param kernel_size: size of filters
        :param kernel_init: weight initialization
        :param activation: activation function (applied after batch normalization)
        :param strides: strides of the convolutional filters
        :param noise: whether to add gaussian noise to the output
        :param noise_std: stadnard deviation of added noise
        :param padding: padding in the conv layer
        :return: output data after applying the conv layer, batch norm, activation
                function and possibly Gaussian noise
        """
    
    _tmp = Conv2D(
        filters,
        kernel_size,
        kernel_initializer=kernel_init,
        strides=strides,
        padding=padding,
        activation=None)(inputs)
    # TODO: set is_training=phase for BatchNormalization
    _tmp = BatchNormalization(
        center=True,
        scale=True)(_tmp)
    _tmp = Activation(activation)(_tmp)
    if noise:
        # TODO: set is_training=phase for GaussianNoise
        _tmp = GaussianNoise(noise_std)(_tmp)
    
    return _tmp

def deconv2d_bn_act(
        inputs,
        filters,
        kernel_size,
        kernel_init,
        activation,
        strides,
        padding='same'):
    """
        Shortcut for a module of transposed convolutional layer, batch
            normalization.
        :param inputs: input data
        :param filters: number of convolutional filters
        :param kernel_size: size of filters
        :param kernel_init: weight initialization
        :param activation: activation function (applied after batch normalization)
        :param strides: strides of the convolutional filters
        :param padding: padding in the conv layer
        :return: output data after applying the transposed conv layer,
            batch norm, and activation function
        """
    _tmp = Conv2DTranspose(
        filters,
        kernel_size,
        kernel_initializer=kernel_init,
        activation=None,
        strides=strides,
        padding=padding)(inputs)
    # TODO: set is_training=phase for BatchNormalization
    _tmp = BatchNormalization(
        center=True,
        scale=True)(_tmp)
    _tmp = Activation(activation)(_tmp)
    
    return _tmp

def dense_bn_act(
        inputs,
        units,
        activation,
        kernel_init,
        noise=False,
        noise_std=0.5):
    """
        Shortcut for a module of dense layer, batch normalization and possibly
            adding of Gaussian noise.
        :param inputs: input data
        :param units: number of units
        :param activation: activation function (applied after batch normalization)
        :param kernel_init: weight initialization
        :return: output data after applying the dense layer, batch norm,
            activation function and possibly Gaussian noise
        """
    
    _tmp = Dense(
        units,
        activation=None,
        kernel_initializer=kernel_init)(inputs)
    _tmp = BatchNormalization(
        center=True,
        scale=True)(_tmp)
    _tmp = Activation(activation)(_tmp)
    if noise:
        # TODO: set is_training=phase for GaussianNoise
        _tmp = GaussianNoise(noise_std)(_tmp)
    
    return _tmp

# Discriminator
def createDiscriminator():
    img_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    noise_input = Input(shape=(Z_DIM+C_DIM,))
    
    # Image discriminator
    d_x_conv_0 = conv2d_bn_act(
        inputs=img_input,
        filters=64,
        kernel_size=3,
        kernel_init='he_normal',
        activation=d_activation,
        strides=2,
        noise=True,
        noise_std=0.3)
    d_x_conv_1 = conv2d_bn_act(
        inputs=d_x_conv_0,
        filters=128,
        kernel_size=3,
        kernel_init='he_normal',
        activation=d_activation,
        strides=2,
        noise=True)
    d_x_conv_1 = Flatten()(d_x_conv_1)
    d_x_dense = dense_bn_act(
        inputs=d_x_conv_1,
        units=512,
        activation=d_activation,
        kernel_init='he_normal',
        noise=True)
    
    # noise discriminator
    noise_input_vec = Reshape((1, 1, Z_DIM+C_DIM))(noise_input)
    d_z_conv_0 = conv2d_bn_act(
        inputs=noise_input_vec,
        filters=64,
        kernel_size=1,
        kernel_init='he_normal',
        activation=d_activation,
        strides=1,
        noise=True,
        noise_std=0.3)
    d_z_conv_1 = conv2d_bn_act(
        inputs=d_z_conv_0,
        filters=128,
        kernel_size=1,
        kernel_init='he_normal',
        activation=d_activation,
        strides=1,
        noise=True)
    d_z_conv_1 = Flatten()(d_z_conv_1)
    d_z_dense = dense_bn_act(
        inputs=d_z_conv_1,
        units=512,
        activation=d_activation,
        kernel_init='he_normal',
        noise=True)
    
    # final discriminator
    inp = Concatenate()([d_x_dense, d_z_dense])
    d_final_dense = dense_bn_act(
        inputs=inp,
        units=1024,
        activation=d_activation,
        kernel_init='he_normal',
        noise=True)
    
    # final prediction whether input is real or generated
    d_final_pred = Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer='he_normal')(d_final_dense)
    
    return Model(inputs=[img_input, noise_input], outputs=[d_final_pred])

D = createDiscriminator()
def discriminate(img_input, noise_input):
    return D([img_input, noise_input])

# Encoder + Mutual Information
def createEncoder():
    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    e_conv_0 = conv2d_bn_act(
        inputs=image_input,
        filters=32,
        kernel_size=3,
        kernel_init='he_normal',
        activation=e_activation,
        strides=1)
    e_conv_1 = conv2d_bn_act(
        inputs=e_conv_0,
        filters=64,
        kernel_size=3,
        kernel_init='he_normal',
        activation=e_activation,
        strides=2)
    e_conv_2 = conv2d_bn_act(
        inputs=e_conv_1,
        filters=128,
        kernel_size=3,
        kernel_init='he_normal',
        activation=e_activation,
        strides=2)
    e_conv_2 = Flatten()(e_conv_2)
    e_dense_0 = dense_bn_act(
        inputs=e_conv_2,
        units=1024,
        activation=e_activation,
        kernel_init='he_normal')

    # prediction for z, i.e. the noise part of the representation
    e_dense_z = Dense(
        units=Z_DIM,
        activation='tanh',
        kernel_initializer='he_normal')(e_dense_0)
    
    # prediction for categorical variables of c
    e_dense_c_disc = []
    for idx, classes in enumerate(DISC_VARS):
        e_dense_c_disc.append(Dense(
            units=classes,
            activation='softmax',
            kernel_initializer='he_normal',
            name="e_dense_c_disc_" + str(idx))(e_dense_0))
    print("shape of e_dense_c_disc: ", np.shape(e_dense_c_disc))
    print("size of e_dense_c_disc: ", np.size(e_dense_c_disc))
    if np.size(e_dense_c_disc) > 1:
        e_dense_c_disc_concat = Concatenate()(e_dense_c_disc)
    else:
        e_dense_c_disc_concat = e_dense_c_disc[0]
    
    # prediction for continuous variables of c
    e_dense_c_cont = Dense(
        units=CONT_VARS,
        activation=None,
        kernel_initializer='he_normal',
        name="e_dense_c_cont")(e_dense_0)
    
    e_out = Concatenate()([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont])
    return Model(inputs=image_input, outputs=e_out)

E = createEncoder()
def encode(image_input):
    return E(image_input)

# Generator
def createGenerator():
    noise_input = Input(shape=(Z_DIM+C_DIM,))
    
    g_dense_0 = dense_bn_act(
        inputs=noise_input,
        units=3136,
        activation=g_activation,
        kernel_init='truncated_normal')
    g_dense_0 = Reshape(target_shape=(7,7,64))(g_dense_0)
    
    g_conv_0 = deconv2d_bn_act(
        inputs=g_dense_0,
        filters=128,
        kernel_size=4,
        kernel_init='truncated_normal',
        activation=g_activation,
        strides=2)
    
    g_conv_1 = deconv2d_bn_act(
        inputs=g_conv_0,
        filters=64,
        kernel_size=4,
        kernel_init='truncated_normal',
        activation=g_activation,
        strides=1)
    
    g_conv_out = Conv2DTranspose(
        filters=1,
        kernel_size=4,
        activation='sigmoid',
        padding='same',
        strides=2,
        kernel_initializer='truncated_normal')(g_conv_1)
    
    return Model(inputs=noise_input, outputs=g_conv_out)

G = createGenerator()
def generate(noise_input):
    return G(noise_input)

# Encoding of a real image by the encoder E
Z_hat = encode(X)
# Fake image generated by generator G
X_hat = generate(Concatenate()([z, c]))

# Prediction of D for real images with encoding by E
D_enc = discriminate(X, Z_hat)
# Prediction of D for generated images
D_gen = discriminate(X_hat, Concatenate()([z,c]))


# Minimize crossentropy between z and E(G(z))
# Encoding by E of generated images
Z_gen = encode(X_hat)
# Get disentangled (non-noise) part of the encoding
c_gen = Z_gen[:, Z_DIM:]
# Crossentropy in continuous variables
cont_stddev_c_gen = K.ones_like(c_gen[:, NUM_DISC_VARS:])
eps_c_gen = (c[:, NUM_DISC_VARS:] - c_gen[:, NUM_DISC_VARS:]) / (cont_stddev_c_gen + 1e-8)
crossent_c_gen_cont = K.mean(
    -K.sum(-0.5*np.log(2*np.pi) - mylog(cont_stddev_c_gen) - 0.5*K.square(eps_c_gen), 1))
# Crossentropy in categorical variables
crossent_c_gen_cat = K.mean(-K.sum(mylog(c_gen[:, :NUM_DISC_VARS]) * c[:, :NUM_DISC_VARS], 1))


# Discriminator loss
D_loss = -K.mean(mylog(D_enc) + mylog(1 - D_gen))
# Generator / Encoder loss
G_loss = -K.mean(mylog(D_gen) + mylog(1 - D_enc)) + \
    crossent_c_gen_cat + crossent_c_gen_cont

# Collect the trainable weights
weights_D = D.trainable_weights
weights_GE = G.trainable_weights + E.trainable_weights

training_updates_D = Adam(lr=LR_D, beta_1=BETA1_D, decay=LD_D).get_updates(
    weights_D, [], D_loss)
training_updates_GE = Adam(lr=LR_G, beta_1=BETA1_G, decay=LD_G).get_updates(
    weights_GE, [], G_loss)

train_D_fn = K.function([X,z,c], [D_loss], training_updates_D)
train_GE_fn = K.function([X,z,c], [G_loss], training_updates_GE)


# Start training
(x_train, _), (_,_) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_train = x_train.astype('float32')/255.0
print("Shape of x_train: ", np.shape(x_train))
data_size = np.shape(x_train)[0]
t0 = time.time()
for iteration in range(MAX_ITERS):
    idx = np.random.choice(range(data_size), BATCH_SIZE, replace=False)
    x_batch = x_train[idx,:,:,:]
    z_batch = sample_z(BATCH_SIZE, Z_DIM)
    c_batch = sample_c(BATCH_SIZE, CONT_VARS, DISC_VARS)
    
    lossD = train_D_fn([x_batch, z_batch, c_batch])
    lossGE = train_GE_fn([x_batch, z_batch, c_batch])
    
    if iteration % 10 == 0:
        print("Iteration: %d; D_loss: %f; G_loss: %f; t: %ds" %
            (iteration, lossD[0], lossGE[0], time.time()-t0))
    
    if iteration % 300 == 0:
        # Visualize categorical variables
        num_samples = 10
        fig, axs = plt.subplots(NUM_DISC_VARS, num_samples)
        for idx in range(NUM_DISC_VARS):
            z_test = sample_z_fixed(num_samples, Z_DIM)
            c_test = sample_c_cat(num_samples, disc_var=idx)
            encoding = np.concatenate((z_test, c_test), axis=1)
            gen = G.predict(encoding)
            
            for sample in range(num_samples):
                axs[idx, sample].imshow(gen[sample,:,:,0])
        
        # Add some handy labels
        for rid in range(NUM_DISC_VARS):
            row_title = "Cat" + str(rid)
            axs[rid,0].set_ylabel(row_title,
                rotation=0, size='large', horizontalalignment='right')
        # for cid in range(num_samples):
        #     col_title = "Sample" + str(cid)
        #     axs[0,cid].set_title(col_title)
        for rid in range(NUM_DISC_VARS):
            for cid in range(num_samples):
                axs[rid,cid].set_xticks([])
                axs[rid,cid].set_yticks([])
        
        # Save the figure
        filename = "gen_cat_iter%06d.png" % iteration
        fig.savefig(os.path.join(log_dir, filename))
        plt.close(fig)
        
        # # Sample some real images and plot with encodings
        # idx = np.random.choice(range(data_size), num_samples, replace=False)
        # x_batch = x_train[idx,:,:,:]
        # encoding
        
        # fig, axs = plt.subplots(NUM_DISC_VARS, num_samples)
        # for var in range(NUM_DISC_VARS):
            