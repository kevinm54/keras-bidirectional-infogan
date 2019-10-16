import numpy as np
import datetime
import os
import sys
import dateutil.tz
from shutil import copyfile

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.layers import Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers import GaussianNoise, Concatenate
from keras.initializers import he_normal, truncated_normal
from keras.optimizers import Adam

from utils import mylog

class BidirectionalInfoGAN():
    def __init__(self):
        # Hyperparameters
        p = {}
        p['batch_size'] = 128
        p['max_iters'] = 30000
        p['img_width'] = 28
        p['img_height'] = 28
        p['img_channels'] = 1
        p['lr_D'] = 1e-4
        p['lr_G'] = 5e-4
        p['lr_red'] = 5
        p['ld_D'] = 1e-3
        p['ld_G'] = 1e-3
        p['final_lr_D'] = p['lr_D'] / p['lr_red']
        p['final_lr_G'] = p['lr_G'] / p['lr_red']
        p['decay_steps_D'] = p['max_iters']
        p['decay_steps_G'] = p['max_iters']
        p['beta1_D'] = 0.5
        p['beta1_G'] = 0.5

        p['z_dim'] = 16
        p['num_cont_vars'] = 2
        p['disc_vars'] = [10]
        p['num_disc_vars'] = sum(p['disc_vars'])
        p['c_dim'] = p['num_cont_vars'] + p['num_disc_vars']

        p['g_activation'] = 'elu'
        p['e_activation'] = 'elu'
        p['d_activation'] = 'elu'
        
        self.params = p
        
        self._init_model()
        self.save_hyperparameters()
    
    def _init_model(self):
        # Placeholders for the input values
        self.X = Input(shape=(self.params['img_height'],
                         self.params['img_width'],
                         self.params['img_channels']))
        self.z = Input(shape=(self.params['z_dim'],))
        self.c = Input(shape=(self.params['c_dim'],))
        
        self.D = self.create_discriminator()
        self.E = self.create_encoder()
        self.G = self.create_generator()
        
        # Encoding of real image
        self.Z_hat = self.encode(self.X)
        # Fake image generated by G
        self.X_hat = self.generate(Concatenate()([self.z, self.c]))
        # Encoding of fake image
        self.Z_gen = self.encode(self.X_hat)
        
        # D prediction for real images
        D_enc = self.discriminate(self.X, self.Z_hat)
        # D prediction for generated images
        D_gen = self.discriminate(self.X_hat,
                                  Concatenate()([self.z, self.c]))
        
        # Get disentangled components of the encoding
        c_gen = self.Z_gen[:, self.params['z_dim']:]
        c_gen_cont = c_gen[:, self.params['num_disc_vars']:]
        c_cont = self.c[:, self.params['num_disc_vars']:]
        c_gen_cat = c_gen[:, :self.params['num_disc_vars']]
        c_cat = self.c[:, :self.params['num_disc_vars']]
        
        # Crossentropy in continuous variables
        cont_stddev_c_gen = K.ones_like(c_gen_cont)
        eps_c_gen = (c_cont - c_gen_cont) / (cont_stddev_c_gen + 1e-8)
        crossent_c_gen_cont = K.mean(
            -K.sum(0.5*np.log(2*np.pi) - mylog(cont_stddev_c_gen) \
            - 0.5*K.square(eps_c_gen), 1))
        # Crossentropy in categorical variables
        crossent_c_gen_cat = K.mean(-K.sum(mylog(c_gen_cat) * c_cat, 1))
        
        # Loss for Discriminator and Generator/Encoder
        D_loss = -K.mean(mylog(D_enc) + mylog(1 - D_gen))
        G_loss = -K.mean(mylog(D_gen) + mylog(1 - D_enc)) + \
            crossent_c_gen_cat + crossent_c_gen_cont
        
        # Collect the trainable weights
        weights_D = self.D.trainable_weights
        weights_GE = self.G.trainable_weights + self.E.trainable_weights
        
        training_updates_D = Adam(
            lr=self.params['lr_D'],
            beta_1=self.params['beta1_D'],
            decay=self.params['ld_D']
        ).get_updates(weights_D, [], D_loss)
        training_updates_GE = Adam(
            lr=self.params['lr_G'],
            beta_1=self.params['beta1_G'],
            decay=self.params['ld_G']
        ).get_updates(weights_GE, [], G_loss)
        
        self.train_D_fn = K.function(
            inputs=[self.X, self.z, self.c],
            outputs=[D_loss],
            updates=training_updates_D)
        self.train_GE_fn = K.function(
            inputs=[self.X, self.z, self.c],
            outputs=[G_loss],
            updates=training_updates_GE)
    
    ############################################################
    ################# Initialization Utilities #################
    ############################################################
    
    def create_log_dir(self):
        src_dir = os.path.dirname(__file__)
        base_dir = os.path.join(src_dir, "../")
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        timestamp = "temp"

        self.log_dir = base_dir + "/logs/" + timestamp

        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_log_dir(self):
        if not hasattr(self, 'log_dir'):
            self.create_log_dir()
        return self.log_dir
    
    def save_hyperparameters(self):
        if not hasattr(self, 'log_dir'):
            self.create_log_dir()
        os.makedirs(self.log_dir, exist_ok=True)
        
        with open(self.log_dir + "/hyperparameters.csv", "w") as f:
            for param in self.params:
                f.write(param + "," + str(self.params[param]) + "\n")

        # Copy pertinent source files as log
        copyfile(sys.argv[0], self.log_dir + "/" + sys.argv[0])
        copyfile(__file__, self.log_dir + "/" + os.path.basename(__file__))
    
    def save(self, suffix=None):
        if suffix is None:
            suffix = ""
        else:
            suffix = "_" + suffix
        
        path = self.get_log_dir()
        self.G.save(path + "/model_G" + suffix + ".h5")
        self.G.save(path + "/model_E" + suffix + ".h5")
        self.G.save(path + "/model_D" + suffix + ".h5")
    
    ############################################################
    ##################### Layer Shortcuts ######################
    ############################################################
    
    def conv2d_bn_act(
            self,
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
            self,
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
            self,
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
    
    ############################################################
    #################### Define GED Models ### #################
    ############################################################
    
    def create_discriminator(self):
        image_input = Input(shape=(self.params['img_height'],
                                 self.params['img_width'],
                                 self.params['img_channels']))
        noise_input = Input(
            shape=(self.params['z_dim']+self.params['c_dim'],))
        
        # Image discriminator
        d_x_conv_0 = self.conv2d_bn_act(
            inputs=image_input,
            filters=64,
            kernel_size=3,
            kernel_init='he_normal',
            activation=self.params['d_activation'],
            strides=2,
            noise=True,
            noise_std=0.3)
        d_x_conv_1 = self.conv2d_bn_act(
            inputs=d_x_conv_0,
            filters=128,
            kernel_size=3,
            kernel_init='he_normal',
            activation=self.params['d_activation'],
            strides=2,
            noise=True)
        d_x_conv_1 = Flatten()(d_x_conv_1)
        d_x_dense = self.dense_bn_act(
            inputs=d_x_conv_1,
            units=512,
            activation=self.params['d_activation'],
            kernel_init='he_normal',
            noise=True)
        
        # noise discriminator
        noise_input_vec = Reshape((1, 1, -1))(noise_input)
        d_z_conv_0 = self.conv2d_bn_act(
            inputs=noise_input_vec,
            filters=64,
            kernel_size=1,
            kernel_init='he_normal',
            activation=self.params['d_activation'],
            strides=1,
            noise=True,
            noise_std=0.3)
        d_z_conv_1 = self.conv2d_bn_act(
            inputs=d_z_conv_0,
            filters=128,
            kernel_size=1,
            kernel_init='he_normal',
            activation=self.params['d_activation'],
            strides=1,
            noise=True)
        d_z_conv_1 = Flatten()(d_z_conv_1)
        d_z_dense = self.dense_bn_act(
            inputs=d_z_conv_1,
            units=512,
            activation=self.params['d_activation'],
            kernel_init='he_normal',
            noise=True)
        
        # final discriminator
        inp = Concatenate()([d_x_dense, d_z_dense])
        d_final_dense = self.dense_bn_act(
            inputs=inp,
            units=1024,
            activation=self.params['d_activation'],
            kernel_init='he_normal',
            noise=True)
        
        # final prediction whether input is real or generated
        d_final_pred = Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer='he_normal')(d_final_dense)
        
        return Model(
            inputs=[image_input, noise_input],
            outputs=[d_final_pred])

    def create_encoder(self):
        image_input = Input(shape=(self.params['img_height'],
                                 self.params['img_width'],
                                 self.params['img_channels']))
        
        e_conv_0 = self.conv2d_bn_act(
            inputs=image_input,
            filters=32,
            kernel_size=3,
            kernel_init='he_normal',
            activation=self.params['e_activation'],
            strides=1)
        e_conv_1 = self.conv2d_bn_act(
            inputs=e_conv_0,
            filters=64,
            kernel_size=3,
            kernel_init='he_normal',
            activation=self.params['e_activation'],
            strides=2)
        e_conv_2 = self.conv2d_bn_act(
            inputs=e_conv_1,
            filters=128,
            kernel_size=3,
            kernel_init='he_normal',
            activation=self.params['e_activation'],
            strides=2)
        e_conv_2 = Flatten()(e_conv_2)
        e_dense_0 = self.dense_bn_act(
            inputs=e_conv_2,
            units=1024,
            activation=self.params['e_activation'],
            kernel_init='he_normal')

        # prediction for z, i.e. the noise part of the representation
        e_dense_z = Dense(
            units=self.params['z_dim'],
            activation='tanh',
            kernel_initializer='he_normal')(e_dense_0)
        
        # prediction for categorical variables of c
        e_dense_c_disc = []
        for idx, classes in enumerate(self.params['disc_vars']):
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
            units=self.params['num_cont_vars'],
            activation=None,
            kernel_initializer='he_normal',
            name="e_dense_c_cont")(e_dense_0)
        
        e_out = Concatenate()([e_dense_z, e_dense_c_disc_concat, e_dense_c_cont])
        return Model(inputs=image_input, outputs=e_out)

    def create_generator(self):
        noise_input = Input(
            shape=(self.params['z_dim']+self.params['c_dim'],))
        
        g_dense_0 = self.dense_bn_act(
            inputs=noise_input,
            units=3136,
            activation=self.params['g_activation'],
            kernel_init='truncated_normal')
        g_dense_0 = Reshape(target_shape=(7,7,64))(g_dense_0)
        
        g_conv_0 = self.deconv2d_bn_act(
            inputs=g_dense_0,
            filters=128,
            kernel_size=4,
            kernel_init='truncated_normal',
            activation=self.params['g_activation'],
            strides=2)
        
        g_conv_1 = self.deconv2d_bn_act(
            inputs=g_conv_0,
            filters=64,
            kernel_size=4,
            kernel_init='truncated_normal',
            activation=self.params['g_activation'],
            strides=1)
        
        g_conv_out = Conv2DTranspose(
            filters=1,
            kernel_size=4,
            activation='sigmoid',
            padding='same',
            strides=2,
            kernel_initializer='truncated_normal')(g_conv_1)
        
        return Model(inputs=noise_input, outputs=g_conv_out)

    def discriminate(self, image_input, noise_input):
        return self.D([image_input, noise_input])

    def encode(self, image_input):
        return self.E(image_input)

    def generate(self, noise_input):
        return self.G(noise_input)
    
    ############################################################
    #################### Training Utilities ####################
    ############################################################
    
    def train(self, X, z, c):
        lossD = self.train_D_fn([X, z, c])
        lossGE = self.train_GE_fn([X, z, c])
        
        return lossD, lossGE
    
    ############################################################
    #################### Sampling Utilities ####################
    ############################################################
    
    def sample_z(self, count):
        z_dim = self.params['z_dim']
        return np.random.uniform(-1.0, 1.0, size=[count, z_dim])
    
    def sample_c(self, count):
        c = np.zeros(shape=(count, 0))
        
        disc_classes = self.params['disc_vars']
        for class_size in disc_classes:
            selected_classes = np.random.multinomial(
                n=1,
                pvals=class_size * [1.0 / class_size],
                size=count)
            c = np.concatenate((c, selected_classes), axis=1)
        
        num_cont_vars = self.params['num_cont_vars']
        for vid in range(num_cont_vars):
            selected_cont = np.random.uniform(-1.0, 1.0, size=(count, 1))
            c = np.concatenate((c, selected_cont), axis=1)
        
        return c