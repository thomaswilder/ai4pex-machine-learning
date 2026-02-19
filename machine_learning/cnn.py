#!/usr/bin/python3

import tensorflow as tf
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
from keras.saving import register_keras_serializable
from keras import ops

#-------------------------------------------------

# Clear all previously registered custom objects
#//keras.saving.get_custom_objects().clear()

@dataclass
@keras.saving.register_keras_serializable(package="cnn", name="Scenario")
class Scenario:
    '''
    Scenario dataclass. Can easily setup the design, 
        adjusting input variables, target variables, cnn filters and kernels.
    '''
    input_var: Iterable[str]
    target: Iterable[str]
    filters: List[int]
    kernels: Tuple[int, int]
    padding: Tuple[int, int]
    name: str

    # do some checks on dataclass
    def __post_init__(self):
        self._validate_kernel_padding()

    def _validate_kernel_padding(self):
        for idx, (kernel, pad) in enumerate(zip(self.kernels, self.padding)):
            # Convert to tuples if they're lists
            kernel = tuple(kernel) if isinstance(kernel, list) else kernel
            pad = tuple(pad) if isinstance(pad, list) else pad

            expected_pad = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
            if pad != expected_pad:
                raise ValueError(
                    f"Padding at index {idx} {pad} \
                    does not match expected {expected_pad} \
                    for kernel {kernel}"
                )
            
    def get_config(self):
        # This method converts the Scenario object into a serializable dictionary.
        # It's important to convert Iterables to Lists/Tuples for reliable serialization.
        return {
            'input_var': list(self.input_var),
            'target': list(self.target),
            'filters': self.filters,
            'kernels': self.kernels,
            'padding': self.padding,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config):
        # This class method reconstructs the Scenario object from the dictionary
        # saved by get_config. Dataclasses make this very straightforward.
        # Remove any keys not in the dataclass fields
        #//allowed_keys = {f.name for f in cls.__dataclass_fields__.values()}
        #//filtered_config = {k: v for k, v in config.items() if k in allowed_keys}
        return cls(**config) #filtered_

@keras.saving.register_keras_serializable(package="cnn", name="ReplicationPadding2D")    
class ReplicationPadding2D(keras.layers.Layer):
    '''
        2D Replication padding

        Attributes:

            - padding : (padding_width, padding_height) tuple

        From:
            https://github.com/christianversloot/machine-learning-articles/blob/main/using-constant-padding-reflection-padding-and-replication-padding-with-keras.md
    '''
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], 
                input_shape[2] + 2 * self.padding[1], input_shape[3])
    
    def get_config(self):
        config = super().get_config()
        config.update({'padding': self.padding})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], 
                                     [padding_height, padding_height], 
                                     [padding_width, padding_width], 
                                     [0,0] ], 
                                     'SYMMETRIC')
    
@keras.saving.register_keras_serializable(package="cnn", name="MSESSIMLoss")
class MSESSIMLoss(keras.losses.Loss):
    '''
    Custom loss function that combines MSE and SSIM losses.
    '''
    def __init__(self, alpha=0.5, beta=0.5, **kwargs):
        super(MSESSIMLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        mse_loss = ops.mean(ops.square(y_true - y_pred))
        max_value = ops.abs(tf.reduce_max(y_true) - tf.reduce_min(y_true))
        ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=max_value)
        return self.alpha * mse_loss + (1 - self.beta) * ssim_loss

@keras.saving.register_keras_serializable(package="cnn", name="MaskedMSELoss")
class MaskedMSELoss(keras.losses.Loss):
    '''
    Custom loss function that masks the input so land values
        do not contribute to the loss.
    '''
    def __init__(self, mask=None, **kwargs):
        super(MaskedMSELoss, self).__init__(**kwargs)

        if mask is not None:
            self.mask = tf.constant(mask, dtype=tf.float32)
        else:
            self.mask = None

    def call(self, y_true, y_pred):

        # print(f"Mask shape: {self.mask.shape}, dtype: {self.mask.dtype}")

        # print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        # print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")

        # mask true and predicted values if mask is provided
        y_true = y_true * self.mask if self.mask is not None else y_true
        y_pred = y_pred * self.mask if self.mask is not None else y_pred

        if self.mask is not None:
            mse = ops.mean(ops.square(y_true - y_pred)) / ops.sum(self.mask)
            # print("ops.square(y_true - y_pred):", ops.square(y_true - y_pred))
            # print("ops.sum(self.mask):", ops.sum(self.mask))
            # print("Masked MSE:", mse)
        else:
            mse = ops.mean(ops.square(y_true - y_pred))

        return mse


@keras.saving.register_keras_serializable(package="cnn", name="CNN")
class CNN(keras.Model):
    '''
    A fully convolutional neural network class using keras api
    '''
    def __init__(self, sc, input_shape, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs) # inherit from keras.Model class
        self.sc = sc
        self.input_shape = input_shape #// this doesn't look it is used?
        self.dropout_rate = dropout_rate

        # //self.input_layer = ReplicationPadding2D(padding=(2, 2), input_shape=input_shape)

        self.conv_blocks = []
        for i, (f, k, p) in enumerate(zip(sc.filters[:-1], sc.kernels[:-1], sc.padding[:-1])):
            if i<2:
                use_dropout = True
            block = self._make_conv_block(f, k, p, use_dropout=use_dropout)
            self.conv_blocks.append(block)
            use_dropout = False # reset to false to ensure only first two blocks have dropout
            # self.conv_blocks = [self._make_conv_block(f, k, p) for f, k, p in 
            #                 zip(sc.filters[:-1], sc.kernels[:-1], sc.padding[:-1])]

        self.output_layer = (keras.layers.Conv2D(
                        sc.filters[-1], 
                        sc.kernels[-1], 
                        use_bias=False,
                        kernel_initializer=keras.initializers.GlorotUniform(),
                        activation='linear')
                        )

    def _make_conv_block(self, filters, kernel, padding, use_dropout=False):
        layers = [
            ReplicationPadding2D(padding=padding),
            keras.layers.Conv2D(
                filters, 
                kernel_size=kernel, 
                use_bias=False,
                kernel_initializer=keras.initializers.HeNormal(),
                activation="relu"), 
            keras.layers.BatchNormalization()
        ]
        if use_dropout:
            # layers.append(keras.layers.Dropout(self.dropout_rate))
            layers.append(keras.layers.SpatialDropout2D(self.dropout_rate))
        return keras.Sequential(layers)

    # def get_padded_feature_map(self, inputs):
    #     '''
    #         Returns a feature map with 2d replication padding applied.
    #     '''
    #     x = inputs
    #     x = ReplicationPadding2D(padding=(2, 2))(x) 
    #     return x.numpy()  # Returns as numpy array

    #/ custom objects require the below to be serializable
    def get_config(self):
        config = super().get_config()
        config.update({
            'sc': self.sc,
            'input_shape': self.input_shape,
            'dropout_rate': self.dropout_rate,
        })
        return config

    #/ then must deserialize the custom object
    @classmethod
    def from_config(cls, config):
        sc_config = config.pop('sc')
        # extracts the inner config from the Scenario object
        if isinstance(sc_config, dict) and 'config' in sc_config:
            sc_config = sc_config['config']
        sc = Scenario.from_config(sc_config) # reconstruct Scenario object
        input_shape = config.pop('input_shape')
        dropout_rate = config.pop('dropout_rate', 0.2)
        return cls(sc, input_shape, **config)

    def call(self, inputs, training=False):
        x = inputs
        # //x = self.input_layer(inputs) # adding replication padding
        for block in self.conv_blocks:
            x = block(x, training = training) # training flag is passed depending on .fit or .evaluate/.predict
        x = ReplicationPadding2D(padding=self.sc.padding[-1])(x) # add padding before final convolution
        y = self.output_layer(x) # final convolution layer with padding
        #// y = keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(yp) # crop the padding in final layer
        return y