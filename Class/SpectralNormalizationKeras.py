from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Conv3D, Embedding
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
import tensorflow as tf


def l2normalize(v, eps=1e-12):
    return v / (K.sqrt(K.sum(K.square(v))) + eps)


def power_iteration(W, u, iters=1):
    for _ in range(iters):
        v = l2normalize(K.dot(u, K.transpose(W)))
        u = l2normalize(K.dot(v, W))
    return u, v

class DenseSN(Dense):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        W_reshaped = K.reshape(self.kernel, [-1, self.kernel.shape[-1]])
        u_hat, v_hat = power_iteration(W_reshaped, self.u)
        sigma = K.dot(K.dot(v_hat, W_reshaped), K.transpose(u_hat))
        W_bar = W_reshaped / sigma
        if training:
            self.u.assign(u_hat)

        W_bar = K.reshape(W_bar, self.kernel.shape)
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation is not None:
            return self.activation(output)
        return output
class _ConvSN(Layer):
    def compute_spectral_norm(self, kernel, u):
        W_reshaped = K.reshape(kernel, [-1, kernel.shape[-1]])
        u_hat, v_hat = power_iteration(W_reshaped, u)
        sigma = K.dot(K.dot(v_hat, W_reshaped), K.transpose(u_hat))
        W_bar = W_reshaped / sigma
        return K.reshape(W_bar, kernel.shape), u_hat
class ConvSN1D(Conv1D, _ConvSN):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        W_bar, u_hat = self.compute_spectral_norm(self.kernel, self.u)
        if training:
            self.u.assign(u_hat)

        outputs = K.conv1d(
            inputs,
            W_bar,
            strides=self.strides[0],
            padding=self.padding.upper(),
            data_format='channels_last',
            dilation_rate=self.dilation_rate[0]
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
class ConvSN2D(Conv2D, _ConvSN):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        W_bar, u_hat = self.compute_spectral_norm(self.kernel, self.u)
        if training:
            self.u.assign(u_hat)

        outputs = K.conv2d(
          inputs,
          W_bar,
          strides=self.strides,
          padding=self.padding,
          data_format='channels_last'
        )

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
class ConvSN3D(Conv3D, _ConvSN):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(shape=(1, self.kernel.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        W_bar, u_hat = self.compute_spectral_norm(self.kernel, self.u)
        if training:
            self.u.assign(u_hat)

        outputs = K.conv3d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format='channels_last',
            dilation_rate=self.dilation_rate
        )
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format='channels_last')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs
class EmbeddingSN(Embedding):
    def build(self, input_shape):
        super().build(input_shape)
        self.u = self.add_weight(shape=(1, self.embeddings.shape[-1]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn_u',
                                 trainable=False)

    def call(self, inputs, training=None):
        W_reshaped = K.reshape(self.embeddings, [-1, self.embeddings.shape[-1]])
        u_hat, v_hat = power_iteration(W_reshaped, self.u)
        sigma = K.dot(K.dot(v_hat, W_reshaped), K.transpose(u_hat))
        W_bar = W_reshaped / sigma
        if training:
            self.u.assign(u_hat)

        W_bar = K.reshape(W_bar, self.embeddings.shape)
        return K.gather(W_bar, inputs)
