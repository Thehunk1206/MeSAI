'''
=================================LICENSE====================================

MIT License

Copyright (c) 2021 Tauhid Khan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints



class GroupNormalization(tf.keras.layers.Layer):

    def __init__(
        self,
        groups:int = 32,
        axis:int = -1,
        smoothing:float = 1e-8,
        shift_center: bool = True,
        scale: bool = True,
        beta_initializer:initializers.Initializer = initializers.Zeros(),
        gamma_initializer:initializers.Initializer = initializers.Ones(),
        beta_regularizer:regularizers.Regularizer = None,
        gamma_regularizer:regularizers.Regularizer = None,
        beta_constraint:constraints.Constraint = None,
        gamma_constraint:constraints.Constraint = None,
        **kwargs
    ):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.smoothing = smoothing
        self.shift_center = shift_center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer) 
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
    
    def build(self, input_shape):
        '''
        Create learnable variable 'Gamma' and 'Beta' for scaling and shifting the normalized data x.
        The shape of Gamma and Beta wil  be (C,) where C is the channel of input Tensor.
        '''

        var_dim = input_shape[self.axis]

        if var_dim is None:
            raise ValueError(f'Axis {str(self.axis)} of input Tensor should have a proper defined dimension. the shape of the input tensor recieved {str(input_shape)}')
        
        if var_dim < self.groups:
            raise ValueError(f'Number of groups {str(self.groups)} cannot be greater than number of channels ({str(var_dim)})')

        if var_dim % self.groups != 0:
            raise ValueError(f'Number of groups {str(self.groups)} must a multiple of number of channels {str(var_dim)}')

        # override the input tensor signature
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape),axes={self.axis: var_dim})
        
        #shape of Gamma and Beta variable
        var_shape = (var_dim,)

        if self.scale:
            # create Gamma trainable variable
            self.gamma = self.add_weight(shape=var_shape,
                                        name='gamma',
                                        initializer=self.gamma_initializer,
                                        regularizer=self.gamma_regularizer,
                                        constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.shift_center:
            # create beta trainable variable
            self.beta = self.add_weight(shape=var_shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        
        self.built = True
        super().build(input_shape)

    def call(self, inputs:tf.Tensor, **kwargs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        #Create group shape and reshape the inputs in to group shape
        #if input shape is [1,64,64,32] then group shape will be [1,64,64,group,32//group]
        #and the reshaped input will be of shaope [1,64,64,group,32//group]
        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = group_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs,group_shape)

        #Create reduction axes for calculating moments(mean,var)
        # If input shape is of 4-dim then group reduction axes will be [1,2,3]
        group_reduction_axes = list(range(1,len(reshaped_inputs.shape.as_list())))
        group_reduction_axes.pop(self.axis)

        #create a broadcast shape for gamma and beta weights
        #if input shape is [1,64,64,32] the broadcast shape will be [1,1,1,groups,32//groups]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)

        # Calculate mean and variance of reshaped inputs
        mean, variance = tf.nn.moments(reshaped_inputs,axes=group_reduction_axes, keepdims=True)


        # Normalize the reshaped_inputs
        normalized_inputs = (reshaped_inputs-mean) / (tf.sqrt(variance + self.smoothing))
        
        # Reshape the weight params Gamma and Beta to broadcast_shape
        # We explicitly broadcast Gamma and Beta terms, because they are tf.Variable which cannot broadcast automatically
        if self.scale:
            broadcast_gamma = tf.reshape(self.gamma, broadcast_shape)
            normalized_inputs = normalized_inputs * broadcast_gamma
        
        if self.shift_center:
            broadcast_beta = tf.reshape(self.beta, broadcast_shape)
            normalized_inputs = normalized_inputs + broadcast_beta
        
        # reshape the normalize inputs back to origina input shape
        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs
    
    def get_config(self):
        config = super(GroupNormalization, self).get_config()
        config.update({
            'groups': self.groups,
            'axis': self.axis,
            'smoothing': self.smoothing,
            'center': self.shift_center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

if __name__ == "__main__":
    group_norm = GroupNormalization(groups=4)
    # first call to the `group_norm` will create weights
    inputs = tf.ones(shape=[1, 2, 2, 2, 8])
    normalize_input = group_norm(inputs)

    print("weights:", len(group_norm.weights))
    print("trainable weights:", len(group_norm.trainable_weights))
    print("config:", group_norm.get_config())
    print(f"normalize_input shape: {normalize_input.shape}")