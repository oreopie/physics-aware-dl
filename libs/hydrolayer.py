"""
This file is part of the accompanying code to our manuscript:

Jiang S., Zheng Y., & Solomatine D.. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. Geophysical Research Letters, 47, e2020GL088229. https://doi.org/10.1029/2020GL088229

Copyright (c) 2020 Shijie Jiang. All rights reserved.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf


class PRNNLayer(Layer):
    """Implementation of the standard P-RNN layer
    Hyper-parameters
    ----------
    mode: if in "normal", the output will be the generated flow;
          if in "analysis", the output will be a tensor containing all state variables and process variables
    ==========
    Parameters
    ----------
    f: Rate of decline in flow from catchment bucket | Range: (0, 0.1)
    smax: Maximum storage of the catchment bucket      | Range: (100, 1500)
    qmax: Maximum subsurface flow at full bucket     | Range: (10, 50)
    ddf: Thermal degree‐day factor                     | Range: (0, 5.0)
    tmax: Temperature above which snow starts melting  | Range: (0, 3.0)
    tmin: Temperature below which precipitation is snow| Range: (-3.0, 0)
    """

    def __init__(self, mode='normal', **kwargs):
        self.mode = mode
        super(PRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.f = self.add_weight(name='f', shape=(1,),  #
                                 initializer=initializers.Constant(value=0.5),
                                 constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                 trainable=True)
        self.smax = self.add_weight(name='smax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=1 / 15, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.qmax = self.add_weight(name='qmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.2, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.ddf = self.add_weight(name='ddf', shape=(1,),
                                   initializer=initializers.Constant(value=0.5),
                                   constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                   trainable=True)
        self.tmin = self.add_weight(name='tmin', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)
        self.tmax = self.add_weight(name='tmax', shape=(1,),
                                    initializer=initializers.Constant(value=0.5),
                                    constraint=constraints.min_max_norm(min_value=0.0, max_value=1.0, rate=0.9),
                                    trainable=True)

        super(PRNNLayer, self).build(input_shape)

    def heaviside(self, x):
        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """

        return (K.tanh(5 * x) + 1) / 2

    def rainsnowpartition(self, p, t, tmin):
        """
        Equations to partition incoming precipitation into rain or snow
            if t < tmin:
                psnow = p
                prain = 0
            else:
                psnow = 0
                prain = p
        """
        tmin = tmin * -3  # (-3.0, 0)

        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s0, t, ddf, tmax):
        """
        Equations for the snow bucket
            if t > tmax:
                if s0 > 0:
                    melt = min(s0, ddf*(t - tmax))
                else:
                    melt = 0
            else:
                melt = 0
        """
        ddf = ddf * 5  # (0, 5.0)
        tmax = tmax * 3  # (0, 3.0)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    def soilbucket(self, s1, pet, f, smax, qmax):
        """
        Equations for the soil bucket
            if s1 < 0:
                et = 0
                qsub = 0
                qsurf = 0
            elif s1 > smax:
                et = pet
                qsub = qmax
                qsurf = s1 - smax
            else:
                et = pet * (s1 / smax)
                qsub = qmax * exp(-f * (smax - s1))
                qsurf = 0
        """
        f = f / 10  # (0, 0.1)
        smax = smax * 1500  # (100, 1500)
        qmax = qmax * 50  # (10, 50)

        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]

    def step_do(self, step_in, states):  # Define step function for the RNN
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current input column
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]

        # Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, self.tmin)
        # Snow bucket
        _m = self.snowbucket(s0, t, self.ddf, self.tmax)
        # Soil bucket
        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        # Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)

        # Concatenate pprcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # Recursively calculate state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)

        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # Calculate final process variables
        m = self.snowbucket(s0, tmean, self.ddf, self.tmax)
        [et, qsub, qsurf] = self.soilbucket(s1, pet, self.f, self.smax, self.qmax)

        if self.mode == "normal":
            return qsub + qsurf
        elif self.mode == "analysis":
            return K.concatenate([s0, s1, m, et, qsub, qsurf], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == "normal":
            return (input_shape[0], input_shape[1], 1)
        elif self.mode == "analysis":
            return (input_shape[0], input_shape[1], 6)


class ConvLayer(Layer):
    """Implementation of the standard 1D-CNN layer

    Hyper-parameters (same as the Conv1D in https://keras.io/layers/convolutional/)
    ----------
    filters: The dimensionality of the output space (i.e. the number of output filters in the convolution)
    kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window
    padding: One of "valid", "causal" or "same"
    seed: Random seed for initialization
    """

    def __init__(self, filters, kernel_size, padding, seed=200, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.seed = seed
        super(ConvLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.kernel_size, input_shape[-1], self.filters),
                                      initializer=initializers.random_uniform(seed=self.seed),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(self.filters,),
                                    initializer=initializers.Zeros(),
                                    trainable=True)

        super(ConvLayer, self).build(input_shape)

    def call(self, inputs):
        #self.kernel = K.concatenate((self.kernel_black, self.kernel_white), axis=1)
        outputs = K.conv1d(inputs, self.kernel, strides=1, padding=self.padding)
        outputs = K.elu(K.bias_add(outputs, self.bias))
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)


class ScaleLayer(Layer):
    """
    Scale the inputs with the mean activation close to 0 and the standard deviation close to 1
    """

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):
        met = inputs[:, :, :-1]
        flow = inputs[:, :, -1:]

        self.met_center = K.mean(met, axis=-2, keepdims=True)
        self.met_scale = K.std(met, axis=-2, keepdims=True)
        self.met_scaled = (met - self.met_center) / self.met_scale

        return K.concatenate([self.met_scaled, flow], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

    
class RegionalPRNNLayer(Layer):
    """Implementation of the P-RNN layer with parameterization pipeline
    The main is the same as the PRNNLayer (contains 6 theta_p)
    The parameterization pipeline contains a 2-layer fully-connected neural network to map
    27 catchment attributes [sample size, 27] into theta_p [sample size, 6].

    Hyper-parameters
    ----------
    h_nodes: The number of hidden nodes in the fully-connected neural network (default: 32 neurons)
    seed: Random seed for initialization
    ==========
    Parameters
    ----------
    The paramters of the RegionalPRNNLayer are the weights and biases for the fully-connected neural network
    w1, b1 are for the firts layer, and w2, b2 are for the second layer
    """
    
    def __init__(self, h_nodes=32, seed=200, **kwargs):
        self.h_nodes = h_nodes
        self.seed = seed
        super(RegionalPRNNLayer, self).__init__(**kwargs)
        

    def build(self, input_shape):
        self.prnn_w1 = self.add_weight(name='prnn_w1', 
                                  shape=(input_shape[1][1], self.h_nodes),
                                  initializer=initializers.RandomUniform(seed=self.seed - 5), # avoid same initial w1 and w2
                                  trainable=True)
           
        self.prnn_b1 = self.add_weight(name='prnn_b1',
                                  shape=(self.h_nodes,), 
                                  initializer=initializers.zeros(),
                                  trainable=True)

        self.prnn_w2 = self.add_weight(name='prnn_w2', 
                                  shape=(self.h_nodes, 6),
                                  initializer=initializers.RandomUniform(seed=self.seed + 5), # avoid same initial w1 and w2
                                  trainable=True)
       
        self.prnn_b2 = self.add_weight(name='prnn_b2', 
                                  shape=(6,), 
                                  initializer=initializers.zeros(),
                                  trainable=True)
        self.shape   = input_shape

        super(RegionalPRNNLayer, self).build(input_shape)

    def heaviside(self, x):
        """Same as the PRNNLayer"""
        
        return (K.tanh(5 * x) + 1) / 2

    def rainsnowpartition(self, p, t, tmin):
        """Same as the PRNNLayer"""
        
        tmin = (tmin + 1) * -1.5         # scale (-1, 1) into (-3, 0)
        
        psnow = self.heaviside(tmin - t) * p
        prain = self.heaviside(t - tmin) * p

        return [psnow, prain]

    def snowbucket(self, s0, t, ddf, tmax):
        """Same as the PRNNLayer"""
        
        ddf = (ddf + 1) * 2.5            # scale (-1, 1) into (0, 5)
        tmax = (tmax + 1) * 1.5          # scale (-1, 1) into (0, 3)

        melt = self.heaviside(t - tmax) * self.heaviside(s0) * K.minimum(s0, ddf * (t - tmax))

        return melt

    def soilbucket(self, s1, pet, f, smax, qmax):
        """Same as the PRNNLayer"""
        
        f = (f + 1)  / 20                # scale (-1, 1) into (0, 0.1)
        smax = (smax + 1) * 700 + 100    # scale (-1, 1) into (100, 1500)
        qmax = (qmax + 1) * 20 + 10      # scale (-1, 1) into (10, 50)    
        
        et = self.heaviside(s1) * self.heaviside(s1 - smax) * pet + \
            self.heaviside(s1) * self.heaviside(smax - s1) * pet * (s1 / smax)
        qsub = self.heaviside(s1) * self.heaviside(s1 - smax) * qmax + \
            self.heaviside(s1) * self.heaviside(smax - s1) * qmax * K.exp(-1 * f * (smax - s1))
        qsurf = self.heaviside(s1) * self.heaviside(s1 - smax) * (s1 - smax)

        return [et, qsub, qsurf]        
        
        
    def step_do(self, step_in, states):
        # Load the current paras
        tmin = step_in[:, 3:4]
        tmax = step_in[:, 4:5]
        ddf  = step_in[:, 5:6]
        f    = step_in[:, 6:7]
        smax = step_in[:, 7:8]
        qmax = step_in[:, 8:9]
        
        """ The below is the same as the PRNNLayer"""
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket

        # Load the current forcing columns
        p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]
        
        # Partition precipitation into rain and snow
        [_ps, _pr] = self.rainsnowpartition(p, t, tmin)
        # Snow bucket
        _m = self.snowbucket(s0, t, ddf, tmax)
        # Soil bucket
        [_et, _qsub, _qsurf] = self.soilbucket(s1, pet, f, smax, qmax)

        # Water balance equations
        _ds0 = _ps - _m
        _ds1 = _pr + _m - _et - _qsub - _qsurf

        # Record all the state variables which rely on the previous step
        next_s0 = s0 + K.clip(_ds0, -1e5, 1e5)
        next_s1 = s1 + K.clip(_ds1, -1e5, 1e5)

        step_out = K.concatenate([next_s0, next_s1], axis=1)

        return step_out, [step_out]

    def call(self, inputs):
        """
        The inputs should contain two lists: 
        -- inputs[0] is the forcing dataset with a shape of [sample size, sequence length, 5]
           (Please do ensure the first three variables are prec, tmean, and dayl)
        -- inputs[1] is the attributes dataset with a shape of [sample size, 27]
           (The attributes dataset should be scaled in advance)
        """
        assert isinstance(inputs, list)
        forcing, attrs = inputs
        
        # Load the input vector
        prcp  = forcing[:, :, 0:1]
        tmean = forcing[:, :, 1:2]
        dayl  = forcing[:, :, 2:3]

        # Calculate PET using Hamon’s formulation
        pet = 29.8 * (dayl * 24) * 0.611 * K.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)
        
        # Calculate the theta_p by a 2-layer fully-connected neural network
        paras = K.tanh(K.dot(attrs, self.prnn_w1)+ self.prnn_b1) # layer 1
        paras = K.tanh(K.dot(paras, self.prnn_w2)+ self.prnn_b2) # layer 2
        paras = K.repeat(paras, n=self.shape[0][1]) # expand [sample size, 6] into [sample size, sequence length, 6]
        
        # concatenate forcings (prec, tmean, and pet) and paras as a new input
        new_inputs = K.concatenate((prcp, tmean, pet, paras), axis=-1) 
                
        # Define initial [s0, s1] at the begining
        init_states = [K.zeros((K.shape(forcing)[0], 2))]
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        
        # Calculate final process variables
        s1 = outputs[:, :, 1:2]
        f    = paras[:, :, 3:4]
        smax = paras[:, :, 4:5]
        qmax = paras[:, :, 5:6]

        [_, qsub, qsurf] = self.soilbucket(s1, pet, f, smax, qmax)

        return qsub + qsurf

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], 1)

    
class RegionalConv1Layer(Layer):
    """
    Implementation of the first Conv1D layer with parameterization pipeline
    The parameterization pipeline contains a 2-layer fully-connected neural network to map
    27 catchment attributes [sample size, 27] into theta_n (for the first Conv1d layer).
    Theoretically, we only need one fully-connected neural network for the mapping; however, since 
    weights for forcings, weights for q_hat, and biases may follow different distributions, it is 
    more reasonable to use three separate neural networks to respectively generate 
    weights for forcings, weights for q_hat, and biases.
    
    Hyper-parameters
    ----------
    h_nodes: The number of hidden nodes in the fully-connected neural network (default: 8 neurons)
    seed: Random seed for initialization
    ==========
    Parameters
    ----------
    The paramters of the RegionalConv1Layer are the weights and biases for the three fully-connected neural networks:
    c1_black_w1, c1_black_b1, c1_black_w2, c1_black_b2 are for generating first Conv1D's weights for forcings
    c1_white_w1, c1_white_b1, c1_white_w2, c1_white_b1 are for generating first Conv1D's weights for q_hat
    c1_bias_w1,  c1_bias_b1,  c1_bias_w2,  c1_bias_b2  are for generating first Conv1D's biases
    """

    def __init__(self, h_nodes=8, seed=200, **kwargs):
        self.h_nodes = h_nodes
        self.seed = seed
        super(RegionalConv1Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        
        # The first Conv1D's weights for forcings contains 400 weights (50 weights in each of the 8 convolutional filter)
        self.c1_black_w1 = self.add_weight(name='c1_black_w1', shape=(27, self.h_nodes),
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l1(0.001), 
                                           trainable=True)
           
        self.c1_black_b1 = self.add_weight(name='c1_black_b1', 
                                           shape=(self.h_nodes,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        self.c1_black_w2 = self.add_weight(name='c1_black_w2', 
                                           shape=(self.h_nodes, 400), 
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
       
        self.c1_black_b2 = self.add_weight(name='c1_black_b2', 
                                           shape=(400,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        # The first Conv1D's weights for q_hat contains 80 weights (10 weights in each of the 8 convolutional filter)
        self.c1_white_w1 = self.add_weight(name='c1_white_w1', shape=(27, self.h_nodes),
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l1(0.001), 
                                           trainable=True)
           
        self.c1_white_b1 = self.add_weight(name='c1_white_b1', 
                                           shape=(self.h_nodes,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        self.c1_white_w2 = self.add_weight(name='c1_white_w2', 
                                           shape=(self.h_nodes, 80), 
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
       
        self.c1_white_b2 = self.add_weight(name='c1_white_b2', 
                                           shape=(80,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        # The first Conv1D's biases contains 8 biases (1 bias in each of the 8 convolutional filter)
        self.c1_bias_w1 = self.add_weight(name='c1_bias_w1', shape=(27, self.h_nodes),
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l1(0.001), 
                                           trainable=True)
           
        self.c1_bias_b1 = self.add_weight(name='c1_bias_b1', 
                                           shape=(self.h_nodes,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        self.c1_bias_w2 = self.add_weight(name='c1_bias_w2', 
                                           shape=(self.h_nodes, 8), 
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
       
        self.c1_bias_b2 = self.add_weight(name='c1_bias_b2', 
                                           shape=(8,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
        
        super(RegionalConv1Layer, self).build(input_shape)

    def call(self, inputs):   
        assert isinstance(inputs, list)
        x, attrs = inputs
        
        # Calculate c1_black in shape of [10,5,8]
        c1_attrs1 = K.dropout(attrs, 0.1, seed=self.seed+1) 
        c1_black = K.relu(K.dot(c1_attrs1, self.c1_black_w1)+ self.c1_black_b1)
        c1_black = K.dropout(c1_black, 0.1, seed=self.seed-1)        
        c1_black = K.tanh(K.dot(c1_black, self.c1_black_w2)+ self.c1_black_b2)
        c1_black = tf.reshape(c1_black,(-1,10,5,8))
        
        # Calculate c1_white in shape of [10,1,8]  
        c1_attrs2 = K.dropout(attrs, 0.1, seed=self.seed+1) 
        c1_white = K.relu(K.dot(c1_attrs2, self.c1_white_w1)+ self.c1_white_b1)
        c1_white = K.dropout(c1_white, 0.1, seed=self.seed-1)        
        c1_white = K.tanh(K.dot(c1_white, self.c1_white_w2)+ self.c1_white_b2)
        c1_white = tf.reshape(c1_white,(-1,10,1,8))
        
        # Calculate c1_bias in shape of [8]
        c1_attrs3 = K.dropout(attrs, 0.1, seed=self.seed+1) 
        c1_bias = K.relu(K.dot(c1_attrs3, self.c1_bias_w1)+ self.c1_bias_b1)
        c1_bias = K.dropout(c1_bias, 0.1, seed=self.seed-1)   
        c1_bias = K.tanh(K.dot(c1_bias, self.c1_bias_w2)+ self.c1_bias_b2)
        
        # Merge c1_black and c1_white into the weights for the first Conv1D
        c1_weights = tf.concat((c1_black, c1_white), axis=2)
        
        # This is to implement the scale the forcing into the range of q_hat
        met_mean = K.mean(x[:,:,0:5], axis = 1, keepdims=True)
        met_std  = K.std(x[:,:,0:5], axis = 1, keepdims=True)
        q_mean   = K.mean(x[:,:,5:6], axis = 1, keepdims=True)
        q_std    = K.std(x[:,:,5:6], axis = 1, keepdims=True)  

        met_rescaled = (x[:,:,0:5] - met_mean) / met_std * q_std + q_mean
        
        new_inputs = K.concatenate((met_rescaled, x[:,:,5:6]), axis=-1)
        
        # The following is to implement conv1d based on the calculated weights and bises.
        # See the help information in https://www.tensorflow.org/api_docs/python/tf/map_fn
        def fn(x_slice, kernel, bias):
            x_slice = K.expand_dims(x_slice, axis=0)
            conv = K.conv1d(x_slice, kernel, strides=1, padding='causal')
            conv = K.bias_add(conv, bias)
            conv = K.elu(conv)
            return conv[0,:]

        outputs = tf.map_fn(lambda x: fn(x[0], x[1], x[2]), (new_inputs, c1_weights, c1_bias), infer_shape=True, dtype=tf.float32)  
        
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], 8)
    
class RegionalConv2Layer(Layer):
    """
    Implementation of the second Conv1D layer with parameterization pipeline
    The parameterization pipeline contains a 2-layer fully-connected neural network to map
    27 catchment attributes [sample size, 27] into theta_n (for the second Conv1d layer).
    We use two seperate neural networks to respectively generate weights and biases 
    (for the same reason as stated in RegionalConv1Layer)
    
    Hyper-parameters
    ----------
    h_nodes: The number of hidden nodes in the fully-connected neural network (default: 8 neurons)
    seed: Random seed for initialization
    ==========
    Parameters
    ----------
    The paramters of the RegionalConv2Layer are the weights and biases for the two fully-connected neural networks:
    c2_weight_w1, c2_weight_b1, c2_weight_w2, c2_weight_b2 are for generating first Conv1D's weights
    c2_bias_w1,  c2_bias_b1,  c2_bias_w2,  c2_bias_b2  are for generating first Conv1D's biases
    """
    def __init__(self, h_nodes=8, seed=200, **kwargs):
        self.h_nodes = h_nodes
        self.seed = seed
        super(RegionalConv2Layer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        
        # The second Conv1D's weights contains 8 weights (8 weights in the convolutional filter)
        self.c2_weight_w1 = self.add_weight(name='c2_weight_w1', shape=(27, self.h_nodes),
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l1(0.001), 
                                           trainable=True)
           
        self.c2_weight_b1 = self.add_weight(name='c2_weight_b1', 
                                           shape=(self.h_nodes,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        self.c2_weight_w2 = self.add_weight(name='c2_weight_w2', 
                                           shape=(self.h_nodes, 8), 
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
       
        self.c2_weight_b2 = self.add_weight(name='c2_weight_b2', 
                                           shape=(8,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
        
        # The second Conv1D's biases contains 1 bias (1 bias in the convolutional filter)
        self.c2_bias_w1 = self.add_weight(name='c2_bias_w1', shape=(27, self.h_nodes),
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l1(0.001), 
                                           trainable=True)
           
        self.c2_bias_b1 = self.add_weight(name='c2_bias_b1', 
                                           shape=(self.h_nodes,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)

        self.c2_bias_w2 = self.add_weight(name='c2_bias_w2', 
                                           shape=(self.h_nodes, 1), 
                                           initializer=initializers.RandomUniform(seed=self.seed),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
       
        self.c2_bias_b2 = self.add_weight(name='c2_bias_b2', 
                                           shape=(1,), 
                                           initializer=initializers.zeros(),
                                           regularizer=regularizers.l2(0.000), 
                                           trainable=True)
        
        super(RegionalConv2Layer, self).build(input_shape)
    

    def call(self, inputs):
        assert isinstance(inputs, list)
        x, attrs = inputs
        
        # Calculate weights in shape of [1,8,1]
        c2_attrs1 = K.dropout(attrs, 0.1, seed=self.seed+1) 
        c2_weight = K.relu(K.dot(c2_attrs1, self.c2_weight_w1)+ self.c2_weight_b1)
        c2_weight = K.dropout(c2_weight, 0.1, seed=self.seed-1)        
        c2_weight = K.tanh(K.dot(c2_weight, self.c2_weight_w2)+ self.c2_weight_b2)
        c2_weight = tf.reshape(c2_weight,(-1,1,8,1))     
        
        # Calculate c1_bias in shape of [8]
        c2_attrs2 = K.dropout(attrs, 0.1, seed=self.seed+1) 
        c2_bias = K.relu(K.dot(c2_attrs2, self.c2_bias_w1)+ self.c2_bias_b1)
        c2_bias = K.dropout(c2_bias, 0.1, seed=self.seed-1)        
        c2_bias = K.tanh(K.dot(c2_bias, self.c2_bias_w2)+ self.c2_bias_b2) 
            
        # The following is to implement conv1d based on the calculated weights and bises.
        # See the help information in https://www.tensorflow.org/api_docs/python/tf/map_fn
        
        def fn(x_slice, kernel, bias):
            x_slice = K.expand_dims(x_slice, axis=0)
            conv = K.conv1d(x_slice, kernel, strides=1, padding='valid')
            conv = K.bias_add(conv, bias)
            conv = K.elu(conv)
            return conv[0,:]

        outputs = tf.map_fn(lambda x: fn(x[0], x[1], x[2]), (x, c2_weight, c2_bias), infer_shape=True, dtype=tf.float32)  

        return outputs

    def compute_output_shape(self, input_shape):
        
        return (input_shape[0][0], input_shape[0][1], 1)
