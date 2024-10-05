import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math

class Linear(object):
    def __init__(self, units: int, name: str, activation=None, input_shape=None):
        self.units = units
        self.training = True
        self.name = name
        self.activation = activation if activation is not None else lambda x: x
        self.weight = None
        self.bias = None
        if input_shape is not None:
            self.create_vars(input_shape)

    def __call__(self, input: tf.Tensor):
        if self.weight is None or self.bias is None:
            self.create_vars(input.shape.as_list())
        return self.activation(tf.matmul(input, self.weight) + self.bias)
    
    def create_vars(self, input_shape: list | tuple, variance=2.0):
        fan_in = np.prod(input_shape[1:])
        std = math.sqrt(variance / fan_in)
        self.weight = tf.Variable(tf.random.normal(shape=[input_shape[-1], self.units], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)
        self.bias = tf.Variable(tf.zeros(shape=[1, self.units]), name=self.name+"_bias", trainable=True)

    def toggle_inference(self):
        self.training = False

    def get_variables(self) -> list:
        return [self.weight, self.bias]
    
    def get_trainable_variables(self) -> list:
        return [self.weight, self.bias]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var
            elif self.name+"_bias" in name:
                self.bias = var

class LayerNorm(object):
    def __init__(self, *axis, name: str, epsilon=1.0e-3, input_shape=None):
        self.axis = axis
        self.epsilon = epsilon
        self.training = True
        self.name = name
        self.gamma, self.beta = None, None
        if input_shape is not None:
            self.create_vars(input_shape)

    def toggle_inference(self):
        self.training = False

    def __call__(self, input: tf.Tensor):
        if self.gamma is None or self.beta is None:
            self.create_vars(input.shape.as_list())
        mean, variance = tf.nn.moments(input, self.axis, keepdims=True)
        x_norm = (input - mean) / tf.sqrt(variance + self.epsilon)
        return x_norm * self.gamma + self.beta
    
    def create_vars(self, input_shape: list | tuple):
        param_shape = [1] * len(input_shape)
        for axis in self.axis:
            param_shape[axis] = input_shape[axis]
        self.gamma = tf.Variable(tf.ones(param_shape), name=self.name+"_gamma", trainable=True)
        self.beta = tf.Variable(tf.zeros(param_shape), name=self.name+"_beta", trainable=True)

    def get_variables(self) -> list:
        return [self.gamma, self.beta]
    
    def get_trainable_variables(self) -> list:
        return [self.gamma, self.beta]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_gamma" in name:
                self.gamma = var
            elif self.name+"_beta" in name:
                self.beta = var

# feed forward part of the transformer
class TransformerFFN(object):
    def __init__(self, in_dim, hidden_dim, name: str, dropout=0.1):
        self.in_dim = in_dim
        self.name = name
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.training = True
        self.dense0 = Linear(self.hidden_dim, self.name+'_dense0', tf.nn.relu)
        self.dense1 = Linear(self.in_dim, self.name+'_dense1') # need to match the input shape
        self.norm = LayerNorm(-1, name=self.name+'_norm')
        self.layers = [self.dense0, self.dense1, self.norm]

    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()

    def __call__(self, input):
        x = self.dense0(input)
        x = self.dense1(x)
        # apply a dropout mask (only when training though)
            
        if self.training:
            x = tf.nn.dropout(x, rate=self.dropout)
        #     mask = tf.cast(tfp.distributions.Bernoulli(probs=1.0-self.dropout).sample(sample_shape=x.shape.as_list()), dtype=tf.float32)
        #     x *= mask/(1.0-self.dropout)
        x += input # residual connection
        x = self.norm(x)
        return x
    
    def get_trainable_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_trainable_variables())
        return vars
    
    def get_variables(self) -> list:
        vars = []
        for layer in self.layers:
            vars.extend(layer.get_variables())
        return vars

    def load_variables(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)

# embedding + positional encoding
class PositionalEmbedding(object):
    def __init__(self, dim: int, name: str, axis=0, max_len=512, input_shape=None):
        self.dim = dim
        self.axis = axis
        self.training = True
        self.name = name
        self.max_len = max_len
        self.weight = None
        self.pos_encoding = None
        if input_shape is not None:
            self.create_vars(input_shape)

    def toggle_inference(self):
        self.training = False

    def __call__(self, input: tf.Tensor):
        if self.weight is None:
            self.create_vars(input.shape.as_list())
        if self.pos_encoding is None:
            self.create_pos_encoding()
        # the last dimension of the input should match the vocabulary size
        return tf.gather(self.weight, input, axis=self.axis) * math.sqrt(self.dim) + self.pos_encoding[:, :input.shape.as_list()[-1], :]
    
    def create_vars(self, input_shape: list | tuple, variance=1.0):
        std = math.sqrt(variance / self.dim)
        # we want embeddings for all possible words in the vocab
        self.weight = tf.Variable(tf.random.normal(shape=[self.max_len, self.dim], mean=0.0, stddev=std), name=self.name+"_weight", trainable=True)

    def create_pos_encoding(self):
        pos = tf.expand_dims(tf.range(self.max_len, dtype=tf.float32), axis=1)
        dep = tf.expand_dims(tf.range(self.dim/2, dtype=tf.float32), axis=0)
        angle_rads = pos / (10000.0**dep)
        self.pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        self.pos_encoding = tf.expand_dims(self.pos_encoding, axis=0)

    def get_variables(self) -> list:
        return [self.weight]
    
    def get_trainable_variables(self) -> list:
        return [self.weight]
    
    def load_variables(self, vars: dict):
        for name, var in vars.items():
            if self.name+"_weight" in name:
                self.weight = var

class MultiHeadAttention(object):
    def __init__(self, name: str, dim=512, n_heads=8, dropout=0.1):
        self.name = name
        self.dropout = dropout
        self.dim = dim
        self.n_heads = n_heads
        assert self.dim % self.n_heads == 0
        self.d_head = self.dim // self.n_heads # the part of the embeddings processed by each head
        self.training = True
        self.query_linear = Linear(dim, self.name+"_q_linear")
        self.key_linear = Linear(dim, self.name+"_k_linear")
        self.value_linear = Linear(dim, self.name+"_v_linear")
        self.out_linear = Linear(dim, self.name+"_out_linear")
        self.layers = [self.query_linear, self.key_linear, self.value_linear, self.out_linear]

    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()

    def __call__(self, queries, keys=None, values=None, use_mask=False):
        # queries -> [bs, seq_len, dim], where dim is the length of the embedding dimension
        q = self.split(self.query_linear(queries))
        if (keys is None or values is None):
            k = v = q # self-attention
        else: # cross attention
            k = self.split(self.key_linear(keys))
            v = self.split(self.value_linear(values))
        att = self.dp_attention(q, k, v, use_mask) # [bs, n_heads, seq_len, dim_head]
        att = tf.transpose(att, perm=[0, 2, 1, 3]) # [bs, seq_len, n_heads, dim_head]
        shape = att.shape.as_list()
        att = tf.reshape(att, shape=[shape[0], shape[1], shape[2]*shape[3]]) # [bs, seq_len, dim]
        return self.out_linear(att) # [bs, seq_len, dim]
    
    def split(self, input: tf.Tensor):
        # [bs, seq_len, dim] -> [bs, n_heads, seq_len, d_head]
        shape = input.shape.as_list()
        t = tf.reshape(input, shape=[shape[0], shape[1], self.n_heads, self.d_head]) # [bs, seq_len, n_heads, d_head]
        t = tf.transpose(t, perm=[0, 2, 1, 3]) # [bs, n_heads, seq_len, d_head]
        return t

    # scaled dot product attention
    def dp_attention(self, queries, keys, values, use_mask=False):
        # matmul bc it's along the last and 2nd to last dimensions respectively
        # queries = [bs, n_heads, seq_len_q, dim_head]
        # keys = [bs, n_heads, seq_len_k, dim_head]
        # values = [bs, n_heads, seq_len_v, dim_head]
        scores = tf.matmul(queries, tf.transpose(keys, perm=[0, 1, 3, 2])) # [bs, n_heads, seq_len_q, seq_len_k]
        if use_mask:
            # this generates a lower triangular matrix (all elements above the diagonal = 0)
            # a nice way of masking outputs from future timesteps
            score_shape = scores.shape.as_list()
            mask = tf.linalg.band_part(tf.ones(shape=[1, 1, score_shape[2], score_shape[3]]), -1, 0)
            # use an arbitrarily large masking value so that masked values become 0 in softmax
            scores = scores * mask - 1.0e9 * (1.0 - mask) # if masked then make the entry -infinity
        dk = float(queries.shape.as_list()[-1])
        weights = tf.nn.softmax(scores/math.sqrt(dk), axis=-1)
        if self.training:
            weights = tf.nn.dropout(weights, rate=self.dropout)
            # mask = tf.cast(tfp.distributions.Bernoulli(probs=1.0-self.dropout).sample(sample_shape=weights.shape.as_list()), dtype=tf.float32)
            # weights *= mask/(1.0-self.dropout)
        return tf.matmul(weights, values) # [bs, n_heads, seq_len_q, dim_head]
    
    def get_trainable_variables(self) -> list:
        vars = []
        for layer in self.layers:
            if layer.weight is not None and layer.bias is not None: # accounts for self-attention
                vars.extend(layer.get_trainable_variables())
        return vars
    
    def get_variables(self) -> list:
        vars = []
        for layer in self.layers:
            if layer.weight is not None and layer.bias is not None: # accounts for self-attention
                vars.extend(layer.get_variables())
        return vars

    def load_variables(self, vars: dict):
        for layer in self.layers:
            layer.load_variables(vars)