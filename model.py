from blocks import *
from opt import *
import tensorflow as tf
import numpy as np
import math
import os

class EncoderBlock(object):
    def __init__(self, name: str, d_model=512, n_heads=8, hidden_dim=512, dropout=0.1):
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.training = True
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.self_attn = MultiHeadAttention(self.name+'_self_attn', d_model, n_heads, dropout)
        self.norm = LayerNorm(-1, name=self.name+'_self_attn_layer_norm')
        self.ffn = TransformerFFN(d_model, hidden_dim, self.name+'_ffn', dropout)
        self.layers = [self.self_attn, self.norm, self.ffn]

    def __call__(self, input):
        x = self.self_attn(input) # this already performs dropout
        x += input # then residual connection
        x = self.norm(x) # and layer normalization
        x = self.ffn(x) # this already has dropout, residual, and layer norm added
        return x

    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()

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

class DecoderBlock(object):
    def __init__(self, name: str, d_model=512, n_heads=8, hidden_dim=2048, dropout=0.1):
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.training = True
        self.dropout = dropout
        self.masked_attn = MultiHeadAttention(self.name+'_masked_attn', d_model, n_heads, dropout)
        self.norm1 = LayerNorm(-1, name=self.name+'_masked_attn_layer_norm')
        self.cross_attn = MultiHeadAttention(self.name+'_cross_attn', d_model, n_heads, dropout)
        self.norm2 = LayerNorm(-1, name=self.name+'_cross_attn_layer_norm')
        self.ffn = TransformerFFN(d_model, hidden_dim, self.name+'_ffn', dropout)
        self.layers = [self.masked_attn, self.norm1, self.cross_attn, self.norm2, self.ffn]

    def __call__(self, input, kv): # note: kv (encoder output) serves as both keys and values for decoder cross attn
        x = self.masked_attn(input, use_mask=True) # masked (causal) self-attention
        x += input
        x = self.norm1(x)
        y = self.cross_attn(x, kv, kv)
        y += x
        y = self.norm2(y)
        y = self.ffn(y)
        return y

    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()

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

class Encoder(object):
    def __init__(self, name: str, n_layers=4, d_model=512, max_len=256, n_heads=8, hidden_dim=512, dropout=0.1):
        self.name = name
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.training = True
        self.dropout = dropout
        self.positional_embedding = PositionalEmbedding(self.d_model, self.name+'_positional_embedding', 0, max_len)
        self.layers = [self.positional_embedding] + [EncoderBlock(self.name+'_block'+str(i), self.d_model, self.n_heads, self.hidden_dim, self.dropout) for i in range(self.n_layers)]

    def __call__(self, input):
        x = self.positional_embedding(input)
        # apply dropout
        if self.training:
            x = tf.nn.dropout(x, rate=self.dropout)
            # mask = tf.cast(tfp.distributions.Bernoulli(probs=1.0-self.dropout).sample(sample_shape=x.shape.as_list()), dtype=tf.float32)
            # x *= mask/(1.0-self.dropout)
        for i in range(1, self.n_layers+1):
            x = self.layers[i](x) # the first layer is embedding so we skip it
        return x
    
    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()

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

class Decoder(object):
    def __init__(self, name: str, n_layers=4, d_model=512, max_len=256, n_heads=8, hidden_dim=2048, dropout=0.1):
        self.name = name
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.n_layers = n_layers
        self.training = True
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.positional_embedding = PositionalEmbedding(self.d_model, self.name+'_positional_embedding', 0, self.max_len)
        self.layers = [self.positional_embedding] + [DecoderBlock(self.name+'_block'+str(i), self.d_model, self.n_heads, self.hidden_dim, self.dropout) for i in range(self.n_layers)]

    def __call__(self, input, kv): # note: kv (encoder output) serves as both keys and values for decoder cross attn
        x = self.positional_embedding(input)
        if self.training:
            x = tf.nn.dropout(x, rate=self.dropout)
            # mask = tf.cast(tfp.distributions.Bernoulli(probs=1.0-self.dropout).sample(sample_shape=x.shape.as_list()), dtype=tf.float32)
            # x *= mask/(1.0-self.dropout)
        for i in range(1, self.n_layers+1):
            x = self.layers[i](x, kv) # the first layer is embedding so we skip it
        return x
    
    def toggle_inference(self):
        self.training = False
        for layer in self.layers:
            layer.toggle_inference()
            
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

class Transformer(object):
    def __init__(self, CHKPT_DIR='C:/Users/saraa/Desktop/integrator/checkpoints/', vocab_size=20, n_layers=4, n_heads=8, d_model=512, max_len=256, hidden_dim=2048, dropout=0.1, warmup_steps=4000, lr=8.75e-4, beta_1=0.9, beta_2=0.98):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.training = True
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.steps = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.warmup_steps = warmup_steps
        self.dropout = dropout
        self.encoder = Encoder('transformer_encoder', n_layers, d_model, max_len, n_heads, hidden_dim, dropout)
        self.decoder = Decoder('transformer_decoder', n_layers, d_model, max_len, n_heads, hidden_dim, dropout)
        self.output_lin = Linear(self.vocab_size, 'transformer_output_dense', None) # no activation, softmax will happen later
        self.optimizer = Adam(lr=self.lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.optimizer.lr_func = lambda steps: self.d_model ** -0.5 * min(steps ** -0.5, steps * self.warmup_steps ** -1.5)
        self.CHKPT_DIR = CHKPT_DIR

    def __call__(self, input, output):
        x = self.encoder(input)
        return self.output_lin(self.decoder(output, x))

    def toggle_inference(self):
        self.training = False
        self.encoder.toggle_inference()
        self.decoder.toggle_inference()
        self.output_lin.training = False

    def predict(self, input):  # Autoregressive decoding with batch size 1
        # input: indices of shape [1, seq_len]
        seq_len = input.shape[1]  # As batch size is 1, input shape is [1, seq_len]
    
        # Start with the [START] token, represented by a 1
        output = tf.ones([1, 1], dtype=input.dtype)  
    
        for _ in range(self.max_len - 2):  # Loop to generate a sequence of max_len tokens
            # Predict the next token based on current output
            pred = tf.nn.softmax(self(input, output))  # Shape [1, seq_len, vocab_size]
        
            # Extract the prediction for the last timestep
            pred = pred[:, -1:, :]  # Shape [1, 1, vocab_size]
        
            # Get the token with the highest probability
            new_token = tf.argmax(pred, axis=-1, output_type=input.dtype)  # Shape [1, 1]
        
            # Append the new token to the output sequence
            output = tf.concat([output, new_token], axis=-1)
        
            # Stop if the [END] token (2) is generated
            if new_token == 2:
                break
    
        # If the loop finishes without generating an [END] token, append it manually
        if output.shape[1] >= self.max_len-1:
            output = tf.concat([output, tf.constant([[2]], dtype=output.dtype)], axis=-1)
    
        return output

    # def predict(self, input): # this will generate a whole sequence via autoregressive encoding
    #     # input: indices of shape [bs, seq_len]
    #     # that means the other input (aka output) has dimension [bs, seq_len+1] for each timestep
    #     bs, seq_len = input.shape.as_list()
    #     output = tf.ones([bs, 1], dtype=input.dtype) # use ones because [START] token is represented by a 1
    #     stop_flags = tf.zeros([bs, 1], dtype=tf.bool)  # track sequences that should stop
    #     for _ in range(self.max_len-1): # the max amount of tokens we can generate
    #         pred = tf.nn.softmax(self(input, output)) # [bs, seq_len, vocab_size]
    #         pred = pred[:, -1:, :]  # [bs, 1, vocab_size], take the last predictions
    #         pred = tf.argmax(pred, axis=-1, output_type=input.dtype) # [bs, 1]
    #         new_tokens = tf.where(stop_flags, tf.zeros_like(pred, dtype=pred.dtype), pred)  # shape [bs, 1]
    #         output = tf.concat([output, new_tokens], axis=-1)
    #         # this checks if all sequences have reached stopping conditions: 2="[END]"
    #         stop_flags = tf.logical_or(stop_flags, new_tokens == 2)
    #         if tf.reduce_all(stop_flags):
    #             break
    #     if not tf.reduce_all(stop_flags):
    #         output = tf.concat([output, tf.zeros([bs, 1], dtype=output.dtype)+2], axis=-1)
    #     return output
        

    def save_checkpoint(self):
        checkpoint = tf.train.Checkpoint(**self.get_vars_dict())
        size = str(len(os.listdir(self.CHKPT_DIR)))
        newpath = self.CHKPT_DIR+'chkpt'+size+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        checkpoint.save(newpath+'transformer_math_checkpoint'+size)

    def get_vars_dict(self) -> dict:
        # gets all the variables in the model
        vars = {}
        for variable in self.get_all_variables():
            vars[variable.name] = variable
        return vars
    
    def get_trainable_variables(self) -> list:
        # gets all the trainable variables in the model
        vars = self.encoder.get_trainable_variables()
        vars.extend(self.decoder.get_trainable_variables())
        vars.extend(self.output_lin.get_trainable_variables())
        return vars
    
    def get_all_variables(self) -> list:
        # gets all the variables in the model
        vars = self.encoder.get_variables()
        vars.extend(self.decoder.get_variables())
        vars.extend(self.output_lin.get_variables())
        return vars

    def load_checkpoint(self, index=-1):
        vars = {}
        if index < 0:
            index = str(len(os.listdir(self.CHKPT_DIR))+index)
        else:
            index = str(index)
        checkpoint_reader = tf.train.load_checkpoint(self.CHKPT_DIR+'chkpt'+index+'/')
        var_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
        for var_name in var_to_shape_map:
            name = var_name[:str.find(var_name, '/')]
            variable = tf.Variable(initial_value=checkpoint_reader.get_tensor(var_name), trainable=True, name=name)
            vars[name] = variable
        self.encoder.load_variables(vars)
        self.decoder.load_variables(vars)
        self.output_lin.load_variables(vars)