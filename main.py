from train import Trainer
import tensorflow as tf
import datetime
from infix import parse_expr

hyper_params = dict(
    seed=6983, # seed for reproducibility
    n_layers=4, # layers for the encoder and decoder
    n_heads=4, # how many attention heads
    d_model=256, # length of the embedding dimension
    max_len=512, # the number of positional encodings to calculate
    max_input_len=128, # the max length of an input in tokens
    hidden_dim=512, # the hidden dimension for feed forward parts of the network
    dropout=0.0, # dropout rate for the model
    warmup_steps=4000, # for learning rate scheduler
    lr=1.0e-4, # initial learning rate: 0.0001
    beta_1=0.9, # normal adam optimizer parameters
    beta_2=0.98, # normal adam optimizer parameters
    batch_size=32, # number of equations per batch
    epoch_size=1000000, # number of equations per epoch
    start_idx=1000000, # what iine to start at in each file
    epoch_start=0 # how many epochs have been completed so far
)


trainer = Trainer(hyper_params, chkpt_dir='C:/Users/saraa/Desktop/integrator/checkpoints/', dataset_dir='C:/Users/saraa/Desktop/integrator/dataset/')
trainer.model.load_checkpoint()
# trainer.model.toggle_inference()
trainer.train_rd_file() # the main training loop
# expr = input('input an expression to integrate: ')
# print(f'the antiderivative of {expr} is {trainer.infer(expr)}')
