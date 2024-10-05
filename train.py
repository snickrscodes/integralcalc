from model import Transformer
import re
import numpy as np
import sys
import tensorflow as tf
import datetime
from infix import parse_expr

OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'neg': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'cbrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'csc': 1,
        'sec': 1,
        'cot': 1,
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acsc': 1,
        'asec': 1,
        'acot': 1,
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'csch': 1,
        'sech': 1,
        'coth': 1,
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acsch': 1,
        'asech': 1,
        'acoth': 1
}

class Trainer(object):
    def __init__(self, params: dict, chkpt_dir='C:/Users/saraa/Desktop/integrator/checkpoints/', dataset_dir='C:/Users/saraa/Desktop/integrator/dataset/'):
        self.batch_size = params['batch_size']
        self.operators = list(OPERATORS.keys())
        self.epoch_size = params['epoch_size']
        self.max_input_len = params['max_input_len']
        self.seed = params['seed']
        self.num_eq = 0 # number of trained equations so far
        self.n_epochs = params['epoch_start']
        self.start_idx = params['start_idx']
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        # by default 49 words
        self.vocab = ['[PAD]', '[START]', '[END]', 'INT+', 'INT-', 'pi', 'E', 'I', 'x', '.', '-'] + [str(i) for i in range(10)] + self.operators
        self.id2word = dict(enumerate(self.vocab))
        self.word2id = {word: id for id, word in self.id2word.items()}
        self.dataset_dir = dataset_dir

        self.filenames = [self.dataset_dir+'prim_ibp.train', self.dataset_dir+'prim_fwd.train', self.dataset_dir+'prim_ibp.valid', self.dataset_dir+'prim_ibp.valid', self.dataset_dir+'prim_ibp.test', self.dataset_dir+'prim_fwd.test']
        
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.filenames[:2]).interleave(lambda x: tf.data.TextLineDataset(x).batch(self.batch_size//2).map(self.process_data), cycle_length=2, block_length=1)
        self.valid_dataset = tf.data.Dataset.from_tensor_slices(self.filenames[2:4]).interleave(lambda x: tf.data.TextLineDataset(x).batch(self.batch_size//2).map(self.process_data), cycle_length=2, block_length=1)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(self.filenames[4:]).interleave(lambda x: tf.data.TextLineDataset(x).batch(self.batch_size//2).map(self.process_data), cycle_length=2, block_length=1)

        self.model = Transformer(chkpt_dir, len(self.vocab), params['n_layers'], params['n_heads'], params['d_model'], params['max_len'], params['hidden_dim'], params['dropout'], params['warmup_steps'], params['lr'], params['beta_1'], params['beta_2'])

    def generator(self, dataset):
        iterator = iter(dataset)
        try:
            prev_input, prev_output = next(iterator)
            while True:
                current_input, current_output = next(iterator)
                concatenated_input = tf.concat([prev_input, current_input], axis=0)
                concatenated_output = tf.concat([prev_output, current_output], axis=0)
                yield (concatenated_input, concatenated_output)
                prev_input, prev_output = current_input, current_output  
        except StopIteration:
            if prev_input is not None:
                yield (prev_input, prev_output)

    def get_data(self, bs, func='train', start_idx=0): # generator that gets a batched version of the data
        def file_reader(filename, nlines):
            with open(filename, 'r') as file:
                # with mmap.mmap(raw_file.fileno(), 0, access=mmap.ACCESS_READ) as file:
                    batch = []
                    for i, line in enumerate(file):
                        if i < start_idx:
                            continue
                        batch.append(line.strip())
                        if (i + 1) % nlines == 0:
                            yield batch
                            batch = []
                    if batch:
                        yield batch
        filepath1 = self.dataset_dir + 'prim_ibp.' + func
        filepath2 = self.dataset_dir + 'prim_fwd.' + func
        filepath3 = self.dataset_dir + 'prim_bwd.' + func
        reader1 = file_reader(filepath1, bs//3)
        reader2 = file_reader(filepath2, bs//3)
        reader3 = file_reader(filepath3, bs//3)
        while True:
            combined_batch = []
            try:
                combined_batch = next(reader1) + next(reader2) + next(reader2)
            except StopIteration:
                break
            if combined_batch:
                yield combined_batch
            if len(combined_batch) < 3 * (bs//3):  # Check if there are fewer lines than expected
                break
    def reformat(self, input): # this is mainly to convert from prefix to infix
        # print(input)
        # invalid_int_plus = re.findall(r'INT\+\s*\D', input) # for expressions with INT+ but no following digit
        # invalid_int_minus = re.findall(r'INT-\s*\D', input) # expressions with INT- but no following digit
        # if invalid_int_plus or invalid_int_minus:
        #     raise ValueError("invalid format 'INT+' or 'INT-' must be followed by an integer")
    
        output = re.sub(r'INT\+ (\d+) (\d+)', lambda m: f'{m.group(1)}{m.group(2)}', input) # replace 'INT+ X Y' with XY
        output = re.sub(r'INT-\s*(\d+) (\d+)', lambda m: f'neg {m.group(1)}{m.group(2)}', output) # replace 'INT- X Y' with neg XY
        output = re.sub(r'INT\+ (\d+)', lambda m: m.group(1), output) # replace INT+ X with X
        output = re.sub(r'INT-\s*(\d+)', lambda m: f'neg {m.group(1)}', output) # replace INT- X with neg X
        return output
    def prefix_to_infix(self, input: str): # for sanity checking
        tokens = self.reformat(input).split(' ')
        stack = []
        for i in range(len(tokens)-1, -1, -1):
            if tokens[i] in self.operators:
                if OPERATORS[tokens[i]] == 1:
                    match tokens[i]:
                        case 'neg':
                            stack.append(f'(-{stack.pop()})')
                        case 'inv':
                            stack.append(f'(1/{stack.pop()})')
                        case 'pow2':
                            stack.append(f'({stack.pop()}**2)')
                        case 'pow3':
                            stack.append(f'({stack.pop()}**3)')
                        case 'pow4':
                            stack.append(f'({stack.pop()}**4)')
                        case 'pow5':
                            stack.append(f'({stack.pop()}**5)')
                        case _:
                            stack.append(f'{tokens[i]}({stack.pop()})')
                else:
                    match tokens[i]:
                        case 'add':
                            stack.append(f'({stack.pop()}+{stack.pop()})')
                        case 'sub':
                            stack.append(f'({stack.pop()}-{stack.pop()})')
                        case 'mul':
                            stack.append(f'({stack.pop()}*{stack.pop()})')
                        case 'div':
                            stack.append(f'({stack.pop()}/{stack.pop()})')
                        case 'pow':
                            stack.append(f'({stack.pop()}**{stack.pop()})')
                        case 'rac':
                            stack.append(f'({stack.pop()}**(1/{stack.pop()}))')
            else:
                stack.append(tokens[i])
        return stack[-1]
    
    def infer(self, input: str):
        expr = ['[START]'] + parse_expr(input) + ['[END]']
        integrand = [list(map(lambda x: self.word2id[x], expr))]
        print(integrand)
        raw_pred = self.model.predict(tf.convert_to_tensor(integrand, dtype=tf.int32))
        print(raw_pred)
        pred = raw_pred.numpy().tolist()[0]
        if 2 in pred:
            eq = pred[:pred.index(2)] # discard everything after the end token (2)
        else:
            eq = pred
        eq = list(filter(lambda x: x > 1, eq)) # remove padding and tokens - 0, 1
        eq = list(map(lambda x: self.id2word[x], eq)) # map ids to their words
        print(eq)
        eq = ' '.join(eq) # concatenate the words to make a prefix expression
        return self.prefix_to_infix(eq)

    def clean_data(self, batch):
        raw = batch
        if isinstance(batch, tf.Tensor):
            raw = batch.numpy().tolist()
        input, output = [], []
        max_in_len = max_out_len = 0
        for equation in raw:
            if isinstance(equation, bytes):
                eq = equation.decode('utf-8')
            else:
                eq = equation
            if '\t' not in eq:
                continue
            integrand, antiderivative = eq.split('\t')
            integrand = '[START] ' + integrand.split("Y' ")[1] + ' [END]' # add start and end tokens
            antiderivative = '[START] ' + antiderivative + ' [END]' # add start and end tokens
            integrand = integrand.split(' ')
            integrand = list(map(lambda x: self.word2id[x], integrand))
            antiderivative = antiderivative.split(' ')
            antiderivative = list(map(lambda x: self.word2id[x], antiderivative))
            input.append(integrand)
            output.append(antiderivative)
            max_in_len = max(max_in_len, len(integrand))
            max_out_len = max(max_out_len, len(antiderivative))
        # valid_indices = [i for i in range(len(input)) if len(input[i]) <= self.max_input_len and len(output[i]) <= self.max_output_len]
        # 0 is the index of the [PAD] token
        input = [i + [0] * (max_in_len - len(i)) for i in input]
        output = [i + [0] * (max_out_len - len(i)) for i in output]
        # dtypes so that inputs get embedded, outputs are also embedded
        return (tf.convert_to_tensor(input, dtype=tf.int32), tf.convert_to_tensor(output, dtype=tf.int32))
    
    def process_data(self, tensor):
        return tf.py_function(self.clean_data, [tensor], [tf.int32, tf.int32])
    
    def optimize(self, input, output, labels): # labels are just the output shifted by 1 timestep
        with tf.GradientTape() as tape:
            y_pred = self.model(input, output)
            mask = tf.cast(labels != 0, dtype=tf.float32) # so that we can multiply it with the loss
            cce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, y_pred) * mask
            loss = tf.reduce_sum(cce) / tf.reduce_sum(mask)
        variables = self.model.get_trainable_variables()
        gradients = tape.gradient(loss, variables)
        self.model.optimizer.apply_gradients(zip(gradients, variables))
        self.model.steps += 1
        self.model.optimizer.lr = self.model.optimizer.lr_func(self.model.steps)

    def decode_pred(self, pred: tf.Tensor):
        # pred has shape [bs, seq_len]
        tokens = pred.numpy().tolist()
        equations = []
        for eq in tokens:
            if 2 in eq:
                eq = eq[:eq.index(2)] # discard everything after the end token (2)
            eq = list(filter(lambda x: x > 1, eq)) # remove padding and tokens - 0, 1
            eq = list(map(lambda x: self.id2word[x], eq)) # map ids to their words
            eq = ' '.join(eq) # concatenate the words to make a prefix expression
            try:
                eq = self.prefix_to_infix(eq) # parse the expression
                equations.append(eq)
            except:
                pass # the expression wasn't valid
        return equations
    
    def evaluate(self, inputs, outputs):
        pred = self.model(inputs, outputs[:, :-1])
        pred = tf.argmax(pred, axis=-1, output_type=outputs.dtype)
        labels = outputs[:, 1:]
        match = labels == pred
        mask = labels != 0
        match = tf.cast(match & mask, dtype=tf.float32) # only count non-padding predictions
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(match) / tf.reduce_sum(mask) # ratio of non-padding predictions

    def validate(self, dataset):
        accuracies = []
        for data in self.generator(dataset):
            accuracies.append(self.evaluate(data[0], data[1]).numpy())
        accuracies = np.array(accuracies)
        return np.mean(accuracies)*100.0
    
    def validate_rd_file(self, func='valid'):
        accuracies = []
        for data in self.get_data(self.batch_size, func):
            input, output = self.clean_data(data)
            accuracies.append(self.evaluate(input, output).numpy())
        accuracies = np.array(accuracies)
        return np.mean(accuracies)*100.0

    def train(self):
        # print(f'{datetime.datetime.now()}: begin training with validation accuracy of {self.validate(self.valid_dataset)}%')
        print(f'{datetime.datetime.now()}: begin epoch {self.n_epochs+1}')
        for batch in self.generator(self.train_dataset):
            # train on the data
            self.optimize(batch[0], batch[1][:, :-1], batch[1][:, 1:])
            self.num_eq += self.batch_size
            # validate the model every epoch
            if self.num_eq >= self.epoch_size:
                # self.model.save_checkpoint()
                self.n_epochs += 1
                if self.n_epochs % 50 == 0:
                    print(f'{datetime.datetime.now()}: completed epoch {self.n_epochs} with validation accuracy of {self.validate(self.valid_dataset)}%')
                else:
                    print(f'{datetime.datetime.now()}: completed epoch {self.n_epochs}')
                self.num_eq = 0
                print(f'{datetime.datetime.now()}: begin epoch {self.n_epochs+1}')
        self.model.save_checkpoint()
        print(f'{datetime.datetime.now()}: finished training! :)')
        print(f'{datetime.datetime.now()}: test accuracy of {self.validate(self.test_dataset)}%')

    def train_rd_file(self):
        # print(f'{datetime.datetime.now()}: begin training with validation accuracy of {self.validate_rd_file()}%')
        print(f'{datetime.datetime.now()}: begin epoch {self.n_epochs+1}')
        for batch in self.get_data(self.batch_size, 'train', self.start_idx):
            # train on the data
            input, output = self.clean_data(batch)
            self.optimize(input, output[:, :-1], output[:, 1:])
            self.num_eq += self.batch_size
            # validate the model every epoch
            if self.num_eq >= self.epoch_size:
                # self.model.save_checkpoint()
                self.n_epochs += 1
                self.model.save_checkpoint()
                print(f'{datetime.datetime.now()}: completed epoch {self.n_epochs} with validation accuracy of {self.validate_rd_file()}%')
                self.num_eq = 0
        self.model.save_checkpoint()
        print(f'{datetime.datetime.now()}: finished training! :)')
        print(f'{datetime.datetime.now()}: test accuracy of {self.validate_rd_file('test')}%')