import tensorflow as tf

from tensorflow import keras


### CONSTRUCT CELLS FOR MULTI-LAYER RNNS
def create_rnn_cells(num_units, num_layers, RNN_type, activation_fn):
    '''
        GOAL         : create the list of cells used to construct a multi-layer RNN
        num_units    : number of units in each layer
        num_layers   : number of layers
        RNN_type     : either 'LSTM' or 'GRU'

        Keras 3 has no MultiRNNCell/DropoutWrapper, so the cells are returned as a plain list and
        stacked (with dropout) by rnn_step() below. This keeps the keep probability a run-time value
        instead of baking it into the cell at construction time.
    '''
    if activation_fn == 'None':
        activation_fn = tf.nn.tanh

    cells = []
    for _ in range(num_layers):
        if RNN_type == 'GRU':
            # reset_after=False reproduces the original tf.contrib.rnn.GRUCell formulation
            cell = keras.layers.GRUCell(num_units, activation=activation_fn, reset_after=False)
        elif RNN_type == 'LSTM':
            cell = keras.layers.LSTMCell(num_units, activation=activation_fn,
                                         recurrent_activation='sigmoid', unit_forget_bias=True)
        else:
            raise ValueError('ERROR: WRONG RNN CELL TYPE: {}'.format(RNN_type))
        cells.append(cell)

    return cells


### RUN ONE TIME-STEP THROUGH THE STACKED CELLS
def rnn_step(cells, x_t, states, keep_prob=1.0):
    '''
        GOAL         : run a single time step through the stacked cells
        cells        : list of cells (output of create_rnn_cells)
        x_t          : input at the current time step [mb_size, input_dim]
        states       : list of per-layer states
        keep_prob    : keep probabilty [0, 1]  (if None, dropout is not employed)

        Reproduces MultiRNNCell wrapped in DropoutWrapper(input_keep_prob, output_keep_prob):
        the input of each layer and the output of each layer are both dropped out.
    '''
    new_states = []
    out        = x_t

    for i, cell in enumerate(cells):
        if keep_prob is not None:
            out = tf.nn.dropout(out, rate=1. - keep_prob)
        out, tmp_state = cell(out, states[i])
        if keep_prob is not None:
            out = tf.nn.dropout(out, rate=1. - keep_prob)
        new_states.append(tmp_state)

    return out, new_states


### GET THE INITIAL (ZERO) STATE OF THE STACKED CELLS
def get_initial_state(cells, batch_size):
    return [cell.get_initial_state(batch_size=batch_size) for cell in cells]


### EXTRACT STATE OUTPUT OF THE STACKED CELLS
def create_concat_state(states, num_layers, RNN_type):
    '''
        GOAL	     : concatenate the per-layer states into a single tensor
        states       : list of per-layer states (output of rnn_step)
        num_layers   : number of layers
        RNN_type     : either 'LSTM' or 'GRU'

        Keras cells return [h, c] for LSTM and [h] for GRU, so the hidden state h is states[i][0]
        in both cases (note: the tf.contrib LSTMStateTuple ordering was (c, h) instead).
    '''
    if RNN_type not in ('LSTM', 'GRU'):
        raise ValueError('ERROR: WRONG RNN CELL TYPE: {}'.format(RNN_type))

    return tf.concat([states[i][0] for i in range(num_layers)], axis=1)


### FEEDFORWARD NETWORK
class FCNet(keras.layers.Layer):
    '''
        GOAL             : Create FC network with different specifications
        num_layers       : number of layers in FCNet
        h_dim  (int)     : number of hidden units
        h_fn             : activation function for hidden layers (default: tf.nn.relu)
        o_dim  (int)     : number of output units
        o_fn             : activation function for output layers (defalut: None)
        w_init           : initialization for weight matrix (defalut: Xavier)
        w_reg            : regularizer for the weight matrix (not applied to the bias)

        The layers are built once, at construction time, and reused on every call -- in TF1 the
        equivalent function relied on variable-scope reuse inside raw_rnn's while_loop.
    '''
    def __init__(self, num_layers, h_dim, h_fn, o_dim, o_fn, w_init, w_reg=None, **kwargs):
        super().__init__(**kwargs)

        # default active functions (hidden: relu, out: None)
        if h_fn is None:
            h_fn = tf.nn.relu
        if o_fn is None:
            o_fn = None

        # default initialization functions (weight: Xavier, bias: None)
        if w_init is None:
            w_init = keras.initializers.GlorotUniform() # Xavier initialization

        self.hidden_layers = [
            keras.layers.Dense(h_dim, activation=h_fn, kernel_initializer=w_init, kernel_regularizer=w_reg)
            for _ in range(num_layers - 1)
        ]
        self.out_layer     = keras.layers.Dense(o_dim, activation=o_fn, kernel_initializer=w_init,
                                                kernel_regularizer=w_reg)

    def call(self, inputs, keep_prob=1.0):
        h = inputs
        for layer in self.hidden_layers:
            h = layer(h)
            if keep_prob is not None:
                h = tf.nn.dropout(h, rate=1. - keep_prob)

        return self.out_layer(h)
