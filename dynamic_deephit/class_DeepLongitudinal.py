import tensorflow as tf

from tensorflow import keras

import utils_network as utils

_EPSILON = 1e-08



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.math.log(x + _EPSILON)

def div(x, y):
    return tf.math.divide(x, (y + _EPSILON))

def get_seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    tmp_length = tf.reduce_sum(used, 1)
    tmp_length = tf.cast(tmp_length, tf.int32)
    return tmp_length

def l1_regularizer(scale):
    '''
        tf.contrib.layers.l1_regularizer returned None for scale == 0; keep that behaviour so that
        a zero-scale setting adds no term (and no op) to the regularization loss.
    '''
    if scale == 0.:
        return None
    return keras.regularizers.L1(scale)


class Model_Longitudinal_Attention(keras.layers.Layer):
    '''
        Subclasses keras.layers.Layer rather than keras.Model on purpose: Layer already provides
        everything needed here (variable tracking, .losses for the regularizers, and checkpointing
        of the sublayers and the optimizers) while keras.Model would additionally define predict(),
        fit() and evaluate() -- names this class overrides with incompatible signatures.
    '''
    def __init__(self, name, input_dims, network_settings):
        super().__init__(name=name)

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.x_dim_cont         = input_dims['x_dim_cont']
        self.x_dim_bin          = input_dims['x_dim_bin']

        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.max_length         = input_dims['max_length']

        # NETWORK HYPER-PARMETERS
        self.h_dim1             = network_settings['h_dim_RNN']
        self.h_dim2             = network_settings['h_dim_FC']
        self.num_layers_RNN     = network_settings['num_layers_RNN']
        self.num_layers_ATT     = network_settings['num_layers_ATT']
        self.num_layers_CS      = network_settings['num_layers_CS']

        self.RNN_type           = network_settings['RNN_type']

        self.FC_active_fn       = network_settings['FC_active_fn']
        self.RNN_active_fn      = network_settings['RNN_active_fn']
        self.initial_W          = network_settings['initial_W']

        self.reg_W              = l1_regularizer(network_settings['reg_W'])
        self.reg_W_out          = l1_regularizer(network_settings['reg_W_out'])

        self._build_net()
        self._build_optimizers()
        self._build_functions()


    def _build_net(self):
        ##### SHARED SUBNETWORK: RNN w/ TEMPORAL ATTENTION
        #create the cells with RNN hyper-parameters (RNN types, #layers, #nodes, activation functions)
        self.cells        = utils.create_rnn_cells(self.h_dim1, self.num_layers_RNN,
                                                   self.RNN_type, self.RNN_active_fn)

        #the temporal attention network; built once here and called at every time step
        self.att_net      = utils.FCNet(self.num_layers_ATT, self.h_dim2, tf.nn.tanh, 1, None,
                                        self.initial_W, name='Attention')

        self.z_mean_layer = keras.layers.Dense(self.x_dim, activation=None,
                                               kernel_initializer=self.initial_W, name='RNN_out_mean1')
        self.z_std_layer  = keras.layers.Dense(self.x_dim, activation=None,
                                               kernel_initializer=self.initial_W, name='RNN_out_std1')

        ##### CS-SPECIFIC SUBNETWORK w/ FCNETS
        #1 layer for combining inputs
        self.layer1       = keras.layers.Dense(self.h_dim2, activation=self.FC_active_fn,
                                               kernel_initializer=self.initial_W, name='Layer1')

        # (num_layers_CS-1) layers for cause-specific (num_Event subNets)
        self.cs_nets      = [
            utils.FCNet(self.num_layers_CS, self.h_dim2, self.FC_active_fn, self.h_dim2,
                        self.FC_active_fn, self.initial_W, self.reg_W, name='CS_{}'.format(e))
            for e in range(self.num_Event)
        ]

        self.out_layer    = keras.layers.Dense(self.num_Event * self.num_Category,
                                               activation=tf.nn.softmax,
                                               kernel_initializer=self.initial_W,
                                               kernel_regularizer=self.reg_W_out, name='Output')

        #run the network once so that every variable exists before the optimizers are built
        dummy_x = tf.random.normal([2, self.max_length, self.x_dim])
        self(dummy_x, tf.zeros_like(dummy_x), keep_prob=1.0)


    def _build_optimizers(self):
        # two independent optimizers, mirroring the two separate Adam ops of the original graph
        self.solver         = keras.optimizers.Adam(learning_rate=1e-4)
        self.solver_burn_in = keras.optimizers.Adam(learning_rate=1e-4)

        self.solver.build(self.trainable_variables)
        self.solver_burn_in.build(self.trainable_variables)


    def _build_functions(self):
        # explicit signatures (batch dimension left free) so that varying minibatch sizes never retrace
        spec_x  = tf.TensorSpec([None, self.max_length, self.x_dim], tf.float32)
        spec_1  = tf.TensorSpec([None, 1], tf.float32)
        spec_m  = tf.TensorSpec([None, self.num_Event, self.num_Category], tf.float32)
        spec_m3 = tf.TensorSpec([None, self.num_Category], tf.float32)
        spec_s  = tf.TensorSpec([], tf.float32)

        self._forward_fn    = tf.function(self._forward, input_signature=[spec_x, spec_x, spec_s])
        self._cost_fn       = tf.function(
            self._cost, input_signature=[spec_x, spec_x, spec_1, spec_1, spec_m, spec_m, spec_m3,
                                         spec_s, spec_s, spec_s, spec_s])
        self._train_fn      = tf.function(
            self._train_step, input_signature=[spec_x, spec_x, spec_1, spec_1, spec_m, spec_m, spec_m3,
                                               spec_s, spec_s, spec_s, spec_s])
        self._burn_in_fn    = tf.function(self._burn_in_step, input_signature=[spec_x, spec_x, spec_s])


    ### FORWARD PASS
    def call(self, x, x_mi, keep_prob=1.0):
        mb_size        = tf.shape(x)[0]

        seq_length     = get_seq_length(x)
        tmp_range      = tf.expand_dims(tf.range(0, self.max_length, 1), axis=0)

        rnn_mask1      = tf.cast(tf.less_equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)
        rnn_mask2      = tf.cast(tf.equal(tmp_range, tf.expand_dims(seq_length - 1, axis=1)), tf.float32)

        # divide into the last x and previous x's
        x_last = tf.reduce_sum(tf.tile(tf.expand_dims(rnn_mask2, axis=2), [1,1,self.x_dim]) * x, axis=1)    #sum over time since all others time stamps are 0
        x_last = tf.slice(x_last, [0,1], [-1,-1])                               #remove the delta of the last measurement
        x_hist = x * (1.-tf.tile(tf.expand_dims(rnn_mask2, axis=2), [1,1,self.x_dim]))                                    #since all others time stamps are 0 and measurements are 0-padded
        x_hist = tf.slice(x_hist, [0, 0, 0], [-1,(self.max_length-1),-1])

        # do same thing for missing indicator
        mi_last = tf.reduce_sum(tf.tile(tf.expand_dims(rnn_mask2, axis=2), [1,1,self.x_dim]) * x_mi, axis=1)    #sum over time since all others time stamps are 0
        mi_last = tf.slice(mi_last, [0,1], [-1,-1])                               #remove the delta of the last measurement
        mi_hist = x_mi * (1.-tf.tile(tf.expand_dims(rnn_mask2, axis=2), [1,1,self.x_dim]))                                    #since all others time stamps are 0 and measurements are 0-padded
        mi_hist = tf.slice(mi_hist, [0, 0, 0], [-1,(self.max_length-1),-1])

        all_hist = tf.concat([x_hist, mi_hist], axis=2)
        all_last = tf.concat([x_last, mi_last], axis=1)

        #extract inputs for the temporal attention: mask (to incorporate only the measured time) and x_{M}
        rnn_mask_att   = tf.cast(tf.not_equal(tf.reduce_sum(x_hist, axis=2), 0), dtype=tf.float32)  #[mb_size, max_length-1], 1:measurements 0:no measurements


        ##### SHARED SUBNETWORK: RNN w/ TEMPORAL ATTENTION
        # explicit time-step loop (replaces tf.nn.raw_rnn + loop_fn_att): at step j the attention power
        # e_{j} is computed from the state *after* consuming the j-th history measurement.
        states       = utils.get_initial_state(self.cells, mb_size)
        rnn_outputs  = []   #cell outputs
        rnn_states   = []   #hidden states (h_{j})
        att_powers   = []   #att power (e_{j})

        for time in range(self.max_length-1):
            cell_output, states = utils.rnn_step(self.cells, all_hist[:, time], states, keep_prob)

            tmp_h = utils.create_concat_state(states, self.num_layers_RNN, self.RNN_type)

            e = self.att_net(tf.concat([tmp_h, all_last], axis=1), keep_prob=keep_prob)
            e = tf.exp(e)

            rnn_outputs.append(cell_output)
            rnn_states.append(tmp_h)
            att_powers.append(e)

        rnn_final_state = states
        rnn_outputs     = tf.stack(rnn_outputs, axis=1)   #[mb_size, max_length-1, h_dim1]
        rnn_states      = tf.stack(rnn_states, axis=1)    #[mb_size, max_length-1, num_layers_RNN*h_dim1]

        att_weight  = tf.reshape(tf.stack(att_powers, axis=1), [-1, self.max_length-1]) * rnn_mask_att # masking to set 0 for the unmeasured e_{j}

        #get a_{j} = e_{j}/sum_{l=1}^{M-1}e_{l}
        att_weight  = div(att_weight, (tf.reduce_sum(att_weight, axis=1, keepdims=True) + _EPSILON)) #softmax (tf.exp is done, previously)

        # 1) expand att_weight to hidden state dimension, 2) c = \sum_{j=1}^{M} a_{j} x h_{j}
        context_vec = tf.reduce_sum(tf.tile(tf.reshape(att_weight, [-1, self.max_length-1, 1]), [1, 1, self.num_layers_RNN*self.h_dim1]) * rnn_states, axis=1)


        z_mean      = self.z_mean_layer(rnn_outputs)
        z_std       = tf.exp(self.z_std_layer(rnn_outputs))

        epsilon     = tf.random.normal([mb_size, self.max_length-1, self.x_dim], mean=0.0, stddev=1.0, dtype=tf.float32)
        z           = z_mean + z_std * epsilon


        ##### CS-SPECIFIC SUBNETWORK w/ FCNETS
        inputs = tf.concat([x_last, context_vec], axis=1)

        #1 layer for combining inputs
        h = self.layer1(inputs)
        h = tf.nn.dropout(h, rate=1.-keep_prob)

        # (num_layers_CS-1) layers for cause-specific (num_Event subNets)
        out = [cs_net(h, keep_prob=keep_prob) for cs_net in self.cs_nets]
        out = tf.stack(out, axis=1) # stack referenced on subject
        out = tf.reshape(out, [-1, self.num_Event*self.h_dim2])
        out = tf.nn.dropout(out, rate=1.-keep_prob)

        out = self.out_layer(out)
        out = tf.reshape(out, [-1, self.num_Event, self.num_Category])

        return {'out'            : out,
                'z'              : z,
                'z_mean'         : z_mean,
                'z_std'          : z_std,
                'att_weight'     : att_weight,
                'context_vec'    : context_vec,
                'rnn_final_state': rnn_final_state,
                'rnn_mask1'      : rnn_mask1}


    ### LOSS-FUNCTION 1 -- Log-likelihood loss
    def loss_Log_Likelihood(self, out, k, fc_mask1, fc_mask2):
        sigma3 = tf.constant(1.0, dtype=tf.float32)

        I_1 = tf.sign(k)
        denom = 1 - tf.reduce_sum(tf.reduce_sum(fc_mask1 * out, axis=2), axis=1, keepdims=True) # make subject specific denom.
        denom = tf.clip_by_value(denom, tf.cast(_EPSILON, dtype=tf.float32), tf.cast(1.-_EPSILON, dtype=tf.float32))

        #for uncenosred: log P(T=t,K=k|x,Y,t>t_M)
        tmp1 = tf.reduce_sum(tf.reduce_sum(fc_mask2 * out, axis=2), axis=1, keepdims=True)
        tmp1 = I_1 * log(div(tmp1,denom))

        #for censored: log \sum P(T>t|x,Y,t>t_M)
        tmp2 = tf.reduce_sum(tf.reduce_sum(fc_mask2 * out, axis=2), axis=1, keepdims=True)
        tmp2 = (1. - I_1) * log(div(tmp2,denom))

        return - tf.reduce_mean(tmp1 + sigma3*tmp2)


    ### LOSS-FUNCTION 2 -- Ranking loss
    def loss_Ranking(self, out, k, t, fc_mask3):
        sigma1 = tf.constant(0.1, dtype=tf.float32)

        eta = []
        for e in range(self.num_Event):
            one_vector = tf.ones_like(t, dtype=tf.float32)
            I_2 = tf.cast(tf.equal(k, e+1), dtype = tf.float32) #indicator for event
            I_2 = tf.linalg.tensor_diag(tf.squeeze(I_2, axis=1))
            tmp_e = tf.reshape(tf.slice(out, [0, e, 0], [-1, 1, -1]), [-1, self.num_Category]) #event specific joint prob.

            R = tf.matmul(tmp_e, tf.transpose(fc_mask3)) #no need to divide by each individual dominator
            # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

            diag_R = tf.reshape(tf.linalg.tensor_diag_part(R), [-1, 1])
            R = tf.matmul(one_vector, tf.transpose(diag_R)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
            R = tf.transpose(R)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

            T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(t)) - tf.matmul(t, tf.transpose(one_vector))))
            # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

            T = tf.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i

            tmp_eta = tf.reduce_mean(T * tf.exp(-R/sigma1), axis=1, keepdims=True)

            eta.append(tmp_eta)
        eta = tf.stack(eta, axis=1) #stack referenced on subjects
        eta = tf.reduce_mean(tf.reshape(eta, [-1, self.num_Event]), axis=1, keepdims=True)

        return tf.reduce_sum(eta) #sum over num_Events


    ### LOSS-FUNCTION 3 -- RNN prediction loss
    def loss_RNN_Prediction(self, z, x, x_mi, rnn_mask1):
        tmp_x  = tf.slice(x, [0,1,0], [-1,-1,-1])  # (t=2 ~ M)
        tmp_mi = tf.slice(x_mi, [0,1,0], [-1,-1,-1])  # (t=2 ~ M)

        tmp_mask1  = tf.tile(tf.expand_dims(rnn_mask1, axis=2), [1,1,self.x_dim]) #for hisotry (1...J-1)
        tmp_mask1  = tmp_mask1[:, :(self.max_length-1), :]

        zeta = tf.reduce_mean(tf.reduce_sum(tmp_mask1 * (1. - tmp_mi) * tf.pow(z - tmp_x, 2), axis=1))  #loss calculated for selected features.

        return zeta


    ### REGULARIZATION LOSS (replaces tf.losses.get_regularization_loss)
    def regularization_loss(self):
        reg_losses = self.losses
        if len(reg_losses) == 0:
            return tf.constant(0., dtype=tf.float32)
        return tf.add_n(reg_losses)


    ##### GRAPH FUNCTIONS (traced once each, see _build_functions)
    def _forward(self, x, x_mi, keep_prob):
        return self(x, x_mi, keep_prob=keep_prob)

    def _total_loss(self, x, x_mi, k, t, m1, m2, m3, a, b, c, keep_prob):
        res    = self(x, x_mi, keep_prob=keep_prob)
        LOSS_1 = self.loss_Log_Likelihood(res['out'], k, m1, m2)
        LOSS_2 = self.loss_Ranking(res['out'], k, t, m3)
        LOSS_3 = self.loss_RNN_Prediction(res['z'], x, x_mi, res['rnn_mask1'])
        return a*LOSS_1 + b*LOSS_2 + c*LOSS_3 + self.regularization_loss()

    def _cost(self, x, x_mi, k, t, m1, m2, m3, a, b, c, keep_prob):
        return self._total_loss(x, x_mi, k, t, m1, m2, m3, a, b, c, keep_prob)

    def _apply_gradients(self, optimizer, loss, tape):
        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(
            [(g, v) for g, v in zip(gradients, variables) if g is not None])

    def _train_step(self, x, x_mi, k, t, m1, m2, m3, a, b, c, keep_prob):
        with tf.GradientTape() as tape:
            LOSS_TOTAL = self._total_loss(x, x_mi, k, t, m1, m2, m3, a, b, c, keep_prob)
        self._apply_gradients(self.solver, LOSS_TOTAL, tape)
        return LOSS_TOTAL

    def _burn_in_step(self, x, x_mi, keep_prob):
        with tf.GradientTape() as tape:
            res         = self(x, x_mi, keep_prob=keep_prob)
            LOSS_3      = self.loss_RNN_Prediction(res['z'], x, x_mi, res['rnn_mask1'])
            LOSS_BURNIN = LOSS_3 + self.regularization_loss()
        self._apply_gradients(self.solver_burn_in, LOSS_BURNIN, tape)
        return LOSS_3


    ##### PUBLIC API
    def get_cost(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma)      = PARAMETERS
        cost = self._cost_fn(_f32(x_mb), _f32(x_mi_mb), _f32(k_mb), _f32(t_mb),
                             _f32(m1_mb), _f32(m2_mb), _f32(m3_mb),
                             _f32(alpha), _f32(beta), _f32(gamma), _f32(keep_prob))
        return cost.numpy()

    def train(self, DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (m1_mb, m2_mb, m3_mb)     = MASK
        (x_mi_mb)                 = MISSING
        (alpha, beta, gamma)      = PARAMETERS
        self.solver.learning_rate.assign(lr_train)
        loss = self._train_fn(_f32(x_mb), _f32(x_mi_mb), _f32(k_mb), _f32(t_mb),
                              _f32(m1_mb), _f32(m2_mb), _f32(m3_mb),
                              _f32(alpha), _f32(beta), _f32(gamma), _f32(keep_prob))
        return None, loss.numpy()

    def train_burn_in(self, DATA, MISSING, keep_prob, lr_train):
        (x_mb, k_mb, t_mb)        = DATA
        (x_mi_mb)                 = MISSING

        self.solver_burn_in.learning_rate.assign(lr_train)
        loss = self._burn_in_fn(_f32(x_mb), _f32(x_mi_mb), _f32(keep_prob))
        return None, loss.numpy()

    def predict(self, x_test, x_mi_test, keep_prob=1.0):
        return self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))['out'].numpy()

    def predict_z(self, x_test, x_mi_test, keep_prob=1.0):
        return self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))['z'].numpy()

    def predict_rnnstate(self, x_test, x_mi_test, keep_prob=1.0):
        '''
            Returns the final RNN state as [layer][state], where the Keras cell state ordering is
            [h, c] for LSTM and [h] for GRU -- i.e. the hidden state is state[i][0].
            NOTE: under TF1 this returned tf.contrib LSTMStateTuple(c, h), where the hidden state was
                  state[i][1]. Code carried over from the TF1 version must swap that index.
        '''
        state = self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))['rnn_final_state']
        return [[s.numpy() for s in layer_state] for layer_state in state]

    def predict_att(self, x_test, x_mi_test, keep_prob=1.0):
        return self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))['att_weight'].numpy()

    def predict_context_vec(self, x_test, x_mi_test, keep_prob=1.0):
        return self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))['context_vec'].numpy()

    def get_z_mean_and_std(self, x_test, x_mi_test, keep_prob=1.0):
        res = self._forward_fn(_f32(x_test), _f32(x_mi_test), _f32(keep_prob))
        return res['z_mean'].numpy(), res['z_std'].numpy()


def _f32(value):
    return tf.convert_to_tensor(value, dtype=tf.float32)
