__author__ = 'liushuman'

import tensorflow as tf
import yaml

from tensorflow.python.ops import tensor_array_ops, control_flow_ops


FLAGS = tf.flags.FLAGS


def set_params(value):
    return tf.Variable(value)


def get_params(shape, scale=0.01):
    return tf.Variable(tf.random_normal(shape, stddev=scale))


class GraphBase(object):
    def __init__(self, hyper_params=None, params=None):
        if not hyper_params:
            self.hyper_params = {}
        else:
            self.hyper_params = hyper_params
        if not params:
            self.params = []
        else:
            self.params = params

    def print_params(self):
        classname = self.__class__.__name__
        tf.logging.info("==In class %s", classname)
        tf.logging.info("\nHyper params: %s", yaml.dump({classname: self.hyper_params}))


class LSTM(GraphBase):
    def __init__(self, in_dim, h_dim, c_dim=None, hyper_params=None, params=None):
        GraphBase.__init__(self, hyper_params, params)
        self.hyper_params["in_dim"] = in_dim
        self.hyper_params["h_dim"] = h_dim
        self.hyper_params["c_dim"] = c_dim

        self.recurrent_unit = self.create_recurrent_unit(c_dim)

    def step(self, x_t, x_m_t, cell_tm1):
        """
        :param x_t: input in batch_size * in_dim
        :param x_m_t: mask in batch_size * 1
        :param cell_tm1: hidden with memory in batch_size * h_dim
        :return: new hidden with memory
        """
        h_t, m_t = self.recurrent_unit(x_t, x_m_t, cell_tm1)
        return [h_t, m_t]

    def step_with_content(self, x_t, x_m_t, content, cell_tm1):
        """
        :param x_t: input in batch_size * in_dim
        :param x_m_t: mask in batch_size * 1
        :param content: content in batch_size * c_dim
        :param cell_tm1: hidden with memory in batch_size * h_dim
        :return: new hidden with memory
        """
        h_t, m_t = self.recurrent_unit(x_t, x_m_t, content, cell_tm1)
        return [h_t, m_t]

    def _step(self, i, cell_tm1,
              x_ta, x_mask_ta, hidden_ta):
        x_t = tf.reshape(x_ta.read(i), [-1, self.hyper_params["in_dim"]])
        x_mask_t = tf.reshape(x_mask_ta.read(i), [-1, 1])

        h_t, m_t = self.recurrent_unit(x_t, x_mask_t, cell_tm1)
        hidden_ta = hidden_ta.write(i, h_t)

        return i+1, [h_t, m_t], \
               x_ta, x_mask_ta, hidden_ta

    def forward(self, x, x_mask):
        """
        :param x: input matrix in           max_len * batch_size * in_dim
                    with start token and end token
        :param x_mask: mask matrix in       max_len * batch_size
        :return: hidden matrix in           max_len * batch_size * h_dim
        """
        x_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x)
        x_mask_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True).unstack(x_mask)
        hidden_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        _, _, _, _, hidden_ta = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < x_ta.size(),
            body=self._step,
            loop_vars=(
                tf.constant(0, dtype=tf.int32),
                [tf.zeros([FLAGS.batch_size, self.hyper_params["h_dim"]], dtype=tf.float32),
                 tf.zeros([FLAGS.batch_size, self.hyper_params["h_dim"]], dtype=tf.float32)],
                x_ta, x_mask_ta, hidden_ta
            )
        )

        hidden_ta = hidden_ta.stack()
        return hidden_ta

    def create_recurrent_unit(self, con=False):
        self.W_i = get_params([self.hyper_params["in_dim"], self.hyper_params["h_dim"]])
        self.U_i = get_params([self.hyper_params["h_dim"], self.hyper_params["h_dim"]])
        self.b_i = get_params([self.hyper_params["h_dim"]])

        self.W_f = get_params([self.hyper_params["in_dim"], self.hyper_params["h_dim"]])
        self.U_f = get_params([self.hyper_params["h_dim"], self.hyper_params["h_dim"]])
        self.b_f = get_params([self.hyper_params["h_dim"]])

        self.W_o = get_params([self.hyper_params["in_dim"], self.hyper_params["h_dim"]])
        self.U_o = get_params([self.hyper_params["h_dim"], self.hyper_params["h_dim"]])
        self.b_o = get_params([self.hyper_params["h_dim"]])

        self.W_c = get_params([self.hyper_params["in_dim"], self.hyper_params["h_dim"]])
        self.U_c = get_params([self.hyper_params["h_dim"], self.hyper_params["h_dim"]])
        self.b_c = get_params([self.hyper_params["h_dim"]])

        self.params.extend([
            self.W_i, self.U_i, self.b_i,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
            self.W_c, self.U_c, self.b_c
        ])

        if con:
            self.V_i = get_params([self.hyper_params["c_dim"], self.hyper_params["h_dim"]])
            self.V_f = get_params([self.hyper_params["c_dim"], self.hyper_params["h_dim"]])
            self.V_o = get_params([self.hyper_params["c_dim"], self.hyper_params["h_dim"]])
            self.V_c = get_params([self.hyper_params["c_dim"], self.hyper_params["h_dim"]])
            self.params.extend([self.V_i, self.V_f, self.V_o, self.V_c])

        def unit(x, x_m, cell_tm1):
            """
            :param x: input in batch_size * in_dim
            :param x_m: mask in batch_size * 1
            :param cell_tm1: hidden with memory in batch_size * h_dim
            :return: new hidden with memory
            """
            hidden_tm1, memory_tm1 = cell_tm1

            # Input Gate
            i = tf.sigmoid(tf.matmul(x, self.W_i) + tf.matmul(hidden_tm1, self.U_i) + self.b_i)

            # Forget Gate
            f = tf.sigmoid(tf.matmul(x, self.W_f) + tf.matmul(hidden_tm1, self.U_f) + self.b_f)

            # Output Gate
            o = tf.sigmoid(tf.matmul(x, self.W_o) + tf.matmul(hidden_tm1, self.U_o) + self.b_o)

            # Candidate memory
            memory_ = tf.nn.tanh(tf.matmul(x, self.W_c) + tf.matmul(hidden_tm1, self.U_c) + self.b_c)

            # Memory Cell
            memory = f * memory_tm1 + i * memory_

            # Hidden
            hidden = o * tf.nn.tanh(memory)

            # Mask
            memory = x_m * memory + (1. - x_m) * memory_tm1
            hidden = x_m * hidden + (1. - x_m) * hidden_tm1

            return [hidden, memory]

        def unit_with_content(x, x_m, content, cell_tm1):
            """
            :param x: input in batch_size * in_dim
            :param x_m: mask in batch_size * 1
            :param content: content in batch_size * c_dim
            :param cell_tm1: hidden with memory in batch * h_dim
            :return: new hidden with memory
            """
            hidden_tm1, memory_tm1 = cell_tm1

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.W_i) + tf.matmul(hidden_tm1, self.U_i) + tf.matmul(content, self.V_i) + self.b_i)

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.W_f) + tf.matmul(hidden_tm1, self.U_f) + tf.matmul(content, self.V_f) + self.b_f)

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.W_o) + tf.matmul(hidden_tm1, self.U_o) + tf.matmul(content, self.V_o) + self.b_o)

            # Candidate memory
            memory_ = tf.nn.tanh(
                tf.matmul(x, self.W_c) + tf.matmul(hidden_tm1, self.U_c) + tf.matmul(content, self.V_c) + self.b_c)

            # Memory Cell
            memory = f * memory_tm1 + i * memory_

            # Hidden
            hidden = o * tf.nn.tanh(memory)

            # Mask
            memory = x_m * memory + (1. - x_m) * memory_tm1
            hidden = x_m * hidden + (1. - x_m) * hidden_tm1

            return [hidden, memory]

        if not con:
            return unit
        return unit_with_content


class CNN(GraphBase):
    def __init__(self, kernel_size=3, width=512, units=512):
        GraphBase.__init__(self)
        self.hyper_params["kernel_size"] = kernel_size
        self.hyper_params["width"] = width
        self.hyper_params["units"] = units

        self.convolutional_unit, self.activate_unit, self.pooling_unit\
            = self.create_convolutional_unit()

    def create_convolutional_unit(self):
        filter_shape = [self.hyper_params["kernel_size"], self.hyper_params["width"],
                        1, self.hyper_params["units"]]
        self.W = set_params(tf.truncated_normal(filter_shape, stddev=0.1))
        self.params.extend([self.W])

        def convolutional_unit(x):
            """
            deep CNN unit without pooling
            :param x: input in batch_size * seq_len * width * 1
            :return: hidden in batch_size * seq_len-kernel_size+1 * unit * 1
            """
            conv = tf.nn.conv2d(
                x, self.W,
                strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC"
            )
            return conv

        def activate_unit(conv):
            hidden = tf.nn.relu(conv)
            return hidden

        def pooling_unit(hidden):
            """
            mean pooling in the highest layer
            :param hidden: highest layer hidden in batch_size * len * width * 1
            :return: pooling state in batch_size * width
            """
            final_state = tf.squeeze(tf.reduce_mean(hidden, 1))
            return final_state

        return convolutional_unit, activate_unit, pooling_unit
