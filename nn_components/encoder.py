__author__ = 'liushuman'

import graph_base
import tensorflow as tf


class Encoder(graph_base.GraphBase):

    def __init__(self, layer_num, in_dim, h_dim, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["layer_num"] = layer_num
        self.hyper_params["in_dim"] = in_dim
        self.hyper_params["h_dim"] = h_dim

        self.lstms = []
        for i in range(self.hyper_params["layer_num"]):
            if i == 0:
                llstm = graph_base.LSTM(self.hyper_params["in_dim"], self.hyper_params["h_dim"])
            else:
                llstm = graph_base.LSTM(self.hyper_params["h_dim"], self.hyper_params["h_dim"])
            rlstm = graph_base.LSTM(self.hyper_params["h_dim"], self.hyper_params["h_dim"])
            self.lstms.append([llstm, rlstm])
            self.params.extend(llstm.params)
            self.params.extend(rlstm.params)

    def forward(self, x, x_mask):
        """
        :param x: input in max_len * batch_size * in_dim
        :param x_mask: mask in max_len * batch_size
        :return: hidden in max_len * batch_size * h_dim
        """
        hidden = x
        for i in range(self.hyper_params["layer_num"]):
            llstm, rlstm = self.lstms[i]

            # TODO: residual didn't add here
            hidden = llstm.forward(hidden, x_mask)
            hidden = rlstm.forward(hidden[::-1], x_mask[::-1])

        return hidden


class CNNEncoder(graph_base.GraphBase):

    def __init__(self, layer_num, kernel_size, in_dim, h_dim, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["layer_num"] = layer_num
        self.hyper_params["kernel_size"] = kernel_size
        self.hyper_params["in_dim"] = in_dim                                # embedding size
        self.hyper_params["h_dim"] = h_dim                                  # unit size

        # position embedding set outside

        self.cnns = []
        for i in range(self.hyper_params["layer_num"]):
            if i == 0:
                cnn = graph_base.CNN(
                    self.hyper_params["kernel_size"], self.hyper_params["in_dim"], self.hyper_params["h_dim"])
            else:
                cnn = graph_base.CNN(
                    self.hyper_params["kernel_size"], self.hyper_params["h_dim"], self.hyper_params["h_dim"])
            self.cnns.append(cnn)
            self.params.extend(cnn.params)

    def forward(self, x):
        """
        :param x: input in max_len * batch_size * in_dim(here is embedding + position, element-wise according to gnmt)
        :return: hidden in batch_size * h_dim
        """
        hidden = tf.expand_dims(tf.transpose(x, perm=[1, 0, 2]), -1)
        for i in range(self.hyper_params["layer_num"]):
            cnn = self.cnns[i]
            conv = cnn.convolutional_unit(hidden)

            # residual connection
            if i > 0:
                conv += hidden
            hidden = cnn.activate_unit(conv)
        final_state = self.cnns[-1].pooling_unit(hidden)

        return final_state
