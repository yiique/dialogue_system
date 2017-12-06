__author__ = 'liushuman'

import graph_base


class HRED(graph_base.GraphBase):
    def __init__(self, in_dim, h_dim, c_dim, hyper_params=None, params=None):
        graph_base.GraphBase.__init__(self, hyper_params, params)

        self.hyper_params["in_dim"] = in_dim
        self.hyper_params["h_dim"] = h_dim
        self.hyper_params["c_dim"] = c_dim

        self.lstm = graph_base.LSTM(self.hyper_params["in_dim"], self.hyper_params["h_dim"], self.hyper_params["c_dim"])
        self.params.extend(self.lstm.params)

    # TODO: here step in LSTM class
    # TODO: rl part didn't implementation