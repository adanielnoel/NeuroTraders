import json
import logging
import re
from datetime import datetime

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

'''
 Tensorflow model for stock price prediction using a recurrent neural network with LSTM architecture
 To visualise results using Tensorboard fire up a terminal window in this folder and run:
    tensorboard --logdir=./logs
 Note: The output shows the test MSE and the baseline MSE. As long as the test MSE is higher than
       the baseline the neural net is not predicting anything.
'''


class RNNPredictor:
    def __init__(self, name, input_time_steps,
                 input_length=1, output_length=1, lstm_size=128, num_layers=1):
        """
        :param input_time_steps: Length in time of the input
        :param input_length: size of each of the input data points
        :param output_length: size of each of the output data points
        :param lstm_size: Size of the hidden LSTM state, per layer
        :param num_layers: Number of layers
        """
        self.name = name
        self.save_dir_models = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
        self.save_dir_current_model = None
        self.save_dir_logs = os.path.join(os.path.dirname(os.path.realpath(__file__)), "logs")
        self.input_time_steps = input_time_steps
        self.input_length = input_length
        self.output_length = output_length
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.graph = None
        self.selected_nodes = {}
        self.saver = None
        self.sess = None
        self.writer = None
        self.ridge_coeff = 100.0

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_node = tf.placeholder(tf.float32, [None, self.input_time_steps, self.input_length],
                                              name="inputs")
            self.targets_node = tf.placeholder(tf.float32, [None, self.output_length],
                                               name="targets")
            self.learning_rate_node = tf.placeholder(tf.float32, name="learning_rate")
            self.keep_prob_node = tf.placeholder(tf.float32, name="keep_prob")
            self.prediction_node = self._prediction_node()
            self.loss_node, self.baseline_loss = self._loss_node()
            self.optimize_node = self._optimize_node()
            self.summary_node = self._summary_node()
            self.initializer_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.input_scaler = None
        self.output_scaler = None
        self.sess = None
        self.current_train_session_date = datetime.now().strftime("_%Y-%m-%d_%H:%M:%S")

    def launch(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.graph.as_default()
        # Same as tf.global_variables_initializer(), but that didn't work without the <with ...> syntax
        self.sess.run(self.initializer_op)
        self.load_model()

    def close(self, step=None):
        self.save_model(step=step)
        self.sess.close()
        tf.reset_default_graph()
        self.sess = None

    def _prediction_node(self):
        def _create_lstm_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob_node)

        if self.num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([_create_lstm_cell() for _ in range(self.num_layers)],
                                               state_is_tuple=True)
        else:
            cell = _create_lstm_cell()

        # all_outputs contains the outputs of all the lstm cells
        # state = tf.Variable(cell.zero_state(50, tf.float32), trainable=False)
        all_outputs, _ = tf.nn.dynamic_rnn(cell, self.inputs_node, dtype=tf.float32)

        with tf.name_scope("Last_layer"):
            # rearrange from (batch_size, num_steps, lstm_size) to (num_steps, batch_size, lstm_size)
            all_outputs = tf.transpose(all_outputs, [1, 0, 2])
            # Get only the output of the last step for each batch
            last = tf.gather(all_outputs,
                             int(all_outputs.get_shape()[0]) - 1,
                             name="last_lstm_output")

            # Define weights and biases between the hidden and output layers
            # TODO: check why this step
            self.weights = tf.Variable(tf.random_uniform([self.lstm_size, self.output_length], -0.7, 0.7), name="w")
            # bias = tf.Variable(tf.constant(0.1, shape=[self.output_length]), name="b")
            self.bias = tf.Variable(tf.random_uniform([self.output_length], -0.7, 0.7), name="b")
            # TODO: make the bias also normally distributed around 0
            prediction = tf.matmul(last, self.weights) + self.bias

        return prediction

    def _loss_node(self):
        loss = tf.reduce_mean(tf.square(self.prediction_node - self.targets_node), name="loss_mse")
        baseline_loss = tf.reduce_mean(tf.square(self.targets_node), name="no_predict_loss")
        return loss, baseline_loss

    def _loss_node2(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction_node, labels=self.targets_node))
        return loss

    def _optimize_node2(self):
        minimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate_node).minimize(self.loss_node)
        return minimize

    def _optimize_node(self):
        minimize = tf.train.RMSPropOptimizer(self.learning_rate_node).minimize(
            self.loss_node + self.ridge_coeff * tf.reduce_sum(tf.abs(self.weights)), name="rmsprop_optim")
        return minimize

    def _summary_node(self):
        tf.summary.scalar("test_loss_MSE", self.loss_node)
        tf.summary.scalar("learning_rate", self.learning_rate_node)
        tf.summary.histogram("Weights", self.weights)
        tf.summary.histogram("biases", self.bias)
        summary_op = tf.summary.merge_all()
        return summary_op

    def train(self,
              train_X,
              train_Y,
              test_X,
              test_Y,
              keep_prob=1.0,  # Units to keep in dropout operation
              dropout_stop_epoch=1000000,
              init_learning_rate=0.001,  # Learning rate to start with
              learning_rate_decay=1.0,  # Decay ratio per epoch
              init_decay_epoch=0,  # Epoch number at which to start decay
              minimum_learning_rate=0.0,  # Minimum learning rate at which to stop decay
              max_epochs=100,  # Maximum number of epoch for training
              batch_size=1):

        self.current_train_session_date = datetime.now().strftime("_%Y-%m-%d_%H:%M:%S")

        self.save_dir_current_model = os.path.join(self.save_dir_models,
                                                   self.name + self.current_train_session_date)

        if self.sess is None:
            self.launch()

        self.writer = tf.summary.FileWriter(self.new_log_dir())
        self.writer.add_graph(self.graph)

        logger.info("Training started")

        global_step = 0

        for epoch in range(max_epochs):
            learning_rate = init_learning_rate * (
                1.0 if epoch < init_decay_epoch else learning_rate_decay ** (epoch - init_decay_epoch)
            )
            learning_rate = max(minimum_learning_rate, learning_rate)
            if epoch > dropout_stop_epoch:
                keep_prob = 1.0

            train_loss = 0
            for batch in range(train_X.shape[0]):
                train_loss, _, summary = self.sess.run([self.loss_node,
                                                        self.optimize_node,
                                                        self.summary_node],
                                                       {self.inputs_node: train_X[batch],
                                                        self.targets_node: train_Y[batch],
                                                        self.learning_rate_node: learning_rate,
                                                        self.keep_prob_node: keep_prob})
                global_step += 1

                if batch % 10 == 0:
                    test_loss, baseline_los, summary = self.sess.run([self.loss_node, self.baseline_loss, self.summary_node],
                                                                     {self.inputs_node: test_X[0],
                                                                      self.targets_node: test_Y[0],
                                                                      self.learning_rate_node: learning_rate,
                                                                      self.keep_prob_node: keep_prob
                                                                      })
                    print("Epoch:{:4}  lr: {:.6f} train_loss:{:.8f} test_loss:{:.8f}  baseline_loss:{:.8f}".format(
                        epoch, learning_rate, train_loss, test_loss, baseline_los
                    ))

                    self.writer.add_summary(summary, global_step=epoch * batch)
            self.writer.flush()
        logger.info("Training ended")
        self.save_model(global_step)

    def format_inputs(self, data, format_for_training=True):
        # Compute the offset of rows to ignore such that all the rest are a multiple of self.input_time_steps
        offset = data.shape[0] % self.input_time_steps
        if format_for_training:
            # ensure there is one extra element (the last element, which is ignored in training
            # since there is no data of the next data point to compare the prediction with)
            if offset == 0:
                data = data[self.input_time_steps - 1:]
            else:
                offset -= 1
        data = np.array([data[offset + i * self.input_time_steps: offset + (i + 1) * self.input_time_steps]
                         for i in range(data.shape[0] // self.input_time_steps)])
        return data

    def format_targets(self, data, reshape=True, format_for_training=True):
        offset = data.shape[0] % self.input_time_steps
        # Make sure to use the same range of data as in self.format_inputs(), only now do not add
        # an extra element since we are using the last one as target (see the discussion in self.format_inputs())
        if format_for_training:
            if offset == 0:
                data = data[self.input_time_steps:]
        # Create a tensor with output cases in the first dimension and target vectors in the second
        # The targets correspond to an index more that the last of the inputs in that case
        data = np.array([data[offset + i * self.input_time_steps - 1]
                         for i in range(1, data.shape[0] // self.input_time_steps + 1)])
        if reshape:
            data = data.reshape((*data.shape, -1))
        return data

    def predict(self, input_x):
        if self.sess is None:
            print("Session was closed")
            self.launch()
        prediction = self.sess.run([self.prediction_node],
                                   {
                                       self.inputs_node: input_x,
                                       self.learning_rate_node: 0.0,
                                       self.keep_prob_node: 1.0
                                   })[0]
        return prediction

    def new_log_dir(self, log_info=None):
        if log_info is None:
            log_info = {}
        log_info_file = os.path.join(self.save_dir_logs, "info.json")
        log_id = 0
        if not os.path.exists(self.save_dir_logs):
            os.makedirs(self.save_dir_logs)
            log_info["id"] = log_id
            with open(log_info_file, "w") as f:
                json.dump([log_info], f)
        else:
            with open(log_info_file, "r") as f:
                last_log_info = json.load(f)[-1]
            with open(log_info_file, "w") as f:
                log_id = last_log_info["id"] + 1
                log_info["id"] = log_id
                json.dump([log_info], f)
        return os.path.join(self.save_dir_logs, "{}_{}".format(log_id, self.current_train_session_date))

    def save_model(self, step):
        if not os.path.exists(self.save_dir_current_model):
            os.makedirs(self.save_dir_current_model)

        self.saver.save(self.sess, self.save_dir_current_model, global_step=step)

    def load_model(self):
        model_name = "RNNPredictor" + self.name + ".model"
        model_ckpt_dir = os.path.join(self.save_dir_models, model_name)

        if not os.path.exists(model_ckpt_dir):
            return False, 0

        ckpt = tf.train.get_checkpoint_state(model_ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_ckpt_dir, ckpt_name))
            # TODO: check what this does and if it's relevant enough to keep
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0


if __name__ == "__main__":
    import os
    from pandas import read_csv
    import math as m

    # """
    aapl_dataset = read_csv(os.path.dirname(os.path.realpath(__file__)) + "/../new_database/aapl/time_data.csv",
                            header=0, index_col=0)

    x = aapl_dataset[["relative_intraday",
                      "relative_overnight_tomorrow",
                      "vol_rel_m_ave_10",
                      "sentiment_p",
                      "sentiment_n",
                      "sentiment_u"]].values
    y = aapl_dataset[["relative_intraday_tomorrow"]].values.reshape(-1)
    # x = np.delete(x, [0, 1], 1)
    train_x = x[:int(0.8 * x.shape[0])]
    train_y = y[:int(0.8 * x.shape[0])]
    test_x = x[m.ceil(0.8 * x.shape[0]):]
    test_y = y[m.ceil(0.8 * x.shape[0]):]
    # Todo: split 60/25/15

    # train_x = np.random.normal(0.0, 1.0, (300, 6))
    # train_y = train_x[:, 1]
    # test_x = np.random.normal(0.0, 1.0, (200, 6))
    # test_y = test_x[:, 1]

    price_change_predictor = RNNPredictor(name="Price predictor",
                                          input_time_steps=5,
                                          input_length=x.shape[1],
                                          output_length=1,
                                          lstm_size=50,
                                          num_layers=1)
    price_change_predictor.train(train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y,
                                 max_epochs=600, batch_size=100,
                                 init_learning_rate=0.0005, minimum_learning_rate=0.00002,
                                 init_decay_epoch=100, learning_rate_decay=0.990,
                                 keep_prob=0.95, dropout_stop_epoch=700)

    import matplotlib.pyplot as plt

    predictions_test = price_change_predictor.predict(test_x)
    actual_test = price_change_predictor.format_targets(test_y, format_for_training=False)
    predictions_train = price_change_predictor.predict(train_x)
    actual_train = price_change_predictor.format_targets(train_y, format_for_training=False)
    # actual_test = actual_test.reshape((len(actual_test), 1))
    # actual_train = actual_train.reshape((len(actual_train), 1))

    # plt.plot(range(len(actual)), predictions - actual, label="absolute error")
    plt.subplot(211)
    plt.plot(range(len(actual_train)), predictions_train, label="prediction (train)")
    plt.plot(range(len(actual_train)), actual_train, label="truth (train)")
    plt.legend()
    plt.subplot(212)
    plt.plot(range(len(actual_test)), predictions_test, label="prediction (test)")
    plt.plot(range(len(actual_test)), actual_test, label="truth (test)")
    plt.legend()
    plt.show()

    # price_train = aapl_dataset["adj_close"].values[int(0.5 * x.shape[0]): int(0.8 * x.shape[0])]
    # price_train = price_train[price_change_predictor.input_time_steps - 1:]
    # price_train = price_change_predictor.format_targets(price_train, format_for_training=False)
    # price_pred_train = price_train[:-1] * (predictions_train + 1.0)
    # price_train = price_train[1:]
    #
    # price_test = aapl_dataset["adj_close"].values[m.ceil(0.8 * x.shape[0]):]
    # price_test = price_test[price_change_predictor.input_time_steps - 1:]
    # price_test = price_test.reshape((len(price_test), 1))
    # price_pred_test = price_test[:-1] * (predictions_test + 1.0)
    # price_test = price_test[1:]
    #
    # plt.subplot(211)
    # plt.plot(range(len(price_train)), price_train, label="Truth (train)")
    # plt.plot(range(len(price_train)), price_pred_train, label="Prediction (train)")
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(range(len(price_test)), price_test, label="Truth (test)")
    # plt.plot(range(len(price_test)), price_pred_test, label="Prediction (test)")
    # plt.legend()
    # plt.show()




#
# num_epochs = 100
# total_series_length = 50000
# truncated_backprop_length = 15
# state_size = 4
# num_classes = 2
# echo_step = 3
# batch_size = 5
# num_batches = total_series_length//batch_size//truncated_backprop_length
