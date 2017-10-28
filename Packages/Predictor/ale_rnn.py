import re

import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RNNPredictor:
    def __init__(self, name, input_time_steps, input_length=1, output_length=1, lstm_size=128, num_layers=1):
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
        self.X = None
        self.Y = None
        self.graph = None
        self.selected_nodes = {}
        self.saver = None
        self.sess = None
        self.writer = None
        self._setup_graph()
        # TODO: ability to store a sequence and ask for the prediction at a specific index
        # TODO: use pandas labeled indexes

    def _setup_graph(self):
        logger.info("Setting up graph for {}".format(self.name))
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Number of data examples is unknown, therefore is set None for now
            inputs = tf.placeholder(tf.float32, [None, self.input_time_steps, self.input_length], name="inputs")
            targets = tf.placeholder(tf.float32, [None, self.output_length], name="targets")
            learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            def _create_lstm_cell():
                lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

            if self.num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([_create_lstm_cell() for _ in range(self.num_layers)],
                                                   state_is_tuple=True)
            else:
                cell = _create_lstm_cell()

            # all_outputs contains the outputs of all the lstm cells
            all_outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

            # rearrange from (batch_size, num_steps, lstm_size) to (num_steps, batch_size, lstm_size)
            all_outputs = tf.transpose(all_outputs, [1, 0, 2])
            # Get only the output of the last step for each batch
            last = tf.gather(all_outputs,
                             int(all_outputs.get_shape()[0]) - 1,
                             name="last_lstm_output")

            # Define weights and biases between the hidden and output layers
            # TODO: check why this step
            weights = tf.Variable(tf.truncated_normal([self.lstm_size, self.output_length]), name="w")
            bias = tf.Variable(tf.constant(0.1, shape=[self.output_length]), name="b")
            prediction = tf.matmul(last, weights) + bias

            # Define loss function and optimizer
            loss = tf.reduce_mean(tf.square(prediction - targets), name="loss_mse")
            minimize = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, name="rmsprop_optim")

            # Create summaries
            # tf.summary.histogram(name="weights", values=weights)
            # tf.summary.histogram(name="biases", values=bias)
            tf.summary.histogram(name="loss_mse", values=loss)
            tf.summary.histogram(name="learning_rate", values=learning_rate)
            summary_op = tf.summary.merge_all()

            # Hold reference to important nodes
            self.selected_nodes["inputs"] = inputs
            self.selected_nodes["targets"] = targets
            self.selected_nodes["loss"] = loss
            self.selected_nodes["minimize"] = minimize
            self.selected_nodes["prediction"] = prediction
            self.selected_nodes["learning_rate"] = learning_rate
            self.selected_nodes["keep_prob"] = keep_prob
            self.selected_nodes["summary_op"] = summary_op

            # Create saver to hold training checkpoints
            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)
            logger.info("Graph done")

        # Unselect current graph
        tf.reset_default_graph()

    def train(self,
              train_X,
              train_Y,
              test_X,
              test_Y,
              keep_prob=1.0,  # Units to keep in dropout operation
              init_learning_rate=0.001,  # Learning rate to start with
              learning_rate_decay=1.0,  # Decay ratio per epoch
              init_decay_epoch=0,  # Epoch number at which to start decay
              max_epochs=100,  # Maximum number of epoch for training
              auto_terminate=False,  # If True training will terminate when no improvement is shown
              batch_size=1):
        train_X = self.format_inputs(train_X)
        train_Y = self.format_outputs(train_Y)
        test_X = self.format_inputs(test_X)
        test_Y = self.format_outputs(test_Y)
        self.save_dir_current_model = os.path.join(self.save_dir_models,
                                                   self.name + datetime.now().strftime("_%Y-%m-%d_%H:%M:%S"))
        with self.sess as sess:
            self.writer = tf.summary.FileWriter(os.path.join(self.save_dir_logs))
            self.writer.add_graph(self.graph)

            tf.global_variables_initializer().run()

            logger.info("Training started")

            global_step = 0
            for epoch in range(max_epochs):
                learning_rate = init_learning_rate * (
                    1.0 if epoch < init_decay_epoch else learning_rate_decay ** (epoch - init_decay_epoch)
                )
                train_loss = 0
                for batch in range(train_X.shape[0] // batch_size):
                    batch_X = train_X[batch * batch_size:(batch + 1) * batch_size]
                    batch_Y = train_Y[batch * batch_size:(batch + 1) * batch_size]
                    train_loss, _, summary = sess.run([self.selected_nodes["loss"],
                                              self.selected_nodes["minimize"],
                                              self.selected_nodes["summary_op"]],
                                             {self.selected_nodes["inputs"]: batch_X,
                                              self.selected_nodes["targets"]: batch_Y,
                                              self.selected_nodes["learning_rate"]: learning_rate,
                                              self.selected_nodes["keep_prob"]: keep_prob})
                    self.writer.add_summary(summary,
                                            global_step=global_step)

                test_loss = sess.run([self.selected_nodes["loss"]],
                                     {self.selected_nodes["inputs"]: test_X,
                                      self.selected_nodes["targets"]: test_Y,
                                      self.selected_nodes["learning_rate"]: learning_rate,
                                      self.selected_nodes["keep_prob"]: keep_prob
                                      })
                print("Step:{} [Epoch:{}] [Learning rate: {}] train_loss:{} test_loss:{}".format(
                    global_step, epoch, learning_rate, train_loss, test_loss
                ))
        logger.info("Training ended")
        self.save_model(global_step)

    def format_inputs(self, data):
        input_count = data.shape[0] - self.input_time_steps + 1
        if len(data.shape) == 2:
            input_vec_length = data.shape[1]
        else:
            input_vec_length = 1
        data_formatted = np.zeros([input_count, self.input_time_steps, input_vec_length])
        for i in range(input_count):
            for j in range(self.input_time_steps):
                data_formatted[i][j] = data[i + j]
        return data_formatted

    def format_outputs(self, data):
        if len(data.shape) > 1:
            inner_dims = data.shape[1:]
        else:
            inner_dims = [1]
        data = data[self.input_time_steps-1:]
        data_formatted = data.reshape((data.shape[0], *inner_dims))
        return data_formatted

    def predict(self, input_x):
        return self.sess.run([self.selected_nodes["prediction"]],
                             {
                                 self.selected_nodes["inputs"]: input_x,
                                 self.selected_nodes["learning_rate"]: 0.0,
                                 self.selected_nodes["keep_prob"]: 1.0
                             })

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

    aapl_dataset = read_csv(os.path.dirname(os.path.realpath(__file__)) + "/../../Database/AAPL/record.csv",
                            header=0, index_col=0)

    import math as m

    def softmax(vector):
        """
        Softmax function that favours the highest score
        :param vector: input vector of scores
        :return: output of the softmax
        """
        if sum(vector) == 0.0:
            return np.zeros(len(vector))
        vector = np.array(vector) / (0.3 * sum(vector))
        e_scaled = []
        for value in vector:
            e_scaled.append(m.exp(value))
        sum_e = sum(e_scaled)

        return np.array(e_scaled) / sum_e

    selected_cols = aapl_dataset[["volume",
                                  "relative_change",
                                  "positive_sentiment",
                                  "negative_sentiment",
                                  "uncertainty_sentiment",
                                  "news_volume"]].values
    for row in range(selected_cols.shape[0]):
        selected_cols[row][-4: -1] = softmax(selected_cols[row][-4: -1])
        if selected_cols[row][-1] == 0:
            continue
        elif selected_cols[row][-1] < 30:
            selected_cols[row][-1] = 0.33
        elif selected_cols[row][-1] < 100:
            selected_cols[row][-1] = 0.66
        else:
            selected_cols[row][-1] = 1.0

    # Relative volume diff from the 50-day volume moving average
    selected_cols[:, 0] = selected_cols[:, 0] / aapl_dataset["volume_m_ave_50"] - np.ones(selected_cols.shape[0])

    x = selected_cols
    y = x[:, 1]
    train_x = x[:int(0.8*x.shape[0])]
    train_y = y[:int(0.8*x.shape[0])]
    test_x = x[m.ceil(0.8*x.shape[0]):]
    test_y = y[m.ceil(0.8*x.shape[0]):]

    price_change_predictor = RNNPredictor(name="Price predictor",
                                          input_time_steps=7,
                                          input_length=6,
                                          output_length=1,
                                          lstm_size=28,
                                          num_layers=1)
    price_change_predictor.train(train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y, max_epochs=1)

    print(price_change_predictor.predict(test_x))



#
# num_epochs = 100
# total_series_length = 50000
# truncated_backprop_length = 15
# state_size = 4
# num_classes = 2
# echo_step = 3
# batch_size = 5
# num_batches = total_series_length//batch_size//truncated_backprop_length
