"""Seq2Seq model."""
import tensorflow as tf
import sys

class EncDecModel(object):

    def __init__(self, args):
        self.source_vocab_size = args.source_vocab_size
        self.target_vocab_size = args.target_vocab_size
        self.cell_size = args.cell_size
        self.num_layers = args.num_layers
        self.input_size = args.input_size
        self.learning_rate = tf.Variable(float(args.learning_rate), trainable=False, dtype=tf.float32)
        self.output_projection = args.output_projection
        self.saver = tf.train.Saver()
        self.encoder_time_steps = args.encoder_time_steps
        self.decoder_time_steps = args.decoder_time_steps
        self.keep_prob = args.keep_prob
        self.training = args.training
        self.upper_limit = args.upper_limit
        self.batch_average_loss = args.batch_average_loss

        # Build Model!
        self.build_model()

    def build_model(self):
        rnn = tf.nn.rnn_cell

        cell = rnn.GRUCell(self.cell_size)
        cell = rnn.MultiRNNCell([cell] * self.num_layers)
        cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        #Feed for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.targets = []

        cell = rnn.OutputProjectionWrapper(cell, self.upper_limit+1)
        for i in range(self.encoder_time_steps):
            self.encoder_inputs.append(tf.placeholder(tf.float32,
                    shape=[None, self.upper_limit+1],
                    name="encoder{0}".format(i)))

        for i in xrange(self.decoder_time_steps):
            self.decoder_inputs.append(tf.placeholder(tf.float32,
                    shape=[None, self.upper_limit+1],
                    name="decoder{0}".format(i)))

        for i in xrange(len(self.decoder_inputs)-1):
            self.targets.append(self.decoder_inputs[i+1])
        self.targets.append(self.decoder_inputs[0]) # end token


        logits, state = tf.nn.seq2seq.basic_rnn_seq2seq(
                                    encoder_inputs = self.encoder_inputs,
                                    decoder_inputs = self.decoder_inputs,
                                    cell = cell,
                                    dtype=tf.float32)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits = logits,
                targets = self.targets)
        self.loss = tf.reduce_sum(self.loss, [0,1,2])

        if self.batch_average_loss:
            batch_size = tf.shape(logits[0])[0]
            self.loss /= tf.cast(batch_size, self.loss.dtype)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.predictions  = tf.nn.top_k(logits, 1)

    def set_lr(self, sess, lr_value):
        sess.run(tf.assign(self.learning_rate, lr_value))


    def feed(self, encoder_inputs, decoder_inputs):
        input_feed = {}
        for l in xrange(len(encoder_inputs)):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(len(decoder_inputs)):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        # print(input_feed) # TODO take a look
        return input_feed


