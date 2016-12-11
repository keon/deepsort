"""Seq2seq model example for sorting numbers."""
import sys
from model import EncDecModel
import tensorflow as tf
import numpy as np
import data_utils as du

# def main():
class args(object):
    cell_size = 100
    num_layers = 3
    batch_size = 32
    input_size = 100
    learning_rate = 0.1
    encoder_end_padding = False
    decoder_go_padding = True #This has to be always true.
    seed = 200
    output_projection = None
    epochs = 100
    lr_decay = 0.5
    decay_epoch = 20
    keep_prob = 0.5
    intializations = 0.1
    batch_average_loss = True
    #Dataset-parameters
    #Assuming that the lower limit is always '1'.
    #<go> & <end> padding have index '0'.
    upper_limit = 10
    source_vocab_size = upper_limit + 2#Change to upper_limit+1 if end_padding = True.
    target_vocab_size = upper_limit + 2
    seq_lenght = 5
    training_example_count = 5
    test_example_count = 2
    encoder_time_steps = seq_lenght
    decoder_time_steps = seq_lenght
    number_repetition = True
    one_hot_repr = True
    attention_mechanism = False
    training = True

def run_epoch(sess, model, args, encoder_inputs, decoder_inputs):
    total_batches = (encoder_inputs.shape[0])/args.batch_size
    for batch_number in range(total_batches):
        batch_x = encoder_inputs[:,batch_number*args.batch_size:(batch_number+1)*args.batch_size]
        batch_y = decoder_inputs[:,batch_number*args.batch_size:(batch_number+1)*args.batch_size]
        input_feed = model.feed(batch_x, batch_y)

        sess.run([model.optimizer], input_feed)

    #last batch if len(train_x)/args.batch_size leaves reminder.
    if len(encoder_inputs[0])%args.batch_size != 0:
        last_batch_count = len(encoder_inputs)%args.batch_size
        batch_x = encoder_inputs[:,-last_batch_count:]
        batch_y = decoder_inputs[:,-last_batch_count:]
        input_feed = model.feed(batch_x, batch_y)
        sess.run([model.optimizer], input_feed)


def train(args):
    train, test = du.get_dataset(args)

    enc_train, dec_train = du.pad(train[0], train[1], args)
    enc_test, dec_test = du.pad(test[0], test[1], args)

    if args.decoder_go_padding:
        args.decoder_time_steps += 1
    if args.encoder_end_padding:
        args.encoder_time_steps += 1

    enc_train_oh = du.one_hot(enc_train, args)
    dec_train_oh = du.one_hot(dec_train, args)
    enc_test_oh = du.one_hot(enc_test, args)
    dec_test_oh = du.one_hot(dec_test, args)

    initializer = tf.random_uniform_initializer(
            -args.intializations, args.intializations, seed = args.seed)

    with tf.Session() as sess:
        model = EncDecModel(args)
        tf.initialize_all_variables().run()

        # Input feed: encoder inputs, decoder inputs, as provided.
        train_feed = model.feed(enc_train_oh, dec_train_oh)
        test_feed = model.feed(enc_test_oh, dec_test_oh)
        encoder_inputs, decoder_inputs = enc_train_oh, dec_train_oh

        for epoch in xrange(1, args.epochs):
            run_epoch(sess, model, args, encoder_inputs, decoder_inputs)
            loss = sess.run([model.loss], train_feed)[0]
            test_loss = sess.run([model.loss], test_feed)[0]
            print("[%s] Loss : %s" % (epoch, loss), "test loss : %s"% test_loss)
            if epoch % args.decay_epoch==0:
                lr_value = sess.run([model.learning_rate])[0]*args.lr_decay
                print("New learning rate %s" % lr_value)
                model.set_lr(sess, lr_value)
                args.decay_epoch = args.decay_epoch * 2

            model.training = False
            model.keep_prob = 1.0
            enc_sample = enc_test_oh[:,0,:].reshape([-1, 1, args.upper_limit+1])
            dec_sample = dec_test_oh[:,0,:].reshape([-1, 1, args.upper_limit+1])
            sample_feed = model.feed(enc_sample, dec_sample)
            print(enc_test[:,0], dec_test[:,0],
                sess.run([model.predictions], sample_feed)[0][1].reshape([-1]))
            model.training = True
            model.keep_prob = args.keep_prob

if __name__ == "__main__":
    # main()
    train(args)
