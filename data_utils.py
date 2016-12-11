import numpy as np
import itertools
import random
import sys

def one_hot(inputs, args):
    one_hots = []
    num_classes = args.upper_limit+1
    for vector in inputs:
        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        one_hots.append(np.asarray(result.astype(int)))
    return np.asarray(one_hots)

def pad(encoder_inputs, decoder_inputs, config):
    if config.decoder_go_padding:
        zeros = np.zeros(shape=(1, decoder_inputs.shape[1]), dtype = int)
        decoder_inputs = np.concatenate((zeros, decoder_inputs), axis=0)

    if config.encoder_end_padding:
        padding = np.full(shape=[1,encoder_inputs.shape[1]], fill_value=config.upper_limit+1, dtype=int)
        encoder_inputs = np.concatenate((encoder_inputs, padding), axis=0)

    return encoder_inputs, decoder_inputs

def get_dataset(args):
    np.random.seed(args.seed)
    int_range = range(1,args.upper_limit+1)

    if args.number_repetition:
        encoder_inputs_train = np.random.randint(low = 1,
                                           high = args.upper_limit,
                                           size = (args.seq_lenght, args.training_example_count))
    else:
        encoder_inputs_train = []
        for i in range(args.training_example_count):
            encoder_inputs_train.append(random.sample(int_range, args.seq_lenght))
        encoder_inputs_train = np.transpose(np.asarray(encoder_inputs_train))

    decoder_inputs_train = [sorted(inputs) for inputs in np.transpose(encoder_inputs_train)]
    decoder_inputs_train = np.transpose(decoder_inputs_train)


    if args.number_repetition:
        encoder_inputs_test = np.random.randint(low = 1,
                                           high = args.upper_limit,
                                           size = (args.seq_lenght, args.test_example_count))
    else:
        encoder_inputs_test = []
        for i in range(args.test_example_count):
            encoder_inputs_test.append(random.sample(int_range, args.seq_lenght))
        encoder_inputs_test = np.transpose(np.asarray(encoder_inputs_test))


    decoder_inputs_test = [sorted(inputs) for inputs in np.transpose(encoder_inputs_test)]
    decoder_inputs_test = np.transpose(decoder_inputs_test)

    fwrite("dataset/encoder_train_data.txt", encoder_inputs_train)
    fwrite("dataset/decoder_train_data.txt", decoder_inputs_train)
    fwrite("dataset/encoder_test_data.txt", encoder_inputs_test)
    fwrite("dataset/decoder_test_data.txt", decoder_inputs_test)

    return (encoder_inputs_train, decoder_inputs_train), (encoder_inputs_test, decoder_inputs_test)


def fwrite(filename, data):
    file = open(filename, 'w')
    for entry in data:
        file.write(str(entry) +'\n')
    file.close()

