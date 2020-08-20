__author__ = "NSanjay"

import functools
import json
import logging
from pathlib import *
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
# from tf.lookup import TextFileInitializer

from tf_metrics import precision, recall, f1
from six.moves import reduce

logger = logging.getLogger(__name__)
import argparse

tf.compat.v1.logging.set_verbosity('INFO')

def masked_conv1d_and_max(t, weights, filters, kernel_size):
    """Applies 1d convolution and a masked max-pooling

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool
        A Tensor of shape [d1, d2, dn-1]
    filters : int
        number of filters
    kernel_size : int
        kernel size for the temporal convolution

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, dn-1, filters]

    """
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.compat.v1.to_float(weights)
    # filters =

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    print(f"shape of t:::: {t.shape}")
    print(f"filters::: {filters}")
    print(f"kernel_size:: {kernel_size}")

    # Apply convolution
    t_conv = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(t)
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.compat.v1.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max


def parse_fn(line_words, line_tags):
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]

    chars = [[c.encode() for c in word] for word in line_words.strip().split()]
    lengths = [len(word) for word in chars]
    max_len = max(lengths)

    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]

    return ((words, len(words)), (chars, lengths)), tags

def generator_fn(words, tags):
    with open(Path(words)) as f_words, open(Path(tags)) as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)

def input_fn(words, tags, params=None, shuffle=False):
    params = {} if params is None else params
    shapes = ((([None], ()),
               ([None, None], [None])),
                [None])
    types = (((tf.string, tf.int32),
             (tf.string, tf.int32)),
             tf.string)
    defaults = ((("<pad>", 0),
                 ("<pad>", 0)),
                "O")

    dataset = tf.data.Dataset.from_generator(
              functools.partial(generator_fn, words, tags),
              output_shapes=shapes, output_types=types
    )

    if shuffle:
        dataset = dataset.shuffle(params["buffer"]).repeat(params["epochs"])

    dataset = (dataset.padded_batch(params.get("batch_size", 16), shapes, defaults).prefetch(1))

    return dataset

def fwords(data_dir, name):
    return str(Path(data_dir, '{}.words.txt'.format(name)))


def ftags(data_dir, name):
    return str(Path(data_dir, '{}.tags.txt'.format(name)))

def model_fn(features, labels, mode, params):
    dropout = params['dropout']
    (words, nwords), (chars, nchars) = features

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    with open(params["tags"]) as tags_file:
        indices = [idx for idx, tag in enumerate(tags_file) if tag != "O"]
        num_tags = len(indices) + 1

    with open(params["chars"]) as chars_file:
        num_chars = sum(1 for _ in chars_file) + params["num_oov_buckets"]

    text_initializer = tf.lookup.TextFileInitializer(
        params["chars"],
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        delimiter=" ")

    vocab_chars = tf.lookup.StaticVocabularyTable(text_initializer, params["num_oov_buckets"])
    char_ids = vocab_chars.lookup(chars)

    # char_variable = tf.Variable(name="chars_embedding", shape=[num_chars + 1, params['dim_chars']], dtype=tf.float32)
    uniform_initializer = tf.keras.initializers.GlorotUniform()
    char_variable = tf.Variable \
                    (uniform_initializer(shape=[num_chars + 1, params['dim_chars']], dtype=tf.float32))
    char_embeddings = tf.nn.embedding_lookup(char_variable, char_ids)
    dropout_layer = tf.keras.layers.Dropout(dropout)
    char_embeddings = dropout_layer(char_embeddings, training=training)

    weights = tf.sequence_mask(nchars)
    char_embeddings = masked_conv1d_and_max(
        char_embeddings, weights, params['filters'], params['kernel_size'])

    text_word_initializer = tf.lookup.TextFileInitializer(
        params["words"],
        key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        delimiter=" "
    )

    vocab_words = tf.lookup.StaticVocabularyTable(text_word_initializer, params["num_oov_buckets"])
    word_ids = vocab_words.lookup(words)
    glove_embeddings = np.load(params["glove"])["embeddings"]
    glove_embeddings = np.vstack([glove_embeddings, [[0.] * params["dim"]]])  # add extra entry for oov
    word_variable = tf.Variable(glove_embeddings, dtype=tf.float32, trainable=False)
    word_embeddings = tf.nn.embedding_lookup(word_variable, word_ids)

    # Concat Word and Char Embeddings
    embeddings = tf.concat([word_embeddings, char_embeddings], axis=-1)

    embeddings = dropout_layer(embeddings, training=training)

    print(f"embeddings_shape::{embeddings.shape}")
    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    fw_lstm_layer = tf.keras.layers.LSTM(params["lstm_size"], return_sequences=True)
    bw_lstm_layer = tf.keras.layers.LSTM(params["lstm_size"], return_sequences=True, go_backwards=True)
    outputs = tf.keras.layers.Bidirectional(fw_lstm_layer, backward_layer=bw_lstm_layer,
                                            merge_mode="concat")(t)
    outputs = tf.transpose(outputs, perm=[1, 0, 2])
    outputs = dropout_layer(outputs, training=training)

    # CRF
    dense_layer = tf.keras.layers.Dense(num_tags)
    logits = dense_layer(outputs)

    crf_params = tf.Variable(uniform_initializer(shape=[num_tags, num_tags], dtype=tf.float32))

    pred_ids, _ = tfa.text.crf_decode(logits, crf_params, nwords)

    print(f"mode is:: {mode} and training:: {training}")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_tags_reverse = tf.lookup.TextFileInitializer(
            params["tags"],
            value_dtype=tf.string, value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            key_dtype=tf.int64, key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter=" "
        )
        predicted_tags_reverse = tf.lookup.StaticVocabularyTable(predicted_tags_reverse, 0)
        pred_strings = predicted_tags_reverse.lookup(tf.to_int64(pred_ids))

        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # loss

        predicted_tags = tf.lookup.TextFileInitializer(
            params["tags"],
            key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            delimiter=" "
        )
        predicted_tags = tf.lookup.StaticHashTable(predicted_tags, -1)

        gold_ids = predicted_tags.lookup(labels)
        log_likelihood, _ = tfa.text.crf_log_likelihood(logits, gold_ids, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)

        metrics = {
            "acc": tf.compat.v1.metrics.accuracy(gold_ids, pred_ids, weights),
            "precision": precision(gold_ids, pred_ids, num_tags, indices, weights),
            "recall": recall(gold_ids, pred_ids, num_tags, indices, weights),
            "f1": f1(gold_ids, pred_ids, num_tags, indices, weights)
        }

        for metric_name, op in metrics.items():
            print(f"here")
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        train_op = tf.compat.v1.train.AdamOptimizer().minimize(
            loss, global_step=tf.compat.v1.train.get_or_create_global_step())

        train_hook_list = []
        train_log = {"accuracy" : tf.compat.v1.metrics.accuracy(gold_ids, pred_ids, weights),
                     "loss" : loss,
                     "global_step" : tf.compat.v1.train.get_global_step()
                     }

        train_hook_list.append(tf.compat.v1.train.LoggingTensorHook(
            tensors=train_log, every_n_iter=100))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



def main():
    # Params

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../../data")

    args = parser.parse_args()
    params = {
        'dim_chars': 100,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'filters': 50,
        'kernel_size': 3,
        'lstm_size': 100,
        'words': str(Path(args.data_dir, 'vocab.words.txt')),
        'chars': str(Path(args.data_dir, 'vocab.chars.txt')),
        'tags': str(Path(args.data_dir, 'vocab.tags.txt')),
        'glove': str(Path(args.data_dir, 'glove.npz'))
    }

    train_input_fn = functools.partial(input_fn, fwords(args.data_dir, "train_split"), ftags(args.data_dir, "train_split"),
                                       shuffle=True)
    eval_input_fn = functools.partial(input_fn, fwords(args.data_dir, "test_split"), ftags(args.data_dir, "test_split"))

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, "results/model", cfg, params)

    hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator, "f1", 500, min_steps=8000, run_every_secs=120)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=120)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print("done")

if __name__ == "__main__":
    main()
