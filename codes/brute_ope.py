import os
import time
import pickle
import argparse
import numpy as np
import ADmetrics as adm
import tensorflow as tf


# Brute-Forth OPE
class Bope(tf.keras.Model):
    def __init__(self, n_units, n_features, activation, name='bope', **kwargs):
        super(Bope, self).__init__(name=name, **kwargs)
        self.n_units = n_units
        self.n_features = n_features
        self.activation = activation
        self.dense_1 = tf.keras.layers.Dense(self.n_units, activation=self.activation,
                                             input_shape=(self.n_features,), name='d1')
        self.pred = tf.keras.layers.Dense(1, activation='linear', name='pred')

    def call(self, inputs):
        x = self.dense_1(inputs)
        # x = self.dense_2(x)
        x = self.pred(x)
        return x


def loss_bf(model, supp_min, supp_max, x_batch, x_pos, x_neg,):

    p_pos = model(x_pos)
    p_neg = model(x_neg)

    n_pos, v = x_pos.shape[0], x_pos.shape[1]
    n_neg = x_neg.shape[0]

    x_pseudo = tf.random.uniform(shape=[n_pos, v],
                                 minval=supp_min-3,
                                 maxval=supp_max+3,
                                 dtype='float64',)
    p_pseudo = model(x_pseudo)

    loss_pos = (n_pos / (n_pos + n_neg)) * tf.reduce_sum(tf.nn.softplus(-p_pos))
    loss_neg = (n_neg / (n_neg+n_pos)) * tf.reduce_sum(tf.nn.softplus(p_neg))
    loss_pseudo = 0.001 * tf.reduce_sum(tf.nn.softplus(p_pseudo))

    preds = tf.nn.sigmoid(model(x_batch))

    return loss_pos+loss_neg+loss_pseudo, preds


def apply_bope(n_units, n_features, n_classes, n_epochs, batch_size, learning_rate, activation=None,
               x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None, verbose=False):

    supp_min = np.min(x_train, keepdims=True, axis=0)
    supp_max = np.max(x_train, keepdims=True, axis=0)

    if x_train is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).batch(batch_size)
    else:
        print("No training set is provided!")
        return None

    if x_val is not None:
        valid_dataset = tf.data.Dataset.from_tensor_slices(
            (x_val, y_val)).batch(batch_size)

    if x_test is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(batch_size)

    # Initialization
    # Create an instance of the model
    bope_model = Bope(n_units=n_units, n_features=n_features, activation=activation)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    # print("shapes in BOPE:", n_features, x_train.shape,)
    # print("training_dataset:", train_dataset)

    step = 0

    for epoch in range(1, n_epochs+1):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        step += 1
        for x_batch_tr, y_batch_tr in train_dataset:
            x_pos = tf.convert_to_tensor(x_batch_tr.numpy()[np.where(y_batch_tr.numpy() == 0), :][0])
            # print("x_pos:", "\n", x_pos)
            x_neg = tf.convert_to_tensor(x_batch_tr.numpy()[np.where(y_batch_tr.numpy() == 1), :][0])
            # print("x_neg:", "\n", x_neg)
            with tf.GradientTape() as tape:
                loss_value, preds = loss_bf(model=bope_model, supp_min=supp_min, supp_max=supp_max,
                                            x_batch=x_batch_tr, x_pos=x_pos, x_neg=x_neg)

        grads = tape.gradient(loss_value, bope_model.trainable_weights)
        opt.apply_gradients(zip(grads, bope_model.trainable_weights))
        train_loss(loss_value)
        train_accuracy(y_batch_tr, tf.nn.sigmoid(preds))

        if x_val is not None:
            for x_batch_vl, y_batch_val in valid_dataset:
                predictions = bope_model(x_batch_vl)
                t_loss = loss_object(y_batch_val, predictions)
                test_loss(t_loss)
                test_accuracy(y_batch_val,  tf.nn.softmax(predictions))

        if verbose:
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result()))

    return bope_model

