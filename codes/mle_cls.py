import os
import time
import warnings
import itertools
import numpy as np
import ADmetrics as adm
import tensorflow as tf
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

tf.keras.backend.set_floatx('float64')


class CrossEntropy(tf.keras.Model):

    def __init__(self, n_units, n_features, n_classes, activation, name='bi_cls_adv', *args, **kwargs):
        super(CrossEntropy, self).__init__(name=name, *args, **kwargs)
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_features = n_features
        self.activation = activation
        # Dropout, non-linear activations etc, each, takes into account uncertainty.
        # While in our experiments we want to have Cross Entropy (MLE)
        # without any additional part which has been devoted to considering uncertainty
        # to show how our proposed method, i.e. pz_x, takes into account uncertainty
        self.d1 = tf.keras.layers.Dense(self.n_units, activation=self.activation, input_shape=(self.n_features,), )
        self.d2 = tf.keras.layers.Dropout(0.25)
        self.d3 = tf.keras.layers.Dense(n_classes)

    def call(self, x, training=True):
        x = self.d1(x)
        if training:
            # for future usage
            # x = self.d2(x, training=training)
            x = x
        return self.d3(x)


def apply_mle(n_units, n_features, n_classes, n_epochs, batch_size, learning_rate, activation=None,
              x_train=None, y_train=None, x_val=None, y_val=None, x_test=None, y_test=None, verbose=False):

    if x_train is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    else:
        print("No training set is provided!")
        return None

    if x_val is not None:
        valid_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    if x_test is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Create an instance of the model
    model = CrossEntropy(n_units, n_features, n_classes, activation=activation)  # training=True

    if n_classes == 2:
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(x_batch, training=True)
            loss = loss_object(y_batch, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(y_batch, predictions)

    @tf.function
    def test_step(x_batch, y_batch):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = tf.nn.softmax(model(x_batch, training=False))
        t_loss = loss_object(y_batch, predictions)

        test_loss(t_loss)
        test_accuracy(y_batch, predictions)
        return predictions

    for epoch in range(n_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for x_batch, y_batch in train_dataset:
            train_step(x_batch, y_batch)

        if x_val is not None:
            for x_batch_v, y_batch_v in valid_dataset:
                _ = test_step(x_batch_v, y_batch_v)

        if verbose:
            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result()))

    return model


if __name__ == '__main__':

    n_units = 10
    n_epochs = 200
    n_classes = 2
    n_features = 20
    batch_size = 100
    n_samples = 1000
    learning_rate = 1e-2

    verbose = True
    advance = True
    basic = True
    inside_call = True

    # Generate synthetic data / load data sets
    x_in, y_in = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_classes, n_redundant=0,
                                     n_repeated=0, n_classes=n_classes, n_clusters_per_class=1,
                                     weights=None, flip_y=0.01, class_sep=1.0, hypercube=True,
                                     shift=0.0, scale=1.0, shuffle=False, random_state=42)
    print("y_in:", set(y_in))

    # Normalizing the data points
    x_in = np.divide(x_in, np.ptp(x_in, axis=0))
    x_in = x_in.astype('float64')
    y_in = y_in.astype('float64').reshape(-1, 1)

    one_hot_encoder = OneHotEncoder(sparse=False)
    y_in = one_hot_encoder.fit_transform(y_in)
    y_in = y_in.astype('float64')

    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=0.4, random_state=42, shuffle=True)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42, shuffle=True)

    print("shapes:", x_train.shape, y_train.shape, x_test.shape, y_test.shape, x_val.shape, y_val.shape)

    x_min = np.min(x_train, axis=0)
    x_max = np.max(x_train, axis=0)
    x_range = x_max - x_min

    model = apply_mlp(n_units=n_units, n_features=n_features, n_classes=n_classes, n_epochs=n_epochs,
                      batch_size=batch_size, learning_rate=learning_rate, x_train=x_train, y_train=y_train,
                      x_val=x_val, y_val=y_val, x_test=None, y_test=None, verbose=False)

    py_x_logits = model(x_test, training=False)
    py_x_probs = tf.nn.softmax(py_x_logits)
    py_x_cls_probs = tf.nn.softmax(py_x_logits)
    labels_true = one_hot_encoder.inverse_transform(y_test)
    labels_pred = tf.argmax(tf.nn.softmax(py_x_cls_probs), axis=1)

    if inside_call:
        adm.plot_roc_auv_curve_of_an_algorithm(alg_ms=labels_pred, gt_ms=labels_true,
                                               alg_probs=py_x_probs, gt_ms_onehot=y_test,
                                               data_name='make_cls', alg_name='mlp-dropout',
                                               name_of_auc_roc_fig='mlp-dropout', sample_weight=None, case=0)

        prf = metrics.precision_recall_fscore_support(y_true=labels_true, y_pred=labels_pred, average='weighted')
        print("PRF:", prf)

