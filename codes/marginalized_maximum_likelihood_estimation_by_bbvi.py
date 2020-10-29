import numpy as np
import ADmetrics as adm
import tensorflow as tf
from sklearn import metrics
import tensorflow_probability as tfp
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

tf.keras.backend.set_floatx('float64')


class MargMLVIEstimation(tf.keras.Model):

    """ Marginalized Maximum Likelihood Estimation using Black-Box Variational Inference (Bayes) """

    def __init__(self, n_units, n_features, n_classes, dense_layer_type, name='theta', **kwargs):
        super(MargMLVIEstimation, self).__init__(name=name, **kwargs)
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_features = n_features
        self.dense_layer_type = dense_layer_type

        c = np.log(np.expm1(1.))
        scale = 1e-5 + tf.nn.softplus(c)

        if self.dense_layer_type.lower() == "pw-re".lower() \
                or self.dense_layer_type.lower() == "pw-li".lower():
            if self.dense_layer_type.lower() == "pw-re".lower():
                self.dense_1 = tfpl.DenseReparameterization(self.n_features, activation=tf.nn.relu,
                                                            trainable=True, name='pw_dense_1',)
            else:
                self.dense_1 = tfpl.DenseReparameterization(self.n_features, activation=None,
                                                            trainable=True, name='pw_dense_1',)
            # In order to be able to compute the log likelihood
            # a distribution is needed, and it is instantiated as follow:
            self.pz_x = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=scale,),
                                                trainable=True, name='pz_x')
            self.py_x_z = tfpl.DenseReparameterization(self.n_classes,
                                                       activation=None, name='py_x_z')

        elif self.dense_layer_type.lower() == "fo-re".lower()\
                or self.dense_layer_type.lower() == "fo-li".lower():
            if self.dense_layer_type.lower() == "fo-re".lower():
                # With Relu activation
                self.dense_1 = tfpl.DenseFlipout(self.n_features, activation=tf.nn.relu,
                                                 name='fl_dense_1')
            else:
                # With Linear activation
                self.dense_1 = tfpl.DenseFlipout(self.n_features, activation=None,
                                                 name='fl_dense_1')

            # In order to be able to compute the log likelihood
            # a distribution is needed, and it is instantiated as follow:
            self.pz_x = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=scale, ),
                                                trainable=True, name='pz_x')
            self.py_x_z = tfpl.DenseFlipout(self.n_classes, activation=None,
                                            name='py_x_z')

    def call(self, inputs, training=False):
        x = self.dense_1(inputs, training=training)
        if training:
            pz_x = self.pz_x(x, training=training)
            py_x_z_logits = self.py_x_z(x, training=training)
            return pz_x, py_x_z_logits
        # py_x_z_logits
        return self.py_x_z(x, training=training)


def apply_mmle(n_units, n_features, n_classes, n_epochs, batch_size, learning_rate,
                dense_layer_type='path_wise', x_train=None, y_train=None, x_val=None,
                y_val=None, x_test=None, y_test=None, verbose=False):

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

    # metrics for monitoring the training and validation procedure
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    model_adv = MargMLVIEstimation(n_units=n_units, n_features=n_features,
                                   n_classes=n_classes,
                                   dense_layer_type=dense_layer_type)

    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def training_step(x_batch, y_batch, model):
        with tf.GradientTape() as tape:

            dist_pz_x, py_x_z_logits = model(x_batch, training=True)

            neg_log_lik_pz_x = -tf.reduce_sum(
                dist_pz_x.log_prob(x_batch))

            neg_log_lik_py_x_z = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_batch, logits=py_x_z_logits)

            neg_log_lik_py_x_z_cls = -tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_batch, logits=py_x_z_logits))
            kl_loss = sum(model.losses)  # /(x_train.shape[0])

            total_loss = tf.math.multiply(neg_log_lik_pz_x, neg_log_lik_py_x_z) + neg_log_lik_py_x_z_cls  # + kl_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        _ = train_loss.update_state(total_loss)
        predictions = tf.nn.softmax(py_x_z_logits)
        _ = train_accuracy.update_state(y_batch, predictions)

    @tf.function
    def testing_step(x_batch_t, y_batch_t, model):
        dist_pz_x_t, py_x_z_logits_t = model(x_batch_t, training=True)
        neg_log_lik_pz_x_t = -tf.reduce_sum(dist_pz_x_t.log_prob(x_batch_t))
        neg_log_lik_py_x_z_t = tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_t, logits=py_x_z_logits_t)
        neg_log_lik_py_x_z_cls_t = -tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_t, logits=py_x_z_logits_t))
        kl_loss = sum(model.losses)  # /(x_train.shape[0])

        total_loss_t = tf.math.multiply(neg_log_lik_pz_x_t, neg_log_lik_py_x_z_t) + neg_log_lik_py_x_z_cls_t

        _ = test_loss.update_state(total_loss_t)
        predictions_t = tf.nn.softmax(py_x_z_logits_t)
        _ = test_accuracy.update_state(y_batch_t, predictions_t)

    for epoch in range(n_epochs):

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            training_step(x_batch=x_batch, y_batch=y_batch, model=model_adv,)

        if x_val is not None:
            for step, (x_batch_v, y_batch_v) in enumerate(valid_dataset):
                testing_step(x_batch_t=x_batch_v, y_batch_t=y_batch_v, model=model_adv, )

        if x_test is not None:
            for step, (x_batch_t, y_batch_t) in enumerate(test_dataset):
                testing_step(x_batch_t=x_batch_t, y_batch_t=y_batch_t, model=model_adv, )

        # Log every 100 batches.
        if step % 100 == 0 and verbose:
            template = 'Epoch {}, Train Loss: {}, Train Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result()))
    return model_adv


if __name__ == '__main__':

    n_units = 10
    n_epochs = 500
    n_classes = 2
    n_features = 10
    batch_size = 100
    n_samples = 1000  # 100
    learning_rate = 1e-2

    inside_call = True
    dense_layer_type = 'path_wise'

    # Generate synthetic data / load data sets
    x_in, y_in = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_classes, n_redundant=0,
                                     n_repeated=0, n_classes=n_classes, n_clusters_per_class=1,
                                     weights=[0.2, 0.8], flip_y=0.4, class_sep=1.0, hypercube=True,
                                     shift=0.0, scale=1.0, shuffle=True, random_state=42)
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

    model_adv = apply_mmle(n_units=n_units, n_features=n_features, n_classes=n_classes,
                           n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                           dense_layer_type=dense_layer_type, x_train=x_train, y_train=y_train,
                           x_val=x_val, y_val=y_val, x_test=None, y_test=None, verbose=False)

    py_x_z_logits = model_adv(x_test, training=False)
    py_x_z_probs = tf.nn.softmax(py_x_z_logits)
    labels_pred = tf.argmax(py_x_z_probs, axis=1)
    labels_true = one_hot_encoder.inverse_transform(y_test)

    if inside_call:
        adm.plot_roc_auv_curve_of_an_algorithm(alg_ms=labels_pred, gt_ms=labels_true,
                                               alg_probs=py_x_z_probs, gt_ms_onehot=y_test,
                                               data_name='make_cls', alg_name='fpvi-pw',
                                               name_of_auc_roc_fig=dense_layer_type, sample_weight=None, case=0)
        prf = metrics.precision_recall_fscore_support(y_true=labels_true, y_pred=labels_pred, average='weighted')
        print("PRF:", prf)
