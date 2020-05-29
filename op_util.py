import tensorflow as tf
import numpy as np
import slim_util

def Optimizer(model, weight_decay, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(LR, .9, nesterov=True)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
    ## Experimental compiling is not available on Windows. If your OS is Windows, use "@tf.function" only
    @tf.function(experimental_compile=True)
    def training(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training = True)
            total_loss = loss_object(labels, predictions)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if weight_decay > 0.:
            gradients = [g+v*weight_decay for g,v in zip(gradients, model.trainable_variables)]

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, predictions)

        return optimizer._decayed_lr(var_dtype = tf.float32)
        
    @tf.function(experimental_compile=True)
    def testing(images, labels):
        predictions = model(images, training = False)
        loss = loss_object(labels, predictions)
        
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)
    return training, train_loss, train_accuracy, testing, test_loss, test_accuracy

def Slimmable_optimizer(args, model, weight_decay, LR):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = agg_SGD(LR, .9, nesterov=True)
        optimizer._create_slots(model.trainable_variables)

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        n = 4
        full_width = slim_util.get_initial_width(args.arch, model)
        min_width = [args.minimum_rate for w in range(len(full_width))]
        slim_util.set_width(args.arch, model, full_width)

    @tf.function(experimental_compile=True)
    def training(images, labels):
        slim_util.set_width(args.arch, model, full_width)
        with tf.GradientTape() as tape:
            predictions = model(images, training = True)
            total_loss = loss_object(labels, predictions)
        gradients = tape.gradient(total_loss, model.trainable_variables)

        if weight_decay > 0.:
            gradients = [g+v*weight_decay for g,v in zip(gradients, model.trainable_variables)]
        optimizer.aggregate_grad(gradients, model.trainable_variables)

        in_dist = tf.stop_gradient(predictions)
        id_soft = tf.nn.softmax(in_dist)
        id_log_soft = tf.nn.log_softmax(in_dist)

        width_list = [[tf.random.uniform([],args.minimum_rate, 1.) for _ in range(len(min_width))] for _ in range(n-2)]
        for i, width in enumerate(width_list + [min_width]):
            slim_util.set_width(args.arch, model, width)
            with tf.GradientTape() as tape:
                pred_sub = model(images, training = True)
                loss_sub = tf.reduce_mean(tf.reduce_sum(id_soft*(id_log_soft-tf.nn.log_softmax(pred_sub)), 1))

            gradidents = tape.gradient(loss_sub, model.trainable_variables)
            optimizer.aggregate_grad(gradients, model.trainable_variables)

        optimizer.apply_grad(model.trainable_variables)
        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, predictions)
        slim_util.set_width(args.arch, model, full_width)

        return optimizer._decayed_lr(var_dtype = tf.float32)

    @tf.function(experimental_compile=True)
    def validation(images, labels, bn_statistics_update = False):
        predictions = model(images, training = bn_statistics_update)
        loss = loss_object(labels, predictions)

        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)
    return training, train_loss, train_accuracy, validation, test_loss, test_accuracy


class PiecewiseConstantDecay(tf.optimizers.schedules.LearningRateSchedule):
    """ A LearningRateSchedule that uses a piecewise constant decay schedule for XLA compiling."""
    """ XLA compiling makes iteration faster but "tf.cond" cannot be compiled.                 """
    """ Therfore, I replace it to multiplication of learning rate and conditions.              """
    def __init__(
        self,
        boundaries,
        values,
        name=None):
        super(PiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError(
            "The length of boundaries should be 1 less than the length of values")

        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step):
        from tensorflow.python.framework import ops
        with tf.name_scope(self.name or "PiecewiseConstant"):
            x_recomp = ops.convert_to_tensor(step)

            lr  = self.values[0] * tf.cast( x_recomp < self.boundaries[0], tf.float32)
            for v, b0, b1 in zip(self.values[1:-1], self.boundaries[:-1], self.boundaries[1:]):
                lr += v * tf.cast( (b0 <= x_recomp) & (x_recomp < b1), tf.float32)
            lr += self.values[-1] * tf.cast(self.boundaries[-1] <= x_recomp, tf.float32)
            return lr

    def get_config(self):
        return {
                "boundaries": self.boundaries,
                "values": self.values,
                "name": self.name
                }

from tensorflow.python.framework import ops
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
class agg_SGD(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True
    def __init__(self,
                 learning_rate=0.01,
                 momentum=0.0,
                 nesterov=False,
                 name="SGD",
                 **kwargs):
        super(agg_SGD, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self.learning_rate = learning_rate

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)
        self.momentum = momentum
        self.nesterov = nesterov

    def _create_slots(self, var_list):
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
                self.add_slot(var, "agg_grad")

    def aggregate_grad(self, grad, var, apply_state = None):
        for g, v in zip(grad, var):
            agg = self.get_slot(v, "agg_grad")
            agg.assign_add(g)

    def apply_grad(self, var, apply_state = None):
        for v in var:
            momentum_var = self.get_slot(v, "momentum")
            agg = self.get_slot(v, "agg_grad")
            g = agg + (agg - momentum_var) * (1 - self.momentum)
            g = g + self.momentum * g
            v.assign_add(-self.learning_rate * g)
            agg.assign(agg*0.)

    def get_config(self):
        config = super(agg_SGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config
