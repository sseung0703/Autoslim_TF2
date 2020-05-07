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
    def validation(images, labels):
        predictions = model(images, training = False)
        loss = loss_object(labels, predictions)
        
        test_loss.update_state(loss)
        test_accuracy.update_state(labels, predictions)
    return training, train_loss, train_accuracy, validation, test_loss, test_accuracy

def Slimmable_optimizer(model, weight_decay, LR, min_rate):
    with tf.name_scope('Optimizer_w_Distillation'):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.SGD(LR, .9, nesterov=True)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



    def forward(images, labels, width = None):
        if width is not None:
            predictions = model(images, training = True)
            total_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(labels)*(tf.nn.log_softmax(labels)
                                                                            -tf.nn.log_softmax(predictions)), 1))
        else:
            predictions = model(images, training = True)
            total_loss = loss_object(labels, predictions)
        return total_loss, predictions


    @tf.function
    def training(images, labels):
        with tf.GradientTape() as tape:
            total_loss, predictions = forward(images, labels)
            inplace_distillation = tf.stop_gradient(predictions)

            width_list = tf.random.uniform([2], min_rate, 1.)
            width_list = [tf.reshape(w,[]) for w in tf.split(width_list,2)] + [tf.constant(min_rate)]

            for width in width_list:
                for k in model.Layers.keys():
                    if getattr(model.Layers[k], 'type', False) != 'input':
                        model.Layers[k].in_depth = width
                    model.Layers[k].out_depth = width
                tl, _ = forward(images, inplace_distillation, width)
                for k in model.Layers.keys():
                    if getattr(model.Layers[k], 'type', False) != 'input':
                         delattr(model.Layers[k], 'in_depth')
                    delattr(model.Layers[k], 'out_depth')

                total_loss += tl
        gradients = tape.gradient(total_loss, model.trainable_variables)
        if weight_decay > 0.:
            gradients = [g+v*weight_decay for g,v in zip(gradients, model.trainable_variables)]
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
         
        train_loss.update_state(total_loss)
        train_accuracy.update_state(labels, predictions)

        return optimizer._decayed_lr(var_dtype = tf.float32)
        
    def validation(images, labels, width_list = None, bn_statistics_update = False):
        if width_list is not None:
            slim_util.set_width(model, width_list)
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
