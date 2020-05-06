import tensorflow as tf
from nets import tcl

def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Model(tf.keras.Model):
    def __init__(self, num_class, width_mul = 1.0, name = 'Mobilev2', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.VarianceScaling(2, mode = 'fan_out'),
                                                  use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.DepthwiseConv2d, 'pre_defined', kwargs(kernel_initializer = tf.keras.initializers.VarianceScaling(2, mode = 'fan_out'),
                                                           use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(kernel_initializer = tf.random_normal_initializer(stddev = 1e-2), trainable = trainable))
        
        self.width_mult = 1
        self.Layers = {}
        from math import ceil
        self.setting = [[1,  ceil(16*width_mul), 1, 1],
                        [6,  ceil(24*width_mul), 2, 2],
                        [6,  ceil(32*width_mul), 3, 1],
                        [6,  ceil(64*width_mul), 4, 2],
                        [6,  ceil(96*width_mul), 3, 1],
                        [6, ceil(160*width_mul), 3, 2],
                        [6, ceil(320*width_mul), 1, 1]]

        inc = int(32 * self.width_mult)
        lastc = int(1280*self.width_mult) if self.width_mult > 1. else 1280

        self.Layers['conv'] = tcl.Conv2d([3,3], inc, strides = 1, name = 'conv', layertype = 'input')
        self.Layers['bn']   = tcl.BatchNorm(name = 'bn')

        for i, (t, c, n, s) in enumerate(self.setting):
            outc = c * self.width_mult

            for j in range(n):
                name = 'InvertedResidual_%d_%d/'%(i,j)
                with tf.name_scope(name): 
                    self.Layers[name + 'conv0'] = tcl.Conv2d([1,1], inc * t, name = name + 'conv0')
                    self.Layers[name + 'bn0']   = tcl.BatchNorm(name = name + 'bn0')

                    self.Layers[name + 'conv1'] = tcl.DepthwiseConv2d([3,3], strides = s if j == 0 else 1, name = name + 'conv1')
                    self.Layers[name + 'bn1']   = tcl.BatchNorm(name = name + 'bn1')

                    self.Layers[name + 'conv2'] = tcl.Conv2d([1,1], outc, name = name + 'conv2')
                    self.Layers[name + 'bn2']   = tcl.BatchNorm(name = name + 'bn2')

                    inc = outc

        self.Layers['conv_last'] = tcl.Conv2d([1,1], lastc, name = 'conv_last')
        self.Layers['bn_last']   = tcl.BatchNorm(name = 'bn_last')

        self.Layers['fc'] = tcl.FC(num_class, name = 'fc')
    
    def call(self, x, training=None):
        x = self.Layers['conv'](x)
        x = self.Layers['bn'](x, training = training)
        x = tf.nn.relu6(x)
        inc = x.shape[-1]
        for i, (t, c, n, s) in enumerate(self.setting):
            for j in range(n):
                name = 'InvertedResidual_%d_%d/'%(i,j)
                x_ = self.Layers[name + 'conv0'](x)
                x_ = self.Layers[name + 'bn0'](x_, training = training)
                x_ = tf.nn.relu6(x_)

                x_ = self.Layers[name + 'conv1'](x_)
                x_ = self.Layers[name + 'bn1'](x_, training = training)
                x_ = tf.nn.relu6(x_)

                x_ = self.Layers[name + 'conv2'](x_)
                x_ = self.Layers[name + 'bn2'](x_, training = training)

                if x_.shape == x.shape:
                    x = x_ + x
                else:
                    x = x_

        x = self.Layers['conv_last'](x)
        x = self.Layers['bn_last'](x, training = training )
        x = tf.nn.relu6(x)

        x = tf.reduce_mean(x,[1,2])
        x = self.Layers['fc'](x)
        return x
    
    def corr_update(self, conv, x):
        """
        Compute feature map's correlation for correlation based constraint and initial affinity matrix.

        Arguments:
            conv: Convolutional layer that target filter set is involved in.
            x: target filter set
        Returns:
            None
        """
        if hasattr(conv, 'reg'):
            B, H, W, D = x.shape
            if conv.reg:
                x = tf.reshape(x, [B,H*W,D])
                x = tf.linalg.l2_normalize(x-tf.reduce_mean(x,[0,2],keepdims=True),2)
                x = tf.reduce_mean(tf.matmul(x, x, transpose_a = True),0)
                conv.corr_loss = x

            if conv.stack:
                x = tf.reshape(x, [B,H*W,D])
                x = tf.linalg.l2_normalize(x-tf.reduce_mean(x,[0,2],keepdims=True),2)
                x = tf.reduce_sum(tf.matmul(x, x, transpose_a = True),0)
                if hasattr(conv, 'out_corr'):
                    conv.corr.assign_add(x)
