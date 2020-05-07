import tensorflow as tf
from nets import tcl
from math import ceil

class Model(tf.keras.Model):
    def __init__(self, num_class, width_mul = 1.0, name = 'Mobilev2', trainable = True, expansion = 1., **kwargs):
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
        self.expansion = expansion
        self.Layers = {}
        
        self.setting = [[1,  ceil(16*self.expansion), 1, 1],
                        [6,  ceil(24*self.expansion), 2, 2],
                        [6,  ceil(32*self.expansion), 3, 1],
                        [6,  ceil(64*self.expansion), 4, 2],
                        [6,  ceil(96*self.expansion), 3, 1],
                        [6, ceil(160*self.expansion), 3, 2],
                        [6, ceil(320*self.expansion), 1, 1]]

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

        ## Source codes are fixed. If you want to confirm network is training with various width, uncomments below line.
        #tf.print(getattr(self.Layers['conv'], 'out_depth', 1.))

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
    
