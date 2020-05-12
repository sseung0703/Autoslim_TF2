import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, num_layers, num_class, name = 'WResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(trainable = trainable))

        self.Layers = {}
        network_argments = {56 : {'blocks' : [9,9,9],'depth' : [16,32,64], 'strides' : [1,2,2]}}
        self.net_args = network_argments[num_layers]

        self.Layers['conv'] = tcl.Conv2d([3,3], self.net_args['depth'][0], name = 'conv', layertype = 'input')
        self.Layers['bn']   = tcl.BatchNorm(name = 'bn')
        in_depth = self.net_args['depth'][0]
        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = '/BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if strides > 1 or depth != in_depth:
                    self.Layers[name + 'conv2'] = tcl.Conv2d([1,1], depth, strides = strides, name = name +'conv2')
                    self.Layers[name + 'bn2']   = tcl.BatchNorm( name = name + 'bn2')

                self.Layers[name + 'conv0'] = tcl.Conv2d([3,3], depth, strides = strides, name = name + 'conv0')
                self.Layers[name + 'bn0']   = tcl.BatchNorm( name = name + 'bn0')
                self.Layers[name + 'conv1'] = tcl.Conv2d([3,3], depth, name = name + 'conv1')
                self.Layers[name + 'bn1']   = tcl.BatchNorm( name = name + 'bn1')
                in_depth = depth

        self.Layers['fc'] = tcl.FC(num_class, name = 'fc')

    def call(self, x, training=None):
        x = self.Layers['conv'](x)
        x = self.Layers['bn'](x)
        x = tf.nn.relu(x)
        in_depth = self.net_args['depth'][0]

        for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['blocks'], self.net_args['depth'], self.net_args['strides'])):
            for j in range(nb_resnet_layers):
                name = '/BasicBlock%d.%d/'%(i,j)
                if j != 0:
                    strides = 1

                if strides > 1 or depth != in_depth:
                    residual = self.Layers[name + 'conv2'](x)
                    residual = self.Layers[name + 'bn2'](residual)
                else:
                    residual = x

                x = self.Layers[name + 'conv0'](x)
                x = self.Layers[name + 'bn0'](x)
                x = tf.nn.relu(x)
                x = self.Layers[name + 'conv1'](x)
                x = self.Layers[name + 'bn1'](x)
                x = tf.nn.relu(x + residual)
                in_depth = depth

        x = tf.reduce_mean(x, [1,2])
        x = self.Layers['fc'](x)
        return x
