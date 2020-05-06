import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, num_layer, num_class, name, trainable = False):
        super(Model, self).__init__(num_layer, num_class, name, trainable)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs( use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs( activation_fn = tf.nn.softmax, trainable = trainable))
        
        self.resnet_layers = {}
        network_argments = {56 : {'nb_resnet_layers' : [9,9,9],'depth' : [16,32,64], 'strides' : [1,2,2]}}
        self.net_args = network_argments[num_layer]
        self.net_name = 'ResNet'
        with tf.name_scope(self.net_name):
            self.resnet_layers[self.net_name + '/conv0'] = tcl.Conv2d([3,3], self.net_args['depth'][0])
            self.resnet_layers[self.net_name + '/bn0']   = tcl.BatchNorm(activation_fn = tf.nn.relu)
            
            for i, (nb_resnet_layers, depth, strides) in enumerate(zip(self.net_args['nb_resnet_layers'], 
                                                                       self.net_args['depth'],
                                                                       self.net_args['strides'])):
                for j in range(nb_resnet_layers):           
                    block_name = '/BasicBlock%d.%d'%(i,j)
                    with tf.name_scope(block_name[1:]):
                        nb_name = self.net_name + block_name
                        if j != 0:
                            strides = 1
                        self.resnet_layers[nb_name + '/conv0'] = tcl.Conv2d([3,3], depth, strides = strides)
                        self.resnet_layers[nb_name + '/bn0']   = tcl.BatchNorm(activation_fn = tf.nn.relu)
                        self.resnet_layers[nb_name + '/conv1'] = tcl.Conv2d([3,3], depth)
                        self.resnet_layers[nb_name + '/bn1']   = tcl.BatchNorm()
                        
                        #if strides > 1 or depth != self.net_args['depth'][max(0,i-1)]:
                        #    self.resnet_layers[nb_name + '/conv2'] = tcl.Conv2d([1,1], depth, strides = strides)
            self.resnet_layers['FC'] = tcl.FC(num_class)
                        
    def call(self, x, training=None):
        with tf.name_scope(self.net_name):
            x = self.resnet_layers[self.net_name + '/conv0'](x)
            x = self.resnet_layers[self.net_name + '/bn0'](x)
            for i, (nb_resnet_layers, depth, stride) in enumerate(zip(self.net_args['nb_resnet_layers'], 
                                                                      self.net_args['depth'],
                                                                      self.net_args['strides'])):
                for j in range(nb_resnet_layers):           
                    block_name = '/BasicBlock%d.%d'%(i,j)
                    with tf.name_scope(block_name[1:]):
                        nb_name = self.net_name + block_name
                        if j != 0:
                            stride = 1
                        conv = self.resnet_layers[nb_name + '/conv0'](x)
                        conv = self.resnet_layers[nb_name +   '/bn0'](conv, training = training)
                        conv = self.resnet_layers[nb_name + '/conv1'](conv)
                        conv = self.resnet_layers[nb_name +   '/bn1'](conv, training = training)
                            
                        if stride > 1 or conv.shape[-1] != x.shape[-1]:
                            x = tf.pad(x, [[0,0],[0,0],[0,0],[0,x.shape[-1]]])
                            x = x[:,::2,::2]
                            #x = self.resnet_layers[nb_name + '/conv2'](x)
                                
                        x = tf.nn.relu(x+conv)
                        
            x = tf.reduce_mean(x,[1,2])
            x = self.resnet_layers['FC'](x)
            return x
