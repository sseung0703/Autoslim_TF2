import tensorflow as tf
from nets import tcl
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, num_class, name = 'VGG', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(activation_fn = None, trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(trainable = trainable))
        self.Layers = {}
        self.filter_depth = [64,64,128,128,256,256,256,512,512,512,512,512,512]
        for i, c in enumerate(self.filter_depth):
            self.Layers['conv%d'%i] = tcl.Conv2d([3,3], c, name = 'conv%d'%i)
            self.Layers['bn%d'%i]   = tcl.BatchNorm(name = 'bn%d'%i)

            if i == 0:
                self.Layers['conv%d'%i].type = 'input'
            else:
                self.Layers['conv%d'%i].type = 'mid'
        self.max_pool = tf.keras.layers.MaxPool2D()
        self.Layers['fc'] = tcl.FC(num_class, name = 'fc')
        self.Layers['fc'].type = 'VGG_class'
    
    def call(self, x, training=None):
        for i in range(len(self.filter_depth)):
            conv = self.Layers['conv%d'%i]
            bn = self.Layers['bn%d'%i]

            x = bn(conv(x))
            self.corr_update(conv, x)
            self.compute_prune_rate(conv, x)

            x = tf.nn.relu(x)
            if  i in [1,3,6,9]:
                x = self.max_pool(x)

        x = tf.reduce_mean(x,[1,2])
        x = self.Layers['fc'](x)
        self.logits = x
        return x
    
    def get_feat(self, x, feat, training = None):
        y = self.call(x, training)
        return y, getattr(self, feat)

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

    def compute_prune_rate(self, conv, x):
        """
        compute filter's computational cost and number of parameters.

        Arguments:
            conv: Convolutional layer that target filter set is involved in.
            x: target filter set
        Returns:
            None
        """
        H,W = x.shape[1:3]
        kh,kw,Di,Do = conv.kernel.shape
        if hasattr(conv, 'in_mask'):
            Di = conv.in_mask.size
        if hasattr(conv, 'mask'):
            Do = conv.mask.size
        
        conv.params = kh*kw*Di*Do
        conv.flops  = H*W*conv.params
