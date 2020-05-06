import tensorflow as tf
from nets import tcl

class Model(tf.keras.Model):
    def __init__(self, architecture, num_class, name = 'WResNet', trainable = True, **kwargs):
        super(Model, self).__init__(name = name, **kwargs)
        def kwargs(**kwargs):
            return kwargs
        setattr(tcl.Conv2d, 'pre_defined', kwargs(use_biases = False, activation_fn = None, trainable = trainable))
        setattr(tcl.BatchNorm, 'pre_defined', kwargs(trainable = trainable))
        setattr(tcl.FC, 'pre_defined', kwargs(trainable = trainable))
        
        self.Layers = {}
        depth, widen_factor = architecture
        self.nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.stride = [1,2,2]
        self.n = (depth-4)//6
        
        self.Layers['conv'] = tcl.Conv2d([3,3], self.nChannels[0], name = 'conv')
        self.Layers['conv'].type = 'input'
        in_planes = self.nChannels[0]
        
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                block_name = 'BasicBlock%d.%d'%(i,j)
                with tf.name_scope(block_name):
                    equalInOut = in_planes == c
                    in_planes = c
                    
                    self.Layers[block_name + '/bn']   = tcl.BatchNorm(name = 'bn')

                    if not(equalInOut):
                        self.Layers[block_name + '/conv0'] = tcl.Conv2d([1,1], c, strides = s if j == 0 else 1, name = 'conv0')
                        self.Layers[block_name + '/conv0'].type = 'block_bottle'
                    self.Layers[block_name + '/conv1'] = tcl.Conv2d([3,3], c, strides = s if j == 0 else 1, name = 'conv1')
                    self.Layers[block_name + '/bn1']   = tcl.BatchNorm(activation_fn = tf.nn.relu, name = 'bn1')
                    self.Layers[block_name + '/conv2'] = tcl.Conv2d([3,3], c, strides = 1, name = 'conv2')
                            
                    self.Layers[block_name + '/conv1'].type = 'block_in'
                    self.Layers[block_name + '/conv2'].type = 'block_out'

        self.Layers['bn2']= tcl.BatchNorm(name = 'bn2')
        self.Layers['fc'] = tcl.FC(num_class, name = 'fc')
    
    def call(self, x, training=None):
        self.feature = []
        group_id = 0

        conv = self.Layers['conv']
        x = conv(x)
        self.corr_update(conv, x)
        self.compute_prune_rate(conv, x)
        conv.group = group_id 

        conv_name = 'conv'
        in_planes = self.nChannels[0]
        for i, (c, s) in enumerate(zip(self.nChannels[1:], self.stride)):
            for j in range(self.n):
                equalInOut = in_planes == c
                block_name = 'BasicBlock%d.%d'%(i,j)
                bn = self.Layers[block_name + '/bn']
                self.Layers[conv_name.replace('conv', 'bn')] = bn # for automative pruning

                bn1 = self.Layers[block_name + '/bn1']
                if not(equalInOut):
                    conv0 = self.Layers[block_name + '/conv0']
                conv1 = self.Layers[block_name + '/conv1']
                conv2 = self.Layers[block_name + '/conv2']

                with tf.name_scope(block_name):
                    out = bn(x, training = training)
                    if j == 0 and i > 0:
                        self.feature.append([bn, conv_name.replace('conv', 'bn'), out])
                    out = tf.nn.relu(out)
                    
                    if not(equalInOut):
                        group_id += 1
                        residual = conv0(out)
                        self.corr_update(conv0, residual)
                        self.compute_prune_rate(conv0, residual)
                        conv0.group = group_id 
                    else:
                        residual = x

                    out = conv1(out)
                    self.corr_update(conv1, out)
                    self.compute_prune_rate(conv1, out)
                    out = bn1(out, training = training)
                    out = conv2(out)
                    self.corr_update(conv2, out)
                    self.compute_prune_rate(conv2, out)
                    x = residual+out

                in_planes = c
                conv2.group = group_id 
                conv_name = block_name + '/conv2'
                        
        bn = self.Layers['bn2']
        x = bn(x, training = training)
        self.Layers[conv_name.replace('conv', 'bn')] = bn # for automative pruning
        self.feature.append([bn, conv_name.replace('conv', 'bn'), x])
        x = tf.nn.relu(x)

                            
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

        conv.params = kh*kw*Di*Do
        conv.flops  = H*W*conv.params  
