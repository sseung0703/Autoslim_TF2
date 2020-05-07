import tensorflow as tf

def arg_scope(func):
    def func_with_args(self, *args, **kwargs):
        if hasattr(self, 'pre_defined'):
            for k in self.pre_defined.keys():
                kwargs[k] = self.pre_defined[k]
        return func(self, *args, **kwargs)
    return func_with_args

class Conv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, num_outputs, strides = 1, dilations = 1, padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = tf.nn.relu,
                 name = 'conv',
                 trainable = True,
                 layertype = 'mid',
                 **kwargs):
        super(Conv2d, self).__init__(name = name, **kwargs)
        
        self.kernel_size = kernel_size
        self.num_outputs = num_outputs
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

        self.trainable = trainable
        self.type = layertype
        
        
    def build(self, input_shape):
        super(Conv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
    def call(self, input):
        kernel = self.kernel
        kh,kw,Di,Do = kernel.shape

        if getattr(self, 'in_depth', 1.) < 1.:
            Di = tf.cast(tf.math.ceil(Di * getattr(self, 'in_depth')), tf.int32)
            kernel = tf.slice(kernel, [0,0,0,0], [-1,-1,Di,-1])
       
        if getattr(self, 'out_depth', 1.) < 1.:
            Do = tf.cast(tf.math.ceil(Do * getattr(self, 'out_depth')), tf.int32)
            kernel = tf.slice(kernel, [0,0,0,0], [-1,-1,-1,Do])

        conv = tf.nn.conv2d(input, kernel, self.strides, self.padding,
                            dilations=self.dilations, name=None)
        if self.use_biases:
            biases = self.biases
            if getattr(self, 'out_depth', 1.) < 1.:
                biases = tf.slice(kernel, [0,0,0,0], [-1,-1,-1,Do])

            conv += biases
        if self.activation_fn:
            conv = self.activation_fn(conv)

        H,W = conv.shape[1:3]
        self.params = kh*kw*Di*Do
        self.flops  = H*W*self.params
        
        if self.use_biases:
            self.params += Do

        return conv

class DepthwiseConv2d(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, kernel_size, multiplier = 1, strides = [1,1,1,1], dilations = [1,1], padding = 'SAME',
                 kernel_initializer = tf.keras.initializers.VarianceScaling(scale = 2., mode='fan_out'),         
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = tf.nn.relu,
                 name = 'conv',
                 trainable = True,
                 layertype = 'mid',
                 **kwargs):
        super(DepthwiseConv2d, self).__init__(name = name, **kwargs)
        
        self.kernel_size = kernel_size
        self.strides = strides if isinstance(strides, list) else [1, strides, strides, 1]
        self.padding = padding
        self.dilations = dilations if isinstance(dilations, list) else [dilations, dilations]
        self.multiplier = multiplier
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn

        self.trainable = trainable
        self.type = layertype
        
        
    def build(self, input_shape):
        super(DepthwiseConv2d, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = self.kernel_size + [input_shape[-1], self.multiplier],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,1,1, input_shape[-1]*self.multiplier],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
    def call(self, input):
        kernel = self.kernel
        kh,kw,Di,Do = kernel.shape


        if getattr(self, 'in_depth', 1.) < 1.:
            Di = tf.cast(tf.math.ceil(Di * getattr(self, 'in_depth')), tf.int32)
            kernel = tf.slice(kernel, [0,0,0,0], [-1,-1,Di,-1])

        conv = tf.nn.depthwise_conv2d(input, kernel, strides = self.strides, padding = self.padding, dilations=self.dilations)
        if self.use_biases:
            biases = self.biases
            if getattr(self, 'in_depth', 1.) < 1.:
                biases = tf.slice(biases, [0,0,0,0], [-1,-1,-1,Di])
            conv += biases
        if self.activation_fn:
            conv = self.activation_fn(conv)

        H,W = conv.shape[1:3]

        self.params = kh*kw*Di*Do
        self.flops  = H*W*self.params
        
        if self.use_biases:
            self.params += Do

        return conv

class FC(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, num_outputs, 
                 kernel_initializer = tf.keras.initializers.he_normal(),
                 use_biases = True,
                 biases_initializer  = tf.keras.initializers.Zeros(),
                 activation_fn = None,
                 name = 'fc',
                 trainable = True, **kwargs):
        super(FC, self).__init__(name = name, **kwargs)
        self.num_outputs = num_outputs
        self.kernel_initializer = kernel_initializer
        
        self.use_biases = use_biases
        self.biases_initializer = biases_initializer
        
        self.activation_fn = activation_fn
        
        self.trainable = trainable
        
    def build(self, input_shape):
        super(FC, self).build(input_shape)
        self.kernel = self.add_weight(name  = 'kernel', 
                                      shape = [int(input_shape[-1]), self.num_outputs],
                                      initializer=self.kernel_initializer,
                                      trainable = self.trainable)
        if self.use_biases:
            self.biases  = self.add_weight(name = "biases",
                                           shape=[1,self.num_outputs],
                                           initializer = self.biases_initializer,
                                           trainable = self.trainable)
    def call(self, input):
        kernel = self.kernel
        Di,Do = kernel.shape

        if getattr(self, 'in_depth', 1.) < 1.:
            Di = tf.cast(tf.math.ceil(Di * getattr(self, 'in_depth')), tf.int32)
            kernel = tf.slice(kernel, [0,0], [Di,-1])

        fc = tf.matmul(input, kernel)
        if self.use_biases:
            fc += self.biases
        if self.activation_fn:
            fc = self.activation_fn(fc)

        self.params = Di*Do         
        self.flops  = self.params
        for n in fc.shape[1:-1]:
            self.flops *= n 
        
        if self.use_biases:
            self.params += Do

        return fc

class BatchNorm(tf.keras.layers.Layer):
    @arg_scope
    def __init__(self, param_initializers = None,
                       scale = True,
                       center = True,
                       alpha = 0.9,
                       epsilon = 1e-5,
                       activation_fn = None,
                       name = 'bn',
                       trainable = True,
                       **kwargs):
        super(BatchNorm, self).__init__(name = name, **kwargs)
        if param_initializers == None:
            param_initializers = {}
        if not(param_initializers.get('moving_mean')):
            param_initializers['moving_mean'] = tf.keras.initializers.Zeros()
        if not(param_initializers.get('moving_std')):
            param_initializers['moving_std'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('gamma')) and scale:
            param_initializers['gamma'] = tf.keras.initializers.Ones()
        if not(param_initializers.get('beta')) and center:
            param_initializers['beta'] = tf.keras.initializers.Zeros()
        
        self.param_initializers = param_initializers
        self.scale = scale
        self.center = center
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_fn = activation_fn
        
        self.trainable = trainable



    def build(self, input_shape):
        super(BatchNorm, self).build(input_shape)
        self.moving_mean = self.add_weight(name  = 'moving_mean', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_mean'])
        self.moving_std = self.add_weight(name  = 'moving_std', trainable = False,
                                      shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                      initializer=self.param_initializers['moving_std'])

        if self.scale:
            self.gamma = self.add_weight(name  = 'gamma', 
                                         shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                         initializer=self.param_initializers['gamma'],
                                         trainable = self.trainable)
        else:
            self.gamma = 1.
        if self.center:
            self.beta = self.add_weight(name  = 'beta', 
                                        shape = [1]*(len(input_shape)-1)+[int(input_shape[-1])],
                                        initializer=self.param_initializers['beta'],
                                        trainable = self.trainable)
        else:
            self.beta = 0.
            
        self.mask = tf.range(input_shape[-1])

    def EMA(self, variable, value):
        Do = variable.shape[-1]
        if variable.shape[-1] != value.shape[-1]:
            variable_ = tf.squeeze(variable)
            Do = tf.math.ceil(Do * self.out_depth)
            mask = tf.slice(self.mask, [0],[Do])
            update_delta = (tf.gather(variable_, mask) - tf.squeeze(value)) * (1-self.alpha)
            variable.assign(tf.reshape(tf.tensor_scatter_nd_sub(variable_, tf.expand_dims(mask,1), update_delta),[1,1,1,-1]))
        else:
            update_delta = (variable - value) * (1-self.alpha)
            variable.assign(variable-update_delta)
        
    def call(self, input, training=None):
        Do = self.moving_mean.shape[-1]
        if training:
            mean, var = tf.nn.moments(input, list(range(len(input.shape)-1)), keepdims=True)
            std = tf.sqrt(var + self.epsilon)
            self.EMA(self.moving_mean, mean)
            self.EMA(self.moving_std, std)
        else:
            mean = self.moving_mean
            std = self.moving_std
        gamma, beta = self.gamma, self.beta

        if getattr(self, 'out_depth', 1.) < 1.:
            Do = tf.cast(tf.math.ceil(Do * getattr(self, 'out_depth')), tf.int32)
            if not(training):
                mean = tf.slice(mean, [0,0,0,0], [-1,-1,-1,Do])
                std = tf.slice(std, [0,0,0,0], [-1,-1,-1,Do])
            gamma = tf.slice(self.gamma, [0,0,0,0], [-1,-1,-1,Do])
            beta = tf.slice(self.beta, [0,0,0,0], [-1,-1,-1,Do])

        bn = (input-mean)/std*gamma + beta

        if self.activation_fn:
            bn = self.activation_fn(bn)

        self.params = Do*(2 + self.scale + self.center)
        self.flops  = self.params
        for n in bn.shape[1:-1]:
            self.flops *= n 

        return bn
