import tensorflow as tf
from math import ceil
import scipy.io as sio
import numpy as np
import time

def set_width(model, width_list):
    w_num = 0
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if 'conv1' in k:
            layer.in_depth = in_width
            if k.replace('conv', 'bn') in model.Layers:
                model.Layers[k.replace('conv', 'bn')].out_depth = in_width
        elif 'conv2' in k:
            if '0/conv2' in k:
                width = ceil(width_list[w_num] * layer.kernel.shape[3])
                group_width = width
                w_num += 1

            layer.in_depth  = in_width
            layer.out_depth = group_width
            if k.replace('conv', 'bn') in model.Layers:
                model.Layers[k.replace('conv', 'bn')].out_depth = group_width
            in_width = group_width

        elif 'conv' in k:
            width = ceil(width_list[w_num] * layer.kernel.shape[3])
            if layer.type == 'input':
                layer.out_depth = width
            else:
                layer.in_depth  = in_width
                layer.out_depth = width

            if k.replace('conv', 'bn') in model.Layers:
                model.Layers[k.replace('conv', 'bn')].out_depth = width

            in_width = width
            w_num += 1

        if 'fc' in k:
            layer.in_depth  = in_width

def clear_width(model):
    for k in model.Layers.keys():
        if hasattr(model.Layers[k], 'in_depth'):
            delattr(model.Layers[k], 'in_depth')
        if hasattr(model.Layers[k], 'out_depth'):
            delattr(model.Layers[k], 'out_depth')


def check_complexity(model):
    total_params = 0
    total_flops = 0
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if hasattr(layer, 'params'):
            total_params += layer.params
        if hasattr(layer, 'flops'):
            total_flops += layer.flops
    return total_params, total_flops

def actual_slimming(model, width_list, path):
    w_num = 0
    params = {}

    for k in model.Layers.keys():
        layer = model.Layers[k]
        if 'conv1' in k:
            kernel = layer.kernel.numpy()
            params[k + '/kernel:0'] = kernel[:,:,:in_width]

            if layer.use_biases:
                biases = layer.biases.numpy()
                params[k + '/biases:0'] = biases

            if k.replace('conv', 'bn') in model.Layers:
                k_bn = k.replace('conv', 'bn') 
                bn = model.Layers[k_bn]

                mm = bn.moving_mean.numpy()
                params[k_bn + '/moving_mean:0'] = mm[...,:in_width]
                ms = bn.moving_std.numpy()
                params[k_bn + '/moving_std:0'] = ms[...,:in_width]

                if bn.scale:
                    scale = bn.gamma.numpy()
                    params[k_bn  + '/gamma:0'] = scale[...,:in_width]
                if bn.center:
                    center = bn.beta.numpy()
                    params[k_bn  + '/beta:0'] = center[...,:in_width]
                  

        elif 'conv2' in k:
            if '0/conv2' in k:
                width = ceil(width_list[w_num]*layer.kernel.shape[-1])
                group_width = width
                w_num += 1

            kernel = layer.kernel.numpy()
            params[k + '/kernel:0'] = kernel[:,:,:in_width,:group_width]

            if layer.use_biases:
                biases = model.Layers[k].biases.numpy()
                params[k + '/biases:0'] = biases[:,:,:group_width]

            if k.replace('conv', 'bn') in model.Layers:
                k_bn = k.replace('conv', 'bn') 
                bn = model.Layers[k_bn]

                mm = bn.moving_mean.numpy()
                params[k_bn + '/moving_mean:0'] = mm[...,:group_width]
                ms = bn.moving_std.numpy()
                params[k_bn + '/moving_std:0'] = ms[...,:group_width]

                if bn.scale:
                    scale = bn.gamma.numpy()
                    params[k_bn + '/gamma:0'] = scale[...,:group_width]
                if bn.center:
                    center = bn.beta.numpy()
                    params[k_bn + '/beta:0'] = center[...,:group_width]
            in_width = group_width

        elif 'conv' in k:
            kernel = layer.kernel.numpy()

            width = int(width_list[w_num]*layer.kernel.shape[-1])
            if layer.type == 'input':
                in_width = kernel.shape[2]

            params[k + '/kernel:0'] = kernel[:,:,:in_width,:width]

            if layer.use_biases:
                biases = model.Layers[k].biases.numpy()
                params[k + '/biases:0'] = biases[:,:,:width]

            if k.replace('conv', 'bn') in model.Layers:
                k_bn = k.replace('conv', 'bn') 
                bn = model.Layers[k_bn]

                mm = bn.moving_mean.numpy()
                params[k_bn + '/moving_mean:0'] = mm[...,:width]
                ms = bn.moving_std.numpy()
                params[k_bn + '/moving_std:0'] = ms[...,:width]

                if bn.scale:
                    scale = bn.gamma.numpy()
                    params[k_bn + '/gamma:0'] = scale[...,:width]
                if bn.center:
                    center = bn.beta.numpy()
                    params[k_bn + '/beta:0'] = center[...,:width]

            in_width = width
            w_num += 1

        if 'fc' in k:
            kernel = layer.kernel.numpy()
            params[k + '/kernel:0'] = kernel[:in_width]
            biases = layer.biases.numpy()
            params[k + '/biases:0'] = biases

    sio.savemat(path+'/slimmed_params.mat', params)

def assign_slimmed_param(model, new_param, trainable = False):
    model_name = model.variables[0].name.split('/')[0] + '/'

    for k in model.Layers.keys():
        layer = model.Layers[k]
        if 'conv' in k:
            kernel = new_param[k + '/kernel:0']
            delattr(layer, 'kernel')
            setattr(layer, 'kernel', tf.Variable(kernel, trainable = trainable, name = model_name + k + '/kernel'))

            if layer.use_biases:
                biases = new_param[k + '/biases:0']
                delattr(layer, 'biases')
                setattr(layer, 'biases', tf.Variable(biases, trainable = trainable, name =  model_name + k + 'biases'))

        if 'bn' in k:
            mm = new_param[k + '/moving_mean:0']
            delattr(layer, 'moving_mean')
            setattr(layer, 'moving_mean', tf.Variable(mm, trainable = False, name =  model_name + k + '/moving_mean'))

            ms = new_param[k + '/moving_std:0']
            delattr(layer, 'moving_std')
            setattr(layer, 'moving_std', tf.Variable(ms, trainable = False, name =  model_name + k + '/moving_std'))


            if layer.scale:
                gamma = new_param[k + '/gamma:0']
                delattr(layer, 'gamma')
                setattr(layer, 'gamma', tf.Variable(gamma, trainable = trainable, name =  model_name + k + '/gamma'))

                beta = new_param[k + '/beta:0']
                delattr(layer, 'beta')
                setattr(layer, 'beta', tf.Variable(beta, trainable = trainable, name =  model_name + k + '/beta'))

        if 'fc' in k:
            kernel = new_param[k + '/kernel:0']
            delattr(layer, 'kernel')
            setattr(layer, 'kernel', tf.Variable(kernel, trainable = trainable, name =  model_name + k + '/kernel'))

            if layer.use_biases:
                biases = new_param[k + '/biases:0']
                delattr(layer, 'biases')
                setattr(layer, 'biases', tf.Variable(biases, trainable = trainable, name =  model_name + k + '/biases'))
                
def Warm_up(args, train_step, training_epoch, train_sub_ds, train_loss, train_accuracy,
            validation, test_step, val_ds, test_loss, test_accuracy):
    step = 0
    train_time = time.time()
    for epoch in range(training_epoch):
        for images, labels in train_sub_ds:
            train_step(images, labels)
            step += 1
            if step % args.do_log == 0:
                template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                print (template.format(step, train_loss.result(), (time.time()-train_time)/args.do_log))
                train_time = time.time()

        val_time = time.time()
        validation(test_step, val_ds, test_loss, test_accuracy,
                   train_loss, train_accuracy, epoch = epoch, bn_statistics_update = True)

        train_time += time.time() - val_time
        
        
def Greedly_search(args, model, val_ds, test_step, test_accuracy, test_loss):
    width_list = [1.0 for k in model.Layers.keys() if 'conv' in k and 'conv1' not in k and 'conv2' not in k]
    width_list += [1.0 for k in model.Layers.keys() if '0/conv2' in k]
    
    for test_images, test_labels in val_ds:
        test_step(test_images, test_labels, width_list = width_list, bn_statistics_update = True)
    ori_acc = test_accuracy.result().numpy()
    test_loss.reset_states()
    test_accuracy.reset_states()

    ori_p, ori_f = check_complexity(model)
    
    while(True):
        accuracy_list = []
        for i in range(len(width_list)):
            if width_list[i] > .2:
                width_list[i] -= .2
                for test_images, test_labels in val_ds:
                    test_step(test_images, test_labels, width_list = width_list, bn_statistics_update = True)
                accuracy_list.append(test_accuracy.result().numpy())
                test_loss.reset_states()
                test_accuracy.reset_states()
                width_list[i] += .2
            else:
                accuracy_list.append(0.)
        idx = np.argmax(np.where(np.array(width_list) > .2, accuracy_list, 0.))
        width_list[idx] -= .2
        width_list = [round(w*10)/10 for w in width_list]
        print (width_list, idx)
        set_width(model, width_list)
        model(np.zeros([1]+list(test_images.shape[1:]), dtype=np.float32), training = False)

        p, f = check_complexity(model)
        print ('Ori Acc.: %.2f, Current Acc.: %.2f'%(100*ori_acc, 100*max(accuracy_list)))
        print ('Ori params: %.4fM, Slim params: %.4fM, Ori FLOPS: %.4fM, Slim FLOPS: %.4fM'%(ori_p/1e6, p/1e6, ori_f/1e6, f/1e6))
        if f/ori_f < .5:
            break 
    actual_slimming(model, width_list, args.train_path)
    trained = sio.loadmat(args.train_path + '/slimmed_params.mat')
    assign_slimmed_param(model, trained, trainable = True)
    clear_width(model)
    return ori_p, ori_f, p, f

