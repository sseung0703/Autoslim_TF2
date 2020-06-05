import tensorflow as tf
from math import ceil
import scipy.io as sio
import numpy as np
import time

def get_initial_width(arch, model):
    if arch == 'Mobilev2':
        width_list = [1.0 for k in model.Layers.keys() if 'conv' in k and 'conv1' not in k and 'conv2' not in k]
        width_list += [1.0 for k in model.Layers.keys() if '0/conv2' in k]

    elif 'ResNet' in arch:
        width_list = [1.0 for k in model.Layers.keys() if 'conv' in k and 'conv1' not in k]
    return width_list

def set_slimmed_param(layer, attr, in_depth = None, out_depth = None, actual = False, trainable = False):    
    if actual:
        for a in attr:
            if not(hasattr(layer, a)):
                continue
            tensor = getattr(layer, a).numpy()
            name = getattr(layer, a).name

            if 'fc' in name:
                if in_depth is not None:
                    tensor = tensor[:ceil(in_depth*tensor.shape[0])]
         
                if out_depth is not None:
                    tensor = tensor[:,:ceil(out_depth*tensor.shape[1])]

            else:
                if in_depth is not None:
                    tensor = tensor[:,:,:ceil(in_depth*tensor.shape[2])]
         
                if out_depth is not None:
                    tensor = tensor[:,:,:,:ceil(out_depth*tensor.shape[3])]

            if hasattr(layer, 'in_mask'):
                delattr(layer, 'in_mask')
            if hasattr(layer, 'in_depth'):
                delattr(layer, 'in_depth')
            if hasattr(layer, 'out_mask'):
                delattr(layer, 'out_mask')
            if hasattr(layer, 'out_depth'):
                delattr(layer, 'out_depth')
            delattr(layer, a)
            setattr(layer, a, tf.Variable(tensor, trainable = trainable, name = name[:-2]))

    else:
        # In my experiments, the static execution is much faster than the eager execution eventhough it's FLOPS is larger than the eager.
        # Therefore, I use a mask to implement slimmable tensor.
        tensor = getattr(layer, attr[0])
        name = getattr(layer, attr[0]).name

        if in_depth is not None:
            Di = tensor.shape[-2]
            if not(hasattr(layer, 'in_mask')):
                setattr(layer, 'in_depth', tf.Variable(1., trainable = False))
                layer.in_mask = tf.range(Di, dtype = tf.float32)
            else:
                layer.in_depth.assign(in_depth)

        if out_depth is not None:
            Do = tensor.shape[-1]
            if not(hasattr(layer, 'out_mask')):
                setattr(layer, 'out_depth', tf.Variable(1., trainable = False))
                layer.out_mask = tf.range(Do, dtype = tf.float32)
            else:
                layer.out_depth.assign(out_depth)

def set_width(arch, model, width_list, actual = False):
    if arch == 'Mobilev2':
        w_num = 0
        for k in model.Layers.keys():
            layer = model.Layers[k]
            if 'conv' in k:
                if 'conv1' in k:
                    Di = in_width
                    Do = None

                elif 'conv2' in k:
                    if '0/conv2' in k:
                        group_width = width_list[w_num]
                        w_num += 1

                    Di = in_width
                    Do = group_width
                    in_width = group_width

                else:
                    width = width_list[w_num]
                    Di = None if layer.type == 'input' else in_width
                    Do = width

                    in_width = width
                    w_num += 1

                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Di, out_depth = Do, actual = actual, trainable = True)

                if k.replace('conv', 'bn') in model.Layers:
                    if 'conv1' in k:
                        Do = Di
                    bn = model.Layers[k.replace('conv', 'bn')]
                    set_slimmed_param(bn, ['moving_mean', 'moving_variance'], out_depth = Do, actual = actual, trainable = False)
                    set_slimmed_param(bn, ['gamma', 'beta'], out_depth = Do, actual = actual, trainable = True)

            if 'fc' in k:
                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Do, actual = actual, trainable = True)

    if 'ResNet' in arch:
        w_num = 0
        for k in model.Layers.keys():
            layer = model.Layers[k]          
            if 'conv' in k:
                if 'conv1' in k or 'conv2' in k:
                    if 'conv2' in k:
                        group_width = width_list[w_num]
                        w_num += 1
                    Di = in_width
                    Do = group_width

                    if 'conv1' in k:
                        in_width = group_width
                else:
                    width = width_list[w_num]
                    if layer.type == 'input':
                        group_width = width
                        Di = None
                    else:
                        Di = in_width
                    Do = width
                    in_width = width
                    w_num += 1

                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Di, out_depth = Do, actual = actual, trainable = True)

                if k.replace('conv', 'bn') in model.Layers:
                    bn = model.Layers[k.replace('conv', 'bn')]
                    set_slimmed_param(bn, ['moving_mean', 'moving_variance'], out_depth = Do, actual = actual, trainable = False)
                    set_slimmed_param(bn, ['gamma', 'beta'], out_depth = Do, actual = actual, trainable = True)

            if 'fc' in k:
                set_slimmed_param(layer, ['kernel', 'bias'], in_depth = Do, actual = actual, trainable = True)

def clear_width(model):
    for k in model.Layers.keys():
        if hasattr(model.Layers[k], 'in_depth'):
            delattr(model.Layers[k],'in_depth')
        if hasattr(model.Layers[k], 'out_depth'):
            delattr(model.Layers[k],'out_depth')

def check_complexity(model):
    total_params = []
    total_flops = []
    for k in model.Layers.keys():
        layer = model.Layers[k]
        if hasattr(layer, 'params'):
            p = layer.params
            if not(isinstance(p, int)):
                p = p.numpy()
            total_params.append(p)
        if hasattr(layer, 'flops'):
            f = layer.flops
            if not(isinstance(f, int)):
                f = f.numpy()
            total_flops.append(f)
    return sum(total_params), sum(total_flops)

def Warm_up(args, model, train_step, training_epoch, train_sub_ds, train_loss, train_accuracy,
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
    width_list = get_initial_width(args.arch, model)
    set_width(args.arch, model, width_list)

    model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)
    ori_p, ori_f = check_complexity(model)

    for test_images, test_labels in val_ds:
        test_step(test_images, test_labels, True)
    ori_acc = test_accuracy.result().numpy()
    test_accuracy.reset_states()

    step = 1    
    while(True):
        accuracy_list = []
        for i in range(len(width_list)):
            if width_list[i] > args.minimum_rate:
                width_list[i] -= args.search_step
                set_width(args.arch, model, width_list)
                for test_images, test_labels in val_ds:
                    test_step(test_images, test_labels, True)
                acc = test_accuracy.result().numpy()
                test_accuracy.reset_states()
                accuracy_list.append(acc)
                test_accuracy.reset_states()
                width_list[i] += args.search_step
            else:
                accuracy_list.append(0.)
        idx = np.argmax(np.where(np.array(width_list) > args.minimum_rate, accuracy_list, 0))
        cur_acc = accuracy_list[idx]


        width_list[idx] -= args.search_step
        width_list = [round(w*10)/10 for w in width_list]
        print (width_list, idx)

        set_width(args.arch, model, width_list)
        model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)
        p, f = check_complexity(model)

        print ('Ori Acc.: %.2f, Current Acc.: %.2f'%(100*ori_acc, 100*cur_acc))
        print ('Ori params: %.4fM, Slim params: %.4fM, Ori FLOPS: %.4fM, Slim FLOPS: %.4fM'%(ori_p/1e6, p/1e6, ori_f/1e6, f/1e6))

        if f/ori_f < args.target_rate:
            break 

    set_width(args.arch, model, width_list, actual = True)

    return ori_p, ori_f, p, f
