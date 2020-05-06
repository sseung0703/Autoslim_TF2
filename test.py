import tensorflow as tf
import numpy as np
import scipy.io as sio
import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import Dataloader
import op_util
from nets import WResNet, VGG, ResNet, Mobilev2
#from nets import Multiple
import slim_util

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default="test", type=str)
parser.add_argument("--arch", default='Mobilev2', type=str)
parser.add_argument("--dataset", default="cifar10", type=str)

parser.add_argument("--val_batch_size", default=2000, type=int)
parser.add_argument("--trained_param", type=str)

parser.add_argument("--slimmable", default=True, type=bool)

parser.add_argument("--gpu_id", default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    ### define path and hyper-parameter
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    
    train_images, train_labels, val_images, val_labels, pre_processing = Dataloader(args.dataset, '')
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).cache()
    train_ds = train_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.val_batch_size).batch(args.val_batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size)
    test_ds = test_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
   
    if 'WResNet' in args.arch:
        arch = [int(a) for a in args.arch.split('-')[1:]]
        model = WResNet.Model(architecture=arch, num_class = np.max(train_labels)+1,
                                      name = 'Student', trainable = True)
    elif 'VGG' in args.arch:
        model = VGG.Model(num_class = np.max(train_labels)+1,
                                  name = 'Student', trainable = True)
    elif 'ResNet' in args.arch:
        arch = int(args.arch.split('-')[1])
        model = ResNet.Model(num_layer=arch, num_class = np.max(train_labels)+1,
                                  name = 'Student', trainable = True)
    elif 'Mobilev2' in args.arch:
        model = Mobilev2.Model(num_class = np.max(train_labels)+1, width_mul = 1.0 if args.slimmable else 1.0,
                                       name = 'Student', trainable = True)

    if args.slimmable:
        Opt = op_util.Slimmable_optimizer
    else:
        Opt = op_util.Optimizer
    train_step, train_loss, train_accuracy,\
    test_step,  test_loss,  test_accuracy = Opt(model, 0., 0.)

    model(np.zeros([1]+list(train_images.shape[1:]), dtype=np.float32), training = False)

    model_name = model.variables[0].name.split('/')[0]
    trained = sio.loadmat(args.trained_param+ '/trained_params.mat')
    n = 0
    for v in model.variables:
        v.assign(trained[v.name[len(model_name)+1:]])
        n += 1
    print (n, 'params loaded')

    width_list = [1.0 for k in model.Layers.keys() if 'conv' in k and 'conv1' not in k and 'conv2' not in k] 
    width_list += [1.0 for k in model.Layers.keys() if '0/conv2' in k]
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels, width_list = width_list, bn_statistics_update = True)
    slim_util.clear_width(model)
    ori_acc = test_accuracy.result().numpy()
    test_loss.reset_states()
    test_accuracy.reset_states()

    ori_p, ori_f = slim_util.check_complexity(model)
    while(True):
        accuracy_list = []
        for i in range(len(width_list)):
            if width_list[i] > .2:
                width_list[i] -= .2
                for j, (test_images, test_labels) in enumerate(test_ds):
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
        slim_util.set_width(model, width_list)
        model(np.zeros([1]+list(train_images.shape[1:]), dtype=np.float32), training = False)

        p, f = slim_util.check_complexity(model)
        print ('Ori Acc.: %.2f, Current Acc.: %.2f'%(100*ori_acc, 100*max(accuracy_list)))
        print ('Ori params: %.4fM, Slim params: %.4fM, Ori FLOPS: %.4fM, Slim FLOPS: %.4fM'%(ori_p/1e6, p/1e6, ori_f/1e6, f/1e6))
        if f/ori_f < .5:
            break

    pruned_params = slim_util.actual_slimming(model, width_list, args.trained_param)
 

