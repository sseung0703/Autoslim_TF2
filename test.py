import tensorflow as tf
import numpy as np
import scipy.io as sio
import os, argparse
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

    test_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size)
    test_ds = test_ds.cache().prefetch(tf.data.experimental.AUTOTUNE)
   
    if 'WResNet' in args.arch:
        arch = [int(a) for a in args.arch.split('-')[1:]]
        model = WResNet.Model(architecture=arch, num_class = np.max(train_labels)+1,
                                      name = 'WResNet', trainable = True)
    elif 'VGG' in args.arch:
        model = VGG.Model(num_class = np.max(train_labels)+1,
                                  name = 'VGG', trainable = True)
    elif 'ResNet' in args.arch:
        arch = int(args.arch.split('-')[1])
        model = ResNet.Model(num_layer=arch, num_class = np.max(train_labels)+1,
                                  name = 'ResNet', trainable = True)
    elif 'Mobilev2' in args.arch:
        model = Mobilev2.Model(num_class = np.max(train_labels)+1, width_mul = 1.0 if args.slimmable else 1.0,
                                       name = 'Mobilev2', trainable = True)

    _,_,_, test_step,  test_loss,  test_accuracy = op_util.Optimizer(model, 0., 0.)
    model(np.zeros([1]+list(train_images.shape[1:]), dtype=np.float32), training = False)
    
    trained = sio.loadmat(args.trained_param+ '/trained_params.mat')
    if args.slimmable:
        slim_util.assign_slimmed_param(model, trained, trainable = False)
    else:
        n = 0
        model_name = model.variables[0].name.split('/')[0]
        for v in model.variables:
            v.assign(trained[v.name[len(model_name)+1:]])
            n += 1
        print (n, 'params loaded')

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    slim_util.clear_width(model)
    ori_acc = test_accuracy.result().numpy()
    test_loss.reset_states()
    test_accuracy.reset_states()


