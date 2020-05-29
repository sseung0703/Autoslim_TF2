import tensorflow as tf
import numpy as np
import scipy.io as sio
import os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import Dataloader
import op_util, slim_util
from nets import WResNet, VGG, ResNet, Mobilev2
from math import ceil

home_path = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description='')

parser.add_argument("--train_path", default="test", type=str)
parser.add_argument("--arch", default='Mobilev2', type=str)
parser.add_argument("--dataset", default="cifar10", type=str)

parser.add_argument("--learning_rate", default = 0.1, type=float)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)
parser.add_argument("--weight_decay", default=4e-5, type=float)

parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--val_batch_size", default=2000, type=int)
parser.add_argument("--train_epoch", default=200, type=int)

parser.add_argument("--slimmable", default=False, type=bool)
parser.add_argument("--trained_slimmable", type=str)
parser.add_argument("--target_rate", default=.5, type=float)
parser.add_argument("--search_step", default=.2, type=float)
parser.add_argument("--minimum_rate", default=.2, type=float)

parser.add_argument("--gpu_id", default=0, type=int)
parser.add_argument("--do_log", default=100, type=int)
args = parser.parse_args()

def validation(test_step, test_ds, test_loss, test_accuracy,
               train_loss = None, train_accuracy = None, epoch = None, lr = None, logs = None, bn_statistics_update = False):
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels, bn_statistics_update = bn_statistics_update)

    if logs == None:
        tf.summary.scalar('Warmup/Categorical_loss/train', train_loss.result(), step=epoch+1)
        tf.summary.scalar('Warmup/Categorical_loss/test', test_loss.result(), step=epoch+1)
        tf.summary.scalar('Warmup/Accuracy/train', train_accuracy.result()*100, step=epoch+1)
        tf.summary.scalar('Warmup/Accuracy/test', test_accuracy.result()*100, step=epoch+1)

        template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val loss: {3:0.4f}, val_Acc.: {4:2.2f}'
        print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                         test_loss.result(),  test_accuracy.result()*100))
    else:
        tf.summary.scalar('Categorical_loss/train', train_loss.result(), step=epoch+1)
        tf.summary.scalar('Categorical_loss/test', test_loss.result(), step=epoch+1)
        tf.summary.scalar('Accuracy/train', train_accuracy.result()*100, step=epoch+1)
        tf.summary.scalar('Accuracy/test', test_accuracy.result()*100, step=epoch+1)
        tf.summary.scalar('learning_rate', lr, step=epoch)

        template = 'Epoch: {0:3d}, train_loss: {1:0.4f}, train_Acc.: {2:2.2f}, val_loss: {3:0.4f}, val_Acc.: {4:2.2f}'
        print (template.format(epoch+1, train_loss.result(), train_accuracy.result()*100,
                                         test_loss.result(),  test_accuracy.result()*100))

        logs['training_acc'].append(train_accuracy.result()*100)
        logs['validation_acc'].append(test_accuracy.result()*100)
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()

def build_dataset_proviers(train_images, train_labels, test_images, test_labels, pre_processing, slimmable = False):
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).cache()
    train_ds = train_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.shuffle(100*args.batch_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_ds = test_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(args.val_batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    if args.slimmable:
        num_train = train_images.shape[0]
        num_label = np.max(train_labels)+1

        train_sub = np.vstack([train_images[np.where(train_labels[:,0] == i)][:int(num_train/num_label*.8)] for i in range(num_label)])
        val       = np.vstack([train_images[np.where(train_labels[:,0] == i)][int(num_train/num_label*.8):] for i in range(num_label)])
        train_sub_lb = np.vstack([train_labels[np.where(train_labels[:,0] == i)][:int(num_train/num_label*.8)] for i in range(num_label)])
        val_lb       = np.vstack([train_labels[np.where(train_labels[:,0] == i)][int(num_train/num_label*.8):] for i in range(num_label)])

        sub_idx = np.random.choice(train_sub.shape[0], train_sub.shape[0], replace = False)
        train_sub = train_sub[sub_idx]
        train_sub_lb = train_sub_lb[sub_idx]

        sub_idx = np.random.choice(val.shape[0], val.shape[0], replace = False)
        val = val[sub_idx]
        val_lb = val_lb[sub_idx]

        train_sub_ds = tf.data.Dataset.from_tensor_slices((train_sub, train_sub_lb)).cache()
        train_sub_ds = train_sub_ds.map(pre_processing(is_training = True),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_sub_ds = train_sub_ds.shuffle(100*args.batch_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((val, val_lb))
        val_ds = val_ds.map(pre_processing(is_training = False), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.batch(args.val_batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        return {'train': train_ds, 'test': test_ds, 'train_sub': train_sub_ds, 'val': val_ds}
    return {'train': train_ds, 'test': test_ds}

if __name__ == '__main__':
    ### define path and hyper-parameter
    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    
    summary_writer = tf.summary.create_file_writer(args.train_path)
    
    train_images, train_labels, test_images, test_labels, pre_processing = Dataloader(args.dataset, '')
    datasets = build_dataset_proviers(train_images, train_labels, test_images, test_labels, pre_processing, slimmable = args.slimmable)
    args.input_size = list(train_images.shape[1:])
    
    if 'WResNet' in args.arch:
        arch = [int(a) for a in args.arch.split('-')[1:]]
        model = WResNet.Model(architecture=arch, num_class = np.max(train_labels)+1,
                              name = 'Student', trainable = True)
    elif 'VGG' in args.arch:
        model = VGG.Model(num_class = np.max(train_labels)+1,
                          name = 'Student', trainable = True)
    elif 'ResNet' in args.arch:
        arch = int(args.arch.split('-')[1])
        model = ResNet.Model(num_layers = arch, num_class = np.max(train_labels)+1,
                             name = 'Student', trainable = True)
    elif 'Mobilev2' in args.arch:
        model = Mobilev2.Model(num_class = np.max(train_labels)+1, width_mul = 1.0 if args.slimmable else 1.0,
                               name = 'Student', trainable = True)

    model(np.zeros([1]+args.input_size, dtype=np.float32), training = False)

    cardinality = tf.data.experimental.cardinality(datasets['train']).numpy()
    if args.decay_points is None:
        LR = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, cardinality,
                                                            args.decay_rate, staircase=True)
    else:
        args.decay_points = [int(dp*args.train_epoch) if dp < 1 else int(dp) for dp in args.decay_points]
        LR = op_util.PiecewiseConstantDecay([dp*cardinality for dp in args.decay_points],
                                            [args.learning_rate * args.decay_rate ** i for  i in range(len(args.decay_points)+1)])

    if args.slimmable:
        train_step, train_loss, train_accuracy,\
        test_step,  test_loss,  test_accuracy = op_util.Slimmable_optimizer(args, model, args.weight_decay, args.learning_rate)
    else:
        train_step, train_loss, train_accuracy,\
        test_step,  test_loss,  test_accuracy = op_util.Optimizer(model, args.weight_decay, LR)

    with summary_writer.as_default():
        step = 0
        logs = {'training_acc' : [], 'validation_acc' : []}

        model_name = model.variables[0].name.split('/')[0]
        train_time = time.time()
        init_epoch = 0
        if args.slimmable:
            ## Warm-up training
            if args.trained_slimmable is None or 'None' in args.trained_slimmable:
                print ('Warm-up training starts')            
                slim_util.Warm_up(args, model, train_step, ceil(args.train_epoch *.3), datasets['train_sub'], 
                                  train_loss, train_accuracy, validation, test_step, datasets['val'], test_loss, test_accuracy)
                params = {}
                for v in model.variables:
                    if model_name in v.name:
                        params[v.name[len(model_name)+1:]] = v.numpy()

                sio.savemat(args.train_path+'/slimmable_params.mat', params)
                sio.savemat(args.train_path + '/log.mat',logs)

            else:
                model_name = model.variables[0].name.split('/')[0]
                trained = sio.loadmat(args.trained_slimmable)
                n = 0
                for v in model.variables:
                    if model_name in v.name:
                        v.assign(trained[v.name[len(model_name)+1:]])
                        n += 1
                print (n, 'params loaded')

            ## Greed search
            print ('Greed searching starts')
            ori_p, ori_f, p, f = slim_util.Greedly_search(args, model, datasets['val'], test_step, test_accuracy, test_loss)

            ## Make new optimizer to use ordinary training scheme.
            train_step, train_loss, train_accuracy,\
            test_step,  test_loss,  test_accuracy = op_util.Optimizer(model, args.weight_decay, LR)

        ## Conventional training routine
        for epoch in range(init_epoch, init_epoch + args.train_epoch):
            for images, labels in datasets['train']:
                lr = train_step(images, labels)
                step += 1
                if step % args.do_log == 0:
                    template = 'Global step {0:5d}: loss = {1:0.4f} ({2:1.3f} sec/step)'
                    print (template.format(step, train_loss.result(), (time.time()-train_time)/args.do_log))
                    train_time = time.time()

            val_time = time.time()
            validation(test_step, datasets['test'], test_loss, test_accuracy,
                       train_loss, train_accuracy, epoch = epoch, lr = lr, logs = logs, bn_statistics_update = False)
            train_time += time.time() - val_time

        params = {}
        for v in model.variables:
            if model_name in v.name:
                params[v.name[len(model_name)+1:]] = v.numpy()

        if args.slimmable:
            params['ORI_FLOPS'] = ori_f
            params['ORI_PARAMS'] = ori_p
            params['PRUNED_FLOPS'] = f
            params['PRUNED_PARAMS'] = p

        sio.savemat(args.train_path+'/trained_params.mat', params)
        sio.savemat(args.train_path + '/log.mat',logs)

