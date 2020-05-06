import subprocess
import tensorflow as tf
import glob, os, argparse
import scipy.io as sio
import numpy as np

def get_avg_plot(base_path):
    pathes = glob.glob(base_path + '/*')
    summary_writer = tf.summary.create_file_writer(base_path+'/average')
    with summary_writer.as_default():
        training_acc   = []
        validation_acc = []
        for path in pathes:
            logs = sio.loadmat(path + '/log.mat')
            training_acc.append(logs['training_acc'])
            validation_acc.append(logs['validation_acc'])
        training_acc   = np.mean(np.vstack(training_acc),0)
        validation_acc = np.mean(np.vstack(validation_acc),0)
    
        for i, (train_acc, val_acc) in enumerate(zip(training_acc,validation_acc)):
            tf.summary.scalar('Accuracy/train', train_acc, step=i+1)
            tf.summary.scalar('Accuracy/test', val_acc, step=i+1)

parser = argparse.ArgumentParser()

parser.add_argument("--train_path", default="test", type=str)
parser.add_argument("--train_epoch", default=200, type=int)
parser.add_argument("--decay_points", default = [.3, .6, .8], type=float, nargs = '+')
parser.add_argument("--decay_rate", default=.2, type=float)

parser.add_argument("--arch", default='Mobilev2', type=str)

parser.add_argument("--Distillation", default="None", type=str,
                    help = 'Distillation method : Soft_logits, FitNet, AT, FSP, DML, KD-SVD, FT, AB, RKD')
parser.add_argument("--trained_param", default=None, type=str)

parser.add_argument("--gpu_id", default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':            
    conf = 0
    home_path = os.path.dirname(os.path.abspath(__file__))
    if args.decay_points is not None:
        decay_points = ''
        for dp in args.decay_points:
            decay_points += '%f '%dp
    else:
        decay_points = ''

    if conf == 0:
        for i in range(3):
            subprocess.call('python %s/train_w_pruning.py '%home_path
                           +' --train_path %s/%d'%(args.train_path,i)
                           +' --train_epoch %d'%(args.train_epoch)
                           +' --decay_points %s'%decay_points
                           +' --decay_rate %f'%args.decay_rate
 
                           +' --arch %s'%args.arch

                           +' --gpu_id %d'%args.gpu_id
                           ,shell=True)
            print ('Training Done')
        get_avg_plot(args.train_path)
        
