""" Tensorflow train helper

Author: Mohammad Wasil
Date: April 2021

Some functionalities were adopted from PointNet
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils/pointcnn'))
sys.path.append(os.path.join(BASE_DIR, 'utils/pointnet'))
sys.path.append(os.path.join(BASE_DIR, 'utils/pointnet2'))

from argparse import Namespace
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import importlib
import argparse

import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from tensorflow.python.platform import tf_logging as logging

import tf_util
import visualization
import provider
import utils
import time

import threedmfv_net_cls as fv_model
import dgcnn
import pointcnn_cls

# tensor ops for pointnet, spidercnn needs to be recompiled
import spidercnn_cls_xyz as spidercnn
import pointnet_cls as pointnet
import pointnet2_cls_ssg as pointnet2_ssg

import pointfly as pf
from pointcnn_cls import Net
import modelnet_x3_l4 as setting

slim = tf.contrib.slim 

class PointCloudClassification(object):
    
    def __init__(self, model, config_file, train=True):
        self.model = model
        self.config = utils.get_yaml_contents(config_file)
        self.is_train = train
        
        self.exclude_blocks = ['Mixed_1', 'Mixed_2', 'Mixed_3', 'Mixed_4', 'Mixed_5','Mixed_6',
           'maxpool4', 'maxpool7', 'inception_fc0','inception_fc1',
           'dp0_inception_fc','dp1_inception_fc','pc_fc0', 'pc_fc1',
           'dp0_pc_fc','dp1_pc_fc', 'dp_combined', 'fc1', 'dp1', 'logits', 'Logits', 'Predictions', 'Logits/biases',
            'agg','agg/biases','transform_net1','dgcnn1','dgcnn2','dgcnn3','dgcnn4','fc2','fc3','dp2']
        
        if self.model == "3DmFV":
            self.m = self.config['models']['3DmFV']['n_gaussians']
            self.gmm_variance = self.config['models']['3DmFV']['gmm_variance']
            self.gmm = utils.get_3d_grid_gmm(subdivisions=[self.m, self.m, self.m], variance=self.gmm_variance)    
            
        self.set_config()
        self.set_logdir()
        self.load_train_data()
        self.input_dim = 6 if self.model_config.pointcloud_color else 3
        self.load_model()
    
    def set_config(self):
        self.data_config = Namespace(**self.config['data'])
        self.train_config = Namespace(**self.config['train'])
        self.model_config = Namespace(**self.config['models'][self.model])
        
    def set_logdir(self):
        self.train_logdir = os.path.join(self.train_config.logdir_root, '{}/{}_train'.format(self.data_config.dataset_name,self.model))
        self.test_logdir = os.path.join(self.train_config.logdir_root, '{}/{}_test'.format(self.data_config.dataset_name,self.model))

        self.checkpoint_file = tf.train.latest_checkpoint(self.train_logdir)

        if not os.path.exists(self.train_logdir):
            os.makedirs(self.train_logdir)
        if not os.path.exists(self.test_logdir):
            os.makedirs(self.test_logdir)

        print("Initialization done.....................")
        
    def load_train_data(self):
        print("**** Loading dataset....................")
        train_files = provider.get_data_files(os.path.join(self.data_config.datadir_root, self.data_config.dataset_name, 'train_files.txt'))
        test_files = provider.get_data_files(os.path.join(self.data_config.datadir_root, self.data_config.dataset_name, 'test_files.txt'))
        self.label_map = provider.get_label_map(os.path.join(self.data_config.datadir_root, self.data_config.dataset_name, 'label_map.yaml'))
        
        train_file_idxs = np.arange(0, len(train_files))
        np.random.shuffle(train_file_idxs)
        
        if self.is_train == True:
            pointcloud_data = []
            feature_data = []
            image_data = []
            labels = []
            for fn in range(len(train_files)):
                print (train_files[train_file_idxs[fn]])
                if ".h5" in train_files[train_file_idxs[fn]]:
                    pointcloud, image, mask_rgb, feature, label = provider.load_h5(train_files[train_file_idxs[fn]], 
                                                                       cloud_color=self.model_config.pointcloud_color, load_feature=False)
                elif ".pgz" in train_files[train_file_idxs[fn]]:
                    pointcloud, label = provider.load_pickle_file_with_label(train_files[train_file_idxs[fn]], 
                                                                             compressed=True, cloud_color=self.model_config.pointcloud_color)

                pointcloud, label, idx = provider.shuffle_data(pointcloud, np.squeeze(label))
                label = np.squeeze(label)

                pointcloud_data.extend(pointcloud)
                labels.extend(label)

            print('**** Train dataset loaded....................')
            self.train_pointcloud_data = np.asarray(pointcloud_data)
            self.train_labels = np.asarray(labels)

        print("**** Loading test dataset....................")
        test_pointcloud_data = []
        test_image_data = []
        test_feature_data = []
        test_labels = []
        for fn in range(len(test_files)):
            if ".h5" in test_files[fn]:
                pointcloud, image, mask_rgb, feature, label = provider.load_h5(test_files[fn], cloud_color=self.model_config.pointcloud_color,
                                                                              load_feature=False)
            elif ".pgz" in test_files[fn]:
                pointcloud, label = provider.load_pickle_file_with_label(test_files[fn], compressed=True, 
                                                                         cloud_color=self.model_config.pointcloud_color)

            label = np.squeeze(label)

            test_pointcloud_data.extend(pointcloud)
            test_labels.extend(label)

        test_pointcloud_data = np.asarray(test_pointcloud_data)
        test_labels = np.asarray(test_labels)
        test_pointcloud_data, test_labels, idx = provider.shuffle_data(test_pointcloud_data, test_labels)
 
        self.test_data = {}
        self.test_data['pointcloud_data'] = test_pointcloud_data
        self.test_data['labels'] = test_labels
        
    def load_model(self):
        with tf.Graph().as_default():
            points_pl = tf.compat.v1.placeholder(tf.float32, 
                                       [self.train_config.batch_size, self.data_config.num_points, self.input_dim])

            labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(self.train_config.batch_size))
            is_training_pl = tf.compat.v1.placeholder(tf.bool)

            one_hot_labels = tf.one_hot(indices=labels_pl, depth=self.data_config.num_classes)

            global_step = tf.compat.v1.train.get_or_create_global_step()
            batch = tf.Variable(0)    

            bn_decay = tf_util.get_bn_decay(global_step, #batch
                                            self.model_config.bn_init_decay, 
                                            self.train_config.batch_size, 
                                            self.model_config.bn_decay_decay_step, 
                                            self.model_config.bn_decay_decay_rate, 
                                            self.model_config.bn_decay_clip)
            
            learning_rate = tf_util.get_learning_rate(global_step, #batch
                                                      self.model_config.base_learning_rate, 
                                                      self.train_config.batch_size, 
                                                      self.model_config.decay_step, 
                                                      self.model_config.decay_rate)

            print('**** Model selected  -> {} ****\n'.format(self.model))

            if self.model == "3DmFV":
                w_pl = tf.compat.v1.placeholder(tf.float32, shape=(self.gmm.means_.shape[0]))
                mu_pl = tf.compat.v1.placeholder(tf.float32, shape=(self.gmm.means_.shape[0], self.gmm.means_.shape[1]))
                sigma_pl = tf.compat.v1.placeholder(tf.float32, shape=(self.gmm.means_.shape[0], self.gmm.means_.shape[1]))

                logits, end_points = fv_model.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, 
                            bn_decay=bn_decay, weigth_decay=self.model_config.weight_decay, 
                            add_noise=self.model_config.add_gaussian_noise, 
                            num_classes=self.data_config.num_classes)
                total_loss  = fv_model.get_loss(logits, labels_pl)

            elif self.model == "DGCNN" or self.model == "DGCNNC":
                logits, end_points = dgcnn.get_model(points_pl, self.data_config.num_classes, 
                                                     is_training_pl, bn_decay=bn_decay, 
                                                     color=True if self.model == "DGCNNC" else False)
                total_loss  = dgcnn.get_loss(logits, labels_pl, num_classes=self.data_config.num_classes)

            elif self.model == "SpiderCNN":
                logits, end_points = spidercnn.get_model(points_pl, is_training_pl, bn_decay=bn_decay, num_class=self.data_config.num_classes)
                total_loss  = spidercnn.get_loss(logits, labels_pl)

            elif self.model == "PointNet":
                logits, end_points = pointnet.get_model(points_pl, is_training_pl, bn_decay=bn_decay, num_class=self.data_config.num_classes)
                total_loss  = pointnet.get_loss(logits, labels_pl, end_points)

            elif self.model == "PointNet2":
                logits, end_points = pointnet2_ssg.get_model(points_pl, is_training_pl, bn_decay=bn_decay, num_class=self.data_config.num_classes)
                total_loss  = pointnet2_ssg.get_loss(logits, labels_pl, end_points)
                
            elif self.model == "PointCNN":
                
                xforms = tf.compat.v1.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
                rotations = tf.compat.v1.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
                jitter_range = tf.compat.v1.placeholder(tf.float32, shape=(1), name="jitter_range")
                points_augmented = pf.augment(points_pl, xforms, jitter_range)
                net = Net(points=points_augmented, features=None, is_training=is_training_pl, setting=setting)
                
            if self.is_train:
                variables_to_restore = slim.get_variables_to_restore(exclude=self.exclude_blocks) 
            else:
                variables_to_restore = slim.get_variables_to_restore()
            
            if self.model == "PointCNN":
                total_loss, end_points = net.get_loss(labels_pl)
                probabilities = end_points['Probabilities']
                predictions = tf.argmax(probabilities, axis=-1, name='predictions')
                correct = tf.equal(predictions, tf.to_int64(labels_pl))
                
                with tf.name_scope('metrics'):
                    loss_mean_op, loss_mean_update_op = tf.compat.v1.metrics.mean(total_loss)
                    accuracy, update_op = tf.compat.v1.metrics.accuracy(end_points['labels_tile'], predictions)
                    t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.compat.v1.metrics.mean_per_class_accuracy(labels_pl,
                                                                                                           predictions,
                                                                                                           self.data_config.num_classes)
                reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                             if var.name.split('/')[0] == 'metrics'])
                metrics_op = tf.group(update_op, probabilities)
                        
                bn_decay = tf.compat.v1.train.exponential_decay(setting.learning_rate_base, 
                                                                global_step, 
                                                                setting.decay_steps,
                                                                setting.decay_rate, 
                                                                staircase=True)
                
                learning_rate = tf.maximum(bn_decay, setting.learning_rate_min)
                tf.summary.scalar('learning_rate', tensor=learning_rate, collections=['train'])
                reg_loss = setting.weight_decay * tf.compat.v1.losses.get_regularization_loss()
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=setting.epsilon)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(total_loss + reg_loss, global_step=global_step)
            else:
                #predictions that is not one_hot_encoded.
                probabilities = end_points['Probabilities']
                predictions = tf.argmax(probabilities, 1)
                correct = tf.equal(predictions, tf.to_int64(labels_pl))
                accuracy, update_op = tf.compat.v1.metrics.accuracy(labels_pl, predictions)
                metrics_op = tf.group(update_op, probabilities)
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(total_loss, global_step=global_step)
                tf.compat.v1.summary.scalar('learning_rate', learning_rate)

            tf.compat.v1.summary.scalar('bn_decay', bn_decay)
            tf.compat.v1.summary.scalar('loss', total_loss)
            tf.compat.v1.summary.scalar('accuracy', accuracy)    
            summary_op = tf.compat.v1.summary.merge_all()

            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver(variables_to_restore)

            self.ops = {'labels_pl': labels_pl,
                   'points_pl': points_pl,
                   'w_pl': w_pl if self.model == "3DmFV" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                   'mu_pl': mu_pl if self.model == "3DmFV" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                   'sigma_pl': sigma_pl if self.model == "3DmFV" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                   'is_training_pl': is_training_pl,
                   'loss': total_loss,
                   'train_op': train_op,
                   'summary_op': summary_op,
                   'metrics_op': metrics_op,
                   'predictions': predictions,
                   'probabilities':probabilities,
                   'step': batch,
                   'global_step': global_step,
                   'accuracy': accuracy,
                   'correct': correct,
                   'xforms': xforms if self.model == "PointCNN" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                   'rotations': rotations if self.model == "PointCNN" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                   'jitter_range': jitter_range if self.model == "PointCNN" else tf.compat.v1.placeholder(tf.float16, shape=(1)),
                  }

            def restore_fn(sess):
                if self.is_train:
                    if self.checkpoint_file is not None:
                        return saver.restore(sess, self.checkpoint_file)
                    else:
                        return None
                else:
                    return saver.restore(sess, self.checkpoint_file)

            if self.is_train:
                self.sv = tf.train.Supervisor(logdir = self.train_logdir, summary_op = None, init_fn = restore_fn)
            else:
                self.sv = tf.train.Supervisor(logdir = self.test_logdir, saver = None, summary_op = None, init_fn = restore_fn)

            sess_config = tf_util.get_sess_conf(self.train_config.gpu_selection, limit_gpu=self.train_config.limit_gpu)
            self.sv.PrepareSession(config=sess_config)  
            with self.sv.managed_session() as self.sess:
                if self.is_train:
                    self.train()
                else:
                    self.evaluate(export=True)


    def train(self):
        print("**** Start training ")
        is_training = True

        print('**** Training with max epoch %03d' %(self.train_config.max_epoch))
        for epoch in range(self.train_config.max_epoch):
            pointcloud_data, labels, idx = provider.shuffle_data(self.train_pointcloud_data, self.train_labels)

            file_size = pointcloud_data.shape[0]
            num_batches = file_size / self.train_config.batch_size

            curr_step = 0
            total_correct = 0
            total_seen = 0
            loss_sum = 0

            for batch_idx in range(int(num_batches)):
                start_idx = batch_idx * self.train_config.batch_size
                end_idx = (batch_idx + 1) * self.train_config.batch_size

                points_batch = pointcloud_data[start_idx:end_idx, ...]

                if self.data_config.augment_scale:
                    points_batch = provider.scale_point_cloud(points_batch, smin=0.66, smax=1.5)
                if self.data_config.augment_rotation:
                    points_batch = provider.rotate_point_cloud(points_batch)
                if self.data_config.augment_translation:
                    points_batch = provider.translate_point_cloud(points_batch, tval = 0.2)
                if self.data_config.augment_jitter:
                    points_batch = provider.jitter_point_cloud(points_batch, sigma=0.01,clip=0.05)  
                if self.data_config.augment_outlier:
                    points_batch = provider.insert_outliers_to_point_cloud(points_batch, outlier_ratio=0.02)

                points_batch = utils.scale_to_unit_sphere(points_batch)
                label_batch = labels[start_idx:end_idx]
                
                xforms_np, rotations_np = pf.get_xforms(self.train_config.batch_size,
                                                rotation_range=setting.rotation_range,
                                                scaling_range=setting.scaling_range,
                                                order=setting.rotation_order)

                feed_dict = {self.ops['points_pl']: points_batch,
                             self.ops['labels_pl']: label_batch,
                             self.ops['w_pl']: self.gmm.weights_ if self.model == "3DmFV" else [1],
                             self.ops['mu_pl']: self.gmm.means_ if self.model == "3DmFV" else [1],
                             self.ops['sigma_pl']: np.sqrt(self.gmm.covariances_ if self.model == "3DmFV" else [1]),
                             self.ops['is_training_pl']: is_training,
                             self.ops['xforms']: xforms_np if self.model == "PointCNN" else [1],
                             self.ops['rotations']: rotations_np if self.model == "PointCNN" else [1],
                             self.ops['jitter_range']: np.array([setting.jitter] if self.model == "PointCNN" else [1])
                            }
                
                #Log the summaries every 100 step.
                if curr_step % 10 == 0 and curr_step > 0:
                    summary, step, gstep, _top, _mop, loss_val, pred_val = self.sess.run([self.ops['summary_op'], self.ops['step'],
                                                                         self.ops['global_step'], self.ops['train_op'], self.ops['metrics_op'],
                                                                         self.ops['loss'], self.ops['predictions']], feed_dict=feed_dict)

                    self.sv.summary_computed(self.sess, summary)
                else:

                    step, gstep, _top, _mop, loss_val, pred_val = self.sess.run([self.ops['step'],self.ops['global_step'],self.ops['train_op'], 
                                                                            self.ops['metrics_op'],self.ops['loss'],
                                                                            self.ops['predictions']], feed_dict=feed_dict)

                    if curr_step % 100 == 0 or curr_step % 75 == 0:
                        print('global step {}: loss: {} '.format(gstep, loss_val))

                correct = np.sum(pred_val == label_batch)

                total_correct += correct
                total_seen += self.train_config.batch_size
                loss_sum += loss_val

                curr_step += 1

            #evaluate
            if epoch % 2 == 0 or epoch == self.train_config.max_epoch-1:
                acc, acc_avg_cls = self.evaluate()                
                
    def plot_batch_image(self, image_batch, predictions, label_batch, probs):
        fig, ax = plt.subplots(nrows=8, ncols=4, figsize=(16,16))
        fig.tight_layout()
        row = 0
        col = 0
        for i in range(1,image_batch.shape[0]+1):
            idx = i-1
            prediction_name = self.label_map[predictions[idx]]
            true_name = self.label_map[label_batch[idx]]
            probability = np.max(probs[idx])
            img = image_batch[idx]
            text = 'Prediction: %s (%.2f) \n Ground truth: %s' %(prediction_name,probability,true_name)
            ax[row,col].set_title(text)
            img_plot = ax[row,col].imshow(img.astype('uint8'))
            img_plot.axes.get_yaxis().set_ticks([])
            img_plot.axes.get_xaxis().set_ticks([])
            if i%4==0:
                row += 1
                col = 0
            else:
                col += 1

        plt.tight_layout()
        plt.show()

    def evaluate(self, data=None, export=False):
        print("Running evaluation....................")

        is_training = False
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0] * self.data_config.num_classes
        total_correct_class = [0] * self.data_config.num_classes

        points_idx = range(self.data_config.num_points)
        current_step = 0

        pointcloud_data = self.test_data['pointcloud_data']
        labels = self.test_data['labels']

        true_labels = []
        all_pred_labels = []

        file_size = pointcloud_data.shape[0]
        num_batches = file_size / self.train_config.batch_size

        for batch_idx in range(int(num_batches)):
            start_idx = batch_idx * self.train_config.batch_size
            end_idx = (batch_idx + 1) * self.train_config.batch_size

            points_batch = pointcloud_data[start_idx:end_idx, ...]
            points_batch = utils.scale_to_unit_sphere(points_batch)

            label_batch = labels[start_idx:end_idx]

            xforms_np, rotations_np = pf.get_xforms(self.train_config.batch_size,
                                                rotation_range=setting.rotation_range,
                                                scaling_range=setting.scaling_range,
                                                order=setting.rotation_order)

            feed_dict = {self.ops['points_pl']: points_batch,
                         self.ops['labels_pl']: label_batch,
                         self.ops['w_pl']: self.gmm.weights_ if self.model == "3DmFV" else [1],
                         self.ops['mu_pl']: self.gmm.means_ if self.model == "3DmFV" else [1],
                         self.ops['sigma_pl']: np.sqrt(self.gmm.covariances_ if self.model == "3DmFV" else [1]),
                         self.ops['is_training_pl']: is_training,
                         self.ops['xforms']: xforms_np if self.model == "PointCNN" else [1],
                         self.ops['rotations']: rotations_np if self.model == "PointCNN" else [1],
                         self.ops['jitter_range']: np.array([setting.jitter]) if self.model == "PointCNN" else [1]
                        }

            summary,_,loss_val, pred_val, probs, accuracy = self.sess.run([self.ops['summary_op'],self.ops['metrics_op'],self.ops['loss'], 
                                                                      self.ops['predictions'], self.ops['probabilities'],
                                                                      self.ops['accuracy']], feed_dict=feed_dict)
            self.sv.summary_computed(self.sess, summary)
            
            correct = np.sum(pred_val == label_batch)
            true_labels.extend(label_batch) 
            all_pred_labels.extend(pred_val)

            total_correct += correct
            total_seen += self.train_config.batch_size
            loss_sum += (loss_val * self.train_config.batch_size)

            for i in range(start_idx, end_idx):
                l = labels[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)[0] if self.model == "PointCNN" else (pred_val[i - start_idx] == l)

            current_step += 1
        
        total_correct_class
        acc = total_correct / float(total_seen)
        acc_per_class = np.asarray(total_correct_class) / np.asarray(total_seen_class)
        acc_avg_cls =  np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        
        print("current eval accuracy: ", accuracy)
        print("correct ", total_correct_class)
        print("seen ", total_seen_class)
        print("acc per class ", acc_per_class)
        
        print('eval mean loss: %f' % (loss_sum / float(total_seen)))
        print('eval accuracy: %f' % (acc))
        print('eval avg class acc: %f' % (acc_avg_cls))

        if export:
            true_labels = np.asarray(true_labels)
            all_pred_labels = np.asarray(all_pred_labels)

            label_map = []
            for i,label in enumerate(self.label_map.values()):
                if "_" in label:
                    label_split = label.split("_")
                    label_map.append("-".join(label_split))
                else:
                    label_map.append(label)

            label_map = np.asarray(label_map)
        
            visualization.visualize_confusion_matrix(true_labels, all_pred_labels, classes=label_map,
                                        normalize=True, export=True,display=False, 
                                        filename=os.path.join('./log/images/','confusion_mat_{}_{}'.format(self.data_config.dataset_name,
                                                                                                          self.model)), 
                                        n_classes=self.data_config.num_classes, acc_fontsize=10, label_fontsize=12)

            visualization.visualize_histogram(total_correct_class, total_seen_class,label_map, self.model,
                                             filepath=os.path.abspath('./log/images/histogram_{}_{}.pdf'.format(self.data_config.dataset_name,
                                                                                                            self.model)))

        return (acc, acc_avg_cls)