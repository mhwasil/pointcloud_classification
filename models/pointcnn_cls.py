from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pointfly as pf
import tensorflow as tf
from pointcnn import PointCNN


class Net(PointCNN):
    def __init__(self, points, features, is_training, setting):
        PointCNN.__init__(self, points, features, is_training, setting)
        fc_mean = tf.reduce_mean(self.fc_layers[-1], axis=1, keep_dims=True, name='fc_mean')
        self.fc_layers[-1] = tf.cond(is_training, lambda: self.fc_layers[-1], lambda: fc_mean)
        self.logits = pf.dense(self.fc_layers[-1], setting.num_class, 'logits',
                               is_training, with_bn=False, activation=None)
        self.end_points = {}
        
    def get_loss(self, labels_pl):
        probabilities = tf.nn.softmax(self.logits, name='probs')
        labels_2d = tf.expand_dims(labels_pl, axis=-1, name='labels_2d')
        labels_tile = tf.tile(labels_2d, (1, tf.shape(self.logits)[1]), name='labels_tile')
        total_loss = tf.reduce_mean(tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=self.logits))
        
        self.end_points["Logits"] = self.logits
        self.end_points["Probabilities"] = probabilities
        self.end_points["labels_tile"] = labels_tile
        
        return total_loss, self.end_points