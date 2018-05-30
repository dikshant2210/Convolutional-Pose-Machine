import tensorflow as tf
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS


class ConvolutionalPoseMachine:
    def __init__(self, joints, stages, batch_size=32):
        self.images = []
        self.true_heatmaps = []
        self.image_feature_map = []
        self.joints = joints
        self.heatmaps = []
        self.stages = stages
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.batch_size = batch_size
        self.train_op = 0
        self.merged_summary = 0

    def build_model(self):
        images = tf.placeholder(tf.float32, (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

        with tf.variable_scope('sub_stage'):
            sub_conv1 = tf.layers.conv2d(inputs=images,
                                         filters=64,
                                         kernel_size=[9, 9],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv1,
                                                pool_size=[2, 2],
                                                strides=2,
                                                name='sub_pool1')
            sub_conv2 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[9, 9],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                name='sub_pool2')
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[9, 9],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv3,
                                                pool_size=[2, 2],
                                                strides=2,
                                                name='sub_pool3')
            self.image_feature_map = tf.layers.conv2d(inputs=sub_pool3,
                                                      filters=512,
                                                      kernel_size=[5, 5],
                                                      padding='same',
                                                      activation=tf.nn.relu,
                                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                      name='sub_conv4')

        with tf.variable_scope('stage1'):
            conv1 = tf.layers.conv2d(inputs=self.image_feature_map,
                                     filters=512,
                                     kernel_size=[9, 9],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=1024,
                                     kernel_size=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv2')
            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=self.joints,
                                     kernel_size=[1, 1],
                                     padding='same',
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv3')
            self.heatmaps.append(conv3)

        for stage in self.stages:
            self.intermediate_stage(stage)

    def intermediate_stage(self, stage):
        with tf.variable_scope('stage{}'.format(stage)):
            current_featuremap = tf.concat([self.heatmaps[stage - 2], self.image_feature_map], axis=3)

            mid_conv1 = tf.layers.conv2d(inputs=current_featuremap,
                                         filters=128,
                                         kernel_size=[11, 11],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[11, 11],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[11, 11],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=self.joints,
                                         kernel_size=[1, 1],
                                         padding='same',
                                         activation=tf.nn.sigmoid,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')
            self.heatmaps.append(mid_conv5)

    def build_loss(self, gt_heatmap, lr):
        self.true_heatmaps = gt_heatmap
        self.total_loss = 0

        for stage in range(self.stages):
            with tf.variable_scope('stage{}_loss'.format(stage + 1)):
                self.stage_loss[stage] = tf.nn.l2_loss(self.heatmaps[stage] - self.true_heatmaps,
                                                       name='l2_loss') / self.batch_size
                tf.summary.scalar('stage{}_loss'.format(stage + 1), self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in self.stages:
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total_loss',self.total_loss)

        with tf.variable_scope('train'):
            global_step = tf.contrib.framework.get_or_create_global_step()
            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=global_step,
                                                            learning_rate=lr,
                                                            optimizer='Adam')
        self.merged_summary = tf.summary.merge_all()
