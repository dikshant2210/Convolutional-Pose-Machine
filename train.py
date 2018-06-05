import tensorflow as tf
from model import ConvolutionalPoseMachine as CPM
import os
from data_utils import train_generator, valid_generator
from utils import EPOCHS


def main(argv):
    train_log_save_dir = os.path.join('model', 'logs', 'train')
    test_log_save_dir = os.path.join('model', 'logs', 'test')
    os.system('mkdir -p {}'.format(train_log_save_dir))
    os.system('mkdir -p {}'.format(test_log_save_dir))

    model = CPM(stages=3, joints=16)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_log_save_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_save_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=None)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        print('Starting training...')
        for epoch in range(EPOCHS):
            loss = 0
            count = 0
            try:
                for x, y in train_generator():
                    stage_loss, total_loss, train_op, summaries, heatmaps, \
                        global_step = sess.run([model.stage_loss,
                                               model.total_loss,
                                               model.train_op,
                                               merged_summary,
                                               model.heatmaps,
                                               model.global_step],
                                               feed_dict={model.images: x, model.true_heatmaps: y})
                    train_writer.add_summary(summaries, global_step)
                    print('\tEpoch: {}, Iteration: {}, Total loss: {}'.format(epoch + 1, count+1, total_loss))
                    loss += total_loss
                    count += 1
                    if count % 100 == 0:
                        print('Save condition reached!')
                        saver.save(sess=sess, save_path='model/weights/model.ckpt', global_step=global_step+1)
            except ValueError:
                pass
            print('Epoch: {}, Total loss: {}'.format(epoch, loss / count))


if __name__ == '__main__':
    tf.app.run()