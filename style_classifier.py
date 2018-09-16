import numpy as np
import tensorflow as tf
from random import shuffle
from collections import namedtuple
from module import *
from ops import *
from utils import *
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']


class Classifer(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_dir = args.dataset_dir
        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.sample_dir = args.sample_dir
        self.batch_size = args.batch_size
        self.image_size = args.fine_size  # cropped size
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.input_c_dim = args.input_nc  # number of input image channels
        self.sigma_c = args.sigma_c
        self.sigma_d = args.sigma_d
        self.model = args.model

        self.generator = generator_resnet
        self.discriminator = discriminator_classifier
        self.criterionGAN = softmax_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size '
                                        'image_size '
                                        'gf_dim '
                                        'df_dim '
                                        'output_c_dim '
                                        'is_training')
        self.options = OPTIONS._make((args.batch_size,
                                      args.fine_size,
                                      args.ngf,
                                      args.ndf,
                                      args.output_nc,
                                      args.phase == 'train'))
        self._build_model()
        self.now_datetime = get_now_datetime()
        self.saver = tf.train.Saver()

    def _build_model(self):

        # define some placeholders
        self.origin_train = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                        self.input_c_dim])
        self.label_train = tf.placeholder(tf.float32, [self.batch_size, 2])
        self.origin_test = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range, self.input_c_dim])
        self.label_test = tf.placeholder(tf.float32, [None, 2])

        # Origin samples passed through the classifier
        self.D_origin = self.discriminator(self.origin_train, self.options, False, name='classifier')
        self.D_test = self.discriminator(self.origin_test, self.options, True, name='classifier')

        # Discriminator loss
        self.d_loss = self.criterionGAN(self.D_origin, self.label_train)
        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

        # test accuracy
        self.D_test_softmax = tf.nn.softmax(self.D_test)
        self.correct_prediction_test = tf.equal(tf.argmax(self.D_test_softmax, 1), tf.argmax(self.label_test, 1))
        self.accuracy_test = tf.reduce_mean(tf.cast(self.correct_prediction_test, tf.float32))

        # test midi
        self.test_midi = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range, self.input_c_dim])
        self.test_result = self.discriminator(self.test_midi, self.options, True, name='classifier')
        self.test_result_softmax = tf.nn.softmax(self.test_result)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'classifier' in var.name]
        for var in t_vars:
            print(var.name)

    def train(self, args):

        # learning rate
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        # Optimizer
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # log path
        log_dir = './logs/classifier_{}2{}_{}_{}'.format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime,
                                                         str(self.sigma_c))
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)
        counter = 1

        # create training list (origin data with corresponding label)
        # Label for A is (1, 0), for B is (0, 1)
        dataA = glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir))
        labelA = [(1.0, 0.0) for _ in range(len(dataA))]
        labelB = [(0.0, 1.0) for _ in range(len(dataB))]
        data_origin = dataA + dataB
        label_origin = labelA + labelB
        training_list = []
        for pair in zip(data_origin, label_origin):
            training_list.append(pair)
        print('Successfully create training list!')

        # create test list (origin data with corresponding label)
        dataA = glob('./datasets/{}/test/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/test/*.*'.format(self.dataset_B_dir))
        labelA = [(1.0, 0.0) for _ in range(len(dataA))]
        labelB = [(0.0, 1.0) for _ in range(len(dataB))]
        data_origin = dataA + dataB
        label_origin = labelA + labelB
        testing_list = []
        for pair in zip(data_origin, label_origin):
            testing_list.append(pair)
        print('Successfully create testing list!')

        data_test = [np.load(pair[0]) * 2. - 1. for pair in testing_list]
        data_test = np.array(data_test).astype(np.float32)
        gaussian_noise = np.random.normal(0, self.sigma_c, [data_test.shape[0], data_test.shape[1],
                                                          data_test.shape[2], data_test.shape[3]])
        data_test += gaussian_noise
        # data_test = data_test.reshape(data_test.shape[0], data_test.shape[1], data_test.shape[2], 1)
        label_test = [pair[1] for pair in testing_list]
        label_test = np.array(label_test).astype(np.float32).reshape(len(label_test), 2)

        for epoch in range(args.epoch):

            # shuffle the training samples
            shuffle(training_list)

            # get the correct batch number
            batch_idx = len(training_list) // self.batch_size

            # learning rate would decay after certain epochs
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch-epoch) / (args.epoch-args.epoch_step)

            for idx in range(batch_idx):

                # data samples in batch
                batch = training_list[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_data = [np.load(pair[0]) * 2. - 1. for pair in batch]
                batch_data = np.array(batch_data).astype(np.float32)

                # data labels in batch
                batch_label = [pair[1] for pair in batch]
                batch_label = np.array(batch_label).astype(np.float32).reshape(len(batch_label), 2)

                # update classifier network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_loss_sum, self.d_loss],
                                                       feed_dict={self.origin_train: batch_data,
                                                                  self.label_train: batch_label, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                counter += 1

            self.save(args.checkpoint_dir, epoch)

            # test the classifier accuracy based on testing list
            accuracy_test = self.sess.run(self.accuracy_test, feed_dict={self.origin_test: data_test,
                                                                         self.label_test: label_test})
            print('epoch:', epoch, 'testing accuracy:', accuracy_test, 'loss:', d_loss)

    def save(self, checkpoint_dir, step):
        model_name = "classifier.model"
        model_dir = "classifier_{}2{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime,
                                                    str(self.sigma_c))
        # model_dir = "classifier_{}2{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, '2018-06-08',
        #                                             str(self.sigma_c))
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "classifier_{}2{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime,
                                                    str(self.sigma_c))
        # model_dir = "classifier_{}2{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, '2018-06-08',
        #                                             str(self.sigma_c))
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # load the origin samples in npy format and sorted in ascending order
        sample_files_origin = glob('./test/{}2{}_{}_{}_{}/{}/npy/origin/*.*'.format(self.dataset_A_dir,
                                                                                    self.dataset_B_dir,
                                                                                    self.model,
                                                                                    self.sigma_d,
                                                                                    self.now_datetime,
                                                                                    args.which_direction))
        sample_files_origin.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_transfer = glob('./test/{}2{}_{}_{}_{}/{}/npy/transfer/*.*'.format(self.dataset_A_dir,
                                                                                        self.dataset_B_dir,
                                                                                        self.model,
                                                                                        self.sigma_d,
                                                                                        self.now_datetime,
                                                                                        args.which_direction))
        sample_files_transfer.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # load the origin samples in npy format and sorted in ascending order
        sample_files_cycle = glob('./test/{}2{}_{}_{}_{}/{}/npy/cycle/*.*'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  self.now_datetime,
                                                                                  args.which_direction))
        sample_files_cycle.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[0]))

        # put the origin, transfer and cycle of the same phrase in one zip
        sample_files = list(zip(sample_files_origin, sample_files_transfer, sample_files_cycle))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # create a test path to store the generated sample midi files attached with probability
        test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid_attach_prob'.format(self.dataset_A_dir,
                                                                                              self.dataset_B_dir,
                                                                                              self.model,
                                                                                              self.sigma_d,
                                                                                              self.now_datetime,
                                                                                              args.which_direction))
        if not os.path.exists(test_dir_mid):
            os.makedirs(test_dir_mid)

        count_origin = 0
        count_transfer = 0
        count_cycle = 0
        line_list = []

        for idx in range(len(sample_files)):
            print('Classifying midi: ', sample_files[idx])

            # load sample phrases in npy formats
            sample_origin = np.load(sample_files[idx][0])
            sample_transfer = np.load(sample_files[idx][1])
            sample_cycle = np.load(sample_files[idx][2])

            # get the probability for each sample phrase
            test_result_origin = self.sess.run(self.test_result_softmax,
                                               feed_dict={self.test_midi: sample_origin * 2. - 1.})
            test_result_transfer = self.sess.run(self.test_result_softmax,
                                                 feed_dict={self.test_midi: sample_transfer * 2. - 1.})
            test_result_cycle = self.sess.run(self.test_result_softmax,
                                              feed_dict={self.test_midi: sample_cycle * 2. - 1.})

            origin_transfer_diff = np.abs(test_result_origin - test_result_transfer)
            content_diff = np.mean((sample_origin * 1.0 - sample_transfer * 1.0)**2)

            # labels: (1, 0) for A, (0, 1) for B
            if args.which_direction == 'AtoB':
                line_list.append((idx + 1, content_diff, origin_transfer_diff[0][0], test_result_origin[0][0],
                                  test_result_transfer[0][0], test_result_cycle[0][0]))

                # for the accuracy calculation
                count_origin += 1 if np.argmax(test_result_origin[0]) == 0 else 0
                count_transfer += 1 if np.argmax(test_result_transfer[0]) == 0 else 0
                count_cycle += 1 if np.argmax(test_result_cycle[0]) == 0 else 0

                # create paths for origin, transfer and cycle samples attached with probability
                path_origin = os.path.join(test_dir_mid, '{}_origin_{}.mid'.format(idx + 1, test_result_origin[0][0]))
                path_transfer = os.path.join(test_dir_mid, '{}_transfer_{}.mid'.format(idx + 1, test_result_transfer[0][0]))
                path_cycle = os.path.join(test_dir_mid, '{}_cycle_{}.mid'.format(idx + 1, test_result_cycle[0][0]))

            else:
                line_list.append((idx + 1, content_diff, origin_transfer_diff[0][1], test_result_origin[0][1],
                                  test_result_transfer[0][1], test_result_cycle[0][1]))

                # for the accuracy calculation
                count_origin += 1 if np.argmax(test_result_origin[0]) == 1 else 0
                count_transfer += 1 if np.argmax(test_result_transfer[0]) == 1 else 0
                count_cycle += 1 if np.argmax(test_result_cycle[0]) == 1 else 0

                # create paths for origin, transfer and cycle samples attached with probability
                path_origin = os.path.join(test_dir_mid, '{}_origin_{}.mid'.format(idx + 1, test_result_origin[0][1]))
                path_transfer = os.path.join(test_dir_mid, '{}_transfer_{}.mid'.format(idx + 1, test_result_transfer[0][1]))
                path_cycle = os.path.join(test_dir_mid, '{}_cycle_{}.mid'.format(idx + 1, test_result_cycle[0][1]))

            # generate sample MIDI files
            save_midis(sample_origin, path_origin)
            save_midis(sample_transfer, path_transfer)
            save_midis(sample_cycle, path_cycle)

        # sort the line_list based on origin_transfer_diff and write to a ranking txt file
        line_list.sort(key=lambda x: x[2], reverse=True)
        with open(os.path.join(test_dir_mid, 'Rankings_{}.txt'.format(args.which_direction)), 'w') as f:
            f.write('Id  Content_diff  P_O - P_T  Prob_Origin  Prob_Transfer  Prob_Cycle')
            for i in range(len(line_list)):
                f.writelines("\n%5d %5f %5f %5f %5f %5f" % (line_list[i][0], line_list[i][1], line_list[i][2],
                                                            line_list[i][3], line_list[i][4], line_list[i][5]))
        f.close()

        # calculate the accuracy
        accuracy_origin = count_origin * 1.0 / len(sample_files)
        accuracy_transfer = count_transfer * 1.0 / len(sample_files)
        accuracy_cycle = count_cycle * 1.0 / len(sample_files)
        print('Accuracy of this classifier on test datasets is :', accuracy_origin, accuracy_transfer, accuracy_cycle)

    def test_famous(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        song_o = np.load('./datasets/famous_songs/C2J/merged_npy/Scenes from Childhood (Schumann).npy')
        song_t = np.load('./datasets/famous_songs/C2J/transfer/Scenes from Childhood (Schumann).npy')
        print(song_o.shape, song_t.shape)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        sum_o_A = 0
        sum_o_B = 0
        sum_t_A = 0
        sum_t_B = 0
        for idx in range(song_t.shape[0]):
            phrase_o = song_o[idx]
            phrase_o = phrase_o.reshape(1, phrase_o.shape[0], phrase_o.shape[1], 1)
            origin = self.sess.run(self.test_result_softmax, feed_dict={self.test_midi: phrase_o * 2. - 1.})
            phrase_t = song_t[idx]
            phrase_t = phrase_t.reshape(1, phrase_t.shape[0], phrase_t.shape[1], 1)
            transfer = self.sess.run(self.test_result_softmax, feed_dict={self.test_midi: phrase_t * 2. - 1.})

            sum_o_A += origin[0][0]
            sum_o_B += origin[0][1]
            sum_t_A += transfer[0][0]
            sum_t_B += transfer[0][1]

        print("origin, source:", sum_o_A / song_t.shape[0], "target:", sum_o_B / song_t.shape[0])
        print("transfer, source:", sum_t_A / song_t.shape[0], "target:", sum_t_B / song_t.shape[0])

