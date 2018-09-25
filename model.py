from __future__ import division
import os
import time
from shutil import copyfile
from glob import glob
import tensorflow as tf
import numpy as np
import config
from collections import namedtuple
from module import *
from utils import *
from ops import *
from metrics import *
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size  # cropped size
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.input_c_dim = args.input_nc  # number of input image channels
        self.output_c_dim = args.output_nc  # number of output image channels
        self.L1_lambda = args.L1_lambda
        self.gamma = args.gamma
        self.sigma_d = args.sigma_d
        self.dataset_dir = args.dataset_dir
        self.dataset_A_dir = args.dataset_A_dir
        self.dataset_B_dir = args.dataset_B_dir
        self.sample_dir = args.sample_dir

        self.model = args.model
        self.discriminator = discriminator
        self.generator = generator_resnet
        self.criterionGAN = mae_criterion

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
        self.saver = tf.train.Saver(max_to_keep=30)
        self.now_datetime = get_now_datetime()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):

        # define some placeholders
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                     self.input_c_dim + self.output_c_dim], name='real_A_and_B')
        if self.model != 'base':
            self.real_mixed = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='real_A_and_B_mixed')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.gaussian_noise = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                          self.input_c_dim], name='gaussian_noise')
        # Generator: A - B - A
        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        # Generator: B - A - B
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
        # to binary
        self.real_A_binary = to_binary(self.real_A, 0.5)
        self.real_B_binary = to_binary(self.real_B, 0.5)
        self.fake_A_binary = to_binary(self.fake_A, 0.5)
        self.fake_B_binary = to_binary(self.fake_B, 0.5)
        self.fake_A__binary = to_binary(self.fake_A_, 0.5)
        self.fake_B__binary = to_binary(self.fake_B_, 0.5)

        # Discriminator: Fake
        self.DB_fake = self.discriminator(self.fake_B + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A + self.gaussian_noise, self.options,
                                          reuse=False, name="discriminatorA")
        # Discriminator: Real
        self.DA_real = self.discriminator(self.real_A + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorA")
        self.DB_real = self.discriminator(self.real_B + self.gaussian_noise, self.options, reuse=True,
                                          name="discriminatorB")

        self.fake_A_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32, [self.batch_size, self.time_step, self.pitch_range,
                                                         self.input_c_dim], name='fake_B_sample')
        self.DA_fake_sample = self.discriminator(self.fake_A_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample + self.gaussian_noise,
                                                 self.options, reuse=True, name="discriminatorB")
        if self.model != 'base':
            # Discriminator: All
            self.DA_real_all = self.discriminator(self.real_mixed + self.gaussian_noise, self.options, reuse=False,
                                                  name="discriminatorA_all")
            self.DA_fake_sample_all = self.discriminator(self.fake_A_sample + self.gaussian_noise,
                                                         self.options, reuse=True, name="discriminatorA_all")
            self.DB_real_all = self.discriminator(self.real_mixed + self.gaussian_noise, self.options, reuse=False,
                                                  name="discriminatorB_all")
            self.DB_fake_sample_all = self.discriminator(self.fake_B_sample + self.gaussian_noise,
                                                         self.options, reuse=True, name="discriminatorB_all")
        # Generator loss
        self.cycle_loss = self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) + self.cycle_loss
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) + self.cycle_loss
        self.g_loss = self.g_loss_a2b + self.g_loss_b2a - self.cycle_loss
        # Discriminator loss
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        if self.model != 'base':
            self.db_all_loss_real = self.criterionGAN(self.DB_real_all, tf.ones_like(self.DB_real_all))
            self.db_all_loss_fake = self.criterionGAN(self.DB_fake_sample_all, tf.zeros_like(self.DB_fake_sample_all))
            self.db_all_loss = (self.db_all_loss_real + self.db_all_loss_fake) / 2
            self.da_all_loss_real = self.criterionGAN(self.DA_real_all, tf.ones_like(self.DA_real_all))
            self.da_all_loss_fake = self.criterionGAN(self.DA_fake_sample_all, tf.zeros_like(self.DA_fake_sample_all))
            self.da_all_loss = (self.da_all_loss_real + self.da_all_loss_fake) / 2
            self.d_all_loss = self.da_all_loss + self.db_all_loss
            self.D_loss = self.d_loss + self.gamma * self.d_all_loss

        # Define all summaries
        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.cycle_loss_sum = tf.summary.scalar("cycle_loss", self.cycle_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum, self.cycle_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        if self.model != 'base':
            self.d_all_loss_sum = tf.summary.scalar("d_all_loss", self.d_all_loss)
            self.D_loss_sum = tf.summary.scalar("D_loss", self.d_loss)
            self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                                           self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                           self.d_loss_sum, self.d_all_loss_sum, self.D_loss_sum])
        else:
            self.d_sum = tf.summary.merge([self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
                                           self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
                                           self.d_loss_sum])

        # Test
        self.test_A = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32, [None, self.time_step, self.pitch_range,
                                                  self.output_c_dim], name='test_B')
        # A - B - A
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA_ = self.generator(self.testB, self.options, True, name='generatorB2A')
        # B - A - B
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")
        self.testB_ = self.generator(self.testA, self.options, True, name='generatorA2B')
        # to binary
        self.test_A_binary = to_binary(self.test_A, 0.5)
        self.test_B_binary = to_binary(self.test_B, 0.5)
        self.testA_binary = to_binary(self.testA, 0.5)
        self.testB_binary = to_binary(self.testB, 0.5)
        self.testA__binary = to_binary(self.testA_, 0.5)
        self.testB__binary = to_binary(self.testB_, 0.5)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars:
            print(var.name)

    def train(self, args):

        # Learning rate
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')

        # Discriminator and Generator Optimizer
        if self.model == 'base':
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
        else:
            self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.D_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # define the path which stores the log file, format is "{A}2{B}_{date}_{model}_{sigma}".
        log_dir = './logs/{}2{}_{}_{}_{}'.format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime,
                                                 self.model, self.sigma_d)
        # log_dir = './logs/{}2{}_{}_{}_{}'.format(self.dataset_A_dir, self.dataset_B_dir, '2018-06-10',
        #                                          self.model, self.sigma_d)
        self.writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        # Data from domain A and B, and mixed dataset for partial and full models.
        dataA = glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir))
        if self.model == 'partial':
            data_mixed = dataA + dataB
        if self.model == 'full':
            data_mixed = glob('./datasets/JCP_mixed/*.*')

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):

            # Shuffle training data
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            if self.model != 'base': np.random.shuffle(data_mixed)

            # Get the proper number of batches
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size

            # learning rate starts to decay when reaching the threshold
            lr = args.lr if epoch < args.epoch_step else args.lr * (args.epoch-epoch) / (args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):

                # To feed real_data
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_npy_data(batch_file) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # To feed gaussian noise
                gaussian_noise = np.abs(np.random.normal(0, self.sigma_d, [self.batch_size, self.time_step,
                                                                         self.pitch_range, self.input_c_dim]))

                if self.model == 'base':

                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_str, g_loss_a2b, g_loss_b2a, cycle_loss, g_loss = self.sess.run([self.fake_A,
                        self.fake_B, self.g_optim, self.g_sum, self.g_loss_a2b, self.g_loss_b2a, self.cycle_loss,
                        self.g_loss], feed_dict={self.real_data: batch_images, self.gaussian_noise: gaussian_noise,
                                                 self.lr: lr})

                    # Update D network
                    _, summary_str, da_loss, db_loss, d_loss = self.sess.run([
                        self.d_optim, self.d_sum, self.da_loss, self.db_loss, self.d_loss],
                        feed_dict={self.real_data: batch_images, self.fake_A_sample: fake_A, self.fake_B_sample: fake_B,
                                   self.lr: lr, self.gaussian_noise: gaussian_noise})

                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %6.2f, G_loss: %6.2f" %
                          (epoch, idx, batch_idxs, time.time() - start_time, d_loss, g_loss)))
                    print(("++++++++++G_loss_A2B: %6.2f G_loss_B2A: %6.2f Cycle_loss: %6.2f DA_loss: %6.2f DB_loss: %6.2f" %
                           (g_loss_a2b, g_loss_b2a, cycle_loss, da_loss, db_loss)))

                else:

                    # To feed real_mixed
                    batch_files_mixed = data_mixed[idx * self.batch_size:(idx + 1) * self.batch_size]
                    batch_images_mixed = [np.load(batch_file) * 1. for batch_file in batch_files_mixed]
                    batch_images_mixed = np.array(batch_images_mixed).astype(np.float32)

                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_str, g_loss_a2b, g_loss_b2a, cycle_loss, g_loss = self.sess.run([
                        self.fake_A,self.fake_B, self.g_optim, self.g_sum, self.g_loss_a2b, self.g_loss_b2a,
                        self.cycle_loss, self.g_loss], feed_dict={self.real_data: batch_images,
                                                                  self.gaussian_noise: gaussian_noise, self.lr: lr,
                                                                  self.real_mixed: batch_images_mixed})
                    self.writer.add_summary(summary_str, counter)
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])

                    # Update D network
                    _, summary_str, da_loss, db_loss, d_loss, da_all_loss, db_all_loss, d_all_loss, D_loss = self.sess.run([
                        self.d_optim, self.d_sum, self.da_loss, self.db_loss, self.d_loss, self.da_all_loss,
                        self.db_all_loss, self.d_all_loss, self.D_loss],
                        feed_dict={self.real_data: batch_images, self.fake_A_sample: fake_A, self.fake_B_sample: fake_B,
                                   self.lr: lr, self.gaussian_noise: gaussian_noise, self.real_mixed: batch_images_mixed})
                    self.writer.add_summary(summary_str, counter)

                    print('=================================================================')
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f D_loss: %6.2f, d_loss: %6.2f, d_all_loss: %6.2f, "
                           "G_loss: %6.2f" %
                           (epoch, idx, batch_idxs, time.time() - start_time, D_loss, d_loss, d_all_loss, g_loss)))
                    print(("++++++++++G_loss_A2B: %6.2f G_loss_B2A: %6.2f Cycle_loss: %6.2f DA_loss: %6.2f DB_loss: %6.2f, "
                            "DA_all_loss: %6.2f DB_all_loss: %6.2f" %
                           (g_loss_a2b, g_loss_b2a, cycle_loss, da_loss, db_loss, da_all_loss, db_all_loss)))

                counter += 1

                if np.mod(counter, args.print_freq) == 1:
                    sample_dir = os.path.join(self.sample_dir, '{}2{}_{}_{}_{}'.format(self.dataset_A_dir,
                                                                                       self.dataset_B_dir,
                                                                                       self.now_datetime,
                                                                                       self.model,
                                                                                       self.sigma_d))
                    # sample_dir = os.path.join(self.sample_dir, '{}2{}_{}_{}_{}'.format(self.dataset_A_dir,
                    #                                                                    self.dataset_B_dir,
                    #                                                                    '2018-06-10',
                    #                                                                    self.model,
                    #                                                                    self.sigma_d))
                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                    self.sample_model(sample_dir, epoch, idx)

                if np.mod(counter, batch_idxs) == 1:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "{}2{}_{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime, self.model,
                                            self.sigma_d)
        # model_dir = "{}2{}_{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, '2018-06-14', self.model,
        #                                     self.sigma_d)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "{}2{}_{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, self.now_datetime, self.model,
                                            self.sigma_d)
        # model_dir = "{}2{}_{}_{}_{}".format(self.dataset_A_dir, self.dataset_B_dir, '2018-06-14', self.model,
        #                                     self.sigma_d)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            # self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'cyclegan.model-7011'))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):

        print('Processing sample......')

        # Testing data from 2 domains A and B and sorted in ascending order
        dataA = glob('./datasets/{}/train/*.*'.format(self.dataset_A_dir))
        dataB = glob('./datasets/{}/train/*.*'.format(self.dataset_B_dir))
        dataA.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        dataB.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_npy_data(batch_file) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        real_A_binary, fake_A_binary, fake_B_binary = self.sess.run([self.real_A_binary, self.fake_A_binary,
                                                                     self.fake_B_binary],
                                                                    feed_dict={self.real_data: sample_images})
        real_B_binary, fake_A__binary, fake_B__binary = self.sess.run([self.real_B_binary, self.fake_A__binary,
                                                                       self.fake_B__binary],
                                                                      feed_dict={self.real_data: sample_images})

        if not os.path.exists(os.path.join(sample_dir, 'B2A')):
            os.makedirs(os.path.join(sample_dir, 'B2A'))
        if not os.path.exists(os.path.join(sample_dir, 'A2B')):
            os.makedirs(os.path.join(sample_dir, 'A2B'))

        save_midis(real_A_binary, './{}/A2B/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
        save_midis(fake_B_binary, './{}/A2B/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
        save_midis(fake_A__binary, './{}/A2B/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))
        save_midis(real_B_binary, './{}/B2A/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
        save_midis(fake_A_binary, './{}/B2A/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
        save_midis(fake_B__binary, './{}/B2A/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_A_dir))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/test/*.*'.format(self.dataset_B_dir))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')
        sample_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if args.which_direction == 'AtoB':
            out_origin, out_var, out_var_cycle, in_var = (self.test_A_binary, self.testB_binary, self.testA__binary,
                                                          self.test_A)
        else:
            out_origin, out_var, out_var_cycle, in_var = (self.test_B_binary, self.testA_binary, self.testB__binary,
                                                          self.test_B)

        test_dir_mid = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/mid'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_mid):
            os.makedirs(test_dir_mid)

        test_dir_npy = os.path.join(args.test_dir, '{}2{}_{}_{}_{}/{}/npy'.format(self.dataset_A_dir,
                                                                                  self.dataset_B_dir,
                                                                                  self.now_datetime,
                                                                                  self.model,
                                                                                  self.sigma_d,
                                                                                  args.which_direction))
        if not os.path.exists(test_dir_npy):
            os.makedirs(test_dir_npy)

        for idx in range(len(sample_files)):
            print('Processing midi: ', sample_files[idx])
            sample_npy = np.load(sample_files[idx]) * 1.
            sample_npy_re = sample_npy.reshape(1, sample_npy.shape[0], sample_npy.shape[1], 1)
            midi_path_origin = os.path.join(test_dir_mid, '{}_origin.mid'.format(idx + 1))
            midi_path_transfer = os.path.join(test_dir_mid, '{}_transfer.mid'.format(idx + 1))
            midi_path_cycle = os.path.join(test_dir_mid, '{}_cycle.mid'.format(idx + 1))
            origin_midi, fake_midi, fake_midi_cycle = self.sess.run([out_origin, out_var, out_var_cycle],
                                                                    feed_dict={in_var: sample_npy_re})
            save_midis(origin_midi, midi_path_origin)
            save_midis(fake_midi, midi_path_transfer)
            save_midis(fake_midi_cycle, midi_path_cycle)

            npy_path_origin = os.path.join(test_dir_npy, 'origin')
            npy_path_transfer = os.path.join(test_dir_npy, 'transfer')
            npy_path_cycle = os.path.join(test_dir_npy, 'cycle')
            if not os.path.exists(npy_path_origin):
                os.makedirs(npy_path_origin)
            if not os.path.exists(npy_path_transfer):
                os.makedirs(npy_path_transfer)
            if not os.path.exists(npy_path_cycle):
                os.makedirs(npy_path_cycle)
            np.save(os.path.join(npy_path_origin, '{}_origin.npy'.format(idx + 1)), origin_midi)
            np.save(os.path.join(npy_path_transfer, '{}_transfer.npy'.format(idx + 1)), fake_midi)
            np.save(os.path.join(npy_path_cycle, '{}_cycle.npy'.format(idx + 1)), fake_midi_cycle)

    def test_famous(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        song = np.load('./datasets/famous_songs/P2C/merged_npy/YMCA.npy')
        print(song.shape)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if args.which_direction == 'AtoB':
            out_var, in_var = (self.testB_binary, self.test_A)
        else:
            out_var, in_var = (self.testA_binary, self.test_B)

        transfer = self.sess.run(out_var, feed_dict={in_var: song * 1.})
        save_midis(transfer, './datasets/famous_songs/P2C/transfer/YMCA.mid', 127)
        np.save('./datasets/famous_songs/P2C/transfer/YMCA.npy', transfer)