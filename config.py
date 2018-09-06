from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

print('[*] config...')

# class Dataset:
TRACK_NAMES = ['bass', 'drums', 'guitar', 'piano', 'strings']

def get_colormap():
    colormap = np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [1., .5, 0.],
                         [0., .5, 1.]])
    return tf.constant(colormap, dtype=tf.float32, name='colormap')

###########################################################################
# Training
###########################################################################

print_vars = True
is_eval = True
epoch = 20
iter_to_save = 100
sample_size = 64
print_batch = True
drum_filter = np.tile([1, 0.3, 0, 0, 0, 0.3], 16)
scale_mask = [1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 1.]
inter_pair = [(0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (3, 4)]
track_names = TRACK_NAMES
track_dim = len(track_names)
eval_map = np.array([
                [1, 1, 1, 1, 1],  # metric_is_empty_bar
                [1, 1, 1, 1, 1],  # metric_num_pitch_used
                [1, 0, 1, 1, 1],  # metric_too_short_note_ratio
                [1, 0, 1, 1, 1],  # metric_polyphonic_ratio
                [1, 0, 1, 1, 1],  # metric_in_scale
                [0, 1, 0, 0, 0],  # metric_drum_pattern
                [1, 0, 1, 1, 1]   # metric_num_chroma_used
            ])

exp_name = 'exp'
gpu_num = '1'


###########################################################################
# Model Config
###########################################################################


output_w = 96
output_h = 84
lamda = 10
batch_size = 64
beta1 = 0.5
beta2 = 0.9
lr = 2e-4
is_bn = True
colormap = get_colormap()

# nowbar

# NowBarHybridConfig
# track_names = TRACK_NAMES
# track_dim = 5
# acc_idx = 4
# z_inter_dim = 64
# z_intra_dim = 64
# output_dim = 1

# NowBarJammingConfig
# track_names = TRACK_NAMES
# track_dim = 5
# acc_idx = 4
# z_intra_dim = 128
# output_dim = 1

# NowBarComposerConfig
# track_names = ['All']
# track_dim = 1
# acc_idx = 4
# z_inter_dim = 128
# output_dim = 5

# Temporal
# TemporalHybridConfig
track_names = TRACK_NAMES
output_bar = 4
z_inter_dim = 32
z_intra_dim = 32
acc_idx = None
output_dim = 1

# TemporalJammingConfig
# track_names = TRACK_NAMES
# track_dim = 5
# output_bar = 4
# z_intra_dim = 64
# acc_idx = None
# output_dim = 1

# TemporalComposerConfig
# track_names = ['All']
# track_dim = 1
# output_bar = 4
# z_inter_dim = 64
# acc_idx = None
# output_dim = 5

# NowBarTemporalHybridConfig
# track_names = TRACK_NAMES
# acc_idx = 4
# track_dim = 5
# output_bar = 4
# z_inter_dim = 32
# z_intra_dim = 32
# acc_idx = 4
# output_dim = 1
