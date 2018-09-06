import numpy as np
import glob
import datetime
import math
import random
import os
import shutil
from dataprocessing import pretty_midi_to_piano_roll
import matplotlib.pyplot as plt
import pretty_midi
from pypianoroll import Multitrack, Track
import librosa.display
from utils import *

ROOT_PATH = '/Users/sumuzhao/Downloads/'
converter_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_train/converter')
cleaner_path = os.path.join(ROOT_PATH, 'MIDI/pop/pop_train/cleaner')
test_ratio = 0.1
LAST_BAR_MODE = 'remove'


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 64) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((64 - piano_roll.shape[0] % 64, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 64):], axis=0)
    piano_roll = piano_roll.reshape(-1, 64, 128)
    return piano_roll


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_midi'))]
# print(l)
# idx = np.random.choice(len(l), int(test_ratio * len(l)), replace=False)
# print(len(idx))
# for i in idx:
#     shutil.move(os.path.join(ROOT_PATH, 'MIDI/pop/pop_midi', l[i]),
#                 os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi', l[i]))
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner'))]
# print(l)
# print(len(l))
# # idx = np.random.choice(len(l), 5000, replace=False)
# # print(len(idx))
# for i in l:
#     shutil.copy(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/origin_midi', os.path.splitext(i)[0] + '.mid'),
#                 os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi', os.path.splitext(i)[0] + '.mid'))
# now = datetime.datetime.now().strftime('%Y-%m-%d')
# print('a' + str(now))

# x = np.load(os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_test/tra_phr_jazz_test.npy'))
# piano = x[:, :, :, 3]
# piano_re = piano.reshape(piano.shape[0], piano.shape[1], piano.shape[2], 1)
# print(piano.shape, piano_re.shape)
# for i in range(piano_re.shape[0]):
#     if np.max(piano_re[i]):
#         sample = piano_re[i].reshape(1, piano_re[i].shape[0], piano_re[i].shape[1], piano_re[i].shape[2])
#         save_midis(sample, os.path.join(ROOT_PATH, 'jazz_piano/train/jazz_piano_train_{}.mid'.format(i+1)))
# save_midis(piano_re, os.path.join(ROOT_PATH, 'jazz_piano.mid'))


# x = np.load('/Users/sumuzhao/Downloads/5_transfer.npy')
# print(x.shape)
# save_midis(x, '/Users/sumuzhao/Downloads/5_transfer.mid')

# filepaths = []
# msd_id_list = []
# for dirpath, _, filenames in os.walk(os.path.join(ROOT_PATH, 'MIDI/Sinfonie Data')):
#     for filename in filenames:
#         if filename.endswith('.mid'):
#             msd_id_list.append(filename)
#             filepaths.append(os.path.join(dirpath, filename))
# print(filepaths)
# print(msd_id_list)
# for i in range(len(filepaths)):
#     shutil.copy(filepaths[i], os.path.join(ROOT_PATH, 'MIDI/classic/classic_midi/{}'.format(msd_id_list[i])))

# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi'))]
# print(l)
# count = 0
# for i in range(len(l)):
#     try:
#         multitrack = Multitrack(beat_resolution=4, name=os.path.splitext(l[i])[0])
#         x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi', l[i]))
#         multitrack.parse_pretty_midi(x)
#
#         category_list = {'Piano': [], 'Drums': []}
#         program_dict = {'Piano': 0, 'Drums': 0}
#
#         for idx, track in enumerate(multitrack.tracks):
#             if track.is_drum:
#                 category_list['Drums'].append(idx)
#             else:
#                 category_list['Piano'].append(idx)
#         tracks = []
#         merged = multitrack[category_list['Piano']].get_merged_pianoroll()
#
#         # merged = multitrack.get_merged_pianoroll()
#         print(merged.shape)
#         # tracks = [(Track(merged, program=0, is_drum=False, name=os.path.splitext(l[i])[0]))]
#         # mt = Multitrack(None, tracks, multitrack.tempo, multitrack.downbeat, multitrack.beat_resolution, multitrack.name)
#         # mt.write(os.path.join(ROOT_PATH, '-Study No.2 opus.105.mid'))
#
#         pr = get_bar_piano_roll(merged)
#         print(pr.shape)
#         pr_clip = pr[:, :, 24:108]
#         print(pr_clip.shape)
#         if int(pr_clip.shape[0] % 4) != 0:
#             pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
#         pr_re = pr_clip.reshape(-1, 64, 84, 1)
#         print(pr_re.shape)
#         save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_midi_gen', os.path.splitext(l[i])[0] +
#                                        '.mid'))
#         np.save(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', os.path.splitext(l[i])[0] + '.npy'), pr_re)
#     except:
#         count += 1
#         print('Wrong', l[i])
#         continue
# print(count)
#
# l = [f for f in os.listdir(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy'))]
# print(l)
# train = np.load(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', l[0]))
# print(train.shape, np.max(train))
# for i in range(1, len(l)):
#     print(i, l[i])
#     t = np.load(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/cleaner_npy', l[i]))
#     train = np.concatenate((train, t), axis=0)
# print(train.shape)
# np.save(os.path.join(ROOT_PATH, 'MIDI/pop/pop_test/pop_test_piano_16.npy'), (train > 0.0))

# x1 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_1.npy'))
# x2 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_2.npy'))
# x3 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_3.npy'))
# x4 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_4.npy'))
# x5 = np.load(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano_5.npy'))
# x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
# print(x.shape)
# np.save(os.path.join(ROOT_PATH, 'MIDI/classic/classic_train/classic_train_piano.npy'), x)

# x = np.load(os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_test/jazz_test_piano.npy'))
# print(x.shape)
# # np.save(os.path.join(ROOT_PATH, 'MIDI/jazz/jazz_test/jazz_test_piano_b.npy'), (x > 0.0))
# save_midis(x, os.path.join(ROOT_PATH, '2.mid'))

# with open('./t.txt', 'w') as f:
#     f.write('Id     Prob_Origin     Prob_Transfer     Prob_Cycle     ')
#     for i in range(10):
#         f.writelines('\n{}'.format(i))
# f.close()

multitrack = Multitrack(beat_resolution=4, name='YMCA')
x = pretty_midi.PrettyMIDI(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/origin/YMCA.mid'))
multitrack.parse_pretty_midi(x)

category_list = {'Piano': [], 'Drums': []}
program_dict = {'Piano': 0, 'Drums': 0}

for idx, track in enumerate(multitrack.tracks):
    if track.is_drum:
        category_list['Drums'].append(idx)
    else:
        category_list['Piano'].append(idx)
tracks = []
merged = multitrack[category_list['Piano']].get_merged_pianoroll()

# merged = multitrack.get_merged_pianoroll()
print(merged.shape)

pr = get_bar_piano_roll(merged)
print(pr.shape)
pr_clip = pr[:, :, 24:108]
print(pr_clip.shape)
if int(pr_clip.shape[0] % 4) != 0:
    pr_clip = np.delete(pr_clip, np.s_[-int(pr_clip.shape[0] % 4):], axis=0)
pr_re = pr_clip.reshape(-1, 64, 84, 1)
print(pr_re.shape)
save_midis(pr_re, os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_midi/YMCA.mid'), 127)
np.save(os.path.join(ROOT_PATH, 'MIDI/famous_songs/P2C/merged_npy/YMCA.npy'), (pr_re > 0.0))
