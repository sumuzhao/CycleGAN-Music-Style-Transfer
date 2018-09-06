import pypianoroll as pp
import os
import scipy.sparse
import numpy as np

path = '/Users/sumuzhao/Downloads/MIDI/jazz/jazz_test/tracks/data_phr/tra'
# Bass = np.load(os.path.join(path, 'Bass.npy'))
# Drum = np.load(os.path.join(path, 'Drum.npy'))
# Guitar = np.load(os.path.join(path, 'Guitar.npy'))
# Other = np.load(os.path.join(path, 'Other.npy'))
Piano = np.load(os.path.join(path, 'Piano.npy'))
# train = np.concatenate((Bass.reshape(-1, 1), Drum.reshape(-1, 1), Guitar.reshape(-1, 1), Other.reshape(-1, 1),
#                         Piano.reshape(-1, 1)), axis=-1)
# train = train.reshape(Bass.shape[0], 8, 96, 128, -1)
# print(train.shape)
#
# clip = train[:, :, :, 24:108, :]
# print(clip.shape)
#
# tra_phr = clip.reshape(-1, 384, 84, 5)
# print(tra_phr.shape)
# np.save(os.path.join(path, 'tra_phr_pop_train.npy'), tra_phr)

# x = np.load('/Users/sumuzhao/Downloads/MIDI/pop/tracks/data_phr/tra/tra_phr_pop_test.npy')
# print(x.shape)
# # np.random.shuffle(x)
#
# # np.save('/Users/sumuzhao/Downloads/MIDI/pop/tracks/data_phr/tra/tra_phr_pop_test.npy', x[:72])
# # np.save('/Users/sumuzhao/Downloads/MIDI/pop/tracks/data_phr/tra/tra_phr_pop_train.npy', x[72:])

print(Piano.shape)
Piano = Piano.reshape(-1, 384, 128, 1)
print(Piano.shape)
Piano = Piano[:, :, 24:108, :]
print(Piano.shape)
np.save(os.path.join(path, 'tra_phr_jazz_piano_test.npy'), Piano)