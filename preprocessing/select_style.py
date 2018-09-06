import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import librosa
import tables
import IPython.display
import os
import shutil
import json

# Local path constants
DATA_PATH = 'data'
RESULTS_PATH = '/Users/sumuzhao/Downloads/'
# Path to the file match_scores.json distributed with the LMD
SCORE_FILE = os.path.join(RESULTS_PATH, 'match_scores.json')

# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_mp3(msd_id):
    """Given an MSD ID, return the path to the corresponding mp3"""
    return os.path.join(DATA_PATH, 'msd', 'mp3',
                        msd_id_to_dirs(msd_id) + '.mp3')

def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5',
                        msd_id_to_dirs(msd_id) + '.h5')

def get_midi_path(msd_id, midi_md5, kind):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file.
    kind should be one of 'matched' or 'aligned'. """
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind),
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

# msd_id = 'TRJIKIH128F422AA45'
# with tables.open_file(msd_id_to_h5(msd_id)) as h5:
#     print('{} by {} on {}, {}, {}'.format(
#         h5.root.metadata.songs.cols.title[0],
#         h5.root.metadata.songs.cols.artist_name[0],
#         h5.root.metadata.songs.cols.release[0],
#         h5.root.metadata.artist_terms[:],
#         len(h5.root.metadata.artist_terms[:])))
#     x = h5.root.metadata.artist_terms[:]
#     x_str = [i.decode("utf-8") for i in x]
#     print(x_str)
#     if 'rap' in x_str[:10]:
#         print('True')


filepaths = []
msd_id_list = []
for dirpath, _, filenames in os.walk(os.path.join(RESULTS_PATH, 'lmd_matched_h5')):
    for filename in filenames:
        if filename.endswith('.h5'):
            msd_id_list.append(filename)
            filepaths.append(os.path.join(dirpath, filename))
print(len(filepaths), len(msd_id_list))
x = filepaths[12]
y = msd_id_list[12]
print(x, y, os.path.splitext(y)[0], msd_id_to_dirs(os.path.splitext(y)[0]))

# filepaths_midi = []
# for dirpath, _, filenames in os.walk(os.path.join(RESULTS_PATH, 'lmd_matched')):
#     for filename in filenames:
#         if filename.endswith('.mid'):
#             filepaths_midi.append(os.path.join(dirpath, filename))
# print(len(filepaths_midi))

with open(os.path.join(RESULTS_PATH, 'match_scores.json')) as infile:
    midi_dict = json.load(infile)
print(len(midi_dict))
print(list(midi_dict["TRRNARX128F4264AEB"].keys()))
count_jazz = 0
count_pop = 0
count_rock = 0
count_rap = 0
for key in midi_dict:
    with tables.open_file(msd_id_to_h5(key)) as h5:
        x = h5.root.metadata.artist_terms[:1]
        x = [i.decode("utf-8") for i in x]
        if 'jazz' in x:
            for sub_key in midi_dict[key]:
                try:
                    shutil.copy(os.path.join(RESULTS_PATH, 'lmd_matched', msd_id_to_dirs(key), sub_key + '.mid'),
                                os.path.join(RESULTS_PATH, 'piano/jazz/jazz_midi', sub_key + '.mid'))
                except:
                    continue
            count_jazz += 1
        if 'pop' in x:
            for sub_key in midi_dict[key]:
                try:
                    shutil.copy(os.path.join(RESULTS_PATH, 'lmd_matched', msd_id_to_dirs(key), sub_key + '.mid'),
                                os.path.join(RESULTS_PATH, 'piano/pop/pop_midi', sub_key + '.mid'))
                except:
                    continue
            count_pop += 1
        if 'rock' in x:
            for sub_key in midi_dict[key]:
                try:
                    shutil.copy(os.path.join(RESULTS_PATH, 'lmd_matched', msd_id_to_dirs(key), sub_key + '.mid'),
                                os.path.join(RESULTS_PATH, 'piano/rock/rock_midi', sub_key + '.mid'))
                except:
                    continue
            count_rock += 1
        if 'rap' in x or 'hip hop' in x:
            for sub_key in midi_dict[key]:
                try:
                    shutil.copy(os.path.join(RESULTS_PATH, 'lmd_matched', msd_id_to_dirs(key), sub_key + '.mid'),
                                os.path.join(RESULTS_PATH, 'piano/rap/rap_midi', sub_key + '.mid'))
                except:
                    continue
            count_rap += 1
print(count_jazz, count_pop, count_rock, count_rap)
