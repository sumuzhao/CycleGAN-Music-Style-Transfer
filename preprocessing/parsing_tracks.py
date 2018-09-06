import os
import json
from collections import defaultdict
import numpy as np
from pypianoroll import Multitrack
import pretty_midi
import scipy.sparse

ROOT_PATH = '/Users/sumuzhao/Downloads/MIDI'
RESULT_PATH = os.path.join(ROOT_PATH, 'jazz/jazz_train/tracks')
MIDI_DICT_PATH = os.path.join(ROOT_PATH, 'jazz/jazz_train/midis_clean.json')
LAST_BAR_MODE = 'fill'  # 'fill' or 'remove'
FILETYPE = 'npz'  # or 'csv'
MODE = '2D'  # or '3D'
prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano']


def get_bar_piano_roll(piano_roll):
    if int(piano_roll.shape[0] % 96) is not 0:
        if LAST_BAR_MODE == 'fill':
            piano_roll = np.concatenate((piano_roll, np.zeros((96 - piano_roll.shape[0] % 96, 128))), axis=0)
        elif LAST_BAR_MODE == 'remove':
            piano_roll = np.delete(piano_roll,  np.s_[-int(piano_roll.shape[0] % 96):], axis=0)
    piano_roll = piano_roll.reshape(-1, 96, 128)
    return piano_roll


def save_flat_piano_roll(piano_roll, postfix, midi_name):
    filepath = os.path.join(RESULT_PATH, postfix, midi_name + '.' + FILETYPE)
    if FILETYPE == 'npz':  # compressed scipy sparse matrix
        piano_roll = piano_roll.reshape(-1, 128)
        sparse_train_data = scipy.sparse.csc_matrix(piano_roll)
        scipy.sparse.save_npz(filepath, sparse_train_data)
    else:
        if MODE == '2D':
            piano_roll = piano_roll.reshape(-1, 128)
        elif MODE == '3D':
            if FILETYPE == 'csv':
                np.savetxt(filepath, piano_roll, delimiter=',')
            elif FILETYPE == 'npy':  # uncompressed numpy matrix
                np.save(filepath, piano_roll)
        else:
            print( 'Error: Unknown file saving setting')


def main():
    with open(MIDI_DICT_PATH) as f:
        midi_dict = json.load(f)

    # make dirs
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    for p in prefix:
        instr_dir = os.path.join(RESULT_PATH, p)
        if not os.path.exists(instr_dir):
            os.makedirs(instr_dir)

    counter = 0
    for key in midi_dict:
        multi_track = Multitrack(beat_resolution=24, name=key)
        multi_track.load(os.path.join(ROOT_PATH, 'jazz/jazz_train/cleaner', key + '.npz'))
        for track in multi_track.tracks:
            piano_roll = get_bar_piano_roll(track.pianoroll)
            if track.name == 'Guitar':
                save_flat_piano_roll(piano_roll, 'Guitar', key)
            elif track.name == 'Piano':
                save_flat_piano_roll(piano_roll, 'Piano', key)
            elif track.name == 'Bass':
                save_flat_piano_roll(piano_roll, 'Bass', key)
            elif track.name == 'Strings':
                save_flat_piano_roll(piano_roll, 'Other', key)
            else:
                save_flat_piano_roll(piano_roll, 'Drum', key)

        counter += 1
        print('%d' % counter, 'OK!')


if __name__ == "__main__":
    main()
