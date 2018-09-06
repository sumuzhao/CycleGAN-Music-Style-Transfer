from __future__ import division
from __future__ import print_function

import scipy.sparse
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import os

ROOT_TRACKS = '/Users/sumuzhao/Downloads/MIDI/jazz/jazz_train/tracks'
PATH_PIANO_ROLL = join(ROOT_TRACKS, 'Piano_Roll')
prefix = ['Bass', 'Drum', 'Guitar', 'Other', 'Piano']
PATH_INSTRU_ACT = join(ROOT_TRACKS, 'act_instr')
PATH_ALL_ACT = join(ROOT_TRACKS, 'act_all')


def csc_to_array(csc):
    return scipy.sparse.csc_matrix((csc['data'], csc['indices'], csc['indptr']), shape=csc['shape']).toarray()


def reshape_to_bar(flat_array):
    return flat_array.reshape(-1, 96, 128)


def is_empty_bar(bar):
    return not np.sum(bar)


def main():

    if not os.path.exists(PATH_INSTRU_ACT):
        os.makedirs(PATH_INSTRU_ACT)
    if not os.path.exists(PATH_ALL_ACT):
        os.makedirs(PATH_ALL_ACT)
    if not os.path.exists(PATH_PIANO_ROLL):
        os.makedirs(PATH_PIANO_ROLL)

    song_list = [f.split('.')[0] for f in listdir(join(ROOT_TRACKS, 'Drum')) if isfile(join(join(ROOT_TRACKS, 'Drum'), f))]

    thres = 3
    numOfSong = len(song_list)
    count = 0
    for song_idx in range(numOfSong):

        midi_name = song_list[song_idx]
        sys.stdout.write('{0}/{1}\r'.format(song_idx, numOfSong))
        sys.stdout.flush()

        song_piano_rolls = []
        list_is_empty = []

        try:
            piano_roll = reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS, prefix[0], midi_name + '.npz'))))
            song_piano_rolls.append(piano_roll)

            for idx in range(1, 5):
                piano_roll_tmp = reshape_to_bar(csc_to_array(np.load(join(ROOT_TRACKS, prefix[idx], midi_name + '.npz'))))
                piano_roll += piano_roll_tmp
                song_piano_rolls.append(piano_roll_tmp)
        except:
            count += 1
            print('Wrong Analysis, skip this song:', midi_name)
            continue

        piano_roll = np.concatenate((piano_roll[:]))
        piano_roll = piano_roll.T

        numOfBar = song_piano_rolls[0].shape[0]
        instr_act = np.zeros((numOfBar, 5))
        all_act = np.zeros(numOfBar)
        chroma = np.zeros_like(song_piano_rolls[0])

        for bar_idx in range(numOfBar):
            for pre_idx in range(5):
                bar = song_piano_rolls[pre_idx][bar_idx, :, :]
                instr_act[bar_idx, pre_idx] = not is_empty_bar(bar)
                all_act[bar_idx] = np.sum(instr_act[bar_idx, :]) >= thres

        sio.savemat(os.path.join(PATH_PIANO_ROLL, midi_name + '.mat'), {'piano_roll': piano_roll})
        np.save(join(PATH_INSTRU_ACT, midi_name + '.npy'), instr_act)
        np.save(join(PATH_ALL_ACT, midi_name + '.npy'), all_act)

    print('skip songs:', count)


if __name__ == "__main__":
    main()
