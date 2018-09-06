import os
import tensorflow as tf
from os.path import join, split
import pretty_midi as pm
import pypianoroll as pp
import numpy as np
from settings import *
import midi_functions as mf
import sys
import utils
import shutil

Root_Path = '/Users/sumuzhao/Downloads/MIDI'
SMALLEST_NOTE = 16
VELOCITY = 80
print_anything = False


def to_binary(bars, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = tf.equal(bars, tf.reduce_max(bars, axis=-1, keep_dims=True))
    track_pass_threshold = (bars > threshold)
    out_track = tf.logical_and(track_is_max, track_pass_threshold)
    return out_track


def normalization(bars, lower=0.0, upper=127.0):
    """Turn velocity value into range [lower, upper]"""
    bars_normalization = (bars - np.min(bars)) / (np.max(bars) - np.min(bars))
    out_bars = bars_normalization * (upper - lower)
    return out_bars


def pretty_midi_to_piano_roll(pathname):
    # try loading the midi file
    # if it fails, return all None objects
    try:
        mid = pm.PrettyMIDI(pathname)
    except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
        exception_str = 'Unexpected error in ' + pathname + ':\n', e, sys.exc_info()[0]
        print(exception_str)

    if print_anything:
        print("Time signature changes: ", mid.time_signature_changes)

    # determine start and end of the song
    # if there are tempo changes in the song, only take the longest part where the tempo is steady
    # this cuts of silent starts and extended ends
    # this also makes sure that the start of the bars are aligned through the song
    tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
    song_start = 0
    song_end = mid.get_end_time()
    # there will always be at least one tempo change to set the first tempo
    # but if there are more than one tempo changes, that means that the tempos are changed
    if len(tempo_change_times) > 1:
        longest_part = 0
        longest_part_start_time = 0
        longest_part_end_time = song_end
        longest_part_tempo = 0
        for i, tempo_change_time in enumerate(tempo_change_times):
            if i == len(tempo_change_times) - 1:
                end_time = song_end
            else:
                end_time = tempo_change_times[i + 1]
            current_part_length = end_time - tempo_change_time
            if current_part_length > longest_part:
                longest_part = current_part_length
                longest_part_start_time = tempo_change_time
                longest_part_end_time = end_time
                longest_part_tempo = tempo_change_bpm[i]
        song_start = longest_part_start_time
        song_end = longest_part_end_time
        tempo = longest_part_tempo
    else:
        tempo = tempo_change_bpm[0]

    # cut off the notes that are not in the longest part where the tempo is steady
    for instrument in mid.instruments:
        new_notes = []  # list for the notes that survive the cutting
        for note in instrument.notes:
            # check if it is in the given range of the longest part where the tempo is steady
            if note.start >= song_start and note.end <= song_end:
                # adjust to new times
                note.start -= song_start
                note.end -= song_start
                new_notes.append(note)
        instrument.notes = new_notes

    # (descending) order the piano_rolls according to the number of notes per track
    number_of_notes = []
    piano_rolls = [i.get_piano_roll(fs=100) for i in mid.instruments]
    for piano_roll in piano_rolls:
        number_of_notes.append(np.count_nonzero(piano_roll))
    permutation = np.argsort(number_of_notes)[::-1]  # descending order
    mid.instruments = [mid.instruments[i] for i in permutation]
    instrument_with_max_notes = mid.instruments[0]  # select the instrument with max number of notes

    if print_anything:
        print("Song start: ", song_start)
    if print_anything:
        print("Song end: ", song_end)
    if print_anything:
        print("Tempo: ", tempo)

    quarter_note_length = 1. / (tempo / 60.)  # the lenth of one quarter note
    # fs is is the frequency for the song at what rate notes are picked
    # the song will be sampled by (0, song_length_in_seconds, 1./fs)
    # fs should be the inverse of the length of the note, that is to be sampled
    # the value should be in beats per seconds, where beats can be quarter notes or whatever...
    fs = 1. / (quarter_note_length * 4. / SMALLEST_NOTE)

    if print_anything:
        print("fs: ", fs)
    total_ticks = math.ceil(song_end * fs)  # number of total beats
    if print_anything:
        print("Total ticks: ", total_ticks)

    piano_roll = np.zeros((total_ticks, 128))
    for note in instrument_with_max_notes.notes:
        note_tick_start = note.start * fs
        note_tick_end = note.end * fs
        absolute_start = int(round(note_tick_start))
        absolute_end = int(round(note_tick_end))
        decimal = note_tick_start - absolute_start
        # see if it starts at a tick or not
        # if it doesn't start at a tick (decimal > 10e-3) but is longer than one tick, include it anyways
        if decimal < 10e-3 or absolute_end - absolute_start >= 1:
            piano_roll[absolute_start:absolute_end, note.pitch] = 1
    if print_anything:
        print('piano_roll: ', piano_roll, piano_roll.shape)

    Y = piano_roll[:, low_crop:high_crop]
    if include_silent_note:
        Y = np.append(Y, np.zeros((Y.shape[0], 1)), axis=1)
        for step in range(Y.shape[0]):
            if np.sum(Y[step]) == 0:
                Y[step, -1] = 1
        # assert that there is now a 1 at every step
        for step in range(Y.shape[0]):
            assert (np.sum(Y[step, :]) == 1)
    # split the song into chunks of size output_length or input_length
    # pad them with silent notes if necessary
    if output_length > 0:
        # split Y
        padding_length = output_length - (Y.shape[0] % output_length)
        if output_length == padding_length:
            padding_length = 0
        # pad to the right..
        Y = np.pad(Y, ((0, padding_length), (0, 0)), 'constant', constant_values=(0, 0))
        if include_silent_note:
            Y[-padding_length:, -1] = 1
        number_of_splits = Y.shape[0] // output_length
        Y = np.split(Y, number_of_splits)
        Y = np.asarray(Y)
        print(Y.shape)
    return Y.reshape(Y.shape[1], Y.shape[2], 1)


def select_instrument(prettymidi):
    """Select the piano track with max # notes"""
    song_start = 0
    song_end = prettymidi.get_end_time()
    instrument_list = prettymidi.instruments
    instrument_selected = []
    instrument_notes = []
    for instrument in instrument_list:
        if instrument.program < 8 and instrument.is_drum is False:
            instrument_selected.append(instrument)
            instrument_notes.append(len(instrument.notes))
    if len(instrument_selected) != 0:
        pass_this_midi = False
        index_max_notes = instrument_notes.index(max(instrument_notes))
        tempo_change_times, tempo_change_bpm = prettymidi.get_tempo_changes()
        tempo = np.ceil(np.max(tempo_change_bpm))
        quarter_note_length = 1. / (tempo / 60.)
        fs = 1. / (quarter_note_length * 4. / SMALLEST_NOTE)
        total_beats = int(np.ceil(song_end * fs))
        print('tempo:', tempo, ', quarter_note_length:', quarter_note_length, ', fs:', fs,
              ', total_beats:', total_beats, ', song_end:', song_end)
        piano_roll = np.zeros((total_beats, 128))
        for note in instrument_selected[index_max_notes].notes:
            note_tick_start = note.start * fs
            note_tick_end = note.end * fs
            absolute_start = int(round(note_tick_start))
            absolute_end = int(round(note_tick_end))
            decimal = note_tick_start - absolute_start
            if decimal < 10e-3 or absolute_end - absolute_start >= 1:
                piano_roll[absolute_start:absolute_end, note.pitch] = VELOCITY
        return [pass_this_midi, normalization(piano_roll), fs, tempo]
    else:
        pass_this_midi = True
        return [pass_this_midi]


def merge_instrument(prettymidi):
    """Merge the piano tracks into one single track"""
    instruments = select_instrument(prettymidi)
    instrument_length = [round((instrument.get_piano_roll().shape[1]) / 768) for instrument in instruments]
    length = int((max(instrument_length) + min(instrument_length)) / 2)
    merged_track = np.zeros((128, length * 768))
    for idx in range(len(instruments)):
        piano_roll = instruments[idx].get_piano_roll()
        if piano_roll.shape[1] < length * 768:
            piano_roll = np.concatenate((piano_roll, np.zeros((128, length * 768 - piano_roll.shape[1]))), axis=1)
        elif piano_roll.shape[1] > length * 768:
            piano_roll = piano_roll[:, :length * 768]
        else:
            piano_roll = piano_roll
        merged_track += piano_roll
        print("Single piano track size: ", merged_track.shape)
    return merged_track.reshape(-1, 128, 768)


def piano_roll_to_pretty_midi(piano_roll, pathname='./', filename='out.mid', fs=100, program=0, initial_tempo=120.0):
    """
    Convert a Piano Roll array into a PrettyMidi object with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    """

    midi = pm.PrettyMIDI(initial_tempo=initial_tempo, resolution=200)
    midi.time_signature_changes.append(pm.TimeSignature(4, 4, 0))
    piano = pm.Instrument(program=program, is_drum=False)

    # pad 1 column of zeros so we can acknowledge initial and ending events
    piano_roll = np.pad(np.copy(piano_roll), ((low_crop, num_notes - high_crop), (1, 1)),
                        mode='constant', constant_values=0)
    print(piano_roll.shape)
    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    notes, frames = piano_roll.shape
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)
    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        # velocity = velocity.astype(float)
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity + 99
                # prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(velocity=prev_velocities[note], pitch=note, start=note_on_time[note], end=time)
            piano.notes.append(pm_note)
            prev_velocities[note] = 0
    midi.instruments.append(piano)
    midi.write(pathname + filename)
    '''
    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        print(velocity)
        # velocity = velocity.astype(float)
        time = time / fs
        if include_velocity:
            note_on_time[note] = time
            prev_velocities[note] = velocity * 100
        else:
            prev_velocities[note] = 80
        pm_note = pm.Note(velocity=prev_velocities[note], pitch=note, start=note_on_time[note], end=time)
        piano.notes.append(pm_note)
        prev_velocities[note] = 0
    midi.instruments.append(piano)
    midi.write(pathname + filename)
    '''
    """
    notes, frames = piano_roll.shape
    prettymidi = pm.PrettyMIDI(initial_tempo=initial_tempo)
    instrument = pm.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge initial and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        # velocity = velocity.astype(float)
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                # prev_velocities[note] = velocity + 99
                prev_velocities[note] = velocity
        else:
            pm_note = pm.Note(velocity=prev_velocities[note], pitch=note, start=note_on_time[note], end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    prettymidi.instruments.append(instrument)
    return prettymidi
    """


"""1. Generate training MIDI files"""
# jazz_list = [f for f in os.listdir(join(Root_Path, 'Jazz_midkar'))]
# classic_list = [f for f in os.listdir(join(Root_Path, 'Classic'))]
# pop_list = [f for f in os.listdir(join(Root_Path, 'pop songs'))]
# for f in classic_list:
#     try:
#         PM = pm.PrettyMIDI(join(Root_Path, 'Classic', f))
#         sample = select_instrument(PM)
#         print(sample[1].shape, np.max(sample[1]))
#         if not sample[0]:
#             for idx in range(sample[1].shape[0] // 128):
#                 pm_out = piano_roll_to_pretty_midi(np.transpose(sample[1][idx * 128:(idx + 1) * 128, :]),
#                                                    fs=sample[2], program=0, initial_tempo=sample[3])
#                 pm_out.write('./datasets/jazz2classic_realvalue/trainB/{}_{}_out_midi_test.mid'.format(split(f)[1], idx))
#         print(f, 'Processing succeed!')
#     except:
#         print(f, 'Wrong MID file!')
#         continue
# print('All classic midi files finish!')

"""2. Remove empty MIDI files"""
# l = [f for f in os.listdir('./datasets/jazz2classic_npy/trainA/')]
# for idx in range(len(l)):
#     if l[idx] != '.DS_Store':
#         try:
#             print(idx, l[idx])
#             piano_roll_split = pretty_midi_to_piano_roll('./datasets/jazz2classic_npy/trainA/', l[idx])
#             # PM = pm.PrettyMIDI(join('./datasets/jazz2classic_npy/trainB', l[idx]))
#             # p = select_instrument(PM)[1]
#         except:
#             print(idx, l[idx], "Wrong!!")
#             os.remove(join('./datasets/jazz2classic_npy/trainA', l[idx]))

"""3. Adapt the piano_roll size"""
# l = [f for f in os.listdir('./datasets/jazz2classic_realvalue/trainB/')]
# count = 0
# threshold = 4
# for idx in range(len(l)):
#     if l[idx] != '.DS_Store':
#         PM = pm.PrettyMIDI(join('./datasets/jazz2classic_realvalue/trainB', l[idx]))
#         p = select_instrument(PM)[1]
#         print(p.shape)
#         if p.shape[0] > 128 + threshold or p.shape[0] < 128 - threshold:
#             os.remove(join('./datasets/jazz2classic_realvalue/trainB', l[idx]))
#             # p = p
#             count += 1
#         # elif p.shape[0] < 128:
#         #     p = np.concatenate((p, np.zeros((128 - p.shape[0], 128))), axis=0)
#         # elif p.shape[0] > 128:
#         #     p = p[0:128, :]
#         # else:
#         #     p = p
#         # print(p.shape)
#         print(count)

"""4. Generate some test MIDI files"""
# l = [f for f in os.listdir('./datasets/jazz2classic_npy/trainB/')]
# x = np.random.choice(range(len(l)), 100, replace=False)
# print(len(x))
# for i in list(x):
#     shutil.move(join('./datasets/jazz2classic_npy/trainB', l[i]),
#                 join('./datasets/jazz2classic_npy/testB', l[i]))

# x_split = pretty_midi_to_piano_roll('/Users/sumuzhao/Downloads/', '1.Bass Aria.mid')
# np.save('/Users/sumuzhao/Downloads/chameleon-hhancock_ov.npy', x_split)
# piano_roll_to_pretty_midi(x_split.reshape(-1, 64).T, fs=8)
# for idx in range(len(jazz_list)):
#     print(idx)
#     try:
#         piano_roll_split = pretty_midi_to_piano_roll('/Users/sumuzhao/Downloads/MIDI/Jazz_midkar/', jazz_list[idx])
#     except:
#         print(idx, jazz_list[idx], 'wrong midi!')
#         continue
#     for i, pr in enumerate(piano_roll_split):
#         piano_roll_to_pretty_midi(pr.T, fs=8, pathname='./datasets/jazz2classic_npy/trainA/',
#                                   filename='{}_'.format(i) + jazz_list[idx])
# print('All processed!')

# piano_roll_split = pretty_midi_to_piano_roll('./datasets/jazz2classic_npy/trainB/', '17_Mirror Fugue n1.mid')

# x = np.load('/Users/sumuzhao/Downloads/MIDI/pop/pop_train/tra_phr_pop_train.npy')
# for i in range(x.shape[0]):
#     sample = x[i].reshape(1, x[i].shape[0], x[i].shape[1], x[i].shape[2])
#     utils.save_midis(sample, '/Users/sumuzhao/Downloads/MIDI/pop/pop_train/train_midi/pop_train_{}.mid'.format(i + 1))
#     print('Finished!', i + 1)
# print('All finished!')
# l = [f for f in os.listdir('../datasets/JAZZ/train/')]
# idx = np.random.choice(len(l), 68, replace=False)
# print(idx)
# for i in idx:
#     shutil.move(os.path.join('../datasets/JAZZ/train/jazz_{}.mid'.format(i)),
#                 os.path.join('../datasets/JAZZ/test/jazz_{}.mid'.format(i)))
