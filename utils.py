'''Utilites

2021-01-05 first created
'''

import os
import re
import tgt
import pickle
import numpy as np
import pandas as pd
from parselmouth.praat import call

def check_nan(df):
    '''Return row indices where nan occurs as a list'''
    bools = df.isnull().sum(axis=1).values
    row_idx = np.argwhere(bools).squeeze()
    return row_idx


def load_pkl(file_name):
    '''Load .pkl file'''
    with open(file_name, 'rb') as pkl:
        pkl = pickle.load(pkl)
    assert isinstance(pkl, dict), 'pickle data is not python-dictionary'
    assert len(pkl.keys()) > 0, 'there is no valid keys in data'
    return pkl


def load_textgrid(file_name, tier_name='phone'):
    '''Load textgrid & return times and labels'''
    tg = tgt.read_textgrid(file_name)
    tier = tg.get_tier_by_name(tier_name)

    times = []
    labels = []
    for t in tier:
        times.append([round(t.start_time, 4), round(t.end_time, 4)])
        labels.append(t.text)
    assert len(times) > 0, f'"times" is empty: len={len(times)}'
    assert len(labels) > 0, f'"{tier_name}" is empty: len={len(labels)}'
    return np.array(times, dtype='float32'), labels


def load_dictionary(dict_file, field_names, channel_names, audio_channel):
    '''Load pickled dictionary'''
    D = load_pkl(dict_file)
    check_dictionary(D, field_names, channel_names, audio_channel)
    return D 


def check_dictionary(dt, field_names, channel_names, audio_channel):
    '''Check dictionary structure'''
    channels = [audio_channel]+channel_names
    assert set(dt.keys()) & set([audio_channel]+channel_names) == set([audio_channel]+channel_names), \
        f'channel_names are not matching\n   Target{channels} <==>\nProvided:{dt.keyes()}'
    for key in dt.keys():
        keys = dt[key].keys()
        assert set(keys) == set(field_names), \
            f'field_names are not matching at key={key}\n   Target{field_names} <==>\nProvided:{keys}'


def parse_ieee_filename(filename):
    '''Parse IEEE data filename
    Returns: fid, speaker, block, sentence, repetition, rate
    '''
    fid = os.path.basename(filename).split('.')[0]
    speaker = re.sub('([FM][0-9][0-9]).*', '\\1', fid)
    block = int(re.sub('.*_B([0-9][0-9])_.*', '\\1', fid))
    sentence = re.sub('.*_(S[0-9][0-9])_.*', '\\1', fid)
    repetition = int(re.sub('.*_R(0[0-9])_.*', '\\1', fid))
    rate = re.sub('.*_([NF])', '\\1', fid)
    return fid, speaker, block, sentence, repetition, rate


def check_ref_formant_file(filename):
    '''Check reference formant file and its structure'''
    assert os.path.exists(filename), 'file does not exist'

    columns = ['Speaker', 'Nativity', 'Language',
               'Vowel', 'F1ref', 'F2ref', 'F3ref', 'F4ref']
    df = pd.read_excel(filename, index_col=False)
    assert set(columns) == set(df.columns.to_list(
    )), f'Columns do not match: \nTruth:{columns} <==> \nGiven:{df.columns} '
    assert df.shape[0] > 0, f'Data is empty: shape={df.shape}'


def resample(soundObj, target_sampling_rate, precision_ms=50):
    '''Resample soundObj with the target one
    Returns the updated soundObj (praat)
    '''
    raw_sampling_rate = call(soundObj, 'Get sampling frequency')
    soundObj = call(soundObj, 'Resample', target_sampling_rate, precision_ms)
    return soundObj


def init_acous_params(time=None,F1=500,F2=1485,F3=2450,F4=4000,F5=4950,
                        window_length=0.045, step_size=0.002, 
                        num_formants=3, max_formants=5500,
                        max_num_formants=5, pre_emphasis=50,
                        freqcost=1, bwcost=1, transcost=1):
    '''Initialize formant tracking parameters'''
    params = {
        'time': time,
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'F4': F4,
        'F5': F5,
        'window_length': window_length,
        'step_size': step_size,
        'num_formants': num_formants,
        'max_formants': max_formants,
        'max_num_formants': max_num_formants,
        'pre_emphasis': pre_emphasis,
        'freqcost': freqcost,
        'bwcost': bwcost,
        'transcost': transcost,
        'min_f0': 75,
        'max_f0': 500,
    }
    return params    


def init_acous_feats(soundObj, beg_time, end_time, params=None, acous=['f0','F1','F2','F3'], do_resample=True):
    '''Initialize acoustic feature objects using parselmouth (Praat backend)
    Note the case of the alphabet prefix (lower: pitch, upper: formants)
    '''
    # Extract part
    window_shape = 'rectangular'
    relative_width = 1.0
    preserve_time = 1
    soundObjPart = call(soundObj, 'Extract part', float(beg_time), float(end_time), 
                        window_shape, relative_width, preserve_time)
    # Check conditions
    if params is None:
        params = init_acous_params()
    if do_resample:
        raw_sampling_rate = call(soundObjPart, 'Get sampling frequency')
        new_sampling_rate = params['max_formants'] * 2
        if new_sampling_rate != raw_sampling_rate:
            soundObjPart = resample(soundObjPart, new_sampling_rate)

    fmtTrackObj = None
    pitchObj = None
    # Formant object
    if 'F' in [val[0] for val in acous]:
        fmtObj = call(soundObjPart, "To Formant (burg)", 
                      params['step_size'], 
                      params['max_num_formants'], 
                      params['max_formants'], 
                      params['window_length'], 
                      params['pre_emphasis'])
        fmtTrackObj = call(fmtObj, "Track...", params['num_formants'], 
                           params['F1'], params['F2'], params['F3'], params['F4'], params['F5'], 
                           params['freqcost'], params['bwcost'], params['transcost'])
    # Pitch object
    if 'f' in [val[0] for val in acous]:
        pitchObj = call(soundObjPart, 'To Pitch', 0, params['min_f0'], params['max_f0'])
        pitchObj = call(pitchObj, 'Down to PitchTier')

    return fmtTrackObj, pitchObj