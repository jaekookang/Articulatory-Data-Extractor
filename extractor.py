'''Extractor

Example:
  python extractor.py example/F01_B01_S01_R01_N.pkl result/result.csv \
                      --vowel IY1,IH1,EH1,AO1,AH1,AA1,AE1,UH1,UW1 \
                      --artic tr,td,tt,ja,ul,ll \
                      --acous f0,f1,f2,f3 \
                      --n_points 5 \
                      --ref example/reference_formants.xlsx \
                      --skip_nans False

Notation:
- tr --> TR, td --> TD, tt --> TT, ja --> JAW, ul --> UL, ll --> LL

2021-01-05 first created
'''
# Basic
import os
import re
import wave
import tempfile
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from pprint import pprint
import parselmouth
from parselmouth.praat import call
# Custom 
from utils import load_pkl, load_dictionary, check_dictionary, \
     parse_ieee_filename, check_ref_formant_file, \
     init_acous_params, init_acous_feats, check_nan

def check_arguments(args, verbose=False):
    '''Check arguments & Update args'''
    # DATAFILE, OUTFILE
    if os.path.isdir(args.DATAFILE):
        assert len(glob(os.path.join(args.DATAFILE, '*.pkl'))) > 0, f'No pickle files were found at {args.DATAFILE}'
    else:
        assert os.path.exists(args.DATAFILE), f'{args.DATAFILE} (file or directory) does not exist'
    # Check --vowel
    args.vowel = ''.join(args.vowel.split())
    assert len(args.vowel.split(',')) > 0, f'No vowels are provided: {args.vowel}'
    args.vowel = args.vowel.split(',')
    # Check --artic
    args.artic = ''.join(args.artic.split())
    assert len(args.artic.split(',')) > 0, f'No artic is provided: {args.artic}'
    args.artic = args.artic.split(',')
    # Check --acous
    args.acous = ''.join(args.acous.split())
    assert len(args.acous.split(',')) > 0, f'No acous is provided: {args.acous}'
    args.acous = args.acous.split(',')
    args.acous = [val.lower() if val[1]=='0' else val.upper() for val in args.acous] # update label names
    # Check --n_points
    assert args.n_points > 0, 'n_points should be larger than 0'
    assert args.n_points < 10, f'time points are too many... {args.n_points}. Maybe reduce it down <10?'
    # Check --ref
    if args.ref is not None:
        assert os.path.exists(args.ref), f'Reference formant file ({args.ref}) does not exist'
        check_ref_formant_file(args.ref)
    else:
        args.ref = None

    if verbose:
        pprint(vars(args))
    return args



def extract(datafile, acous, artic, artic_dict, ref_file, vowels, n_points,
            field_names, channel_names, audio_channel, header_names, 
            result_file=None, skip_nans=False, write_log=True):
    '''Extract articulatory and acoustic features for a single file
    Parameters
    ----------
        datafile: datafile; eg. F01_B01_S01_R01_N.pkl
        acous: list of acoustic features
        artic: list of articulatory features
        artic_dict: articulatory name conversion dictionary; eg. 'tr' --> 'TR'
        ref_file: formant reference file
        vowels: list of vowels to extract
        n_points: number of time points to extract given a vowel
        field_names: name of the MVIEW fields; eg. 'SIGNAL', ...
        channel_names: EMA sensor names; eg. TR, TB, ...
        audio_channel: audio channel name; eg. AUDIO
        header_names: table header for the output result file
        result_file: (optional) output file name; eg. result/F01.csv
        skip_nans: if False (default), it will throw an error when NaNs are found
        write_log: if True (default), it will write a log file
    '''
    # Initialize log data
    logs = []

    # Load data dictionary
    D = load_dictionary(datafile, field_names, channel_names, audio_channel)
    ema_sr = D['TR']['SRATE']

    # Load audio
    sig, sr = D['AUDIO']['SIGNAL'], D['AUDIO']['SRATE']
    sig_int16 = (sig * (2 ** 15 - 1)).astype("<h")
    with tempfile.NamedTemporaryFile() as f:
        # Write audio temporarily
        s = wave.open(f, mode='wb')
        s.setnchannels(1)
        s.setsampwidth(2)
        s.setframerate(sr)
        s.writeframes(sig_int16.tobytes())
        s.close()
        f.flush()
        soundObj = parselmouth.Sound(f.name)

    # Initialization
    fid, spkr, block, sent, rep, rate = parse_ieee_filename(datafile) # ieee
    header = header_names.copy()
    header += [col.upper() if col != 'f0' else col for col in acous]
    header += [artic_dict[col]+c for col in artic for c in ['x','z']] # TR --> TRx, TRz
    params = init_acous_params()
    if ref_file is not None:
        ref = pd.read_excel(ref_file)

    # Get labels
    _phones, _p_times = D['AUDIO']['PHONES']['LABEL'], D['AUDIO']['PHONES']['OFFS']
    _words, _w_times = D['AUDIO']['WORDS']['LABEL'], D['AUDIO']['WORDS']['OFFS']
    p_idx = [i for i, p in enumerate(_phones) if p in vowels]
    if len(p_idx) == 0:
        logs += [','.join([fid, '_', '_', '_', f'No phones {vowels} were matching'])]
        return None, logs
    phones, p_times = [_phones[i] for i in p_idx], _p_times[p_idx,:]
    w_idx = [i for i, (beg, end) in enumerate(_w_times) 
           if len(np.where((p_times.mean(axis=1)<end) * (p_times.mean(axis=1)>beg))[0])]
    assert len(w_idx) > 0, f'No words were matching given phones {vowels}'
    words, w_times = [_words[i] for i in w_idx], _w_times[w_idx,:]
    
    # --- Iterate over phones (vowels)
    # pad before/after the extracted part to prevent nan from formant tracking
    margin = params['window_length'] * 2 # sec
    df = pd.DataFrame(columns=header, index=range(len(phones * n_points)))
    irow = 0
    for i, p_label, p_time, w_label, w_time in zip(p_idx, phones, p_times, words, w_times):
        # Retrieve meta info
        if ref_file is not None:
            params['F1'] = float(ref.loc[(ref.Speaker==spkr)&(ref.Vowel==p_label), 'F1ref'].values)
            params['F2'] = float(ref.loc[(ref.Speaker==spkr)&(ref.Vowel==p_label), 'F2ref'].values)
            params['F3'] = float(ref.loc[(ref.Speaker==spkr)&(ref.Vowel==p_label), 'F3ref'].values)
            params['F4'] = float(ref.loc[(ref.Speaker==spkr)&(ref.Vowel==p_label), 'F4ref'].values)
        prevowel = _phones[i-1] if (i-1) > 0 else 'BEG'
        postvowel = _phones[i+1] if (i+1) >= len(_phones) else 'END'
        duration = p_time[1] - p_time[0]
        percents = np.linspace(0, 1, n_points) # time --> percent point
        ticks = [float(p_time[0] + duration * t) for t in percents]
        fmtTrackObj, pitchObj, error_flag = init_acous_feats(soundObj, p_time[0]-margin, p_time[1]+margin, 
            params=params, acous=acous, do_resample=True)

        # --- Iterate over time points
        if error_flag is None:
            for j, (perc, tick) in enumerate(zip(percents, ticks)):
                # Extract acoustic features
                acous_data = []
                if 'f0' in header:
                    acous_data += [call(pitchObj, 'Get value at time...', tick)]
                if 'F1' in header:
                    acous_data += [call(fmtTrackObj, 'Get value at time...', 1, tick, 'hertz', 'Linear')]
                if 'F2' in header:
                    acous_data += [call(fmtTrackObj, 'Get value at time...', 2, tick, 'hertz', 'Linear')]
                if 'F3' in header:
                    acous_data += [call(fmtTrackObj, 'Get value at time...', 3, tick, 'hertz', 'Linear')]
                if 'F4' in header:
                    acous_data += [call(fmtTrackObj, 'Get value at time...', 4, tick, 'hertz', 'Linear')]

                # Extract articulatory features
                artic_data = []
                sample_idx = round(tick * ema_sr)
                xz_idx = [0,2]
                for arc in [artic_dict[c] for c in artic]:
                    x, z = D[arc]['SIGNAL'][sample_idx, xz_idx]
                    artic_data += [x, z]

                # Update
                row_data = [fid, spkr, block, rate, sent, rep, w_label, 
                            prevowel, p_label, postvowel, duration, perc, tick]
                row_data += acous_data
                row_data += artic_data
                df.loc[irow] = row_data
                irow += 1
        else:
            logs += [','.join([fid, str(p_time[0]), p_label, w_label, error_flag])]

    if result_file is not None:
        df.to_csv(result_file, index=False)
        if args.write_log:
            log_file = re.sub('csv', 'log', result_file)
            with open(log_file, 'wt') as f:
                for line in logs:
                    f.write(line+'\n')
    if not skip_nans:
        assert len(check_nan(df)) == 0, f'NaNs were found ({len(check_nan(df))} rows)'
    
    return df, logs

def run(args):
    '''Run the extractor'''
    # Set meta information
    field_names = ['NAME', 'SRATE', 'SIGNAL', 'SOURCE', 'SENTENCE', 'WORDS', 'PHONES', 'LABELS']
    channel_names = ['TR', 'TB', 'TT','UL', 'LL', 'JAW']
    audio_channel = 'AUDIO'
    header = ['FileID', 'Speaker', 'Block', 'Rate', 'Sent', 'Rep', 'Word', 
              'PreVowel', 'Vowel', 'PostVowel', 'Duration', 'TimeAt', 'TimeSec']
    artic_dict = {'tr':'TR', 'tb':'TB', 'tt':'TT', 'ja':'JAW', 'ul':'UL', 'll':'LL'}

    # Extract features
    if not os.path.isdir(args.DATAFILE):
        # --- for single file
        fid, _, _, _, _, _, = parse_ieee_filename(args.DATAFILE)
        result_file = os.path.join('result', f'{fid}.csv')
        df = extract(args.DATAFILE, args.acous, args.artic, artic_dict, args.ref, args.vowel, args.n_points,
                     field_names, channel_names, audio_channel, header, 
                     result_file=result_file, skip_nans=args.skip_nans, write_log=args.write_log)
    else:
        # --- for multiple files
        dfs = []
        logs = []
        pkl_files = sorted(glob(os.path.join(args.DATAFILE, '*.pkl')))
        basename = os.path.basename(args.OUTFILE)
        for pkl_file in tqdm(pkl_files, desc=f'Result: {basename}', ascii=True, total=len(pkl_files)): 
            df, log = extract(pkl_file, args.acous, args.artic, artic_dict, args.ref, args.vowel, args.n_points,
                               field_names, channel_names, audio_channel, header, 
                               result_file=None, skip_nans=args.skip_nans, write_log=args.write_log)
            logs += [log]
            if df is not None:
                dfs += [df]
        # Save
        pd.concat(dfs).to_csv(args.OUTFILE, index=False)
        if args.write_log & (len(sum(logs, [])) > 0):
            log_file = re.sub('csv', 'log', args.OUTFILE)
            with open(log_file, 'wt') as f:
                for line in sum(logs, []):
                    f.write(line+'\n')

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Extract articulatory and acoustic features from the EMA data',
                                     epilog='See https://github.com/jaekookang/Articulatory-Data-Extractor')
    parser.add_argument('DATAFILE', type=str, 
                        help='Specify filename or directory')
    parser.add_argument('OUTFILE', type=str, 
                        help='Specify output file name (eg. result.csv)')
    parser.add_argument('--vowel', type=str, required=True,
                        help='Specify vowels (eg. IY1,IH1,EH1,AO1,AH1,AA1,AE1,UH1,UW1')
    parser.add_argument('--artic', type=str, required=True,
                        help='Specify articulatory features (eg. tr,td,tt,ja,ul,ll)')
    parser.add_argument('--acous', type=str, required=True,
                        help='Specify acoustic features (eg. f0,f1,f2,f3 or f0,f1,f2)')
    parser.add_argument('--n_points', type=int, required=True, 
                        help='Specify the number of data points to extract given a vowel (<10)')
    parser.add_argument('--ref', type=str, required=False, default=None,
                        help='Specify a formant reference file for accurate tracking')
    parser.add_argument('--skip_nans', type=bool, required=False, default=False,
                        help='If False (default), it will throw errors on NaNs')
    parser.add_argument('--write_log', type=bool, required=False, default=False,
                        help='if True (default: False), it will write a log file {OUTFILE}.log')
    args = parser.parse_args()
    args = check_arguments(args, verbose=True)

    # Run
    run(args)

    print('Done')
