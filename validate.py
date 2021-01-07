'''Validate the result file

This script validates the result file created from extractor.py

2021-01-07 first created
'''

# Basic
import os
import pickle
import argparse
import numpy as np
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
# Custom
from utils import parse_ieee_filename, load_dictionary


def check_arguments(args, verbose=True):
    '''Check arguments'''
    assert os.path.exists(
        args.RESULTFILE), f'result file does not exist ({args.RESULTFILE})'
    assert os.path.exists(
        args.DATAFILE), f'data file does not exist ({args.DATAFILE})'
    df = pd.read_csv(args.RESULTFILE)
    assert ~df.empty, f'result file is empty'
    fid, speaker, block, sentence, repetition, rate = parse_ieee_filename(args.fileid)
    data = df.loc[(
        (df['FileID'] == fid)
        & (df['Speaker'] == speaker)
        & (df['Block'] == block)
        & (df['Sent'] == sentence)
        & (df['Rep'] == repetition)
        & (df['Rate'] == rate)
    )]
    assert ~data.empty, f'Check your fileid. No matching data from fileid is found ({args.fileid}).'

    if verbose:
        pprint(vars(args))
    return args


def plot_wav(ax, sig, sr):
    '''Plot waveform'''
    n_ticks = 5
    librosa.display.waveplot(sig, sr=sr, ax=ax, x_axis='none')

    ax.set_xlabel('')
    xticks = np.linspace(0, len(sig)/sr, n_ticks)
    ax.xaxis.set_ticks(xticks)
    ax.set_xticklabels([f'{t:.2f}' for t in xticks])
    return ax


def plot_specgram(ax, sig, sr):
    '''Plot spectrogram'''
    n_ticks = 5
    cut_off = 5500
    S = librosa.stft(sig)
    S_db = librosa.amplitude_to_db(abs(S))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='gray_r')

    ax.set_ylim([0, cut_off])
    ax.set_xlabel('')
    ax.set_xticks([])
    return ax


def plot_artic_trajectory(arr, D, channel_names):
    '''Plot articulatory trajectories'''
    x_idx = 0
    z_idx = 2
    n_ticks = 5
    ema_sr = int(D['TR']['SRATE'])
    for ax, ch in zip(arr, channel_names):
        data = D[ch]['SIGNAL']
        ax.plot(data[:, x_idx],  '-', color='gray', label='x')
        ax.plot(data[:, z_idx], '--', color='k',  label='y')

        ax.set_xlim([0, len(data)])
        ax.legend(loc='upper right')
        ax.set_ylabel(ch, fontsize=10)
        xticks = np.linspace(0, len(data), n_ticks)
        ax.xaxis.set_ticks(xticks)
        ax.set_xticklabels([f'{t/ema_sr:.2f}' for t in xticks])
        ax.xaxis.set_ticks_position('none') 
    return arr


def add_formants(ax, df):
    '''Add formant values and vowel/word labels'''
    cut_off = 5500
    cols = df.columns.to_list()
    for i, (time, label) in enumerate(zip(df.TimeSec.values, df.Vowel.values)):
        ax.plot([time, time], [0, cut_off], '--', color='gray', alpha=0.5)
        if 'F1' in cols:
            data = df['F1'].loc[df.TimeSec==time].values[0]
            ax.plot(time, data, 'r.', markersize=3, label='F1')
        if 'F2' in cols:
            data = df['F2'].loc[df.TimeSec==time].values[0]
            ax.plot(time, data, 'g.', markersize=3, label='F2')
        if 'F3' in cols:
            data = df['F3'].loc[df.TimeSec==time].values[0]
            ax.plot(time, data, 'b.', markersize=3, label='F3')
        if df.TimeAt.iloc[i] == 0:
            word = df.Word.iloc[i]
            ax.text(time, cut_off*0.9, f'{label}\n({word})', fontsize=7)
    return ax

def add_artic_points(ax, df, ema_sr, ch, channel_names):
    '''Add articulatory sampled data points'''
    vmax = ax.get_ylim()
    artic_col = [ch+c for ch in channel_names for c in ['x','z']]
    cols = df.columns.to_list()
    for i, time in enumerate(df.TimeSec.values):
        seg = int(time * ema_sr)    
        ax.plot([seg, seg], [vmax[0], vmax[1]], '--', 
                color='gray', linewidth=0.5, alpha=0.5)
        x, z = df[ch+'x'].iloc[i], df[ch+'z'].iloc[i]
        ax.plot(seg, x, 'bo', markersize=2)
        ax.plot(seg, z, 'ro', markersize=2)
    return ax


def run(args):
    field_names = ['NAME', 'SRATE', 'SIGNAL', 'SOURCE', 'SENTENCE', 'WORDS', 'PHONES', 'LABELS']
    channel_names = ['TR', 'TB', 'TT', 'JAW', 'UL', 'LL']
    audio_channel = 'AUDIO'
    fid, speaker, block, sentence, repetition, rate = parse_ieee_filename(args.fileid)

    D = load_dictionary(args.DATAFILE, field_names, channel_names, audio_channel)
    sig, sr = D['AUDIO']['SIGNAL'], int(D['AUDIO']['SRATE'])
    ema_sr = int(D['TR']['SRATE'])

    df = pd.read_csv(args.RESULTFILE)
    df = df.loc[(
        (df['FileID'] == fid)
        & (df['Speaker'] == speaker)
        & (df['Block'] == block)
        & (df['Sent'] == sentence)
        & (df['Rep'] == repetition)
        & (df['Rate'] == rate)
    )].reset_index(drop=True)

    # Set figure
    n_axes = 2 + len(channel_names)
    sns.set_theme(context='paper', style='whitegrid', font_scale=0.8, rc={'axes.grid': False})
    fig, arr = plt.subplots(n_axes, 1, figsize=(8,8), facecolor='white')

    # Acoustic plots
    arr[0] = plot_wav(arr[0], sig, sr)
    arr[1] = plot_specgram(arr[1], sig, sr)
    arr[1] = add_formants(arr[1], df)

    # Articulatory plots
    arr[2:] = plot_artic_trajectory(arr[2:], D, channel_names)
    for ax, ch in zip(arr[2:], channel_names):
        ax = add_artic_points(ax, df, ema_sr, ch, channel_names)

    # Prettify
    fig.suptitle(args.fileid, fontsize=15)
    fig.tight_layout(h_pad=-1)
    plt.show()
    print('Done')


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Validate the result file from extractor.py',
                                     epilog='See https://github.com/jaekookang/Articulatory-Data-Extractor')
    parser.add_argument('RESULTFILE', type=str,
                        help='Specify the result file created from extractor.py (*.csv)')
    parser.add_argument('DATAFILE', type=str,
                        help='Specify the data file (*.pkl)')
    parser.add_argument('--fileid', type=str, required=True,
                        help='Specify fileid; eg. F01_B01_S01_R01_N')
    args = parser.parse_args()
    args = check_arguments(args, verbose=True)

    run(args)
