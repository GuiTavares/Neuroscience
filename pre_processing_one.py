#pip install mne autoreject eeg_positions pyriemann
from os import getcwd, listdir
import sys
print(sys.version)
from keras.callbacks import ModelCheckpoint

# Numerical packages import
import scipy.io
import numpy as np


# mne imports
import mne
from mne import io
from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)
from mne.datasets import sample
from autoreject import AutoReject, get_rejection_threshold
from eeg_positions import get_elec_coords, get_available_elec_names

# Other imports
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import shutil

# Loading raw data from a specific subject and run
def load_raw_data(source_directory, subject_id, run_number):
    subject_directory = os.path.join(source_directory, f'S{subject_id:03d}')
    edf_file = os.path.join(subject_directory, f'S{subject_id:03d}R{run_number:02d}.edf')
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    return raw

# Defining the annotations dictionary
annotations_dict = {
    '0': 'rest',
    '1': 'left fist',
    '2': 'right fist',
    '3': 'both fists',
    '4': 'feet',
    '5': 'left fist (visualizing)',
    '6': 'right fist (visualizing)',
    '7': 'both fists (visualizing)',
    '8': 'feet (visualizing)'
}

# Defining the alias dictionary
alias_dict = {
    'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
    'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
    'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
    'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2',
    'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
    'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
    'Ft7.': 'FT7', 'Ft8.': 'FT8',
    'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10',
    'Tp7.': 'TP7', 'Tp8.': 'TP8',
    'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
    'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8',
    'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2',
    'Iz..': 'Iz'
}

def modify_annotations(raw, round_number):
    """Modify annotations based on the round number."""
    # Create a copy of the annotations
    annotations = raw.annotations.copy()

    # Modify annotations based on the round number
    if round_number in [3, 7, 11]:
        for idx, ann in enumerate(annotations):
            if ann['description'] == 'T0':
                annotations.description[idx] = '0'
            elif ann['description'] == 'T1':
                annotations.description[idx] = '1'
            elif ann['description'] == 'T2':
                annotations.description[idx] = '2'
    elif round_number in [5, 9, 13]:
        for idx, ann in enumerate(annotations):
            if ann['description'] == 'T0':
                annotations.description[idx] = '0'
            elif ann['description'] == 'T1':
                annotations.description[idx] = '3'
            elif ann['description'] == 'T2':
                annotations.description[idx] = '4'
    elif round_number in [4, 8, 12]:
        for idx, ann in enumerate(annotations):
            if ann['description'] == 'T0':
                annotations.description[idx] = '0'
            elif ann['description'] == 'T1':
                annotations.description[idx] = '1'
            elif ann['description'] == 'T2':
                annotations.description[idx] = '2'
    elif round_number in [6, 10, 14]:
        for idx, ann in enumerate(annotations):
            if ann['description'] == 'T0':
                annotations.description[idx] = '0'
            elif ann['description'] == 'T1':
                annotations.description[idx] = '3'
            elif ann['description'] == 'T2':
                annotations.description[idx] = '4'

    # Set the modified annotations to the raw data
    raw.set_annotations(annotations)

import mne
import numpy as np
import matplotlib.pyplot as plt

# Set log level
mne.set_log_level('error')

# Definir o diretório fonte contendo as pastas dos sujeitos
#source_directory = '/content/drive/MyDrive/NEUKO/data/BCICIV_2000/files'
#source_directory = './data/BCICIV_2000/files'

# Lista de canais de interesse
channels_of_interest = list(alias_dict.keys())

col_names =  ['time', 'condition', 'epoch', 'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.',
       'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
       'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.',
       'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..',
       'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..',
       'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..',
       'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.',
       'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..', 'volunteer']

my_df  = pd.DataFrame(columns = col_names)


for subject in range(81,91):
    for round in range(3, 15):
        if round not in [4, 8, 12, 6, 10, 14]:  # Pegando apenas os rounds sem evento feito, com visualization.
            raw = load_raw_data(source_directory, subject_id=subject, run_number=round)
            #raw.pick_channels(channels_of_interest)

            # Modificar as anotações com base no número do round
            modify_annotations(raw, round_number=round)

            # Pré-processamento dos dados
            notch_freq = 60  # Hz
            notch_width = 2  # Hz
            filt_raw = raw.resample(sfreq=128) # Resample the data
            filt_raw = filt_raw.copy().notch_filter(freqs=notch_freq, notch_widths=notch_width)
            filt_raw = filt_raw.copy().filter(2, 40, method='fir') # ver mais um pouco, tipo 1
            # filt_raw.plot(start=0, duration=8, scalings={'eeg': 1.5e-4})

            # Definir a montagem e os eventos
            filt_raw.info.set_montage('standard_1005', match_alias=alias_dict)
            unique_labels = np.unique(filt_raw.annotations.description)
            event_id = {label: idx for idx, label in enumerate(unique_labels)}
            events, event_id = mne.events_from_annotations(filt_raw, event_id=int)

            # Interpolação de canais ruins
            threshold = 0.4

            #channel_neighbors = compute_channel_adjacency(filt_raw)
            #bad_channels = find_bad_channels(filt_raw, threshold)
            #for channel in bad_channels:
            #    filt_raw.info['bads'].append(channel)
            #filt_raw.interpolate_bads(method='spline')

            # Normalização dos dados
            #data = filt_raw.get_data()
            #mean = np.mean(data, axis=1, keepdims=True)
            #std = np.std(data, axis=1, keepdims=True)
            #scaled_data = (data - mean) / std

            # Atualizar os dados normalizados de volta ao objeto Raw
            #filt_raw._data = scaled_data

            # Criar epochs
            epoch = mne.Epochs(filt_raw, events=events, tmin=0, tmax=4, proj=False, baseline=None, preload=True, verbose=False, picks='eeg')
            #epoch = epoch_completo.crop(tmin=0.35, tmax=0.55)

            #gui code
            df = epoch.to_data_frame()
            df['volunteer'] = subject

            my_df = pd.concat([my_df.reset_index(drop=True), df.reset_index(drop=True)], axis= 0)


data_frame_poc2 = my_df.copy()
data_frame_poc2["condition"] = data_frame_poc2["condition"].astype(int)
data_frame_poc2["epoch"] = data_frame_poc2["epoch"].astype(int)


data_frame_poc2.rename(columns=alias_dict, inplace=True)

data_frame_poc3 = data_frame_poc2.set_index("epoch")
data_frame_poc3 = data_frame_poc3.reset_index()

data_frame_poc3.head()


df_all_persons_col_names =  ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'Mid_Beta',
       'High_Beta', 'Gamma', 'EEG_condition', 'EEG_channel', 'EEG_epoch',
       'EEG_median_value', 'EEG_average_value', 'EEG_std_value', 'EEG_25_perc',
       'EEG_75_perc']

df_all_persons = pd.DataFrame(columns = df_all_persons_col_names)

for person in data_frame_poc3.volunteer.unique():
    for ch in data_frame_poc3.columns[3:66]:
        for cond in data_frame_poc3['condition'].unique():
            try:
                for epc in data_frame_poc3['epoch'].loc[(data_frame_poc3.condition == cond)].unique():
                    data_full = data_frame_poc3.loc[(data_frame_poc3.condition == cond) & (data_frame_poc3.epoch == epc)].reset_index()

                    data_full = data_full.reset_index()

                    data_full2 = data_full[['epoch', 'time', 'condition', ch,'volunteer']].copy()
                    data_full2[['epoch', 'time', 'condition', ch,'volunteer']]
                    column_index_to_rename = -2  # Index of the 'name' column
                    new_column_name = 'EEG_value'

                    data_full2.rename(columns={data_full2.columns[column_index_to_rename]: new_column_name}, inplace=True)

                    data = data_full2[["EEG_value"]].values

                    fft_vals = np.absolute(np.fft.rfft(data))

                    fft_freq = np.fft.rfftfreq(len(data), 1.0/160)
                    eeg_bands = {'Delta': (0, 4), #lethargic, not moving, not attentive
                             'Theta': (4, 8), #creative, intuitive; but may also be distracted, unfocused
                             'Low_Alpha': (8, 10), #inner-awareness of self, mind/body integration, balance
                             'High_Alpha': (10, 12), #centering, healing, mind/body connection
                             'Low_Beta': (12, 15), #relaxed yet focused, integrated
                             'Mid_Beta': (15, 18), #alert, active, but not agitated
                             'High_Beta': (18, 30), #mental activity, e.g. math, planning; alertness, agitation
                             'Gamma': (30, 45)} #mental activity, e.g. math, planning; alertness, agitation

                    eeg_band_fft = dict()

                    for band in eeg_bands:
                        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
                    df_bands = pd.DataFrame(columns=['band', 'val'])
                    df_bands['band'] = eeg_bands.keys()
                    df_bands['val'] = [eeg_band_fft[band] for band in eeg_bands]
                    df_bands_transp = df_bands.set_index('band').T.reset_index()
                    df_bands_transp['EEG_condition'] = cond
                    df_bands_transp['EEG_channel'] = ch
                    df_bands_transp['EEG_epoch'] = epc
                    df_bands_transp['EEG_person'] = person

                    df_bands_transp = df_bands_transp.drop('index', axis=1)
                    df_bands_transp = df_bands_transp.rename_axis(None, axis=1)

                    df_bands_transp['EEG_median_value'] = data_full2[["EEG_value"]].quantile(0.5).values
                    df_bands_transp['EEG_average_value'] = data_full2[["EEG_value"]].mean().values
                    df_bands_transp['EEG_std_value'] = data_full2[["EEG_value"]].std().values
                    df_bands_transp['EEG_25_perc'] = data_full2[["EEG_value"]].quantile(0.25).values
                    df_bands_transp['EEG_75_perc'] = data_full2[["EEG_value"]].quantile(0.75).values

                    df_all_persons = pd.concat([df_all_persons, df_bands_transp], axis=0).reset_index()
                    df_all_persons = df_all_persons.drop('index', axis=1)
            except:
                pass



df_all_persons.to_csv('/content/drive/My Drive/NEUKO/outputs/df_all_persons_top_90', index=False)

df_verify = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/df_all_persons_top_90')
print(df_verify)
