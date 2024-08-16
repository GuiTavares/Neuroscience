'''
Project: The module contains functions of EEG channel selection analysis for NEUKO Startup

Author: Guilherme Tavares

Date: August 2024
'''

from os import getcwd, listdir
import sys
#from keras.callbacks import ModelCheckpoint
import scipy.io
import numpy as np
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
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import joblib
import csv
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import pickle
import ast

# loading raw data from a specific subject and run
def load_raw_data(source_directory, subject_id, run_number):
    subject_directory = os.path.join(source_directory, f'S{subject_id:03d}')
    edf_file = os.path.join(subject_directory, f'S{subject_id:03d}R{run_number:02d}.edf')
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    return raw
    
# creating targets based on the kind of tasks
def modify_annotations(raw, round_number):
    """ 
    Modify annotations based on 
    the round number.
    
    """
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

def import_data(source_directory):
    '''
    Returns data_frame for the edf file found at source_directory

    input:
            source_directory: a path to the edf file
    output:
            df: pandas data_frame
    '''
    # defining the alias dictionary
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
    'Iz..': 'Iz'}
    
    # defining columns' names
    col_names =  ['time', 'condition', 'epoch', 'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.',
       'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
       'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.',
       'Fp2.', 'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..',
       'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..',
       'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..', 'P3..', 'P1..',
       'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.',
       'Po8.', 'O1..', 'Oz..', 'O2..', 'Iz..', 'volunteer']
    
    # creating an empty data_frame
    my_df  = pd.DataFrame(columns = col_names)
    
    # fixing columns' names
    channels_of_interest = list(alias_dict.keys())
    
    for subject in range(33,35):
        for round in range(3, 15):
            if round not in [4, 8, 12, 6, 10, 14]:  # Pegando apenas os rounds sem evento feito, com visualization.
                
                # loading raw data
                raw = load_raw_data(source_directory, subject_id=subject, run_number=round)
                
                # fixing columns' names
                raw.pick_channels(channels_of_interest)
                
                # Modificar as anotações com base no número do round
                modify_annotations(raw, round_number=round)
                
                # filtering
                notch_freq = 60  # Hz
                notch_width = 2  # Hz
                
                # resampling the data
                filt_raw = raw.resample(sfreq=128) 
                filt_raw = filt_raw.copy().notch_filter(freqs=notch_freq, notch_widths=notch_width)
                filt_raw = filt_raw.copy().filter(2, 40, method='fir') # ver mais um pouco, tipo 1
    
                # defining montage of events 
                filt_raw.info.set_montage('standard_1005', match_alias=alias_dict)
                unique_labels = np.unique(filt_raw.annotations.description)
                event_id = {label: idx for idx, label in enumerate(unique_labels)}
                events, event_id = mne.events_from_annotations(filt_raw, event_id=int)
    
                # interpolation of bad channels
                threshold = 0.4
                
                # creating epochs
                epoch = mne.Epochs(filt_raw, events=events, tmin=0, tmax=4, proj=False, baseline=None, preload=True, verbose=False, picks='eeg')
    
                # creating data_frame
                df = epoch.to_data_frame()
                df['volunteer'] = subject

                # cocatening data_frame vertically
                my_df = pd.concat([my_df.reset_index(drop=True), df.reset_index(drop=True)], axis= 0)

    # saving file
    my_df.to_csv('./pre_ml_data/my_df/my_df.csv', index=False)  
    
    return my_df

def perform_feature_engineering(source_directory):
    '''
    input:
              source_directory: with csv where we can get a pandas dataframe

    output:
              data_frame: pandas dataframe with new features
    '''   
    # load data
    my_df = my_df = pd.read_csv(source_directory)
    
    # copy data_frame
    data_frame_poc2 = my_df.copy()

    # delating data_frame
    del mf_df
    
    # cast some variables to int
    data_frame_poc2["condition"] = data_frame_poc2["condition"].astype(int)
    data_frame_poc2["epoch"] = data_frame_poc2["epoch"].astype(int)

    # defining the alias dictionary
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
    'Iz..': 'Iz'}
    
    # cast some variables to int
    data_frame_poc2.rename(columns = alias_dict, inplace=True)

    # copy data_frame
    data_frame_poc3 = data_frame_poc2.set_index("epoch")

    # deleting data_frame intermediate variable
    del data_frame_poc2

    # reseting index of the data_frame
    data_frame_poc3 = data_frame_poc3.reset_index()

    # creating columns that will be used for new feature computing
    col_names =  ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'Mid_Beta',
                  'High_Beta', 'Gamma', 'EEG_condition', 'EEG_channel', 'EEG_epoch',
                  'EEG_median_value', 'EEG_average_value', 'EEG_std_value', 'EEG_25_perc','EEG_75_perc']
    
    # creating empty data_frame
    df_all_persons = pd.DataFrame(columns = col_names)

    # looping into each participant
    for person in data_frame_poc3.volunteer.unique():
        
        # looping into each EEG channel
        for ch in data_frame_poc3.columns[3:66]:

            # looping into each condition or target
            for cond in data_frame_poc3['condition'].unique():
                try:
                    # looping into each epoch
                    for epc in data_frame_poc3['epoch'].loc[(data_frame_poc3.condition == cond)].unique():
                        
                        # copy data_frame with restrictions from loops and reseting index
                        data_full = data_frame_poc3.loc[(data_frame_poc3.condition == cond) & 
                                                        (data_frame_poc3.epoch == epc)]
                        
                        # reseting index
                        data_full = data_full.reset_index()
                        
                        # copy data_frame
                        data_full2 = data_full[['epoch', 'time', 'condition', ch,'volunteer']].copy()
                        #data_full2[['epoch', 'time', 'condition', ch,'volunteer']]
                        
                        # Index of the 'ch' column
                        column_index_to_rename = -2  
                        new_column_name = 'EEG_value'
                        
                        # renaming column with EEG channel name to value of this channel
                        data_full2.rename(columns={data_full2.columns[column_index_to_rename]: new_column_name}, inplace=True)
                        
                        # copy values from EEG value column 
                        data = data_full2[["EEG_value"]].values
                        
                        # computing the discrete fourier transform 
                        fft_vals = np.absolute(np.fft.rfft(data))
                        
                        # using frequency sampling of 160 Hz
                        fs = 160
                        
                        # computing the discrete fourier transform
                        fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
                        
                        # dictionary of the EEG bands and sub-bands
                        eeg_bands = {'Delta': (0, 4),        # lethargic, not moving, not attentive
                                     'Theta': (4, 8),        # creative, intuitive; but may also be distracted, unfocused
                                     'Low_Alpha': (8, 10),   # inner-awareness of self, mind/body integration, balance
                                     'High_Alpha': (10, 12), # centering, healing, mind/body connection
                                     'Low_Beta': (12, 15),   # relaxed yet focused, integrated
                                     'Mid_Beta': (15, 18),   # alert, active, but not agitated
                                     'High_Beta': (18, 30),  # mental activity, e.g. math, planning; alertness, agitation
                                     'Gamma': (30, 45)}      # mental activity, e.g. math, planning; alertness, agitation
    
                        # associating EEG bands and sub-bands to fourier transforming
                        eeg_band_fft = dict()
                        for band in eeg_bands:
                            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
                            eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
                        
                        # creating an empty data_frame of EEG bands    
                        df_bands = pd.DataFrame(columns=['band', 'val'])
                        
                        # attributing each EEG band to band column
                        df_bands['band'] = eeg_bands.keys()

                        # attributing each EEG band value to val column
                        df_bands['val'] = [eeg_band_fft[band] for band in eeg_bands]

                        # reseting index after transposing data_frame
                        df_bands_transp = df_bands.set_index('band').T.reset_index()

                        # attributing each variable in the loops above to new columns of df_band 
                        df_bands_transp['EEG_condition'] = cond
                        df_bands_transp['EEG_channel'] = ch
                        df_bands_transp['EEG_epoch'] = epc
                        df_bands_transp['EEG_person'] = person
                        
                        # dropping index column and rename index as none
                        df_bands_transp = df_bands_transp.drop('index', axis=1)
                        df_bands_transp = df_bands_transp.rename_axis(None, axis=1)

                        # computing statistical features
                        df_bands_transp['EEG_median_value'] = data_full2[["EEG_value"]].quantile(0.5).values
                        df_bands_transp['EEG_average_value'] = data_full2[["EEG_value"]].mean().values
                        df_bands_transp['EEG_std_value'] = data_full2[["EEG_value"]].std().values
                        df_bands_transp['EEG_25_perc'] = data_full2[["EEG_value"]].quantile(0.25).values
                        df_bands_transp['EEG_75_perc'] = data_full2[["EEG_value"]].quantile(0.75).values

                        # concatening vertically the empty data_frame and the new created from feature engineering process
                        df_all_persons = pd.concat([df_all_persons, df_bands_transp], axis=0).reset_index()
                        df_all_persons = df_all_persons.drop('index', axis=1)
                
                except ZeroDivisionError:
                    print("Some participants have none values for some conditions per epochs. So that can generate zero values for fft computing.")
                    pass

    # saving file
    df_all_persons.to_csv('./pre_ml_data/df_all_persons/df_all_persons_33_34.csv', index=False)
    return df_all_persons
    

def feature_agreggating(source_directory):
    '''
    input:
              source_directory: with csv where we can get a pandas dataframe

    output:
              data_frame: pandas dataframe with new features
    ''' 
    # load data
    df = pd.read_csv(source_directory)
    
    # copy data
    df_verify = df.copy()

    # delating data_frame
    del df

    # copy data_frame selecting columns
    new_df_verify = df_verify[['EEG_channel', 'EEG_condition', 'EEG_person', 'EEG_epoch',
                               'Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'Mid_Beta',
                               'High_Beta', 'Gamma','EEG_median_value', 'EEG_average_value', 'EEG_std_value',
                                 'EEG_25_perc', 'EEG_75_perc']].copy()
    
    # grouping data_frame by EEG channel
    grouped_EEG_channel = new_df_verify.groupby('EEG_channel')
    multi_df = {}
    for name, group in grouped_EEG_channel:
        multi_df[name] = new_df_verify[(new_df_verify.EEG_channel == name)]
        multi_df[name].rename(columns={'EEG_channel': 'Channel_' + name,
                                    'EEG_median_value': 'median_Value_' + name,
                                    'Delta': 'Delta_' + name,
                                    'Theta': 'Theta_' + name,
                                    'Low_Alpha': 'Low_Alpha' + name, 'High_Alpha': 'High_Alpha_' + name,
                                    'Low_Beta': 'Low_Beta_' + name, 'Mid_Beta': 'Mid_Beta_' + name, 'High_Beta': 'High_Beta_' + name,
                                    'Gamma': 'Gamma_' + name,
                                    'EEG_average_value': 'average_value_' + name,'EEG_std_value': 'std_value_' + name,
                                    'EEG_25_perc': '25_perc_value_' + name,'EEG_75_perc': '75_perc_' + name}, inplace=True)
        multi_df[name] = multi_df[name].reset_index()
        multi_df[name] = multi_df[name].drop('index', axis=1)

    # detecting the unique channels
    lista = list(new_df_verify.EEG_channel.unique())
    dfs = []
    for name in lista:
        dfs.append(multi_df[name])
    final = pd.concat(dfs, axis = 1)
    
    df_final = final.loc[:,~final.columns.duplicated()].copy()
    df_final = df_final.dropna(how='any')
    df_final = df_final.drop(df_final.filter(regex='Channel_').columns, axis=1)
    df_final.to_csv('./pre_ml_data/for_ML/df_final.csv', index=False)

    return df_final

def train_models(source_directory):
    '''
    train, store model results: images + scores, and store models
    input:
              source_directory: with csv where we can get a pandas dataframe
    output:
              None
    '''

    # load data
    df = pd.read_csv(source_directory)
    
    # copy data
    final_df = df.copy()

    # delating data_frame
    del df

    # shuffling data to avoid bias in trainings
    final_df2 = final_df.sample(frac = 1)

    # separating a validation data set for assess overfiting
    final_df3_part = final_df2.sample(frac = 0.80)
    valid_final_part = final_df2.drop(final_df2.index)

    # obtaining features fom data set for training
    X = final_df2.drop(['EEG_condition'], axis=1)
    y = final_df2['EEG_condition'].values

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # splitting data set for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
    
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(bootstrap = True, max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 10)
    lrc = LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg")

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf  = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr  = lrc.predict(X_test)

def train_model_gdrive():
    '''
    training using preprocessed data from 'from_gdrive' folder, store model results: images + scores, and store models
    input:
              None
    output:
              None
    '''
    # load data
    df_final_10 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_10.csv')
    df_final_20 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_20.csv')
    df_final_30 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_30.csv')
    df_final_40 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_40.csv')
    df_final_50 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_50.csv')
    df_final_60 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_60.csv')
    df_final_70 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_70.csv')
    df_final_80 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_80.csv')
    df_final_90 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_90.csv')
    df_final_100 = pd.read_csv('./pre_ml_data/from_gdrive/df_final3_top_100.csv')

    # sub data_frames
    all_df = [df_final_10, df_final_20, df_final_30, df_final_40,df_final_50,
             df_final_60, df_final_70, df_final_80, df_final_90, df_final_100]
    
    # concatenating sub data_frames
    final_df2 = pd.concat(all_df, ignore_index=True)
    
    # eliminating garbage columns 
    final_df2 = final_df2.drop(final_df2.filter(regex='Channel_').columns, axis=1)
    final_df2 = final_df2.drop('EEG_person', axis=1)

    # shuffling data to avoid bias in trainings
    final_df2 = final_df2.sample(frac = 1)

    # obtaining features from data set for training
    X = final_df2.drop(['EEG_condition'], axis=1)
    y = final_df2['EEG_condition'].values

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # proportions between train, test, and validation sets
    train_ratio = 0.80
    validation_ratio = 0.10
    test_ratio = 0.10
    
    # splitting data_set for training, testing, and validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_ratio, random_state=42, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), 
                                                    random_state=42, shuffle=True)
    
    # RandomForestClassifier, LogisticRegression, and XGBoost
    rfc = RandomForestClassifier(bootstrap = True, max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, 
                                 n_estimators = 10)
    lrc = LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg")
    xgbc = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, subsample=0.5)

    # fitting Classifiers
    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    xgbc.fit(X_train, y_train) 

    # saving models
    joblib.dump(rfc, './models/rfc_model_all.pkl')
    joblib.dump(lrc, './models/logistic_model_all.pkl')
    joblib.dump(xgbc, './models/xgbc_model_all.pkl')

    # predictions for test and validation sets
    y_test_preds_rf = rfc.predict(X_test)
    y_val_preds_rf = rfc.predict(X_val)
    y_test_preds_lrc = lrc.predict(X_test)
    y_val_preds_lrc = lrc.predict(X_val)
    y_test_preds_xgbc = xgbc.predict(X_test)
    y_val_preds_xgbc = xgbc.predict(X_val)

    # fitting and prediction of LogisticRegression with feature selection
    select_lrc = SelectFromModel(LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg"), threshold='2.25*median')
    select_lrc.fit(X_train, y_train)
    X_train_fs_lrc = select_lrc.transform(X_train)
    
    # saving feature selection LogisticRegression model
    joblib.dump(select_lrc, './models/select_lrc_model_all.pkl')
    
    # obtaining the best features LogisticRegression
    pos = select_lrc.get_support(indices=True)
    colname = final_df2.columns[pos]

    # detecting channels from selected features
    cols = list(colname)
    cols = list(map(lambda x: x.replace('average_value_','').replace('Gamma_','').replace('Theta_','').replace('Low_Beta_','')
                 .replace('Mid_Beta_','').replace('25_perc_value_','').replace('75_perc_','').replace('High_Beta_','')
                 .replace('std_value_','').replace('Low_Alpha','').replace('High_Alpha_','').replace('median_Value_','')
                 .replace('average_value_',''),cols))
    selected_channels = cols

    # saving columns names as txt LogisticRegression
    with open("./images/results_gdrive/after_feature_selection/best_features_lrc.txt", "w") as output:
        output.write(str(list(colname)))
    
    # saving channel names as txt LogisticRegression
    with open("./images/results_gdrive/after_feature_selection/selected_channels_lrc.txt", "w") as output:
        output.write(str(list(selected_channels)))

    # fitting and prediction of RandomForestClassifier with feature selection
    select_rf = SelectFromModel(RandomForestClassifier(bootstrap = True, max_depth = 30, 
                                                        min_samples_leaf = 1,  min_samples_split = 2, 
                                                        n_estimators = 10), threshold='3.95*median')
    select_rf.fit(X_train, y_train)
    X_train_fs_rf = select_rf.transform(X_train)
    
    # saving feature selection RandomForestClassifier model
    joblib.dump(select_rf, './models/select_rf_model_all.pkl')
    
    # obtaining the best features RandomForestClassifier
    pos = select_rf.get_support(indices=True)
    colname = final_df2.columns[pos]

    # detecting channels from selected features
    cols = list(colname)
    cols = list(map(lambda x: x.replace('average_value_','').replace('Gamma_','').replace('Theta_','').replace('Low_Beta_','')
                 .replace('Mid_Beta_','').replace('25_perc_value_','').replace('75_perc_','').replace('High_Beta_','')
                 .replace('std_value_','').replace('Low_Alpha','').replace('High_Alpha_','').replace('median_Value_','')
                 .replace('average_value_',''),cols))
    selected_channels = cols

    # saving columns names as txt RandomForestClassifier
    with open("./images/results_gdrive/after_feature_selection/best_features_rf.txt", "w") as output:
        output.write(str(list(colname)))
        
    # saving channel names as txt RandomForestClassifier
    with open("./images/results_gdrive/after_feature_selection/selected_channels_rf.txt", "w") as output:
        output.write(str(list(selected_channels)))

    # fitting and prediction of XGBoost with feature selection
    select_xgbc = SelectFromModel(xgb.XGBClassifier(learning_rate=0.1, max_depth=5, subsample=0.5),
                                                        threshold='4.95*median')
    select_xgbc.fit(X_train, y_train)
    X_train_fs_xgbc = select_xgbc.transform(X_train)
    
    # saving feature selection XGBoost model
    joblib.dump(select_xgbc, './models/select_xgbc_model_all.pkl')
    
    # obtaining the best features XGBoost
    pos = select_xgbc.get_support(indices=True)
    colname = final_df2.columns[pos]

    # detecting channels from selected features
    cols = list(colname)
    cols = list(map(lambda x: x.replace('average_value_','').replace('Gamma_','').replace('Theta_','').replace('Low_Beta_','')
                 .replace('Mid_Beta_','').replace('25_perc_value_','').replace('75_perc_','').replace('High_Beta_','')
                 .replace('std_value_','').replace('Low_Alpha','').replace('High_Alpha_','').replace('median_Value_','')
                 .replace('average_value_',''),cols))
    selected_channels = cols

    # saving columns names as txt XGBoost
    with open("./images/results_gdrive/after_feature_selection/best_features_xgbc.txt", "w") as output:
        output.write(str(list(colname)))

    # saving channel names as txt XGBoost
    with open("./images/results_gdrive/after_feature_selection/selected_channels_xgbc.txt", "w") as output:
        output.write(str(list(selected_channels)))
        
    # classes from BCI_2000 task
    classes = ['class_0','class_1','class_2','class_3','class_4']

    # saving test_set report for RandomForestClassifier algorithm
    report = classification_report(y_test, y_test_preds_rf, target_names=classes)
    report_path = "./images/results_gdrive/rf_test_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    
    # saving validation_set report for RandomForestClassifier algorithm
    report_val = classification_report(y_val, y_val_preds_rf, target_names=classes)
    report_path = "./images/results_gdrive/rf_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report_val)
    text_file.close()

    # saving test_set report for LogisticRegression algorithm
    report = classification_report(y_test, y_test_preds_lrc, target_names=classes)
    report_path = "./images/results_gdrive/lrc_test_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()

    # saving validation_set report for LogisticRegression algorithm
    report_val = classification_report(y_val, y_val_preds_lrc, target_names=classes)
    report_path = "./images/results_gdrive/lrc_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report_val)
    text_file.close()
    
    # saving test_set report for XGBoost algorithm
    report = classification_report(y_test, y_test_preds_xgbc, target_names=classes)
    report_path = "./images/results_gdrive/xgbc_test_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()

    # saving validation_set report for XGBoost algorithm
    report_val = classification_report(y_val, y_val_preds_xgbc, target_names=classes)
    report_path = "./images/results_gdrive/xgbc_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report_val)
    text_file.close()
    
def feature_selection_lrc(source_directory_data, source_directory_features):
    '''
    performance testing of feature selection for Logistic Regression and storing results: images + scores, and store models
    inputs:
              source_directory_data: pth of a data_frame for validation of feature selection process
              source_directory_features: pth of selected features as a list of strings            
    output:
              None
    ''' 
    # load data
    df = pd.read_csv(source_directory_data)
    
    # concatenating sub data_frames
    final_df2 = df.copy()

    # saving memory
    del df
    
    # eliminating garbage columns 
    final_df2 = final_df2.drop(final_df2.filter(regex='Channel_').columns, axis=1)
    final_df2 = final_df2.drop('EEG_person', axis=1)

    # shuffling data to avoid bias in trainings
    final_df2 = final_df2.sample(frac = 1)

    # obtaining features from data set for training
    X = final_df2.drop(['EEG_condition'], axis=1)
    y = final_df2['EEG_condition'].values

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # load selected columns as the best features
    best_features = [line.strip() for line in open(source_directory_features)] 
    best_features = best_features[0]
    best_features = ast.literal_eval(best_features)

    # selecting data_frame with the best features for testing
    X_test_fs = final_df2[best_features].copy()
    X_test_fs = scaler.fit_transform(X_test_fs)

    # intanciating a lrc model with the best parameters
    lrc = LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg")
    
    # fitting Classifier
    lrc.fit(X_test_fs, y)

    y_preds_fs_lrc = lrc.predict(X_test_fs)

    # classes from BCI_2000 task
    classes = ['class_0','class_1','class_2','class_3','class_4']

    # saving report for LogisticRegression algorithm after feature selection
    report = classification_report(y, y_preds_fs_lrc, target_names=classes)
    report_path = "./images/results_gdrive/after_feature_selection/fs_lrc_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
    
def feature_selection_rf(source_directory_data, source_directory_features):
    '''
    performance testing of feature selection for Random Forest and storing results: images + scores, and store models
    inputs:
              source_directory_data: pth of a data_frame for validation of feature selection process
              source_directory_features: pth of selected features as a list of strings            
    output:
              None
    ''' 
    # load data
    df = pd.read_csv(source_directory_data)
    
    # concatenating sub data_frames
    final_df2 = df.copy()

    # saving memory
    del df
    
    # eliminating garbage columns 
    final_df2 = final_df2.drop(final_df2.filter(regex='Channel_').columns, axis=1)
    final_df2 = final_df2.drop('EEG_person', axis=1)

    # shuffling data to avoid bias in trainings
    final_df2 = final_df2.sample(frac = 1)

    # obtaining features from data set for training
    X = final_df2.drop(['EEG_condition'], axis=1)
    y = final_df2['EEG_condition'].values

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # load selected columns as the best features
    best_features = [line.strip() for line in open(source_directory_features)] 
    best_features = best_features[0]
    best_features = ast.literal_eval(best_features)

    # selecting data_frame with the best features for testing
    X_test_fs = final_df2[best_features].copy()
    X_test_fs = scaler.fit_transform(X_test_fs)

    # intanciating a rfc model with the best parameters
    rf = RandomForestClassifier(bootstrap = True, max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 10)
    
    # fitting Classifier
    rf.fit(X_test_fs, y)

    y_preds_fs_rf = rf.predict(X_test_fs)

    # classes from BCI_2000 task
    classes = ['class_0','class_1','class_2','class_3','class_4']

    # saving report for LogisticRegression algorithm after feature selection
    report = classification_report(y, y_preds_fs_rf, target_names=classes)
    report_path = "./images/results_gdrive/after_feature_selection/fs_rf_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()

def feature_selection_xgb(source_directory_data, source_directory_features):
    '''
    performance testing of feature selection for Random Forest and storing results: images + scores, and store models
    inputs:
              source_directory_data: pth of a data_frame for validation of feature selection process
              source_directory_features: pth of selected features as a list of strings            
    output:
              None
    ''' 
    # load data
    df = pd.read_csv(source_directory_data)
    
    # concatenating sub data_frames
    final_df2 = df.copy()

    # saving memory
    del df
    
    # eliminating garbage columns 
    final_df2 = final_df2.drop(final_df2.filter(regex='Channel_').columns, axis=1)
    final_df2 = final_df2.drop('EEG_person', axis=1)

    # shuffling data to avoid bias in trainings
    final_df2 = final_df2.sample(frac = 1)

    # obtaining features from data set for training
    X = final_df2.drop(['EEG_condition'], axis=1)
    y = final_df2['EEG_condition'].values

    # normalizing features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # load selected columns as the best features
    best_features = [line.strip() for line in open(source_directory_features)] 
    best_features = best_features[0]
    best_features = ast.literal_eval(best_features)

    # selecting data_frame with the best features for testing
    X_test_fs = final_df2[best_features].copy()
    X_test_fs = scaler.fit_transform(X_test_fs)

    # intanciating a xgboost model with the best parameters
    xgbc = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, subsample=0.5)
    
    # fitting Classifier
    xgbc.fit(X_test_fs, y)

    y_preds_fs_xgbc = xgbc.predict(X_test_fs)

    # classes from BCI_2000 task
    classes = ['class_0','class_1','class_2','class_3','class_4']

    # saving report for LogisticRegression algorithm after feature selection
    report = classification_report(y, y_preds_fs_xgbc, target_names=classes)
    report_path = "./images/results_gdrive/after_feature_selection/fs_xgbc_val_set_classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()
       
if __name__ == '__main__':
    
    # Import data
    #MY_DF = import_data(source_directory='./data/BCICIV_2000/files')
    #print(MY_DF.head())
    #print(MY_DF.tail())
    #print(MY_DF)

    # Feature engineering
    #DF_ALL_PERSONS = perform_feature_engineering(source_directory='./pre_ml_data/my_df/my_df.csv')
    #print(DF_ALL_PERSONS.head())

    # Data_frame for ML
    #DF_FINAL = feature_agreggating(source_directory='./pre_ml_data/df_all_persons/df_all_persons_33_34.csv')
    #print(DF_FINAL.head())

    # ML
    #train_models(source_directory='./pre_ml_data/for_ML/df_final.csv')
    #train_model_gdrive()
    
    # feature selection for LOGISTIC REGRESSION
    #feature_selection_lrc(source_directory_data='./pre_ml_data/from_gdrive/for_feature_selection_test/df_final3_top_110.csv',
    #                     source_directory_features='./images/results_gdrive/after_feature_selection/best_features_lrc.txt')

    #feature_selection_rf(source_directory_data='./pre_ml_data/from_gdrive/for_feature_selection_test/df_final3_top_110.csv',
    #                     source_directory_features='./images/results_gdrive/after_feature_selection/best_features_rf.txt')

    #feature_selection_xgb(source_directory_data='./pre_ml_data/from_gdrive/for_feature_selection_test/df_final3_top_110.csv',
    #                     source_directory_features='./images/results_gdrive/after_feature_selection/best_features_xgbc.txt')