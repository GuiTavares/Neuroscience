# Predict EEG Channel Reduction

- Project **EEG Channel Selection** for BCI Tasks

## Project Description

In this project, we will  identify EEG channels that are most likely to classify tasks in BCI applications.
The Project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Running Files
We will use #git and #conda:

1.Clone the repository

2.Run the requirements file

3.In the folder **pre_ml_data/from_gdrive/** there are the preprocessing outcomes already runner previously on TPU colab:

    a. From df_final3_top_10 to df_final3_top_100 csv files used for training of models
        
    b. Folder **/for_feature_selection_test/** with df_final3_top_100 csv file used for validation of Feature Selection 

4.In the folder **data/** there is a sample of BCICIV_2000 folder.

    a. From df_final3_top_10 to df_final3_top_100 csv files used for training of models
        
    b. Folder **/for_feature_selection_test/** with df_final3_top_100 csv file used for validation of Feature Selection 


5.In the folder **models/** there are models saved as pkl file to use in deployment.

    
6.In the folder **images/results_gdrive/after_feature_selection/** there are the outcomes:

    a. Logistic Regression Classifier
        I. best_features_lrc.txt
        II. fs_lrc_val_set_classification_report.txt
        III. selected_channels_lrc.txt
        
    b. Random Forest
        I. best_features_rf.txt
        II. fs_rf_val_set_classification_report.txt
        III. selected_channels_rf.txt
        
    c. XGBoost
        I. best_features_xgbc.txt
        II. fs_xgbc_val_set_classification_report.txt
        III. selected_channels_xgbc.txt

7.In the folder **model_drift/** will be models and new features for later use in case there is model drifting in the future.

Run: python channel_selection.py
python channel_selection_logging_and_tests.py

check the pylint score using the below:
pylint channel_selection.py
pylint channel_selection_logging_and_tests.py

To assist with meeting pep 8 guidelines, use autopep8 via the command line commands below:
autopep8 --in-place --aggressive --aggressive channel_selection_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive channel_selection.py
