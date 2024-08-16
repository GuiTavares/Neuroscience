import pandas as pd

df_verify = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_20.csv')

df_verify_top_20 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/df_all_persons_top_20')

new_df_verify_top_20 = df_verify_top_20[['EEG_channel', 'EEG_condition', 'EEG_person', 'EEG_epoch',
                               'Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'Mid_Beta',
                               'High_Beta', 'Gamma','EEG_median_value', 'EEG_average_value', 'EEG_std_value',
                                 'EEG_25_perc', 'EEG_75_perc',]].copy()

grouped_EEG_channel = new_df_verify_top_20.groupby('EEG_channel')
multi_df = {}

for name, group in grouped_EEG_channel:
    multi_df[name] = new_df_verify_top_20[(new_df_verify_top_20.EEG_channel == name)]
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

lista = list(new_df_verify_top_20.EEG_channel.unique())
#lista

dfs = []
for name in lista:
    # some process happens here resulting in
    # dataframe = result of above process
    dfs.append(multi_df[name])
final = pd.concat(dfs, axis = 1)

df_final3_top_20 = final.loc[:,~final.columns.duplicated()].copy()
df_final3_top_10 = df_final3_top_20.dropna(how='any')
del df_final3_top_20['Channel_FC5']
df_final3_top_20.to_csv(r'/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_20.csv', index = False)
df_final3_top_20
