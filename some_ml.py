clf2 = RandomForestClassifier(bootstrap = True, max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 10)

scaler = StandardScaler()


X = final_df2_part.drop(['EEG_condition'], axis=1)
y = final_df2_part['EEG_condition'].values

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

select = SelectFromModel(RandomForestClassifier(bootstrap = True, max_depth = 30, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 10),
                                                        threshold='3.95*median')
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("Original Feature set", X_train.shape)
print("subset of Features", X_train_selected.shape)

clf2.fit(X_train, y_train)
print("Accuracy on training set is : {}".format(clf2.score(X_train, y_train)))
print("Accuracy on test set is : {}".format(clf2.score(X_test, y_test)))
Y_test_pred = clf2.predict(X_test)
print(classification_report(y_test, Y_test_pred))

x_valid_normal = valid_final_part.drop(['EEG_condition'], axis=1)
x_valid_normal = scaler.fit_transform(x_valid_normal)
y_valid = valid_final_part['EEG_condition'].values
Y_val_pred = clf2.predict(x_valid_normal)
print("Accuracy on validation set is : {}".format(clf2.score(x_valid_normal, y_valid)))
print(classification_report(y_valid, Y_val_pred))

clf2.fit(X_train_selected, y_train)
print("Accuracy AFTER FS on training set is : {}".format(clf2.score(X_train_selected, y_train)))
#print("Accuracy AFTER FS on test set is : {}".format(clf2.score(X_test, y_test)))
#Y_test_pred = clf2.predict(X_test)
#print(classification_report(y_test, Y_test_pred))
print(select.get_support(indices=True))
pos = select.get_support(indices=True)
colname = final_df2_part.columns[pos]
print("Feature names: ",colname)
x = colname
x = list(x)
set_column(x)import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict


def set_column(list_columns):
  for index, value in enumerate(x):
    # if value includes [br] then replace it with <br>
    if 'average_value_' in value:
        x[index] = value.replace('average_value_', '')
    elif  'Gamma_' in value:
      x[index] = value.replace('Gamma_', '')
    elif  'Theta_' in value:
      x[index] = value.replace('Theta_', '')
    elif  'Low_Beta_' in value:
      x[index] = value.replace('Low_Beta_', '')
    elif  'Mid_Beta_' in value:
      x[index] = value.replace('Mid_Beta_', '')
    elif  '25_perc_value_' in value:
      x[index] = value.replace('25_perc_value_', '')
    elif  '75_perc_' in value:
      x[index] = value.replace('75_perc_', '')
    elif  'High_Beta_' in value:
      x[index] = value.replace('High_Beta_', '')
    elif  'std_value_' in value:
      x[index] = value.replace('std_value_', '')
    elif  'Low_Alpha' in value:
      x[index] = value.replace('Low_Alpha', '')
    elif  'High_Alpha_' in value:
      x[index] = value.replace('High_Alpha_', '')
    elif  'median_Value_' in value:
      x[index] = value.replace('median_Value_', '')
    elif  'average_value_' in value:
      x[index] = value.replace('average_value_', '')
  y=set(x)

  return len(y), y




df_final_10 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_10.csv')
df_final_20 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_20.csv')
df_final_30 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_30.csv')
df_final_40 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_40.csv')
df_final_50 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_50.csv')
df_final_60 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_60.csv')
df_final_70 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_70.csv')
df_final_80 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_80.csv')
df_final_90 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_90.csv')
df_final_100 = pd.read_csv('/content/drive/My Drive/NEUKO/outputs/pre_ML/df_final3_top_100.csv')



all_df = [df_final_10, df_final_20, df_final_30, df_final_40,df_final_50,
          df_final_60, df_final_70, df_final_80, df_final_90, df_final_100]

final_df2 = pd.concat(all_df, ignore_index=True)
final_df2 = final_df2.drop(final_df2.filter(regex='Channel_').columns, axis=1)
final_df2 = final_df2.drop('EEG_person', axis=1)
final_df2 = final_df2.sample(frac = 1)
final_df2_part = final_df2.sample(frac = 0.80)
valid_final_part = final_df2.drop(final_df2_part.index)


grid={"C":np.logspace(-3,3,7), "penalty":["l2"], "solver":['liblinear','newton-cg']}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=3,scoring='f1_micro')
logreg_cv.fit(X_train, y_train)

clf2 = LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg")

scaler = StandardScaler()


X = final_df2_part.drop(['EEG_condition'], axis=1)
y = final_df2_part['EEG_condition'].values

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

select = SelectFromModel(LogisticRegression(C=1.e-02,penalty="l2", solver="newton-cg"), threshold='2.25*median')
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
print("Original Feature set", X_train.shape)
print("subset of Features", X_train_selected.shape)

clf2.fit(X_train, y_train)
print("Accuracy on training set is : {}".format(clf2.score(X_train, y_train)))
print("Accuracy on test set is : {}".format(clf2.score(X_test, y_test)))
Y_test_pred = clf2.predict(X_test)
print(classification_report(y_test, Y_test_pred))

x_valid_normal = valid_final_part.drop(['EEG_condition'], axis=1)
x_valid_normal = scaler.fit_transform(x_valid_normal)
y_valid = valid_final_part['EEG_condition'].values
Y_val_pred = clf2.predict(x_valid_normal)
print("Accuracy on validation set is : {}".format(clf2.score(x_valid_normal, y_valid)))
print(classification_report(y_valid, Y_val_pred))

clf2.fit(X_train_selected, y_train)
print("Accuracy AFTER FS on training set is : {}".format(clf2.score(X_train_selected, y_train)))
print(select.get_support(indices=True))
pos = select.get_support(indices=True)
colname = final_df2_part.columns[pos]
print("Feature names: ",colname)
x = colname
x = list(x)
set_column(x)


