import ReadFiles
import GetPerformances as gp
import preprocData as pp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn.pipeline import Pipeline

import xgboost as xgb
from bayes_opt import BayesianOptimization

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

## ajout de variables
addFeat = pp.AddFeatures()
addFeat.fit(df)
df = addFeat.transform(df)

###########
## nettoyage des données
impMedian = pp.ImputeMedian()
impMedian.fit(df)
df = impMedian.transform(df)


### validation croisée
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

allCols = df.columns[(df.isnull().sum()==0)]
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
           "Absolute_wind_direction_c"]
lstKeepCols = allCols.difference(notKeep).tolist()

#lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
#                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']

model = xgb.XGBRegressor(learning_rate=0.5, n_estimators=1500, max_depth=3,
                         colsample_bytree=1, subsample=1, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

kf = KFold(5)
scores = cross_validate(pipe, df, wtPower, cv=kf, scoring='neg_mean_absolute_error',
                        return_train_score=True, n_jobs=-1)

###########
## modèle sur rotor_speed<15
xtrain, xtest, ytrain, ytest = train_test_split(df, wtPower, test_size=0.2, random_state=345)

lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

allCols = df.columns[(df.isnull().sum()==0)]
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
           "Absolute_wind_direction_c"]
lstKeepCols = allCols.difference(notKeep).tolist()

## optimisation des paramètres par méthode bayesienne
# function to be maximized - must find (x=0;y=10)
def targetFunction(lstKeepCols=lstKeepCols, n_estimators=100, max_depth=3,
                   colsample_bytree=1, subsample=1):
    model = xgb.XGBRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth),
                             colsample_bytree=colsample_bytree, subsample=subsample, n_jobs=-1)
    pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                     ('model', model)])
    score = cross_val_score(pipe, xtrain, ytrain, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
    return score.mean()

# define parameters bounds
bounds = {'n_estimators': (100, 700), 'max_depth': (3, 9),
          'colsample_bytree': (0.25, 1.), 'subsample': (0.5, 1.)}
btypes = {'n_estimators':int, 'max_depth':int, 'colsample_bytree':float, 'subsample':float}
bo = BayesianOptimization(targetFunction, bounds) #,  ptypes=btypes)

bo.probe({'n_estimators':500, 'max_depth':6, 'colsample_bytree':0.75, 'subsample':0.75})
bo.probe({'n_estimators':250, 'max_depth':3, 'colsample_bytree':0.5, 'subsample':0.75})
bo.probe({'n_estimators':100, 'max_depth':9, 'colsample_bytree':1, 'subsample':1})

bo.maximize(init_points=5, n_iter=5)


## modèle
# sélectionner toutes les colonnes sans valeur manquante : quelles informations importantes ?
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

allCols = df.columns[(df.isnull().sum()==0)]
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
           "Absolute_wind_direction_c"]
lstKeepCols = allCols.difference(notKeep).tolist()


xtrain, xtest, ytrain, ytest = train_test_split(df, wtPower, test_size=0.2, random_state=345)

model = xgb.XGBRegressor(learning_rate=0.5, n_estimators=1200, max_depth=3,
                         colsample_bytree=1, subsample=1, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(xtrain, ytrain)

predTr = pd.Series(fitted.predict(xtrain), index=xtrain.index)
predTe = pd.Series(fitted.predict(xtest), index=xtest.index)


maeTr = gp.getMAE(ytrain, predTr)
maeTe = gp.getMAE(ytest, predTe)
print(f'MAE train = {maeTr}\nMAE test = {maeTe}')


gp.getAllResidPlot(ytrain, predTr, ytest, predTe)

varImp = fitted.named_steps['model'].feature_importances_
varImp = pd.Series(varImp, index=lstKeepCols).sort_values()


## mae = ?? train ; ?? test
## mape = ?? train ; ?? test
