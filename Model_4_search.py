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

class SplitEvents():
    def __init__(self, condiSplit):
        self.condiSplit = condiSplit
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        X_t = X[self.condiSplit]
        X_f = X[~self.condiSplit]
        return X_t, X_f

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

## ajout de variables
addFeat = pp.AddFeatures()
addFeat.fit(df)
df = addFeat.transform(df)

## séparation des individus suivant rotor_speed => non linéarité entre rotor_speed et target
condi = df.Rotor_speed3>=16**3 # Rotor_speed>=15 ## test avec 16
splitDataset = SplitEvents(condiSplit=condi)
dfsup, dfinf = splitDataset.transform(df)
wtPowersup, wtPowerinf = splitDataset.transform(wtPower)

###########
## nettoyage des données
impMedian = pp.ImputeMedian()
impMedian.fit(dfinf)
dfinf = impMedian.transform(dfinf)

impMedian = pp.ImputeMedian()
impMedian.fit(dfsup)
dfsup = impMedian.transform(dfsup)


### validation croisée
lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

#allCols = dfinf.columns
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
#notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
#           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
#           "Absolute_wind_direction_c"]
#lstKeepCols = allCols.difference(notKeep).tolist()

lstKeepCols = ['Generator_bearing_1_temperature',
               'Outdoor_temperature_max', 'Nacelle_temperature_max', 'Rotor_speed_max',
               'Pitch_angle_x_Rotor_speedStd', 'Pitch_angleStd_x_Rotor_speed',
               'Generator_stator_temperatureMin_x_Pitch_angleStd', 'Pitch_angle',
               'Pitch_angle_max', 'Rotor_speed3']

#lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
#                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
#model = GradientBoostingRegressor(n_estimators=500, max_depth=3)
model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1500, max_depth=3,
                         colsample_bytree=0.75, subsample=1, reg_lambda=0.05, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

kf = KFold(5, shuffle=True)
scores = cross_val_score(pipe, dfinf, wtPowerinf, cv=kf, scoring='neg_mean_absolute_error')

scores = cross_validate(pipe, dfinf, wtPowerinf, cv=kf, scoring='neg_mean_absolute_error',
                        return_train_score=True, n_jobs=-1)

###########
## modèle sur rotor_speed<15
xtrainI, xtestI, ytrainI, ytestI = train_test_split(dfinf, wtPowerinf, test_size=0.2, random_state=345)

#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
lstKeepCols = ['Generator_bearing_1_temperature',
               'Outdoor_temperature_max', 'Nacelle_temperature_max', 'Rotor_speed_max',
               'Pitch_angle_x_Rotor_speedStd', 'Pitch_angleStd_x_Rotor_speed',
               'Generator_stator_temperatureMin_x_Pitch_angleStd', 'Pitch_angle',
               'Pitch_angle_max', 'Rotor_speed3']

## optimisation des paramètres par méthode bayesienne
# function to be maximized - must find (x=0;y=10)
def targetFunction(lstKeepCols=lstKeepCols, n_estimators=100, max_depth=3,
                   colsample_bytree=1, subsample=1):
    model = xgb.XGBRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth),
                             colsample_bytree=colsample_bytree, subsample=subsample, n_jobs=-1)
    pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                     ('model', model)])
    score = cross_val_score(pipe, xtrainI, ytrainI, cv=2, scoring='neg_mean_absolute_error', n_jobs=-1)
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
lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
lstKeepCols = ['Generator_bearing_1_temperature',
               'Outdoor_temperature_max', 'Nacelle_temperature_max', 'Rotor_speed_max',
               'Pitch_angle_x_Rotor_speedStd', 'Pitch_angleStd_x_Rotor_speed',
               'Generator_stator_temperatureMin_x_Pitch_angleStd', 'Pitch_angle',
               'Pitch_angle_max', 'Rotor_speed3']
# meileure sélection :
#lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
#                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
#lstKeepCols = ['Pitch_angle', 'Rotor_speed3', \
#               'Gearbox_bearing_1_temperature', 'Generator_stator_temperature_std', \
#               'Turbulence']
#model = xgb.XGBRegressor(colsample_bytree=1, subsample=1, n_estimators=680, max_depth=9, n_jobs=-1)
model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1500, max_depth=4,
                         colsample_bytree=0.75, subsample=1, reg_lambda=0.05, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(xtrainI, ytrainI)

predTrI = pd.Series(fitted.predict(xtrainI), index=xtrainI.index)
predTeI = pd.Series(fitted.predict(xtestI), index=xtestI.index)


maeTrI = gp.getMAE(ytrainI, predTrI)
maeTeI = gp.getMAE(ytestI, predTeI)
print(f'MAE train = {maeTrI}\nMAE test = {maeTeI}')


Rotor_speed_std                                     0.001036
Generator_stator_temperature                        0.001052
Generator_bearing_2_temperature_max                 0.001100
Generator_bearing_2_temperature_min                 0.001250
Generator_bearing_1_temperature                     0.001311
Generator_bearing_2_temperature                     0.001316
Outdoor_temperature_max                             0.001368
Nacelle_temperature_max                             0.001372
Rotor_speed_max                                     0.001964
Pitch_angle_x_Rotor_speedStd                        0.002274
Pitch_angleStd_x_Rotor_speed                        0.002336
Generator_stator_temperatureMin_x_Pitch_angleStd    0.002984
Pitch_angle                                         0.003566
Generator_speed_max                                 0.004212
Pitch_angle_max                                     0.004867
Generator_speed                                     0.121479
Rotor_speed3                                        0.274829
Rotor_speed                                         0.556355


## optimisation des hyper-paramètres de la forêt
#from sklearn.model_selection import cross_val_score
#
#def objScore(lstKeepCols=lstKeepCols, n_estimators=10, max_depth=3) :
#  model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=int(max_depth), n_jobs=-1)
#  pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
#                 ('model', model)])
#  score = cross_val_score(pipe, xtrainI, ytrainI, cv=3, scoring='mae', n_jobs=-1)
#  #pipe.fit(xtrainI, ytrainI)
#  #pred = pipe.predict(xtestI)
#  #score = gp.getMAE(ytestI, pred)
#  return score.mean()
#
#params = {'max_depth':(9,16)} #'model__n_estimators':(90,110)
#bayesOpt = bayes_opt.BayesianOptimization(objScore, params)
#
#bayesOpt.maximize(nit_points=5, n_iter=5)

## voir pour ajouter info int/float dans dictionnaire passé à bayesOptim

##### validation croisée pour rotor_speed >15
allCols = dfsup.columns
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
           "Absolute_wind_direction_c"]
lstKeepCols = allCols.difference(notKeep).tolist()
modelsup = xgb.XGBRegressor(earning_rate=0.1, n_estimators=1500, max_depth=3,
                            colsample_bytree=0.5, subsample=0.75, n_jobs=-1)
pipeSup = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                    ('model', modelsup)])

kf = KFold(5, shuffle=True)
scores = cross_val_score(pipeSup, dfsup, wtPowersup, cv=kf, scoring='neg_mean_absolute_error')

scores = cross_validate(pipeSup, dfsup, wtPowersup, cv=kf, scoring='neg_mean_absolute_error', return_train_score=True, n_jobs=-1)


###########
## modèle sur rotor_speed>=15
xtrainS, xtestS, ytrainS, ytestS = train_test_split(dfsup, wtPowersup, test_size=0.2, random_state=3456)

lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
lstKeepCols = ['Absolute_wind_direction', 'Nacelle_angle_min', 'Outdoor_temperature',
               'Gearbox_oil_sump_temperature', 'Hub_temperature_max',
               'Outdoor_temperature_min',
               'Rotor_bearing_temperature_max', 'Hub_temperature_min',
               'Generator_stator_temperature_std', 'Outdoor_temperature_std',
               'Rotor_speed_std', 'Nacelle_temperature_min',
               'Gearbox_bearing_2_temperature_std',
               #'Gearbox_bearing_1_temperature_std',
               'Generator_speed_min',
               'Gearbox_bearing_1_temperature', 'Turbulence',
               #'Generator_bearing_2_temperature',
               'Generator_speed_std',
               'Nacelle_temperature', 'Generator_bearing_1_temperature',
               'Generator_stator_temperature_max', 'Pitch_angle_x_Rotor_speedStd',
               'Generator_speed_max',
               'Generator_stator_temperature',
               'Pitch_angleMax_x_Pitch_angleMin', 'Nacelle_temperature_max',
               'Generator_bearing_1_temperature_min', 'Pitch_angle',
               'Pitch_angle_max', 'Rotor_speed', 'Pitch_angle_std', 'Rotor_speed3',
               'Generator_stator_temperatureMin_x_Pitch_angleStd',
               'Pitch_angleStd_x_Rotor_speed']
#allCols = dfsup.columns
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
#notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
#           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
#           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
#cleanCols = allCols[lstCols].difference(notKeep).tolist()

modelsup = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1500, max_depth=3,
                            colsample_bytree=0.5, subsample=0.75, n_jobs=-1)
pipeSup = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                    ('model', modelsup)])

fittedSup = pipeSup.fit(xtrainS, ytrainS)

predTrS = pd.Series(fittedSup.predict(xtrainS), index=xtrainS.index)
predTeS = pd.Series(fittedSup.predict(xtestS), index=xtestS.index)

maeTrS = gp.getMAE(ytrainS, predTrS)
maeTeS = gp.getMAE(ytestS, predTeS)
print(f'MAE train = {maeTrS}\nMAE test = {maeTeS}')

## importance des variables
#gp.plotImportance(xtrain[lstKeepCols], fitted)

##############
## prédiction finale
predTr = pd.concat((predTrI, predTrS),axis=0).sort_index()
predTe = pd.concat((predTeI, predTeS),axis=0).sort_index()


ytrain = pd.concat((ytrainI, ytrainS),axis=0).sort_index()
ytest = pd.concat((ytestI, ytestS),axis=0).sort_index()

maeTr = gp.getMAE(ytrain, predTr)
maeTe = gp.getMAE(ytest, predTe)

print(f'MAE train = {maeTr}\nMAE test = {maeTe}')

gp.getAllResidPlot(ytrain, predTr, ytest, predTe)

## mae = 16 train ; 17 test
## mape = 1.2 train ; 0.90 test


## tester modèle pour rotor_speed > 15
# en calculant combinaisons de variable
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

allCols = dfsup.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
cleanCols = allCols[lstCols].difference(notKeep).tolist()

modelsup = xgb.XGBRegressor(learning_rate=0.3, n_estimators=100, max_depth=3,
                            colsample_bytree=0.5, subsample=0.75, n_jobs=-1)
poly = Pipeline([('selectCols', pp.SelectColumns(cleanCols)),
                 ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                 ('scale', MinMaxScaler(feature_range=(-10, 10))),
                 ('xgbt', modelsup)])


fittedSup = poly.fit(xtrainS, ytrainS)

predTrS = pd.Series(fittedSup.predict(xtrainS), index=xtrainS.index)
predTeS = pd.Series(fittedSup.predict(xtestS), index=xtestS.index)

maeTrS = gp.getMAE(ytrainS, predTrS)
maeTeS = gp.getMAE(ytestS, predTeS)
print(f'MAE train = {maeTrS}\nMAE test = {maeTeS}')



# ajout de combinaisons semble aider avec cleanCols
# : MAE test = 62 sans poly, 62 avec poly(2), 59 avec poly(3)

# avec toutes les colones et poly(2) : MAE test = 49 !!

## voir quelles combinaisons sortent du lot !!
polyFeatNames = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(xtestS[cleanCols].columns,p) for p in poly.steps[1][1].powers_]] # cree liste des noms des nouvelles variables

varImp = fittedSup.named_steps['xgbt'].feature_importances_
varImp = pd.Series(varImp, index=polyFeatNames)

Rotor_speed^1xRotor_speed_max^1                                            0.141641
* Rotor_speed3^1xRotor_speed_max^1                                           0.124594
** Pitch_angle^1xRotor_speed_std^1                                            0.113983
Generator_speed^1xGenerator_speed_max^1                                    0.076287
* Gearbox_bearing_2_temperature^1xRotor_speed3^1                             0.069622
* Generator_speed^1xPitch_angle_std^1                                        0.058760
Generator_bearing_1_temperature^1xGenerator_bearing_1_temperature_max^1    0.049569
Generator_bearing_1_temperature_max^1xPitch_angle^1                        0.046212
Generator_speed^1xRotor_speed_max^1                                        0.018060
 Grid_voltage_max^1xRotor_speed3^1                                          0.015353
* Generator_bearing_1_temperature^1xGrid_frequency_min^1                     0.014971
** Generator_stator_temperature_min^1xPitch_angle_std^1                       0.014888
* Grid_frequency_min^1xPitch_angle^1                                         0.013456
Pitch_angle_max^1xRotor_speed_std^1                                        0.012991
* Pitch_angle^1xTurbulence^1                                                 0.011479
Pitch_angle_max^1xRotor_speed_min^1                                        0.011136
Generator_bearing_1_temperature_max^1xGrid_voltage_max^1                   0.010059
Generator_bearing_1_temperature_max^1xGrid_voltage^1                       0.009134
Generator_speed^1xGrid_voltage^1                                           0.008555
** Pitch_angle_max^1xPitch_angle_min^1                                        0.008338
Generator_speed_std^1xPitch_angle_max^1                                    0.008168
Grid_voltage^1xPitch_angle^1                                               0.006886
* Generator_converter_speed_max^1xGenerator_stator_temperature_min^1         0.006650
Generator_speed_max^1xRotor_speed3^1                                       0.006340
Generator_speed^1xRotor_speed3^1                                           0.006093
Generator_bearing_2_temperature^1xGenerator_speed^1                        0.005260
Generator_bearing_1_temperature_max^1xRotor_speed_min^1                    0.004927
Generator_stator_temperature^1                                             0.004301
Generator_bearing_1_temperature_max^1xPitch_angle_max^1                    0.003798
Generator_stator_temperature^1xRotor_speed^1                               0.003603


col1 = 'Generator_stator_temperatureMin_x_Pitch_angleStd'
col2 = 'Pitch_angleStd_x_Rotor_speed'
f, axes = plt.subplots(2, 2, sharey=False)
axes[0,0].scatter(df[col1], df[col2], s=6, alpha=0.2)
axes[1,0].scatter(df[col1], wtPower, s=6, alpha=0.2)
axes[0,1].scatter(wtPower, df[col2], s=6, alpha=0.2)
axes[1,0].set_title(col1)
axes[0,1].set_title(col2)
plt.show()

## calcul importance des variables avec somme sur trois splits différents
['Grid_frequency_max',
 'Grid_frequency',
 'Grid_frequency_min',
 'Hub_temperature_std',
 'Grid_frequency_std',
 'Gearbox_bearing_2_temperature',
 'Generator_bearing_2_temperature_std',
 'Gearbox_bearing_1_temperature_min',
 'Gearbox_bearing_2_temperature_max',
 'Nacelle_temperature_std',
 'Nacelle_angle_std',
 'Gearbox_bearing_2_temperature_min',
 'Generator_bearing_1_temperature_std',
 'Gearbox_bearing_1_temperature_max',
 'Rotor_bearing_temperature_std',
 'Gearbox_oil_sump_temperature_std',
 'Gearbox_oil_sump_temperature_max',
 'Rotor_bearing_temperature_min',
 'Nacelle_angle',
 'Absolute_wind_direction',
 'Hub_temperature',
 'Rotor_bearing_temperature',
 'Nacelle_angle_min',
 'Outdoor_temperature',
 'Gearbox_oil_sump_temperature',
 'Hub_temperature_max',
 'Gearbox_oil_sump_temperature_min',
 'Outdoor_temperature_min',
 'Rotor_bearing_temperature_max',
 'Hub_temperature_min',
 'Nacelle_angle_max',
 'Outdoor_temperature_max',
 'Generator_stator_temperature_std',
 'Outdoor_temperature_std',
 'Rotor_speed_std',
 'Nacelle_temperature_min',
 'Gearbox_bearing_2_temperature_std',
 'Gearbox_bearing_1_temperature_std',
 'Generator_speed_min',
 'Gearbox_bearing_1_temperature',
 'Turbulence',
 'Generator_bearing_2_temperature',
 'Generator_bearing_2_temperature_min',
 'Pitch_angle_min',
 'Generator_speed_std',
 'Generator_bearing_2_temperature_max',
 'Nacelle_temperature',
 'Generator_bearing_1_temperature',
 'Generator_stator_temperature_max',
 'Pitch_angle_x_Rotor_speedStd',
 'Generator_stator_temperature_min',
 'Generator_speed_max',
 'Rotor_speed_max',
 'Rotor_speed_min',
 'Generator_stator_temperature',
 'Pitch_angleMax_x_Pitch_angleMin',
 'Nacelle_temperature_max',
 'Generator_bearing_1_temperature_min',
 'Pitch_angle',
 'Generator_bearing_1_temperature_max',
 'Generator_speed',
 'Pitch_angle_max',
 'Rotor_speed',
 'Pitch_angle_std',
 'Rotor_speed3',
 'Generator_stator_temperatureMin_x_Pitch_angleStd',
 'Pitch_angleStd_x_Rotor_speed']



### pour rotor_speed>15 : entre 2 modèles différents (par hyper paramètres de modèle) certains points immobiles dans les résidus.
# voir pourquoi
xtrainS, xtestS, ytrainS, ytestS = train_test_split(dfsup, wtPowersup, test_size=0.2, random_state=3456)

lstKeepCols = ['Absolute_wind_direction', 'Nacelle_angle_min', 'Outdoor_temperature',
               'Gearbox_oil_sump_temperature', 'Hub_temperature_max',
               'Outdoor_temperature_min',
               'Rotor_bearing_temperature_max', 'Hub_temperature_min',
               'Generator_stator_temperature_std', 'Outdoor_temperature_std',
               'Rotor_speed_std', 'Nacelle_temperature_min',
               'Gearbox_bearing_2_temperature_std',
               'Generator_speed_min',
               'Gearbox_bearing_1_temperature', 'Turbulence',
               'Generator_speed_std',
               'Nacelle_temperature', 'Generator_bearing_1_temperature',
               'Generator_stator_temperature_max', 'Pitch_angle_x_Rotor_speedStd',
               'Generator_speed_max',
               'Generator_stator_temperature',
               'Pitch_angleMax_x_Pitch_angleMin', 'Nacelle_temperature_max',
               'Generator_bearing_1_temperature_min', 'Pitch_angle',
               'Pitch_angle_max', 'Rotor_speed', 'Pitch_angle_std', 'Rotor_speed3',
               'Generator_stator_temperatureMin_x_Pitch_angleStd',
               'Pitch_angleStd_x_Rotor_speed']

predVarModel = pd.DataFrame(columns=['n_estim200', 'n_estim600'], index=xtrainS.index)
for n_estim in [200, 600] :
  modelsup = xgb.XGBRegressor(learning_rate=0.1, n_estimators=200, max_depth=3,
                              colsample_bytree=1, subsample=0.75, n_jobs=-1)
  pipeSup = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                      ('model', modelsup)])
  
  fittedSup = pipeSup.fit(xtrainS, ytrainS)
  predVarModel.at[:, f'n_estim{n_estim}'] = fittedSup.predict(xtrainS)


#predTrS = pd.Series(fittedSup.predict(xtrainS), index=xtrainS.index)
#predTeS = pd.Series(fittedSup.predict(xtestS), index=xtestS.index)
#
#maeTrS = gp.getMAE(ytrainS, predTrS)
#maeTeS = gp.getMAE(ytestS, predTeS)
#print(f'MAE train = {maeTrS}\nMAE test = {maeTeS}')
#
#gp.getAllResidPlot(ytrainS, predTrS, ytestS, predTeS)
