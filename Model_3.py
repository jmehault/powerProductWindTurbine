import ReadFiles
import GetPerformances as gp
import preprocData as pp

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

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

###########################
## AJOUTER traitement des données test pour soumission

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

testData = ReadFiles.GetInputTestData()

## ajout de variables
addFeat = pp.AddFeatures()
addFeat.fit(df)
df = addFeat.transform(df)

testData = addFeat.transform(testData)

## séparation des individus suivant rotor_speed => non linéarité entre rotor_speed et target
condi = df.Rotor_speed3>=15**3 # Rotor_speed>=15
splitDataset = SplitEvents(condiSplit=condi)
dfsup, dfinf = splitDataset.transform(df)
wtPowersup, wtPowerinf = splitDataset.transform(wtPower)


condi = testData.Rotor_speed3>=15**3 # Rotor_speed>=15
splitTestDataset = SplitEvents(condiSplit=condi)
testDatasup, testDatainf = splitTestDataset.transform(testData)

###########
## nettoyage des données
impMedian = pp.ImputeMedian()
impMedian.fit(dfinf)
dfinf = impMedian.transform(dfinf)

testDatainf = impMedian.transform(testDatainf)

impMedian = pp.ImputeMedian()
impMedian.fit(dfsup)
dfsup = impMedian.transform(dfsup)

testDatasup = impMedian.transform(testDatasup)

###########
## modèle sur rotor_speed<15

## modèle
lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1)
pipeInf = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fittedInf = pipeInf.fit(dfinf, wtPowerinf)

testDatainf_pred = pd.Series(fittedInf.predict(testDatainf), index=testDatainf.index, name='TARGET')


###########
## modèle sur rotor_speed>=15
allCols = dfsup.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
cleanCols = allCols[lstCols].difference(notKeep).tolist()


pipeSup = Pipeline([('selectCols', pp.SelectColumns(cleanCols)),
                    ('model', RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1))])

fittedSup = pipeSup.fit(dfsup, wtPowersup)

testDatasup_pred = pd.Series(fittedSup.predict(testDatainf), index=testDatainf.index, name='TARGET')



##############
## prédiction finale
pred = pd.concat((testDatainf_pred, testDatasup_pred),axis=0).sort_index()

## mae = 16 train ; 17 test
## mape = 1.2 train ; 0.90 test

##############
## ecriture des prédictions pour soumission
dirOutput = '../Data'
outFilename = 'output_testing_model3.csv'
pred.to_csv(os.path.join(dirOutput,outFilename), sep=';', header=True)
