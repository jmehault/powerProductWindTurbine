import ReadFiles
import preprocData as pp
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb

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


## modèle basses vitesses
# sélectionner toutes les colonnes sans valeur manquante
lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

modelInf = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1500, max_depth=4,
                         colsample_bytree=0.75, subsample=1, reg_lambda=0.05, n_jobs=-1)
pipeInf = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', modelInf)])

fittedInf = pipeInf.fit(dfinf, wtPowerinf)

testDatainf_pred = pd.Series(fittedInf.predict(testDatainf), index=testDatainf.index, name='TARGET')


## modèle hautes vitesses
lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])

modelSup = xgb.XGBRegressor(learning_rate=0.1, n_estimators=1500, max_depth=4,
                            colsample_bytree=0.5, subsample=0.75, n_jobs=-1)
pipeSup = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                    ('model', modelSup)])

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
outFilename = 'output_testing_model4.csv'
pred.to_csv(os.path.join(dirOutput,outFilename), sep=';', header=True)
