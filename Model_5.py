import os
import ReadFiles
import preprocData as pp
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import xgboost as xgb

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

testData = ReadFiles.GetInputTestData()

## ajout de variables
addFeat = pp.AddFeatures()
addFeat.fit(df)
df = addFeat.transform(df)

testData = addFeat.transform(testData)

###########
## nettoyage des données
impMedian = pp.ImputeMedian()
impMedian.fit(df)
df = impMedian.transform(df)

testData = impMedian.transform(testData)

# sélection des données
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
allCols = df.columns[(df.isnull().sum()==0)]
#lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["MAC_CODE", "Date_time", "Nacelle_angle", 'Generator_speed', 'Generator_converter_speed',
           "Outdoor_temperature_max", "Outdoor_temperature_min", "Outdoor_temperature",
           "Absolute_wind_direction_c"]
lstKeepCols = allCols.difference(notKeep).tolist()

# modèle
model = xgb.XGBRegressor(learning_rate=0.5, n_estimators=1200, max_depth=3,
                         colsample_bytree=1, subsample=1, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(df, wtPower)

testData_pred = pd.Series(fitted.predict(testData), index=testData.index, name='TARGET')

##############
## ecriture des prédictions pour soumission
dirOutput = '../Data'
outFilename = 'output_testing_model5.csv'
pred.to_csv(os.path.join(dirOutput,outFilename), sep=';', header=True)

## mae = ?? train ; ?? test
## mape = ?? train ; ?? test
