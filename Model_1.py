import ReadFiles
import GetPerformances as gp
import preprocData as pp

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

from statsmodels.discrete.discrete_model import Logit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

import matplotlib.pyplot as plt

## transformation données
#transformer = Pipeline([
#    ('features', FeatureUnion(n_jobs=1, transformer_list=[
#        # Part 1
#        ('numericals', Pipeline([
#            ('selector', TypeSelector('number')),
#            ('traite', PreTraitementContinues())
#            ])
#        ), # numériques
#        # Part 2
#        ('categoricals', Pipeline([
#            ('selector', TypeSelector('object')),
#            ('traite', PreTraiteQualitatives()),
#            ('labeler', ModalitiesEncoder()),
#            ])
#        ) # catégorielles close
#        ])
#    ), # features close
#    ]) # pipeline close

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

#df = df.assign(PitchAng_re = df.Pitch_angle+ np.abs(df.Pitch_angle.min())*df.MAC_CODE.str[-1].astype('int'))

df = df.join(pd.get_dummies(df.MAC_CODE))

xtrain, xtest, ytrain, ytest = train_test_split(df, wtPower, test_size=0.2, stratify=df['MAC_CODE'], random_state=123)

## modèle
# sélectionner toutes les colonnes sans valeur manquante : quelles informations importantes ?
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
#lstKeepCols = ['Generator_speed', 'Rotor_speed', 'Pitch_angle_std', 'Pitch_angle', 'PitchAng_re', \
#                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
lstKeepCols = ['Pitch_angle', 'Rotor_speed', 'Gearbox_bearing_1_temperature', 'Generator_stator_temperature_std']
model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(xtrain, ytrain)


#model = OLS(ytrain, xtrain[lstKeepCols]) # MAE ~ 64
#fitted = model.fit()

## importance des variables
gp.plotImportance(xtrain[lstKeepCols], fitted)
# generator_speed, rotor speed, pitch_angle_std, pitch_angle,
# generator_speed_max, pitch_angle_max, generator_stator_temperature, generator_bearing_1_temperature(_max)

## prédiction
predTr = pd.Series(fitted.predict(xtrain[lstKeepCols]), index=xtrain.index)
predTe = pd.Series(fitted.predict(xtest[lstKeepCols]), index=xtest.index)

maeTr = gp.getMAE(ytrain, predTr)
maeTe = gp.getMAE(ytest, predTe)

print(f'MAE train = {maeTr}\nMAE test = {maeTe}')

gp.getAllResidPlot(ytrain, predTr, ytest, predTe)

## divergence au delà de TARGET~1000

## MAE = 21 train ; 22 test
