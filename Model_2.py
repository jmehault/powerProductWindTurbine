import ReadFiles
#import GetPerformances as gp
import preprocData as pp
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

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

## séparation des individus suivant rotor_speed => non linéarité entre rotor_speed et target
condi = df.Rotor_speed>=15
dfinf = df.loc[~condi, :]
dfsup = df.loc[condi, :]

wtPowerinf = wtPower[~condi]
wtPowersup = wtPower[condi]

###########
## nettoyage des données
dfinf = dfinf.fillna('median')
dfsup = dfsup.fillna('median')

###########
## modèle sur rotor_speed<15
xtrainI, xtestI, ytrainI, ytestI = train_test_split(dfinf, wtPowerinf, test_size=0.2, stratify=dfinf['MAC_CODE'], random_state=123)

## modèle
# sélectionner toutes les colonnes sans valeur manquante : quelles informations importantes ?
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
#lstKeepCols = ['Generator_speed', 'Rotor_speed', 'Pitch_angle_std', 'Pitch_angle', 'PitchAng_re', \
#                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
lstKeepCols = ['Pitch_angle', 'Rotor_speed', 'Gearbox_bearing_1_temperature', 'Generator_stator_temperature_std']
model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1) #LinearRegression()

pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(xtrainI, ytrainI)

predTrI = pd.Series(fitted.predict(xtrainI), index=xtrainI.index)
predTeI = pd.Series(fitted.predict(xtestI), index=xtestI.index)

#model = OLS(ytrain, xtrain[lstKeepCols]) # MAE ~ 64
#fitted = model.fit()

###########
## modèle sur rotor_speed>=15
xtrainS, xtestS, ytrainS, ytestS = train_test_split(dfsup, wtPowersup, test_size=0.2, stratify=dfsup['MAC_CODE'], random_state=123)

allCols = dfsup.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
cleanCols = allCols[lstCols].difference(notKeep).tolist()


pipeSup = Pipeline([('selectCols', pp.SelectColumns(cleanCols)),
                    ('model', RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1))])

fittedSup = pipeSup.fit(xtrainS, ytrainS)

predTrS = pd.Series(fittedSup.predict(xtrainS), index=xtrainS.index)
predTeS = pd.Series(fittedSup.predict(xtestS), index=xtestS.index)


## importance des variables
#gp.plotImportance(xtrain[lstKeepCols], fitted)

## prédiction
predTr = pd.concat((predTrI, predTrS),axis=0).sort_index()
predTe = pd.concat((predTeI, predTeS),axis=0).sort_index()


ytrain = pd.concat((ytrainI, ytrainS),axis=0).sort_index()
ytest = pd.concat((ytestI, ytestS),axis=0).sort_index()

maeTr = gp.getMAE(ytrain, predTr)
maeTe = gp.getMAE(ytest, predTe)

print(f'MAE train = {maeTr}\nMAE test = {maeTe}')

gp.getAllResidPlot(ytrain, predTr, ytest, predTe)

## divergence pour TARGET>600
## variable GcsRatio ne sert à rien pour l'arbre

## essayer boosting

## mae = 16 train ; 18 test
