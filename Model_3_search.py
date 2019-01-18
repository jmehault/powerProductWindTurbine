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

## recupère données
df = ReadFiles.GetInputTrainData()
wtPower = ReadFiles.GetOutputTrainData()

## ajout de variables
addFeat = pp.AddFeatures()
addFeat.fit(df)
df = addFeat.transform(df)

## séparation des individus suivant rotor_speed => non linéarité entre rotor_speed et target
condi = df.Rotor_speed3>=15**3 # Rotor_speed>=15
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
lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

kf = KFold(5)
scores = cross_val_score(pipe, dfinf, wtPowerinf, cv=kf, scoring='neg_mean_absolute_error')



###########
## modèle sur rotor_speed<15
xtrainI, xtestI, ytrainI, ytestI = train_test_split(dfinf, wtPowerinf, test_size=0.2, stratify=dfinf['MAC_CODE'], random_state=123)

## modèle
# sélectionner toutes les colonnes sans valeur manquante : quelles informations importantes ?
#lstKeepCols = df.columns[(df.isnull().sum()==0)].difference(['MAC_CODE', 'Date_time'])
# meileure sélection :
lstKeepCols = ['Generator_speed', 'Rotor_speed3', 'Pitch_angle_std', 'Pitch_angle', \
                'Generator_speed_max', 'Pitch_angle_max', 'Generator_stator_temperature', 'Generator_bearing_1_temperature']
lstKeepCols = ['Pitch_angle', 'Rotor_speed3', \
               'Gearbox_bearing_1_temperature', 'Generator_stator_temperature_std', \
               'Turbulence']
model = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1)
pipe = Pipeline([('selectCols', pp.SelectColumns(lstKeepCols)),
                 ('model', model)])

fitted = pipe.fit(xtrainI, ytrainI)

predTrI = pd.Series(fitted.predict(xtrainI), index=xtrainI.index)
predTeI = pd.Series(fitted.predict(xtestI), index=xtestI.index)


maeTrI = gp.getMAE(ytrainI, predTrI)
maeTeI = gp.getMAE(ytestI, predTeI)
print(f'MAE train = {maeTrI}\nMAE test = {maeTeI}')


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
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
cleanCols = allCols[lstCols].difference(notKeep).tolist()
pipeSup = Pipeline([('selectCols', pp.SelectColumns(cleanCols)),
                    ('model', RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1))])

kf = KFold(5)
scores = cross_val_score(pipeSup, dfsup, wtPowersup, cv=kf, scoring='neg_mean_absolute_error')


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





###################
## analyse des prédictions trop éloignées de réalité
## histogramme erreur absolue

gp.getError(ytestS, predTeS).hist(bins=80)
plt.show()

# quantiles
gp.getError(ytestS, predTeS).quantile([0.5,0.75,0.90])

# sélection des prédictions abérentes en positif

indOutPred = gp.getError(ytestS, predTeS)>gp.getError(ytestS, predTeS).quantile(0.75)
outPred = predTeS[indOutPred]

# sélection dans données originales
outDf = dfsup.loc[outPred.index,:]
outWtPower = wtPower.loc[outPred.index]


plt.scatter(dfsup.Rotor_speed3, wtPowersup, s=6, alpha=0.2)
plt.scatter(outDf.Rotor_speed3, outWtPower, s=6, alpha=0.4)
plt.show()


# déplacement des erreurs entre prediction et vérité en fonction d'une variable
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

errorsS = gp.getError(ytestS, predTeS)

cm = plt.get_cmap('gist_rainbow')
cNorm  = colors.Normalize(vmin=errorsS.min(), vmax=errorsS.max())
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
mycolors = [scalarMap.to_rgba(i) for i in errorsS.values]

fig = plt.figure()
plt.scatter(dfsup.Rotor_speed3, wtPowersup, s=6, alpha=0.2)
plt.scatter(outDf.Rotor_speed3, outWtPower, s=6, alpha=0.4)
for p in range(30) :
#for p in range(len(outPred)) :
   plt.plot([outDf.Rotor_speed3.iloc[p], outDf.Rotor_speed3.iloc[p]], [outPred.iloc[p], outWtPower.iloc[p]], color=mycolors[p])
plt.show()
# ajouter couleur du trait en fonction de l'erreur


## quelles variables sont différentes entre les "bien" prédits et les autres ?
# suppose qu'il n'y a qu'un groupe de mal classés
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

X = xtestS.copy()
Y = errors.copy() #1*indOutPred
discrPredmodel = LinearRegression() #LogisticRegression()
pipeDiscrModel = Pipeline([('selectCols', pp.SelectColumns(cleanCols)),
                           ('model', discrPredmodel)])

fittedDiscModel = pipeDiscrModel.fit(X,Y)

coefLr = pd.Series(fittedDiscModel.named_steps['model'].coef_[0],
                   index=cleanCols).sort_values()



pred = fittedDiscModel.predict(X)

pd.crosstab(Y,pred)


# recherche de correlation entre erreur et variable initiale
improvePred = xtestS.copy()
improvePred = improvePred.assign(errorS = errorsS)

corrErrors = improvePred.corr(method='spearman')
plt.pcolormesh(corrErrors.abs(), cmap='YlOrBr')
plt.xticks(np.arange(corrErrors.shape[0])+0.5, corrErrors.columns, rotation=45, ha='right')
plt.yticks(np.arange(corrErrors.shape[0])+0.5, corrErrors.columns)
plt.colorbar()
plt.show()

# corrélation maximale : 0.2 en valeur absolue => il n'y a rien avec une seule colonne


#Generator_stator_temperature      -0.078976
#Gearbox_bearing_1_temperature     -0.056406
#Rotor_speed                       -0.051835
#Pitch_angle                       -0.043446
#Rotor_bearing_temperature         -0.037774
#Gearbox_oil_sump_temperature      -0.014751
#Rotor_speed3                       0.001744
#Turbulence                         0.004923
#Generator_bearing_1_temperature    0.018187
#Outdoor_temperature                0.020779
#Nacelle_temperature                0.022938

# => #Generator_stator_temperature #Gearbox_bearing_1_temperature sont les colonnes les plus problématiques pour
# prédire TARGET. mais aucune corrélation n'en sors avec TARGET



## tester modèle pour rotor_speed > 15
# en calculant combinaisons de variable
xtrainS, xtestS, ytrainS, ytestS = train_test_split(dfsup, wtPowersup, test_size=0.2, stratify=dfsup['MAC_CODE'], random_state=123)

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

allCols = dfsup.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle",
           'Gearbox_bearing_2_temperature', 'Generator_speed', 'Hub_temperature', 'Gearbox_inlet_temperature',
           'Generator_bearing_2_temperature','Generator_converter_speed', 'Grid_voltage', 'Grid_frequency']
cleanCols = allCols[lstCols].difference(notKeep).tolist()

modelsup = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1))
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


polyFeatNames = ['x'.join([f'{pair[0]}^{pair[1]}' for pair in tuple if pair[1]!=0])
                 for tuple in [zip(xtestS[cleanCols].columns, p) for p in poly.named_steps['poly'].powers_]]

varImp = fittedSup.named_steps['xgbt'].feature_importances_
varImp = pd.Series(varImp, index=polyFeatNames)

