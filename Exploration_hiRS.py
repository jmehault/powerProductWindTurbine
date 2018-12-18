## exploration des individus avec rotor_speed>~10

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ReadFiles

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

df = ReadFiles.GetAllData()

# y a-t-il corrélation entre target(t) et target(t-dt) ?
#dfeol = df.loc[df.MAC_CODE=="WT1", :]
#dfeol = dfeol.set_index('Date_time')

#norm = (df.TARGET**2).sum()
#s = df.TARGET - df.TARGET.mean()
#autocorr = np.correlate(s,s,mode='full')/norm

# liste des colonnes de moyenne
allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)

# séparation des indiv en fonction de rotor_speed
dflow = df.loc[df.Rotor_speed<8.6,:]
dfmed = df.loc[(df.Rotor_speed>=8.6) & (df.Rotor_speed<15),:]
dfhi = df.loc[df.Rotor_speed>=15,:]

def detectOutliersLow(df_c, col='Rotor_speed') :
  targetLim = 10**(0.112*df_c[col] + 1.11)
  outliers = df_c.TARGET > targetLim
  return outliers

outLow = detectOutliersLow(dflow, col='Rotor_speed')


def detectOutliersMed(df_c, col='Rotor_speed') :
  targetLim = 10**(0.115*df_c[col] + 1.15)
  outliers = df_c.TARGET > targetLim
  return outliers

outMed = detectOutliersMed(dfmed, col='Rotor_speed')

def detectOutliersHi(df_c, col='Rotor_speed') :
  x = np.arange(15,17.26,0.25)
  y = np.array([750, 810, 880, 900, 1000, 1100, 1200, 1400, 1700, 3000])
  polyn = np.poly1d(np.polyfit(x,y,4))
  outliers = df_c.TARGET>polyn(df_c[col])
  return outliers

outHi = detectOutliersHi(dfhi, col='Rotor_speed')

dflow_in = dflow.loc[~outLow]
dflow_out = dflow.loc[outLow]
dfmed_in = dfmed.loc[~outMed]
dfmed_out = dfmed.loc[outMed]
dfhi_in = dfhi.loc[~outHi]
dfhi_out = dfhi.loc[outHi]


import matplotlib.cm as mplcm
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def plotACP(xdf, pc1=0, pc2=2) :
  ytarget = xdf.dropna(axis=0).TARGET.values
  xdf = xdf.drop(['TARGET'], axis=1).dropna(axis=0)
  acp = Pipeline([('scale',MinMaxScaler()),('pca',PCA(n_components=8))])
  acp.fit(xdf)
  x_red = acp.transform(xdf)
  norm = plt.Normalize()
  targetColor = plt.cm.plasma(norm(ytarget))
  plt.scatter(x_red[:,pc1],x_red[:,pc2], color=targetColor, s=6)
  plt.show()
  
  def circleOfCorrelations(pc_infos, ebouli, pc1=0, pc2=1):
    plt.figure(figsize=(8,8))
    plt.Circle((0,0),radius=10, color='g', fill=False)
    circle1=plt.Circle((0,0),radius=1, color='g', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    for idx in range(len(pc_infos["PC-0"])):
      x = pc_infos["PC-%s"%pc1][idx]
      y = pc_infos["PC-%s"%pc2][idx]
      plt.plot([0.0,x],[0.0,y],'k-')
      plt.plot(x, y, 'rx')
      plt.annotate(pc_infos.index[idx], xy=(x,y))
    plt.xlabel("PC-%s (%s%%)" %(pc1, str(ebouli[pc1])[:4].lstrip("0.")) )
    plt.ylabel("PC-%s (%s%%)" %(pc2, str(ebouli[pc2])[:4].lstrip("0.")) )
    plt.xlim((-1,1))
    plt.ylim((-1,1))
    plt.title("Circle of Correlations")
  
  eboulis = pd.Series(acp.named_steps['pca'].explained_variance_ratio_)
  coef = np.transpose(acp.named_steps['pca'].components_)
  cols = ['PC-'+str(x) for x in range(len(eboulis))]
  eigenvalues = pd.DataFrame(coef, columns=cols, index=xdf.columns) # vecteurs propres
  print("eboulis\n", eboulis, "\n")
  print("vecteurs propres\n", eigenvalues)
  circleOfCorrelations(eigenvalues, eboulis, pc1=pc1, pc2=pc2)
  plt.show()


acpCols = ['Gearbox_bearing_1_temperature', 'Gearbox_bearing_2_temperature',
       'Gearbox_oil_sump_temperature', 'Generator_bearing_1_temperature',
       'Generator_bearing_2_temperature',
       'Generator_stator_temperature',
       'Nacelle_temperature', 'Outdoor_temperature', 'Pitch_angle',
       'Rotor_bearing_temperature', 'Rotor_speed', 'TARGET']

acpCols = ['Gearbox_bearing_1_temperature',
       'Gearbox_oil_sump_temperature', 'Generator_bearing_1_temperature',
       'Nacelle_temperature', 'Outdoor_temperature', 'Pitch_angle',
       'Rotor_bearing_temperature', 'TARGET']

acpCols = ['Gearbox_bearing_1_temperature',
           'Gearbox_inlet_temperature',
            'Generator_bearing_1_temperature', 'Generator_bearing_2_temperature',
           'Grid_voltage',
           'Hub_temperature', 'Nacelle_temperature',
           'TARGET']
plotACP(dfhi[acpCols], pc1=0, pc2=1)


remCols = ['Generator_speed', 'Hub_temperature',
           'Generator_converter_speed', 'Grid_frequency']
cleanCols = lstCols.difference(remCols).tolist()

cleaneddf = dfhi[cleanCols].dropna(axis=0)
acp = PCA(n_components=6)
tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200, random_state=123)
tsne = umap.UMAP()
fit_sne = pd.DataFrame(tsne.fit_transform(cleaneddf), colnames=['dim0', 'dim1']) #acp.fit_transform(cleaneddf)))
norm = plt.Normalize()
targetColor = plt.cm.plasma(norm(dfhi[cleanCols+['TARGET']].dropna(axis=0).TARGET.values))
f = plt.figure(figsize=(8,8))
plt.scatter(fit_sne.loc[:,'dim0'], fit_sne.loc[:,'dim1'], s=6, c=targetColor)
plt.show()
### ça marche trop bien
# fit_sne enregistré dans "../Data/dfhi_sne_acp10.csv"
## !! projections sans modèle => réutilisation sur nouvelles données impossible

## => comprendre quelles transformations/combinaisons de variables sont à l'origine des projections
norm = plt.Normalize()
targetColor = plt.cm.plasma(norm(dfhi[cleanCols+['TARGET']].dropna(axis=0).TARGET.values))
col1 = 'Nacelle_temperature' #'Gearbox_bearing_1_temperature'
col2 = 'Gearbox_inlet_temperature' #'Generator_bearing_1_temperature'
plt.scatter(dfhi[col1], dfhi[col2],s=6,color=targetColor)
plt.xlabel(col1)
plt.ylabel(col2)
plt.show()



col1 = 'Gearbox_inlet_temperature'
col2 = 'Nacelle_temperature'
dfhi = dfhi.assign(test = dfhi[col1] * dfhi[col2])
plt.scatter(dfhi['test'], dfhi['TARGET'], s=6, alpha=0.3, color=targetColor)
plt.xlabel('test')
plt.ylabel('TARGET')
plt.show()




fit_sne = pd.read_csv("../Data/dfhi_sne_acp10.csv", sep=";", index_col=0)
fit_sne = dfhi.dropna(axis=0).TARGET.to_frame().join(fit_sne, how='left')
norm = plt.Normalize()
targetColor = plt.cm.plasma(norm(fit_sne.TARGET.values))
f = plt.figure(figsize=(8,8))
plt.scatter(fit_sne.loc[:,'0'], fit_sne.loc[:,'1'], s=6, c=targetColor)
plt.show()


# recupère individus types des groupes détectés
from sklearn.cluster import KMeans
km = KMeans(n_clusters=35)
km.fit(fit_sne.drop('TARGET', axis=1))


f = plt.figure(figsize=(8,8))
plt.scatter(fit_sne.loc[:,'0'], fit_sne.loc[:,'1'], s=6, c=targetColor)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1] ,s=16,c='k')
plt.show()


dfhi = dfhi.join(pd.Series(km.labels_,index=fit_sne.index, name='tsneCl'))

# étude différences entre groupes
gr = dfhi.drop(['MAC_CODE', 'Date_time'],axis=1).dropna(axis=0).groupby('tsneCl')

gr.TARGET.agg(['median', 'mean', 'std', 'size']).sort_values('median')


resTarget = gr.TARGET.agg(['mean', 'std']).set_axis(['TargetMean', 'TargetStd'], axis=1, inplace=False)
for col in cleanCols : #lstCols :
   resCol = gr[col].agg(['mean']).set_axis([f'{col}Mean'], axis=1, inplace=False)
   resTarget = pd.concat((resTarget, resCol), axis=1)
   #print(col,"\n", res.sort_values('TargetMean'),"\n\n")

corrGr = resTarget.corr(method='spearman'
plt.pcolormesh(corrGr)
plt.xticks(np.arange(0,len(corrGr.columns))+0.5, corrGr.columns, ha='right', rotation=45)
plt.yticks(np.arange(0,len(corrGr.columns))+0.5, corrGr.columns)
plt.colorbar()
plt.show()


norm = plt.Normalize()
grTargColor = plt.cm.plasma(norm(corrGr.TargetMean.values))
for col in corrGr.columns :
  plt.scatter(corrGr[col], corrGr['TargetMean'], s=6, alpha=0.3, color=grTargColor)
  plt.xlabel(col)
  plt.ylabel('TargetMean')
  plt.show()


249481
508223
41982
142329
180065


from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor

xtrain, xtest, ytrain, ytest = train_test_split(fit_sne.drop('TARGET', axis=1), fit_sne.TARGET,
                                                 test_size=0.2, random_state=123)

rf = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1)
rf.fit(xtrain, ytrain)

import GetPerformances as gp
predTr = pd.Series(rf.predict(xtrain), index=xtrain.index)
predTe = pd.Series(rf.predict(xtest), index=xtest.index)

maeTr = gp.getMAE(ytrain, predTr)
maeTe = gp.getMAE(ytest, predTe)

print(f'MAE train = {maeTr}\nMAE test = {maeTe}')


gp.getAllResidPlot(ytrain, predTr, ytest, predTe)
