import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ReadFiles

df = ReadFiles.GetAllData()
df = df.assign(LogTARGET = np.log10(df.TARGET+20))
df = df.assign(TargetSup600 = df.TARGET>=600)

allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)
extract = df.loc[:,lstCols]
extract = extract.iloc[:,:5]
pd.plotting.scatter_matrix(extract, alpha=0.2, figsize=(10, 10), diagonal='hist')

gr = df.groupby('MAC_CODE')

df = df.assign(IntRotor_speed = ddf.Rotor_speed.round(0))
df = df.assign(Turbulence = np.log10((df.Rotor_speed_std/(df.Rotor_speed+df.Rotor_speed.min()+0.1)+0.1)))
grRs = df.groupby("IntRotor_speed")
grRs.TARGET.plot(kind='box')
plt.show()

a = grRs.boxplot(column='TARGET')
## créer plot joli

boxDf = df.loc[:,["IntRotor_speed", "TARGET"]]
boxDf["IntRs3"] = boxDf["IntRotor_speed"]**3
boxDf.boxplot(by='IntRotor_speed')
plt.show()

#################################
## statistiques simples :
#################################
# 78 colonnes, 1 ID, 1 id éolienne, 1 date
df.shape
# 617386 lignes avec ID différent
df.isnull().sum().sort_values(ascending=False)
# 14 colonnes avec valeurs manquantes
df.isnull().sum(axis=1).value_counts()/df.shape[0]
# 82% des lignes n'ont pas de valeur manquante
# 16% des lignes avec valeurs manquantes en ont 4 (colonnes de grid_voltage systématiquement)
# 1% ------------------------------------------ 8

df.loc[df.isnull().sum(axis=1)==4,:].isnull().sum().sort_values()
# les 16% concernent les colonnes de grid_voltage,min,max,std seules
# les 1 % concernent les colonnes Gearbox_inlet et Generator_converter ,min,max,std seules
# le reste concerent les combinaisons de : Absolute_wind_direction_c, Nacelle_angle_c
# puis  Grid_voltage_max, Grid_voltage, Nacelle_angle_c, Absolute_wind_direction_c,
#        Grid_voltage_min, Grid_voltage_std

## etude de TARGET
df.TARGET.describe()
df.TARGET.quantile([0,.1,.25,.5,.75,.9,1])
(df.TARGET<0).sum()/df.shape[0]
# 19% des valeurs sont négatives !
# médiane à 194
# q90 à 1042, max au double => fort étalement des valeurs

df.loc[(df.TARGET>0) & (df.TARGET<=0.14),'TARGET'].value_counts().sort_index()
# il y a en réalité un pic de production entre 0.045 et 0.1 puis prod/100 après
# indiv entre 0 et 0.1 représentent 2% des données

## !! étudier lien avec variables : d'abord pitch_angle  ?
dfZero = df.loc[(df.TARGET>0) & (df.TARGET<=0.1), :]
## 98% sont le fait de WT4 # voir pourquoi
dfNeg = df.loc[df.TARGET<=0,:]
## 30% pour WT1,2,3 ; 14 % pour WT4

## etude des TARGET nulles pour WT4
## comparaison distributions des variables pour TARGET nulle/non nulle de WT4
dfWT4Nul = df.loc[((df.TARGET>0) & (df.TARGET<=0.1)) & (df.MAC_CODE=='WT4'), :]
dfWT4NNul = df.loc[~((df.TARGET>0) & (df.TARGET<=0.1)) & (df.MAC_CODE=='WT4'), :]


lstCols = dfWT4NNul.columns.difference(['Date_time', 'MAC_CODE', 'TARGET', 'LogTARGET'])
lstCols = ['Pitch_angle', 'Rotor_speed', 'Gearbox_bearing_1_temperature']
for col in lstCols :
    dfWT4NNul[col].plot(kind='hist', bins=120, label='Target non nulle')
    dfWT4Nul[col].plot(kind='hist', bins=120, label='Target nulle')
    plt.xlabel(col)
    plt.legend()
    plt.show()

# variables discriminantes
# rotor_speed(_min), Gearbox_bearing_1_temp, pitch_angle(_std)

df = df.assign(PitchAng_re = df.Pitch_angle+ np.abs(df.Pitch_angle.min())*df.MAC_CODE.str[-1].astype('int'))
# décalage de Pitch en fonction du numéro de l'éolienne
dfWT4NNul['Pitch_angle'].plot(kind='hist', bins=200, label='Target non nulle')
dfWT4Nul['Pitch_angle'].plot(kind='hist', bins=200, label='Target nulle')
plt.xlabel(col)
plt.legend()
plt.show()
# coupure sur pitch_angle efficace à 44.47 pour filtrer TARGET nulles de WT4. on perd 10% des non nulles et on garde 99% des nulles



## étude de TARGET<0
## comparaison distributions des variables pour TARGET nulle/non nulle de WT4
dfWT123Neg = df.loc[(df.TARGET<=0) & (df.MAC_CODE.isin(['WT1','WT2','WT3'])), :]
dfWT4Neg = df.loc[(df.TARGET<=0) & (df.MAC_CODE=='WT4'), :]
dfNonNeg = df.loc[(df.TARGET>0.1), :]
dfNul = df.loc[((df.TARGET>0) & (df.TARGET<=0.1)), :]

lstCols = dfWT123Neg.columns.difference(['Date_time', 'MAC_CODE', 'TARGET', 'LogTARGET'])
#lstCols = ['Pitch_angle', 'Rotor_speed', 'Gearbox_bearing_1_temperature']
for col in lstCols :
    dfNonNeg[col].plot(kind='hist', bins=120, alpha=0.2, label='Positif')
    dfWT123Neg[col].plot(kind='hist', bins=120, alpha=0.3, label='Neg WT123')
    dfWT4Neg[col].plot(kind='hist', bins=120, alpha=0.3, label='Neg WT4')
    dfNul[col].plot(kind='hist', bins=120, alpha=0.3, label='Nul')
    plt.xlabel(col)
    plt.legend()
    plt.show()

# variables discriminantes pour target <0.1 :
# 'Generator_speed' coupure à 250 (voir pour target € [0, 0.1], 'Generator_stator_temperature_std', coupure à 0.27 ou 0.15
# 'Rotor_speed' coupure à 2.316 =  entre 1.5 (MAE minimal= 1.73) et 2.5 (MAE = 1.77) MAE : comparaison entre target vcrai et zeros

## !! trouver bonne coupure
## => essaie de modèle avec rotor_speed (meilleure variable discriminante) avec Model_explo.py
## l'application d'une coupure brutale n'améliore pas ce que RandomForest fait avec 10 variables
## (MAE = 1.78 contre 0.22 sur xtrain où rotor_speed<2.5)

## test en codant 1/0 si TARGET < 0.1
df = df.assign(TNeg = df.TARGET<=0.1)
df.plot('Rotor_speed', by='TNeg', kind='boxplot')

lstCols = ['Pitch_angle', 'Rotor_speed', 'Gearbox_bearing_1_temperature', 'Generator_stator_temperature_std']
extDf = df[lstCols]

from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import confusion_matrix
model = Logit(df.TNeg.values, extDf)
fitted = model.fit()

seuil = 0.85 # maximise accuracy
pred = fitted.predict(df[lstCols])>seuil

acc = []
for seuil in np.arange(0.1,1.,0.05) :
    pred = fitted.predict(df[lstCols])>seuil
    acc.append(np.trace(confusion_matrix(df.TNeg, pred))/617386)

#accMax = 0.986

from sklearn.tree import DecisionTreeClassifier
mytree = DecisionTreeClassifier(max_depth=9)
treeFit = mytree.fit(extDf, df.TNeg)

predTree = treeFit.predict(extDf)

from sklearn import tree
with open("tree.dot", 'w') as f :
   f = tree.export_graphviz(treeFit, out_file=f)
os.system("dot -Tpdf tree.dot -o tree.pdf")
os.system("okular tree.pdf")

## inutile de faire coupure à la main



gr.LogTARGET.describe()
gr.LogTARGET.plot(kind='kde', alpha=0.5)
plt.show()
# pas de différences de production globale entre éoliennes

## plot target par angle de nacelle
plt.polar(df.Nacelle_angle, df.TARGET)
plt.show()
# on voit pas grand chose
# échantillonnage de l'angle de la nacelle
angBins, bSize = np.linspace(0,360,12, retstep=True)
demiAngBins = np.arange(bSize/2, 360, bSize)  ## ajouter valeur du centre du bin pour labl des bins
demiRadBins = demiAngBins*np.pi/180  ## ajouter valeur du centre du bin pour labl des bins
df = df.assign(binNacelle_angle = pd.cut(df.Nacelle_angle, bins=angBins, labels=demiAngBins, \
                                         include_lowest=True, right=True))
grAng = df.groupby('binNacelle_angle')
# calcul médiane de target globale par pas d'angle
medTargetByAngle = grAng.TARGET.median()
ax = plt.subplot(111, projection='polar')
bars = ax.bar(demiRadBins, medTargetByAngle.values, alpha=0.5, width=bSize*np.pi/180)
plt.show()

# faire même chose par éolienne
grEolAng = df.groupby(['MAC_CODE','binNacelle_angle'])
medTargetByAngle = grEolAng.TARGET.median()

ax = plt.subplot(111, projection='polar')
for i, eol in enumerate(np.sort(df.MAC_CODE.unique())) :
  medEol = medTargetByAngle.xs(eol, level=0)
  newRad = (i*bSize/4+angBins[:-1]+bSize/8)*np.pi/180
  bars = ax.bar(newRad, medEol.values, alpha=0.5, width=bSize*np.pi/180/4, label=eol)
plt.legend()
plt.show()
## WT1 plus productive sur angles ~ 225° que les autres (~200°)
# aucune production entre pi/2 et 3pi/4
# faible production entre 3pi/2 et 2pi

# faire un plot par eolienne et ajouter mediane et quantiles 25 et 90
quantEolProd = grEolAng.LogTARGET.quantile([.10,.25,.50,.75,.90])

colors = ['r','g','b','m']
ls = ['-.','-','-.', '--']
for i, eol in enumerate(np.sort(df.MAC_CODE.unique())) :
    for j, q in enumerate([.25,.50,.75, .90]) :
        plt.polar(demiRadBins,quantEolProd.xs([eol, q], level=[0,2]),'%s%s'%(colors[i],ls[j]))
plt.show()

# WT1 moins efficace entre 3pi/4 et pi que les 3 autres

## on peut en déduire implantation des éoliennes !!




#################################
## valeurs manquantes ?
# . si oui aléatoire ou bien structurel ?
# . à traiter ?
#################################
## pourquoi grid_voltage null dans 16% des cas ?
nullVolt = df.loc[df.Grid_voltage.isnull(),:]
notnullVolt = df.loc[df.Grid_voltage.notnull(),:]
nullVolt.TARGET.describe()
notnullVolt.TARGET.describe()
# pas de différence dans distribution de TARGET
notnullVolt.plot('TARGET', kind='kde')
nullVolt.plot('TARGET', kind='kde')
plt.show()
## détail par éolienne (groupby au début):
gr.apply(lambda g : g.Grid_voltage.isnull().sum())/gr.size()
# 19% de nan pour wt1,2,4 ; 8% pour wt3


# corrélations ?
spCorr = df.corr(method='spearman')
plt.pcolormesh(spCorr.abs(), cmap='YlOrBr')
plt.xticks(np.arange(spCorr.shape[0])+0.5, spCorr.columns, rotation=45, ha='right')
plt.yticks(np.arange(spCorr.shape[0])+0.5, spCorr.columns)
plt.colorbar()
plt.show()



####################################
## objectif : supprimer du bruit
## 1 modéliser sans bruit
## 2 modéliser le bruit pour corriger les prédictions
####################################

## couples à regarder :
# TARGET : pitch_angle(_min)
#. hub_temperature : outdoor_temp, nacelle_temp, generator_b_2_temp
#. generator_converter_speed : generator_speed, roto_seepd
#. generator_b_1_temp : generator_b_2_temp, nacelle_temp
#. gearbox_b_1_temp : gearbox_b_2_temp, rotor_speed
#. gearbox_oit_temp : gearbox_oil_temp
#. nacell_angle : wind_direction, nacelle_angle_c, absolute_wind_dir_c
#. nacelle_temp : outdoor_temp


## triplets dont 2 à supprimer
#. pitch_angle
#. hub_temperature, generator_converter_speed, generator_speed, generator_b_1_temp
#. generator_stator_temp, gearbox_b_1(/2), nacelle_angle, outdoor_temp
#. grid_voltage, outdoor_temp

col1 = 'Gearbox_oil_sump_temperature'
col2 = 'Generator_bearing_2_temperature'
df.plot(x=col1 ,y=col2, kind='scatter', s=6, alpha=0.2);plt.show()

## plot avec histo sur axes x et y
def plotScatHist(data, col1='Rotor_speed', col2='TARGET') :
  fig = plt.figure(figsize=(9,8), constrained_layout=True)
  widths = [1, 4]
  heights = [4, 1]
  spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
  
  ax = fig.add_subplot(spec[0,0])
  ax.hist(data[col2], bins=100, orientation='horizontal')
  ax.set_ylabel(col2)
  
  ax = fig.add_subplot(spec[0,1])
  ax.scatter(data[col1], df[col2], s=6, alpha=0.2)
  
  ax = fig.add_subplot(spec[1,1])
  ax.hist(data[col1], bins=100, orientation='vertical')
  ax.set_xlabel(col1)
  plt.show()

plotScatHist(df,'test', 'TARGET')


## comparer min/max avec moyenne sur 10 minutes pour voir valeurs abérantes
allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)
for col in lstCols :
  suf = "_min"
  df.plot(col, col+suf, kind='scatter', s=6)
  minCol = df[col].min()
  maxCol = df[col].max()
  plt.plot([minCol, maxCol], [minCol, maxCol], 'r-')
  plt.show()


######
# toutes les temp_min/max ont des valeurs à 0
# voir si raison évidente
col = 'Generator_bearing_1_temperature'
condi = df[col+'_min']==0
extract = df.loc[condi,:]
for col in lstCols :
  suf = "_min"
  extract.plot(col, col+suf, kind='scatter', s=6)
  minCol = extract[col].min()
  maxCol = extract[col].max()
  plt.plot([minCol, maxCol], [minCol, maxCol], 'r-')
  plt.show()

# faire comptage de 1/0 sur colonnes lstCols
# créer dataframe avec 1/0 pour températures ==0
getNullTemp = df[lstCols]==0
# part d'individus avec nb donné de valeurs nulles
getNullTemp.sum(axis=1).value_counts()/getNullTemp.shape[0] # 7% des lignes ont au moins 1 des colonnes avec valeur nulle
# colonnes avec des valeurs nulles
getNullTemp.sum(axis=0).sort_values()

getNullTemp = getNullTemp.assign(TARGET = df.TARGET)
getNullTemp = getNullTemp.assign(LogTARGET = np.log10(df.TARGET+20))

for col in lstCols :
  getNullTemp.boxplot('LogTARGET', by=col)
  plt.show()



######
## vérifier moy<min ou moy>max
moyNotInBound = pd.DataFrame(index=df.index, columns=lstCols)
for col in lstCols :
   moyNotInBound[col] = ((df[col]>df[col+"_max"]) | (df[col]<df[col+"_min"])).values
   if moyNotInBound[col].any() :
     nbErr = moyNotInBound[col].sum()
     print(f'bornes {col} incorrectes - {nbErr}')

# nb de colonnes ou bornes non respectées
moyNotInBound.sum(axis=1).value_counts()/df.shape[0]
# 49% des individus ont bornes non respectées

selGoodBound = moyNotInBound.any(axis=1)
goodB = df.loc[selGoodBound,:]
badB = df.loc[~selGoodBound,:]
# étudier les différences, voir pour moduler les colonnes prises en compte pour sélectionner les individus


## vérifier max>min
for col in lstCols :
  if (df[col+"_max"]<df[col+"_min"]).any() :
    print(f'bornes {col} incorrectes')
#Pitch_angle (2x)et nacelle_angle (12486) ont max<min
# normal pour nacelle angle : à 180, le min et le max s'inversent

## travailler sur _min==_max => std==0 ?
# pour chaque colonne compte nb min==max et nb std=0
for col in lstCols :
  nbminmax = (df[col+'_min']==df[col+'_max']).sum()
  nbstdnul = (df[col+'_std']==0).sum()
  print(f'{col} -\n    minmax : {nbminmax}, std: {nbstdnul}')
## voir cas où min==max mais std!=0

## comparer coef de variation avec min et max pour toutes les variables
coefVar = pd.DataFrame((df[lstCols+"_std"].values/df[lstCols].values), columns=lstCols+"_cv", index=df.index)
df = df.join(coefVar)

for col in lstCols :
  ser = df[col+"_cv"].replace(np.inf,np.nan)
  ser.hist(bins=120)
  #plt.scatter(np.log10(ser+ser.min()+0.01), df.TARGET, s=6)
  #plt.ylabel("TARGET")
  plt.xlabel(col+"_cv")
  plt.show()
## pas de corrélation entre coef variation et target
## coef variation ~ 0


## voir rapport des vitesses de rotation
#'Generator_speed', 'Generator_converter_speed'
spRatio = df['Generator_speed']/df['Rotor_speed']
spRatio = spRatio.replace(-np.inf,-1).replace(np.inf,-1) # min à -0.15
spRatio.hist(bins=120)
plt.show()

spRatio = (df['Generator_speed']/df['Rotor_speed']).replace(-np.inf,-1).replace(np.inf,-1)
df = df.assign(GsRatio = spRatio.fillna(spRatio.median()))
spRatio = (df['Generator_converter_speed']/df['Rotor_speed']).replace(-np.inf,-1).replace(np.inf,-1)
df = df.assign(GcsRatio = spRatio.fillna(spRatio.median()))
## inf car rotor_speed nul pour 7% des cas


## target!=0 si gcs € 103,106
## target!=0 si gs € 104.5,106
# remplissage des valeurs manquantes par mediane
# ~ 20 % des données non manquantes sont en dehors => target=0
# à intégrer aux données avant modèle !!


### plot histogram 2D
binnedH = df.TARGET.to_frame().copy()
xnbins = 10
binnedH['xbinned'] = pd.cut(np.log10(df['GcsRatio']+np.abs(df['GcsRatio'].min())+0.1), bins=xnbins)
binnedH['ybinned'] = pd.cut(df['Gearbox_bearing_1_temperature'], bins=xnbins)

hist2D = pd.pivot_table(binnedH, 'TARGET', 'xbinned', 'ybinned', aggfunc='median')


plt.pcolormesh(hist2D)
plt.colorbar()
plt.show()
## créer fonction



## projection des variables de températures
## => comprendre le lien entre elles et vitesse de rotation
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

def plotACP(xdf, pc1=0, pc2=2) :
  classe = xdf.TARGET
  classe = (classe - classe.mean())/classe.std()
  xdf = xdf.drop(['TARGET'], axis=1)
  xdf = xdf.fillna(0)
  acp = Pipeline([('scale',MinMaxScaler()),('pca',PCA(n_components=8))])
  acp.fit(xdf)
  x_red = acp.transform(xdf)
  cm = plt.get_cmap('gist_rainbow')
  nbCols = classe.nunique()
  cNorm  = colors.Normalize(vmin=0, vmax=nbCols-1)
  scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
  colCsp = [scalarMap.to_rgba(i) for i in range(len(classe))]
  plt.scatter(x_red[:,pc1],x_red[:,pc2], color=colCsp, s=6)
  #for i, sec in enumerate(xdf.index) : # ajoute nom des secteurs
  #  plt.text(x_red[i,pc1],x_red[i,pc2], sec)
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


notKeep = ['Pitch_angle', 'Rotor_speed', 'Generator_speed','Generator_converter_speed','Grid_frequency','Grid_voltage']
acpCols = lstCols.difference(notKeep)

allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
acpCols = allCols[lstCols].difference(notKeep)

plotACP(df[acpCols.tolist()+["TARGET"]], pc1=0, pc2=1)



# utilité de toutes les colonnes ?

plt.plot(df['Generator_speed'], df['Generator_converter_speed'], '.')
plt.show()
## comprendre pourquoi 4 lignes parallèles : engrenage ?

df.plot('Rotor_speed', 'Generator_speed', kind='scatter', s=6)
plt.show()

plt.plot(df['Generator_speed'], df['TARGET'], '.')
plt.show()
## dispersion à basse vitesse

## montre que TARGET ~nulle ou <0 pour quelques plages de tension et fréquence
# idem pour Pith_angle
plt.plot(df['Grid_voltage'], df['TARGET'], '.')
plt.show()
#  mais variance augmente avec voltage et frequency


## 21 % des inidividus ont TARGET dans ]-5 , 5[
## 2 % des individus ont TARGET<-5 : pourquoi ?
# TARGET borné à 2256
# répartition du log10 ~ gaussienne
# pic à 0

np.log10(df.TARGET+20).hist(bins=80);plt.show()



#############################################################
## modélisation des TARGET < 600 correcte
## pourquoi ça ne marche pas au-dessus de 600 ?
#############################################################

dfinf = df.loc[df.TARGET<600,:]
dfsup = df.loc[df.TARGET>=600,:]

allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)

for col in lstCols :
  try :
    dfinf[col].hist(color='r', bins=80, label='TARGET<600')
    dfsup[col].hist(color='b', alpha=0.5, bins=80, label='TARGET<600')
    plt.legend()
    plt.xlabel(col)
    plt.show()
  except :
    print(f'{col} erreur')

#  plt.scatter(dfinf[col], dfinf.TARGET, s=6, c='r', label='TARGET<600')
#  plt.scatter(dfsup[col], dfsup.TARGET, s=6, c='b', label='TARGET>600')
#  plt.legend()
#  plt.xlabel(col)
#  plt.ylabel('TARGET')
#  plt.show()

# faire coupure brutale sur rotor_speed
## et modèle sur 2 groupes séparés

# éolienne particulière ? => non
# vitesse de rotation ?
# température ? voir gearbox_bearing_2_temperature, grid frequency/voltage


## suprimer les lignes avec des problèmes : valeurs extrèmes, std!=0

## comprendre d'où viens de bruit dans les données
condi = (df.TARGET>450) & (df.Rotor_speed<10)
extractOut = df.loc[condi,:]
extractIn = df.copy() # loc[(df.TARGET<=450) & (df.Rotor_speed<10),:]
for col in lstCols[7:] :
  plt.scatter(extractIn['Rotor_speed'], extractIn[col], s=6, alpha=0.2, label='Normal')
  plt.scatter(extractOut['Rotor_speed'], extractOut[col], s=8, label='Aberrant')
  plt.xlabel('Rotor_speed')
  plt.ylabel(col)
  plt.show()
# Generator_stator_temperature>65.77, Grid_frequency<49.9, Pitch_angle>95 & <-6 (Pitch_angle x Rotor_speed)

## nettoyer graphique target = f(rotor_speed)

# faire regression quantile sur graphique target = f(rotor_speed) ? linéaire en log
rtSpeed = df.Rotor_speed
target = df.TARGET

x = np.arange(rtSpeed.min(), rtSpeed.max(), 0.5)
y = 10**(0.115*x +1.34)
plt.scatter(rtSpeed, target, s=6, alpha=0.2)
plt.plot(x,y, 'r-')
plt.xlabel('Rotor_speed')
plt.ylabel('Target')
plt.show()

# tag des indiv > fonction

target>10**(0.115*rtSpeed+1.34)

## supprimer le bruit et modéliser sans le bruit
valVolt = np.log10(df.Grid_voltage+0.1)
condVolt = (valVolt<2.577) | ((valVolt>2.64) & (valVolt<2.814)) | (valVolt>2.88)
condFreq = df.Grid_frequency<49.9
condPitch = (df.Pitch_angle<-6) | (df.Pitch_angle>95)
spRatio = (df['Generator_speed']/df['Rotor_speed']).replace(-np.inf,-1).replace(np.inf,-1)
condiRatio = (spRatio<104.96) | (spRatio>105.50)

condi = condVolt | condFreq | condPitch | condiRatio
## pas top : condition sur ratio identifie beacoup d'indiv avec target faible


#################################
## découper 2 ou 3 jeux de données suivant rotor_speed : <8.6 ; € [8.6,15] > 15
df.Rotor_speed.hist(bins=160);plt.show()

dflow = df.loc[df.Rotor_speed<8.6,:]
dfmed = df.loc[(df.Rotor_speed>=8.6) & (df.Rotor_speed<15),:]
dfhi = df.loc[df.Rotor_speed>=15,:]

allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)
import matplotlib.cm as cm
purp = cm.PuRd(0.6)
bleu = cm.Blues(0.7)
orange = cm.Oranges(0.7)
for col in lstCols :
  try :
    dflow[col].hist(color=purp, bins=80, alpha=0.5, label=f'{col} petit')
    dfmed[col].hist(color=bleu, bins=80, alpha=0.5,  label=f'{col} moyen')
    dfhi[col].hist(color=orange, bins=80,alpha=0.5, label=f'{col} grand')
    plt.legend()
    plt.xlabel(col)
    plt.show()
  except :
    print(f'{col} erreur')

# gearbox_bearing_1(2)_temperature

#######
## tentative de projection

from sklearn.manifold import TSNE
cleaneddf = dfhi[lstCols.difference(['Grid_voltage', 'Grid_frequency']).tolist()+['TARGET']].dropna(axis=1)
acp = PCA(n_components=10)
tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200, random_state=123)
fit_sne = pd.DataFrame(tsne.fit_transform(cleaneddf)) #acp.fit_transform(cleaneddf)))
f = plt.figure(figsize=(8,8))
plt.scatter(fit_sne.loc[:,0], fit_sne.loc[:,1], s=6)

fit_sne_re = fit_sne.copy()
fit_sne_re = fit_sne_re.set_index(dfhi.index)
fit_sne_re = fit_sne_re.join(dfhi.TARGET)
# plots
# temps de calcul avec n_iter=200, 150000 individus et 10 axes acp ~ 30 min

# pour dfmed : comprendre valeurs abérantes dans graphique target = f(rotor_speed)
rtSpeed = dflow.Rotor_speed.copy()
target = dflow.TARGET.copy()
x = np.arange(rtSpeed.min(), rtSpeed.max()+0.6, 0.5)
y = 10**(0.112*x +1.2)
plt.scatter(rtSpeed, target, s=6, alpha=0.2)
plt.plot(x,y, 'r-')
plt.xlabel('Rotor_speed')
plt.ylabel('Target')
plt.show()

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

rtSpeed = dfhi.Rotor_speed.copy()
target = dfhi.TARGET.copy()
x2 = np.arange(rtSpeed.min(), rtSpeed.max()+0.15, 0.1)
y2 = 6.37155475e+02*x2**6 -6.09279920e+04*x**5  +2.42670353e+06*x**4 -5.15292701e+07*x**3 +\
        6.15250047e+08*x**2 -3.91638360e+09*x  +1.03835092e+10
#y2 = 8.33333e+02*x2**4 -5.303333e+04*x2**3 +1.2648417e+06*x2**2 -1.339849e+07*x2 +5.318886e+07
#plt.scatter(rtSpeed, target, s=6, alpha=0.2)
plt.scatter(x,y, s=8, alpha=0.9)
plt.plot(x2,y2, 'm-')
plt.plot(x,curve, 'r-')
plt.xlabel('Rotor_speed')
plt.ylabel('Target')
plt.show()


# exclusion des outliers
dflow_in = dflow.loc[~outLow]
dflow_out = dflow.loc[outLow]
dfmed_in = dfmed.loc[~outMed]
dfmed_out = dfmed.loc[outMed]
dfhi_in = dfhi.loc[~outHi]
dfhi_out = dfhi.loc[outHi]


allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)
import matplotlib.cm as cm
purp = cm.PuRd(0.6)
bleu = cm.Blues(0.7)
orange = cm.Oranges(0.7)
for col in lstCols :
    dfhi_in[col].hist(color=purp, bins=80, alpha=0.5, label='rotor_speed petit')
    dfhi_out[col].hist(color=bleu, bins=80, alpha=0.5,  label='rotor_speed moyen')
    plt.legend()
    plt.xlabel(col)
    plt.show()



# aucune variable n'explique à elle seule les valeurs abérantes de rotor_speed

# voir pour modéliser targe en supprimant les valeurs abérantes
# rotor_speed : 6421 indiv supprimés
#'Gearbox_oil_sump_temperature' sépare des indiv groupés de rotor_speed
#rapport min/max nacelle temperature, hub et outdoor temp pour séparer individus avec relation linéaire entre rotor_speed et target


for col in lstCols[7:] :
  plt.scatter(extractIn['Rotor_speed'], extractIn[col], s=6, alpha=0.2, label='Normal')
  plt.scatter(extractOut['Rotor_speed'], extractOut[col], s=8, label='Aberrant')
  plt.xlabel('Rotor_speed')
  plt.ylabel(col)
  plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dfmed_out['Rotor_speed'], dfmed_out['Pitch_anlge'], dfmed_out['TARGET'],alpha=0.3,s=6, marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# faire combinaison de rotor_speed et pitch_angle pour voir augmentation de target

def minmaxScale(X, min=0, max=1) :
  X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
  return X_std * (max - min) + min
  
dfmed = dfmed.assign(Combi_RotorSp_PitchAng = minmaxScale(dfmed.Rotor_speed,1,10)**2 + minmaxScale(dfmed.Pitch_angle,1,10)**2)

## !! Réussir à modéliser jusqu'à target =1500


# récupérer individus ayant relation liénaire entre rotor_speed et target pour les extraire et les étudier
dfhi.plot('Rotor_speed', 'TARGET', kind='scatter', s=6, alpha=0.2);plt.show()

rtSpeed = dfhi.Rotor_speed.copy()
target = dfhi.TARGET.copy()
x = np.arange(rtSpeed.min(), rtSpeed.max()+0.6, 0.5)
#y = 10**(0.112*x +1.2)
y1 = 975.*x -14430
y2 = 975.*x -14550
plt.scatter(rtSpeed, target, s=6, alpha=0.2)
plt.plot(x,y1, 'r-')
plt.xlabel('Rotor_speed')
plt.ylabel('Target')
plt.show()


# recupère individus ayant relation liénaire entre rotor_speed et target
y1 = lambda x : 975.*x -14430
y2 = lambda x : 975.*x -14500
condi = (df.TARGET<=df.Rotor_speed.apply(y1)) & (df.TARGET>=df.Rotor_speed.apply(y2))
dfDAff = df.loc[condi,:]

#trouver différences entre dfDAff et dfhi ~condi
allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
#lstCols = ~(allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)

for col in lstCols :
  plt.scatter(dfhi['Rotor_speed'], dfhi[col], s=6, alpha=0.2, label='normaux')
  plt.scatter(dfDAff['Rotor_speed'], dfDAff[col], s=8, label='affine')
  plt.xlabel('Rotor_speed')
  plt.ylabel(col)
  plt.show()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
poly = Pipeline([('poly',PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                 ('scale',MinMaxScaler(feature_range=(-1, 1)))])
poly.fit(df.dropna(axis=0).loc[:,lstCols.tolist()])
polyFeatNames = ['x'.join(['{}^{}'.format(pair[0],pair[1]) for pair in tuple if pair[1]!=0]) for tuple in [zip(lstCols,p) for p in poly.steps[0][1].powers_]] # cree liste des noms des nouvelles variables
xnew = pd.DataFrame(poly.transform(df.dropna(axis=0).loc[:,lstCols.tolist()]), index=df.dropna(axis=0).index, columns=polyFeatNames)

xdfhi = dfhi.join(xnew, how='left')

xdfDAff = dfDAff.join(xnew, how='left')

for col in polyFeatNames :
  plt.scatter(xdfhi[col], xdfhi['Target'], s=6, alpha=0.2, label='normaux')
  plt.scatter(xdfDAff[col], xdfDAff['Target'], s=8, label='affine')
  plt.xlabel(col)
  plt.ylabel('Target')
  plt.show()



## comprendre données Rotor_speed>12.5
# modèle de rotor_speed< 12.5 n'engendre pas de grandes erreurs sur prédiction de target
allCols = df.columns
lstCols = ~(allCols.str.endswith("_min") | allCols.str.endswith("_max") | allCols.str.endswith("_std") | allCols.str.endswith("_c"))
notKeep = ["TARGET", "LogTARGET", "MAC_CODE", "Date_time", "Absolute_wind_direction", "Nacelle_angle"]
lstCols = allCols[lstCols].difference(notKeep)

for col in lstCols :
  plt.scatter(dfhi['Rotor_speed'], dfhi[col], s=6, alpha=0.2, label='rotor_speed élevé')
  plt.scatter(dfmed['Rotor_speed'], dfmed[col], s=8, label='rotor_speed moyen')
  plt.xlabel('Rotor_speed')
  plt.ylabel(col)
  plt.show()

# déviation visible pour rotor_speed>12.5
# voir avec les colonnes min, max std en fonction de rotor_speed
rupture de rotor_speed pour >8.78
gearbox_inlet_temperature_std
generator_bearing_1(/2)_temperature(_max) >54. ; _std>1.64
generator_bearning_2_temperature_std>0.70
generator_converter_speed_min>1500
generator_stator_temperature>67 ; _max>69
nacelle_temperature_std>2.25

# voir si coupures peuvent expliquer TARGET


## étudier rotor_speed * pitch_angle


## supprimer des valeurs abérantes
## supprimer colonnes avec valeurs manquantes




### plus grandes erreurs pour TARGET > 600 ou 800
# faire regression linéaire puis arbres sur prediction >600 ou 800 ?

## voir pour créer des variables de croisement
# croisement entre wt13, wt24 et autres variables
