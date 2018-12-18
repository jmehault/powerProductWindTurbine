import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


## pre traitement pour exploration
class AddFeatures(BaseEstimator, TransformerMixin) :
    def fit(self, Xi, y=None):
        self.colMins = Xi.min().to_dict()
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        ## passage en 1/0
        rSpeedMin = self.colMins['Rotor_speed']
        X = X.assign(Turbulence = np.log10((X.Rotor_speed_std/(X.Rotor_speed+rSpeedMin+0.1)+0.1)))
        X = X.assign(Rotor_speed3 = X['Rotor_speed']**3)
        return X

class ImputeMedian(BaseEstimator, TransformerMixin):
    """
    Impute columns with median
    """
    def fit(self, Xi, y=None):
        X = Xi.copy()
        self.medianDic = X.median().to_dict()
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        return X.fillna(self.medianDic)

## sélection des colonnes par type
class TypeSelector(BaseEstimator, TransformerMixin) :
    ## retourne data frame avec colonnes sélectionnée en fonction du type souhaité
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        return X.select_dtypes(include=[self.dtype])

# codage 1/0 des vairables catégorielles par valeur croissante
class ModalitiesEncoder(BaseEstimator, TransformerMixin):
    ## retourne data frame de variables qualitatives en colonnes de 1/0
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)
        return pd.get_dummies(X)


class SelectColumns(BaseEstimator, TransformerMixin) :
    ## retourne data frame avec colonnes sélectionnée en fonction du type souhaité
    def __init__(self, colnames):
        self.colnames = colnames
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        return X[self.colnames]

class SelectHiRotor(BaseEstimator, TransformerMixin) :
    ## retourne data frame avec individus ayant rotor_speed > 15
    def __init__(self, keepHighRS=True) :
        self.keepHighRS = keepHighRS
    
    def fit(self, Xi, y=None):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        condi = (X['Rotor_speed']>=15.)
        self.keepIndex = condi[condi].index
        self.notKeepIndex = condi[~condi].index
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        if self.keepHighRS :
            return X.loc[self.keepIndex]
        else :
            return X.loc[self.notKeepIndex]

class PreTraitementContinues(BaseEstimator, TransformerMixin) :
    ## pré-traitement des variables continues
    def __init__(self):
        #self.colnames = colnames
        self #return self
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        ## passage en log10
        lstToLog10p1 = ['age', 'capital-gain', 'capital-loss']
        X[lstToLog10p1] = np.log10(X[lstToLog10p1]+1)
        ## passage en 1/0
        X = X.assign(haveGain = 1*(X['capital-gain']!=0))
        X = X.assign(haveLoss = 1*(X['capital-loss']!=0))
        X = X.assign(haveGainLoss = 1*((X['capital-loss']!=0) | (X['capital-gain']!=0)))
        return X


class PreTraiteQualitatives(BaseEstimator, TransformerMixin) :
    ## pré-traitement des variables qualitatives
    def __init__(self):
        #self.colnames = colnames
        self #return self
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        ## on remplace valeurs de workclass Never-worked par nan car n'apparaît par dans test
        ## idem pour 'Holand-Netherlands' dans native-country
        ## on remplace husband en wife dans relationship
        X = X.replace({'native-country':{'Holand-Netherlands':np.nan},
                       'workclass':{'Never-worked':np.nan},
                       'relationship':{'Husband':'Wife'}})
        ## remplace pays par continent
        lstPays = {'United-States':'AmerN', 'Cuba':'AmerC', 'Jamaica':'AmerC', 'India':'Asie', 'Mexico':'AmerC',
         'Puerto-Rico':'AmerC', 'Honduras':'AmerC', 'England':'Europ', 'Canada':'AmerN', 'Germany':'Europ', 'Iran':'MoyOr',
         'Philippines':'Asie', 'Poland':'Europ', 'Columbia':'AmerC', 'Cambodia':'Asie', 'Thailand':'Asie',
         'Ecuador':'AmerC', 'Laos':'Asie', 'Taiwan':'Asie', 'Haiti':'AmerC', 'Portugal':'Europ',
         'Dominican-Republic':'AmerC', 'El-Salvador':'AmerC', 'France':'Europ', 'Guatemala':'AmerC',
         'Italy':'Europ', 'China':'Asie', 'South':'Unknown', 'Japan':'Asie', 'Yugoslavia':'Europ', 'Peru':'AmerS',
         'Outlying-US(Guam-USVI-etc)':'Unknown', 'Scotland':'Europ', 'Trinadad&Tobago':'Unknown',
         'Greece':'Europ', 'Nicaragua':'AmerC', 'Vietnam':'Asie', 'Hong':'Asie', 'Ireland':'Europ', 'Hungary':'AmerC'}
        X_pays = X['native-country'].replace(lstPays)
        X_pays.name = 'continent'
        X = pd.concat((X, X_pays), axis=1)
        X.drop('native-country', axis=1, inplace=True)
        return X

class SelectIndiv(BaseEstimator, TransformerMixin) :
    ## retourne data frame avec individus sélectionnés pour prédiction brute ou non
    def __init__(self, keepIndForBrut) :
        self.keepIndForBrut = keepIndForBrut
    
    def fit(self, Xi, y=None):
        return self
    
    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame) # vérifie que X est un data frame
        cond1 = (X['occupation_Priv-house-serv']==1) | (X['occupation_Armed-Forces']==1)
        cond2 = X['workclass_Without-pay']==1
        cond3 = X['relationship_Own-child']==1
        condFinal = cond1 | cond2 | cond3
        if self.keepIndForBrut :
            return X.loc[condFinal,:]
        else :
            return X.loc[~condFinal,:]


class MyModelBrut(BaseEstimator, TransformerMixin) :
    ## Attribue income <=50K pour les individus sélectionnés par SelectIndiv(True)
    def fit(self, Xi, y=None) :
        return self
    
    def transform(self, Xi, y=None) :
        X = Xi.copy()
        predBrut = pd.Series(np.zeros(X.shape[0]), index=X.index, name='income')
        predBrut = predBrut.loc[~predBrut.index.duplicated()] ## supprime les doublons sur les indices
        return predBrut
