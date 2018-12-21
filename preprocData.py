import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# pré-traitement pour exploration
class AddFeatures(BaseEstimator, TransformerMixin):
    """
    Add features
    """

    def fit(self, Xi, y=None):
        self.colMins = Xi.min().to_dict()
        return self

    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        # passage en 1/0
        rSpeedMin = self.colMins['Rotor_speed']
        X = X.assign(Turbulence=np.log10((X.Rotor_speed_std / (X.Rotor_speed + rSpeedMin + 0.1) + 0.1)))
        X = X.assign(Rotor_speed3=X['Rotor_speed'] ** 3)
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
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        return X.fillna(self.medianDic)


# sélection des colonnes par type
class TypeSelector(BaseEstimator, TransformerMixin):
    # retourne data frame avec colonnes sélectionnée en fonction du type souhaité
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, Xi, y=None):
        return self

    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        return X.select_dtypes(include=[self.dtype])


# codage 1/0 des vairables catégorielles par valeur croissante
class ModalitiesEncoder(BaseEstimator, TransformerMixin):
    # retourne data frame de variables qualitatives en colonnes de 1/0
    def fit(self, Xi, y=None):
        return self

    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)
        return pd.get_dummies(X)


class SelectColumns(BaseEstimator, TransformerMixin):
    # retourne data frame avec colonnes sélectionnée en fonction du type souhaité
    def __init__(self, colnames):
        self.colnames = colnames

    def fit(self, Xi, y=None):
        return self

    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        return X[self.colnames]


class SelectHiRotor(BaseEstimator, TransformerMixin):
    # retourne data frame avec individus ayant rotor_speed > 15
    def __init__(self, keepHighRS=True):
        self.keepHighRS = keepHighRS

    def fit(self, Xi, y=None):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        condi = (X['Rotor_speed'] >= 15.)
        self.keepIndex = condi[condi].index
        self.notKeepIndex = condi[~condi].index
        return self

    def transform(self, Xi):
        X = Xi.copy()
        if self.keepHighRS:
            return X.loc[self.keepIndex]
        else:
            return X.loc[self.notKeepIndex]


class PreTraitementContinues(BaseEstimator, TransformerMixin):
    # pré-traitement des variables continues
    def __init__(self):
        # self.colnames = colnames
        self  # return self

    def fit(self, Xi, y=None):
        return self

    def transform(self, Xi):
        X = Xi.copy()
        assert isinstance(X, pd.DataFrame)  # vérifie que X est un data frame
        # passage en log10
        lstToLog10p1 = ['age', 'capital-gain', 'capital-loss']
        X[lstToLog10p1] = np.log10(X[lstToLog10p1] + 1)
        # passage en 1/0
        X = X.assign(haveGain=1 * (X['capital-gain'] != 0))
        X = X.assign(haveLoss=1 * (X['capital-loss'] != 0))
        X = X.assign(haveGainLoss=1 * ((X['capital-loss'] != 0) | (X['capital-gain'] != 0)))
        return X
