from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getMAE(yt, yp):
    return mean_absolute_error(yt, yp)


def getAE(yt, yp):
    return np.abs(yt - yp)


def getMAPE(yt, yp):
    return np.mean(getAE(yt, yp) / yt)


def getResidPlot(yt, yp):
    mae = getMAE(yt, yp)
    plt.subplot(2, 1, 1)
    plt.plot(yt, yp, '.')
    plt.xlabel('Observé')
    plt.ylabel('Prédiction')
    plt.title(f'Erreur absolue moyenne : {mae}')
    plt.subplot(2, 1, 2)
    plt.plot(yt, getAE(yt, yp), '.')
    plt.xlabel('Observé')
    plt.ylabel('Erreur absolue')
    plt.show()


def getAllResidPlot(ytr, yptr, yte, ypte):
    maetr = np.round(getMAE(ytr, yptr), 2)
    maete = np.round(getMAE(yte, ypte), 2)
    f = plt.figure(figsize=(8, 10))
    plt.subplot(2, 2, 1)
    plt.plot(ytr, yptr, '.')
    plt.plot([0, ytr.max()], [0, ytr.max()], 'r.-')
    plt.ylabel('Prédiction')
    plt.title(f'Erreur absolue moyenne train\n{maetr}')
    plt.subplot(2, 2, 3)
    plt.plot(ytr, getAE(ytr, yptr), '.')
    plt.xlabel('Observé train')
    plt.ylabel('Erreur absolue')
    plt.subplot(2, 2, 2)
    plt.plot(yte, ypte, '.')
    plt.plot([0, yte.max()], [0, yte.max()], 'r.-')
    plt.title(f'Erreur absolue moyenne test\n{maete}')
    plt.subplot(2, 2, 4)
    plt.plot(yte, getAE(yte, ypte), '.')
    plt.xlabel('Observé test')
    plt.show()


def plotImportance(xdat, model):
    try:
        # modèle simple
        importance = pd.Series(model.feature_importances_, index=xdat.columns, name="importVal").sort_values(
            ascending=True)
    except:
        # modèle dans Pipeline
        algo = model.named_steps['model']
        cols = model.named_steps['selectCols'].colnames
        importance = pd.Series(algo.feature_importances_, index=cols, name="importVal").sort_values(ascending=True)
    # posx = np.arange(len(importance))
    importance.plot(kind='barh')
    plt.tight_layout()
    plt.show()
