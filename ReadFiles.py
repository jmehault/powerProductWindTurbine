import os
import pandas as pd


def GetInputTrainData(returnDf=True):
    """
    Get input training data
    """
    dirData = '../Data'
    inputData = os.path.join(dirData, 'input_training.csv')
    try:
        inputDf = pd.read_csv(inputData, sep=';', index_col=['ID'])
        return inputDf if returnDf else ' '
    except:
        print('no input data')


def GetOutputTrainData(returnDf=True):
    """
     Get output training data
    """
    dirData = '../Data'
    filename = 'challenge_output_data_training_file_help_engie_improve_wind_power_production.csv'
    outputData = os.path.join(dirData, filename)
    try:
        outputDf = pd.read_csv(outputData, sep=';', index_col=['ID'], squeeze=True)
        # outputDf = outputDf.set_index('ID')
        return outputDf if returnDf else ' '
    except:
        print('no output data')


def GetAllData():
    inDf = GetInputTrainData(returnDf=True)
    outDf = GetOutputTrainData(returnDf=True)
    df = inDf.join(outDf)
    return df


def GetInputTestData(returnDf=True):
    dirData = '../Data'
    inputData = os.path.join(dirData, 'input_testing.csv')
    try:
        inputDf = pd.read_csv(inputData, sep=';', index_col=['ID'])
        return inputDf if returnDf else ' '
    except:
        print('no input data')
