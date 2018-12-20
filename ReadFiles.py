import os
import pandas as pd


def GetInputTrainData(returnDf=True) :
  dirData = '../Data'
  inputData = os.path.join(dirData, 'input_training.csv')
  try :
    inputDf = pd.read_csv(inputData, sep=';', index_col=['ID'])
    #inputDf = inputDf.set_index('ID')
    return inputDf if returnDf else ' '
  except :
    print('no input data')

def GetOutputTrainData(returnDf=True) :
  dirData = '../Data'
  outputData = os.path.join(dirData, 'challenge_output_data_training_file_help_engie_improve_wind_power_production.csv')
  try :
    outputDf = pd.read_csv(outputData, sep=';', index_col=['ID'], squeeze=True)
    #outputDf = outputDf.set_index('ID')
    return outputDf if returnDf else ' '
  except :
    print('no output data')

def GetAllData() :
  inDf = GetInputTrainData(returnDf=True)
  outDf =   GetOutputTRainData(returnDf=True)
  df = inDf.join(outDf)
  return df

def GetInputTestData(returnDf=True) :
  dirData = '../Data'
  inputData = os.path.join(dirData, 'input_testing.csv')
  try :
    inputDf = pd.read_csv(inputData, sep=';', index_col=['ID'])
    #inputDf = inputDf.set_index('ID')
    return inputDf if returnDf else ' '
  except :
    print('no input data')
