import h5py, torch, time, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.modules import Module
from tabulate import tabulate

class DataFile():
### Reads sample file Info (string), Parameters (list), Values (torch array), Data (torch array) and Weights (torch array)
### FilePath is the path of the input file
### Computes cross-section XS (average weight) and total number of data ND in file
### Checks that files are in correct format (correct Keys)
### and that the length of Parameters and Data equals the one of Values and Weights respectively
    def __init__(self, FilePath, verbose=True):
        if verbose: print('\nReading file ...' + FilePath)
        file = h5py.File(FilePath, 'r')
        if list(file.keys()) == ['Data', 'Info', 'Parameters', 'Process', 'Values', 'Weights']:
            if( (len(file['Parameters'][()]) == len(file['Values'][()])) and (len(file['Data'][()]) == len(file['Weights'][()])) ):
                if verbose: print('##### File Info:\n' + file['Info'][()][0] + '\n#####')
                self.FilePath = FilePath
                self.Info = file['Info'][()][0]
                self.Process = file['Process'][()][0]
                self.Parameters = file['Parameters'][()]
                #print(file['Values'][()].dtype)
                #print('%.15f' % (file['Data'][()][0][0]))
                self.Values = torch.DoubleTensor(file['Values'][()])
                self.Data = torch.DoubleTensor(file['Data'][()])
                self.Weights = torch.DoubleTensor(file['Weights'][()])
                self.XS = self.Weights.mean()
                self.ND = len(self.Weights)
            else: 
                print('--> File not valid:\nunequal lenght of Values and Parameters or of Data and Weights')
                raise ValueError
        else:
            print('--> File format not valid:\nKeys: ' + str(list(file.keys())) + 
                  '\nshould be: ' + str(['Data', 'Info', 'Parameters', 'Process', 'Values', 'Weights']))
            raise ValueError

class OurTrainingData():
### Imports data for training. The Return() methods returns [self.Data, self.Labels, self.Weights, self.ParVal]
### All values are in double precision
### Inputs are the SM and BSM file paths and list of integers to chop the datasets if needed
### Weights are normalized to have average = 1 on the entire training sample
    def __init__(self, SMfilepathlist, BSMfilepathlist, process, parameters, SMNLimits="NA", BSMNLimits="NA", verbose=True): 
        self.Process = process
        self.Parameters = parameters
        if verbose: print('Loading Data Files for Process: ' + str(self.Process) +', with new physics Parameters: ' + str(self.Parameters) ) 
        if len(self.Parameters)!= 1: print('Only 1D Implemented in Training !')   
          
####### Load BSM data (stored in self.BSMDataFiles)
        if type(BSMfilepathlist) == list:
            if all(isinstance(n, str) for n in BSMfilepathlist):
                self.BSMDataFiles = []
                for path in BSMfilepathlist:
                    temp =  DataFile(path, verbose=verbose)
                    if( (temp.Process == self.Process) and (temp.Parameters == self.Parameters) and (temp.Values != 0.) ):
                        self.BSMDataFiles.append(temp)
                    else: 
                        print('File not valid: ' + path)
                        print('Parameters = ' + str(temp.Parameters) + ', Process = ' + str(temp.Process) 
                              +' and Values = ' + str(temp.Values.tolist()))
                        print('should be = ' + str(self.Parameters) + ', = ' + str(self.Process) 
                              + ' and != ' + str(0.))
                        raise ValueError
                        self.BSMDataFiles.append(None) 
            else:
                print('BSMfilepathlist input should be a list of strings !')
                raise FileNotFoundError
        else:
            print('BSMfilepathlist input should be a list !')
            raise FileNotFoundError
                  
###### Chop the BSM data sets (stored in BSMNDList, BSMDataList, BSMWeightsList, BSMParValList, BSMTargetList)
        if type(BSMNLimits) == int:
            BSMNLimits = [BSMNLimits for data in self.BSMDataFiles]
        elif type(BSMNLimits) == list and all(isinstance(n, int) for n in BSMNLimits):
            if len(BSMNLimits) != len(self.BSMDataFiles):
                print("--> Please input %d integers to chop each SM file."%(
                    len(self.BSMDataFiles)))
                raise ValueError
            elif sum([self.BSMDataFiles[i].ND >= BSMNLimits[i] for i in range(len(BSMNLimits))]
                    ) != len(self.BSMDataFiles):
                print("--> Some chop limit larger than available data in the corresponding file.")
                print("--> Lengths of the files: "+str([file.ND for file in self.BSMDataFiles ]))
                raise ValueError
        else:
            BSMNLimits =[file.ND for file in self.BSMDataFiles]   
            
        self.BSMNDList = BSMNLimits
        #self.BSMNData = sum(self.BSMNDataList)
        self.BSMDataList = [DF.Data[:N] for (DF, N) in zip(
            self.BSMDataFiles, self.BSMNDList)]
        self.BSMWeightsList = [DF.Weights[:N] for (DF, N) in zip(
            self.BSMDataFiles, self.BSMNDList)] 
        self.BSMParValList =  [torch.ones(N, dtype=torch.double)*DF.Values for (DF, N) in zip(self.BSMDataFiles, self.BSMNDList)]
        self.BSMTargetList = [torch.ones(N, dtype=torch.double) for N in self.BSMNDList] 
        
####### Load SM data (stored in SMDataFiles)
        if type(SMfilepathlist) == list:
            if all(isinstance(n, str) for n in SMfilepathlist):
                #self.SMFilePathList = SMfilepathlist
                #self.SMNumFiles = len(self.SMFilePathList)
                self.SMDataFiles = []
                for path in SMfilepathlist:
                    temp =  DataFile(path, verbose=verbose)
                    if( (temp.Process == self.Process) and (temp.Parameters == 'SM') and (temp.Values == 0.) ):
                       self.SMDataFiles.append(temp)
                    else:
                        print('File not valid: ' + path)
                        print('Parameters = ' + str(temp.Parameters) + ', Process = ' + str(temp.Process) 
                              +' and Values = ' + str(temp.Values.tolist()))
                        print('should be = ' + 'SM'+ ', = ' + str(self.Process) 
                              + ' and = ' + str(0.))
                        self.SMDataFiles.append(None)                    
            else:
                print('SMfilepathlist input should be a list of strings !')
                raise FileNotFoundError
        else:
            print('SMfilepathlist input should be a list !')
            raise FileNotFoundError
            
####### Chop the SM data sets and join them in one (stored in SMND, SMData and SMWeights)
        if type(SMNLimits) == int:
            SMNLimits = [SMNLimits for data in self.SMDataFiles]
        elif type(SMNLimits) == list and all(isinstance(n, int) for n in SMNLimits):
            if len(SMNLimits) != len(self.SMDataFiles):
                print("--> Please input %d integers to chop each SM file."%(
                    len(self.SMDataFiles)))
                raise ValueError
            elif sum([self.SMDataFiles[i].ND >= SMNLimits[i] for i in range(len(SMNLimits))]
                    ) != len(self.SMDataFiles):
                print("--> Some chop limit larger than available data in the corresponding file.")
                print("--> Lengths of the files: " + str([file.ND for file in self.SMDataFiles]))
                raise ValueError
        else:
            SMNLimits = [file.ND for file in self.SMDataFiles]
        self.SMND = sum(SMNLimits)
        self.SMData = torch.cat(
            [DF.Data[:N] for (DF, N) in zip(self.SMDataFiles, SMNLimits)]
            , 0) 
        self.SMWeights = torch.cat(
            [DF.Weights[:N] for (DF, N) in zip(self.SMDataFiles, SMNLimits)]
            , 0)

####### Break SM data in blocks to be paired with BSM data (stored in UsedSMNDList, UsedSMDataList, UsedSMWeightsList, UsedSMParValList, UsedSMTargetList)
        BSMNRatioDataList = [torch.tensor(1., dtype=torch.double)*n/sum(self.BSMNDList
                                                                       ) for n in self.BSMNDList]
        self.UsedSMNDList = [int(self.SMND*BSMNRatioData) for BSMNRatioData in BSMNRatioDataList] 
        #self.UsedSMNData = sum(self.UsedSMNDataList)
        #self.UsedSMData = self.SMData[: self.UsedSMND]
        self.UsedSMDataList =  self.SMData[:sum(self.UsedSMNDList)].split(self.UsedSMNDList)
        
    ##### Reweighting is performed such that the SUM of the SM weights in each block equals the number of BSM data times the AVERAGE 
    ##### of the original weights. This equals the SM cross-section as obtained in the specific sample at hand, times NBSM
        self.UsedSMWeightsList = self.SMWeights[:sum(self.UsedSMNDList)].split(self.UsedSMNDList)
        self.UsedSMWeightsList = [ self.UsedSMWeightsList[i]*self.BSMNDList[i]/self.UsedSMNDList[i] for i in range(len(BSMNRatioDataList))]      
        self.UsedSMParValList =  [torch.ones(N, dtype=torch.double)*DF.Values for (DF, N) in zip(self.BSMDataFiles, self.UsedSMNDList)]       
        self.UsedSMTargetList = [torch.zeros(N, dtype=torch.double) for N in self.UsedSMNDList]

####### Join SM with BSM data
        self.Data = torch.cat(
            [torch.cat([self.UsedSMDataList[i], self.BSMDataList[i]]
                                  ) for i in range(len(self.BSMDataList))]
            )
        self.Weights = torch.cat(
            [torch.cat([self.UsedSMWeightsList[i], self.BSMWeightsList[i]]
                                  ) for i in range(len(self.BSMWeightsList))]
            )
        self.Labels = torch.cat(
            [torch.cat([self.UsedSMTargetList[i], self.BSMTargetList[i]]
                                  ) for i in range(len(self.BSMTargetList))]
            )
        self.ParVal = torch.cat(
            [torch.cat([self.UsedSMParValList[i], self.BSMParValList[i]]
                                  ) for i in range(len(self.BSMParValList))]
            )
        
####### Final reweighting
        avg = self.Weights.mean()
        self.Weights = self.Weights.div(avg)

####### If verbose, display report
        if verbose: self.Report()
        
####### Return Tranining Data
    def ReturnData(self):
        return [self.Data, self.Labels, self.Weights, self.ParVal]
            
    def Report(self):
        #from tabulate import tabulate
        print('\nLoaded SM Files:')
        print(tabulate({str(self.Parameters): [ file.Values for file in self.SMDataFiles ], 
                        "#Data":[ file.ND for file in self.SMDataFiles ], 
                        "XS[pb](avg.w)":[ file.XS for file in self.SMDataFiles ]}, headers="keys"))
        print('\nLoaded BSM Files:')
        print(tabulate({str(self.Parameters): [ file.Values for file in self.BSMDataFiles ], 
                        "#Data":[ file.ND for file in self.BSMDataFiles ], 
                        "XS[pb](avg.w)":[ file.XS for file in self.BSMDataFiles ]}, headers="keys"))
        print('\nPaired BSM/SM Datasets:\n')
        ### Check should be nearly equal to #EV.BSM. It is computed with the weights BEFORE final reweighting
        print(tabulate({str(self.Parameters): [ file.Values for file in self.BSMDataFiles ], "#Ev.BSM": self.BSMNDList
                        , "#Ev.SM": self.UsedSMNDList,
                        "Check": [(self.UsedSMWeightsList[i].sum())/(self.SMWeights.mean()) for i in range(len(self.BSMDataFiles))]
                       }, headers="keys"))     
        
####### Loss function(s), with "input" in (0,1) interval
class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
class WeightedMSELoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
    def forward(self, input, target, weight):
        return torch.mean(torch.mul(weight, (input - target)**2))
