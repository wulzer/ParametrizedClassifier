import h5py, torch, time, datetime, os
import numpy as np
from string import digits
import matplotlib
#matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss
from tabulate import tabulate




### ===============================================================================================
### ===============================================================================================
### ===============================================================================================

class CombinedDataFile():
### Reads sample file Info (string), Parameters (list), Values (torch array), Data (torch array) and Weights (torch array)
### FilePath is the path of the input file
### Computes cross-section XS (average weight) and total number of data ND in file
### Checks that files are in correct format (correct Keys)
### and that the length of Parameters and Data equals the one of Values and Weights respectively
    def __init__(self, DataFilePath, WeightFilePath, verbose=True, NReadData=int(1e6), NumericalPrecision = torch.float32):
        if verbose: print('\nReading file ...' + DataFilePath)
        
        datafile   = h5py.File(DataFilePath, 'r')
        weightfile = h5py.File(WeightFilePath, 'r')
        
        torch.set_default_dtype(NumericalPrecision)
        print('Data Type = ' + str(NumericalPrecision) + '. Global variable torch.set_default_dtype set to ' \
              + str(torch.get_default_dtype()) + '.')
        
        if list(datafile.keys()) != ['Data', 'Info', 'Parameters', 'Process', 'ReweightCoeffs', 'Values', 'Weights']:
            raise ValueError
        if list(weightfile.keys()) != ['ReweightValues', 'Reweights', 'Weights']:
            raise ValueError
                    
            if verbose: print('##### File Info:\n' + datafile['Info'][()][0] + '\n#####')
                
        self.DataFilePath   = DataFilePath
        self.WeightFilePath = WeightFilePath
        self.Info           = datafile['Info'][()][0]
        self.Process        = datafile['Process'][()][0]
        self.ParameterNames = datafile['Parameters'][()]
        self.Values         = torch.tensor(datafile['Values'][()], dtype = NumericalPrecision)
        self.Data           = torch.tensor(datafile['Data'][()], dtype = NumericalPrecision)
        self.Weights        = torch.tensor(weightfile['Weights'][()], dtype = NumericalPrecision)
        self.ReweightValues = torch.tensor(weightfile['ReweightValues'][()], dtype = NumericalPrecision)
        self.Reweights      = torch.tensor(weightfile['Reweights'][()], dtype = NumericalPrecision)

        self.Data           = self.Data[0:NReadData]
        self.Weights        = self.Weights[0:NReadData]
        self.Reweights      = self.Reweights[0:NReadData]
        self.ReweightValues = self.ReweightValues[0:NReadData]
        self.ND = self.Data.size(0)

        self.scale = torch.sum(self.Weights)*(self.ReweightValues.size(-2)) + torch.sum(self.Reweights)
        
####### Shuffle 
    def RandomShuffle(self, seed):
        ShuffleGenerator = torch.Generator()
        ShuffleGenerator.manual_seed(seed)
        RandIndex = torch.randperm(self.ND, generator=ShuffleGenerator)
    
        self.Data = self.Data[RandIndex]
        self.Weights = self.Weights[RandIndex]
        self.Reweights = self.Reweights[RandIndex]
        
        if len(self.ReweightValues.size()) == 3:
            self.ReweightValues = self.ReweightValues[RandIndex]
    
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[RandIndex]

####### Take
    def Take(self, NDdataStart, NDdataEnd):
        self.Weights = self.Weights[NDdataStart:NDdataEnd]
        self.ND = len(self.Weights)
        self.Data = self.Data[NDdataStart:NDdataEnd]
        self.Reweights = self.Reweights[NDdataStart:NDdataEnd]
        self.ReweightValues = self.ReweightValues[NDdataStart:NDdataEnd]
        self.scale = torch.sum(self.Weights)*self.ReweightValues.size(-2) + torch.sum(self.Reweights)
        
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[NDdataStart:NDdataEnd]
            self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)


class DataFile():
### Reads sample file Info (string), Parameters (list), Values (torch array), Data (torch array) and Weights (torch array)
### FilePath is the path of the input file
### Computes cross-section XS (average weight) and total number of data ND in file
### Checks that files are in correct format (correct Keys)
### and that the length of Parameters and Data equals the one of Values and Weights respectively
    def __init__(self, FilePath, PDFWeights=True, verbose=True, NReadData=-1, NumericalPrecision = torch.float32):
        if verbose: print('\nReading file ...' + FilePath)
        file = h5py.File(FilePath, 'r')
        torch.set_default_dtype(NumericalPrecision)
        print('Data Type = ' + str(NumericalPrecision) + '. Global variable torch.set_default_dtype set to ' \
              + str(torch.get_default_dtype()) + '.')
        if PDFWeights:
            if list(file.keys()) != ['Data', 'Info', 'PDF_weights', 'Parameters', 'Process', 'ReweightValues', 'Reweights', 'Values', 'Weights']:
                print('--> File format not valid:\nKeys: ' + str(list(file.keys())) + 
                  '\nshould be: ' + str(['Data', 'Info', 'PDF_weights', 'Parameters', 'Process', 'ReweightValues', 'Reweights', 'Values', 'Weights']))
                raise ValueError
        else:
            if list(file.keys()) != ['Data', 'Info', 'Parameters', 'Process', 'ReweightValues', 'Reweights', 'Values', 'Weights']:
                if list(file.keys()) != ['Data', 'Info', 'Parameters', 'Process', 'ReweightCoeffs', 'ReweightValues', 'Reweights', 'Values', 'Weights']:
                    print('--> File format not valid:\nKeys: ' + str(list(file.keys())) + 
                  '\nshould be: ' + str(['Data', 'Info', 'Parameters', 'Process', 'ReweightValues', 'Reweights', 'Values', 'Weights']))
                    raise ValueError
                    

        if( (len(file['Data'][()]) == len(file['Weights'][()])) ):
            if verbose: print('##### File Info:\n' + file['Info'][()][0] + '\n#####')
            self.FilePath = FilePath
            self.Info = file['Info'][()][0]
            self.Process = file['Process'][()][0]
            self.ParameterNames = file['Parameters'][()]
            self.Values = torch.tensor(file['Values'][()], dtype = NumericalPrecision)
            self.Data = torch.tensor(file['Data'][()], dtype = NumericalPrecision)
            self.Weights = torch.tensor(file['Weights'][()], dtype = NumericalPrecision)
            self.ReweightValues = torch.tensor(file['ReweightValues'][()], dtype = NumericalPrecision)
            self.Reweights = torch.tensor(file['Reweights'][()], dtype = NumericalPrecision)
            
            self.Data = self.Data[0:NReadData]
            self.Weights = self.Weights[0:NReadData]
            self.Reweights = self.Reweights[0:NReadData]
            self.ReweightValues = self.ReweightValues[0:NReadData]
            if PDFWeights:
                self.PDFWeights = torch.tensor(file['PDF_weights'][()], dtype = NumericalPrecision)
                self.PDFWeights = self.PDFWeights[0:NReadData]
            self.ND = len(self.Weights)
            
            self.scale = torch.sum(self.Weights)*(self.ReweightValues.size(-2)) + torch.sum(self.Reweights)
            if PDFWeights:
                self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)

        else:
            print('--> File not valid:\nunequal lenght of Data and Weights')
            raise ValueError

####### Convert Angles
    def ConvertAngles(self, AnglePos):
        Angles = self.Data[:, AnglePos]
        ConvertedAngles = torch.cat([torch.sin(Angles), torch.cos(Angles)], dim=1)
        OtherPos = list(set(range(self.Data.size(1)))-set(AnglePos))
        self.Data = torch.cat([self.Data[:, OtherPos], ConvertedAngles], dim=1)
        print('####\nAngles at position %s have been converted to Sin and Cos and put at the last columns of the Data.'%(AnglePos))
        print('####')
        
####### Shuffle 
    def RandomShuffle(self, seed):
        ShuffleGenerator = torch.Generator()
        ShuffleGenerator.manual_seed(seed)
        RandIndex = torch.randperm(self.ND, generator=ShuffleGenerator)
    
        self.Data = self.Data[RandIndex]
        self.Weights = self.Weights[RandIndex]
        self.Reweights = self.Reweights[RandIndex]
        
        if len(self.ReweightValues.size()) == 3:
            self.ReweightValues = self.ReweightValues[RandIndex]
    
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[RandIndex]
            
        return RandIndex

####### Take
    def Take(self, NDdataStart, NDdataEnd):
        self.Weights = self.Weights[NDdataStart:NDdataEnd]
        self.ND = len(self.Weights)
        self.Data = self.Data[NDdataStart:NDdataEnd]
        self.Reweights = self.Reweights[NDdataStart:NDdataEnd]
        self.ReweightValues = self.ReweightValues[NDdataStart:NDdataEnd]
        self.scale = torch.sum(self.Weights)*self.ReweightValues.size(-2) + torch.sum(self.Reweights)
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[NDdataStart:NDdataEnd]
            self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)
####### Take
    def TakeByIndex(self, Index):
        self.Weights = self.Weights[Index]
        self.ND = len(self.Weights)
        self.Data = self.Data[Index]
        self.Reweights = self.Reweights[Index]
        self.ReweightValues = self.ReweightValues[Index]
        self.scale = torch.sum(self.Weights)*(self.ReweightValues.size(-2)) + torch.sum(self.Reweights)
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[Index]
            self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)           
            


        
### ===============================================================================================
### ===============================================================================================
### ===============================================================================================

class DataFileNOReweight():
### Reads sample file Info (string), Parameters (list), Values (torch array), Data (torch array) and Weights (torch array)
### FilePath is the path of the input file
### Computes cross-section XS (average weight) and total number of data ND in file
### Checks that files are in correct format (correct Keys)
### and that the length of Parameters and Data equals the one of Values and Weights respectively
    def __init__(self, FilePath, PDFWeights=True, verbose=True, NReadData=-1, NumericalPrecision = torch.float32):
        if verbose: print('\nReading file ...' + FilePath)
        file = h5py.File(FilePath, 'r')
        if PDFWeights:
            if list(file.keys()) != ['Data', 'Info', 'PDF_weights', 'Parameters', 'Process', 'Values', 'Weights']:
                print('--> File format not valid:\nKeys: ' + str(list(file.keys())) + 
                  '\nshould be: ' + str(['Data', 'Info', 'PDF_weights', 'Parameters', 'Process', 'Values', 'Weights']))
                raise ValueError
        else:
            if list(file.keys()) != ['Data', 'Info', 'Parameters', 'Process', 'Values', 'Weights']:
                print('--> File format not valid:\nKeys: ' + str(list(file.keys())) + 
                  '\nshould be: ' + str(['Data', 'Info', 'Parameters', 'Process', 'Values', 'Weights']))
                raise ValueError
                
        #if( (len(file['Parameters'][()]) == len(file['Values'][()])) and (len(file['Data'][()]) == len(file['Weights'][()])) ):
        if( (len(file['Data'][()]) == len(file['Weights'][()])) ):
            if verbose: print('##### File Info:\n' + file['Info'][()][0] + '\n#####')
            self.FilePath = FilePath
            self.Info = file['Info'][()][0]
            self.Process = file['Process'][()][0]
            self.ParameterNames = file['Parameters'][()]
            #self.Values = torch.DoubleTensor(file['Values'][()]).float()
            self.Values = torch.tensor(file['Values'][()], dtype = NumericalPrecision)
            #self.Data = torch.DoubleTensor(file['Data'][()]).float()
            self.Data = torch.tensor(file['Data'][()], dtype = NumericalPrecision)
            #self.Weights = torch.DoubleTensor(file['Weights'][()]).float()
            self.Weights = torch.tensor(file['Weights'][()], dtype = NumericalPrecision)
            #self.ReweightValues = torch.DoubleTensor(file['ReweightValues'][()]).float()
            #self.ReweightValues = torch.tensor(file['Values'][()], dtype = NumericalPrecision)
            #self.Reweights = torch.DoubleTensor(file['Reweights'][()]).float()
            #self.Reweights = torch.tensor(file['Reweights'][()], dtype = NumericalPrecision)
            
            self.Data = self.Data[0:NReadData]
            self.Weights = self.Weights[0:NReadData]
            #self.Reweights = self.Reweights[0:NReadData]
                        
            if PDFWeights:
                #self.PDFWeights = torch.DoubleTensor(file['PDF_weights'][()]).float()
                self.PDFWeights = torch.tensor(file['PDF_weights'][()], dtype = NumericalPrecision)
                self.PDFWeights = self.PDFWeights[0:NReadData]
                self.PDFWeights = 1. + self.PDFWeights
                self.PDFWeights = torch.einsum('ij,i->ij',self.PDFWeights,self.Weights)
            self.ND = len(self.Weights)
            
            #s = torch.sum(self.Weights)*len(self.ReweightValues) + torch.sum(self.Reweights)
            #self.Weights = self.Weights.div(s)
            #self.Reweights = self.Reweights.div(s)
            #if PDFWeights:
            #    self.PDFWeights = self.PDFWeights.div(s)
            
            #self.scale = torch.sum(self.Weights)*len(self.ReweightValues) + torch.sum(self.Reweights)
            if PDFWeights:
                self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)

        else:
            #print('--> File not valid:\nunequal lenght of Values and Parameters or of Data and Weights')
            print('--> File not valid:\nunequal lenght of Data and Weights')
            raise ValueError

####### Convert Angles
    def ConvertAngles(self, AnglePos):
        Angles = self.Data[:, AnglePos]
        ConvertedAngles = torch.cat([torch.sin(Angles), torch.cos(Angles)], dim=1)
        OtherPos = list(set(range(self.Data.size(1)))-set(AnglePos))
        self.Data = torch.cat([self.Data[:, OtherPos], ConvertedAngles], dim=1)
        print('####\nAngles at position %s have been converted to Sin and Cos and put at the last columns of the Data.'%(AnglePos))
        print('####')

    def RandomShuffle(self, seed):
        ShuffleGenerator = torch.Generator()
        ShuffleGenerator.manual_seed(seed)
        RandIndex = torch.randperm(td.ND, generator=ShuffleGenerator)
    
        self.Data = self.Data[RandIndex]
        self.Weights = self.Weights[RandIndex]
        
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[RandIndex]
            
####### Shulffle 
    def RandomShuffleOLD(self, seed):
        np.random.seed(seed)
        arr = np.arange(self.ND)
        np.random.shuffle(arr)
        self.Data = self.Data[arr]
        self.Weights = self.Weights[arr]
        #elf.Reweights = self.Reweights[arr]
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[arr]
        
####### Take
    def Take(self, NDdataStart, NDdataEnd):
        self.Weights = self.Weights[NDdataStart:NDdataEnd]
        self.ND = len(self.Weights)
        self.Data = self.Data[NDdataStart:NDdataEnd]
        if hasattr(self, 'PDFWeights'):
            self.PDFWeights = self.PDFWeights[NDdataStart:NDdataEnd]
            self.PDFscale = torch.sum(self.Weights)*(self.PDFWeights.size()[-1]) + \
                                        torch.sum(self.PDFWeights)

            
            
### ===============================================================================================
### ===============================================================================================       
### ===============================================================================================

class OurModel(nn.Module):
### Input is the architecture (list of integers, the last one being equal to 1) and the activation type ('ReLU' or 'Sigmoid')
    def __init__(self, NumberOfParameters, AR = [1, 3, 3, 1] , AF = 'ReLU', PDF = 'No'):               
        super(OurModel, self).__init__() 
        ValidActivationFunctions = {'ReLU': torch.relu, 'Sigmoid': torch.sigmoid}
        try:
            self.ActivationFunction = ValidActivationFunctions[AF]
        except KeyError:
            print('The activation function specified is not valid. Allowed activations are %s.'
                 %str(list(ValidActivationFunctions.keys())))
            print('Will use ReLU.')
            self.ActivationFunction = torch.relu
        if type(AR) == list:
            if( ( all(isinstance(n, int) for n in AR)) and ( AR[-1] == 1 or PDF != 'No' ) ):
                self.Architecture = AR
            else:
                print('Architecture should be a list of integers, the last one should be 1.')
                raise ValueError             
        else:
            print('Architecture should be a list !')
            raise ValueError
        self.NumberOfParameters = NumberOfParameters
        
        self.PDFLearning = PDF
        
        if PDF == 'Exponential':
            self.PDFLearning =  lambda x: torch.exp(x)
        if PDF == 'Linear':
            self.PDFLearning =  lambda x: 1. + x
        if PDF == 'Quadratic':
            self.PDFLearning =  lambda x: (1. + x)**2

### Define Layers
        if self.PDFLearning == 'No':
            self.NumberOfNetworks = int((2+NumberOfParameters)*(1+NumberOfParameters)/2)-1
        else:
            self.NumberOfNetworks = 1

        LinearLayers = [([nn.Linear(self.Architecture[i], self.Architecture[i+1]) \
                                  for i in range(len(self.Architecture)-1)])\
                        for n in range(self.NumberOfNetworks)]
        LinearLayers = [Layer for SubLayerList in LinearLayers for Layer in SubLayerList]
        self.LinearLayers = nn.ModuleList(LinearLayers)

        
    def Forward(self, Data, Parameters):
### Forward Function. Performs Preprocessing, returns rho, where rho is quadratically parametrized.
        # Checking that data has the right input dimension
        InputDimension = self.Architecture[0]
        if Data.size(1) != InputDimension:
            print('Dimensions of the data and the network input mismatch: data: %d, model: %d'
                  %(Data.size(1), InputDimension))
            raise ValueError

        # Checking that preprocess has been initialised
        if not hasattr(self, 'Shift'):
            print('Please initialize preprocess parameters!')
            raise ValueError
        with torch.no_grad(): 
            Data, Parameters = self.Preprocess(Data, Parameters)  
        
        NumberOfLayers, NumberOfEvents = len(self.Architecture)-1, Data.size(0)
        EntryIterator, NetworkIterator = 0, -1
        MatrixLT = torch.zeros([NumberOfEvents, (self.NumberOfParameters+1)**2])
        
        if Data.is_cuda:
            MatrixLT = OurCudaTensor(MatrixLT)
        
        for i in range(self.NumberOfParameters+1):
            EntryIterator += i
            DiagonalEntry = True
            for j in range(self.NumberOfParameters+1-i):
                if NetworkIterator == -1:
                    MatrixLT[:, EntryIterator] = torch.ones(NumberOfEvents)
                else:
                    x = Data
                    for Layer in self.LinearLayers[NumberOfLayers*NetworkIterator:\
                                                  NumberOfLayers*(NetworkIterator+1)-1]:
                        x = self.ActivationFunction(Layer(x))
                    x = self.LinearLayers[NumberOfLayers*(NetworkIterator+1)-1](x).squeeze()
                    MatrixLT[:, EntryIterator] = x
                EntryIterator += 1
                NetworkIterator += 1
                DiagonalEntry = False

        MatrixLT = MatrixLT.reshape([-1, self.NumberOfParameters+1, self.NumberOfParameters+1])
     
        if len(Parameters.size()) == 3:
            # if the Parameters selected are customised for individual data points
            onel = torch.ones(Parameters.size(0), Parameters.size(1) ,1)
            if Data.is_cuda:
                onel = OurCudaTensor(onel)
            ParMatrix = torch.cat((onel, Parameters),dim=2)
        else:
            # if the Parameters selected are the same for all data points
            onel = torch.ones(Parameters.size(0), 1)
            if Data.is_cuda:
                onel = OurCudaTensor(onel)
            ParMatrix = torch.cat((onel, Parameters),dim=1)
        
        if len(ParMatrix.size()) == 3:
            # if the Parameters selected are customised for individual data points
            MatrixLTP = torch.einsum('bij,bkj->bik', MatrixLT, ParMatrix)
        else:            
            # if the Parameters selected are the same for all data points
            MatrixLTP = torch.einsum('bij,kj->bik', MatrixLT, ParMatrix)
        
        rho = torch.einsum('bij,bij->bj', MatrixLTP, MatrixLTP)
        return rho

    def ForwardPDF(self, Data):
### ForwardPDF Function. Performs Preprocessing, returns rho, where rho is quadratically parametrized.
        # Checking that data has the right input dimension
        InputDimension = self.Architecture[0]
        if Data.size(1) != InputDimension:
            print('Dimensions of the data and the network input mismatch: data: %d, model: %d'
                  %(Data.size(1), InputDimension))
            raise ValueError

        # Checking that preprocess has been initialised
        if not hasattr(self, 'Shift'):
            print('Please initialize preprocess parameters!')
            raise ValueError
        with torch.no_grad(): 
            Data = self.PreprocessPDF(Data)  
        
        NumberOfLayers, NumberOfEvents = len(self.Architecture)-1, Data.size(0)        
        x = Data
        for Layer in self.LinearLayers[0:NumberOfLayers-1]:
            x = self.ActivationFunction(Layer(x))
        x = self.LinearLayers[NumberOfLayers-1](x).squeeze()
        
        rho=self.PDFLearning(x)
 
        return rho
    
    def GetL1Bound(self, L1perUnit):
        self.L1perUnit = L1perUnit
    
    def ClipL1Norm(self, n_features=10):
### Clip the weights      
        def ClipL1NormLayer(DesignatedL1Max, Layer):
            L1 = Layer.weight.abs().sum()
            Layer.weight.masked_scatter_(L1 > DesignatedL1Max, 
                                        Layer.weight*(DesignatedL1Max/L1))
            return
        
        for m in self.children():
            for mm in m:
                if (mm.in_features) != n_features:
                    with torch.no_grad():
                        DesignatedL1Max = mm.weight.size(0)*mm.weight.size(1)*self.L1perUnit
                        ClipL1NormLayer(DesignatedL1Max, mm)
        return 
    
    def DistributionRatio(self, points):
### This is rho. I.e., after training, the estimator of the distribution ratio.
        with torch.no_grad():
            F = self(points)
        return F

    def InitPreprocess(self, datafile):
### This can be run only ONCE to initialize the preprocess (shift and scaling) parameters
### Takes as input the training Data and the training Parameters as Torch tensors.
        if not hasattr(self, 'Scaling'):
            print('Initializing Preprocesses Variables')
            self.Scaling = datafile.Data.std(0)
            self.Shift = datafile.Data.mean(0)
            
            # This handles the situation when any of the features is homogenous
            if torch.sum(self.Scaling==0) > 0:
                print('===== some feature(s) is homogenous, check please!')
                raise ValueError 
            
            # The following change is made to accommodate the case when
            # the Parameters selected are customised for individual data points
            Parameters = datafile.ReweightValues
            self.ParameterScaling = Parameters.std(tuple(range(len(Parameters.size())))[:-1])
            # This handles the situation when one of the parameters is redundant
            self.ParameterScaling[self.ParameterScaling == 0.] = 1.
            
        else: print('Preprocess can be initialized only once. Parameters unchanged.')
            
    def Preprocess(self, Data, Parameters):
### Returns scaled/shifted data and parameters
### Takes as input Data and Parameters as Torch tensors.
        if  not hasattr(self, 'Scaling'): print('Preprocess parameters are not initialized.')
        Data = (Data - self.Shift)/self.Scaling
        Parameters = Parameters/self.ParameterScaling
        return Data, Parameters
    
    def PreprocessPDF(self, Data):
### Returns scaled/shifted data
### Takes as input Data as Torch tensors.
        if  not hasattr(self, 'Scaling'): print('Preprocess parameters are not initialized.')
        Data = (Data - self.Shift)/self.Scaling
        return Data
    
    def Save(self, Name, Folder, csvFormat=False):
### Saves the model in Folder/Name
        FileName = Folder + Name + '.pth'
        torch.save({'StateDict': self.state_dict(), 
                   'Scaling': self.Scaling,
                   'Shift': self.Shift,
                   'ParameterScaling': self.ParameterScaling}, 
                   FileName)
        print('Model successfully saved.')
        print('Path: %s'%str(FileName))
        
        if csvFormat:
            modelparams = np.array([w.detach().tolist() for w in self.parameters()], dtype=object)
            np.savetxt(Folder + Name + ' (StateDict).csv', modelparams, '%s')
            statistics = np.array([self.Shift.detach().tolist(), self.Scaling.detach().tolist(),
                         self.ParameterScaling.detach().tolist()], dtype=object)
            np.savetxt(Folder + Name + ' (Statistics).csv', statistics, '%s')
    
    def Load(self, Name, Folder, verbose = True):
### Loads the model from Folder/Name
        FileName = Folder + Name + '.pth'
        try:
            IncompatibleKeys = self.load_state_dict(torch.load(FileName)['StateDict'])
        except KeyError:
            print('No state dictionary saved. Loading model failed.')
            return 
        
        if list(IncompatibleKeys)[0]:
            print('Missing Keys: %s'%str(list(IncompatibleKeys)[0]))
            print('Loading model failed. ')
            return 
        
        if list(IncompatibleKeys)[1]:
            print('Unexpected Keys: %s'%str(list(IncompatibleKeys)[0]))
            print('Loading model failed. ')
            return 
        
        self.Scaling = torch.load(FileName)['Scaling']
        self.Shift = torch.load(FileName)['Shift']
        self.ParameterScaling = torch.load(FileName)['ParameterScaling']
        
        if verbose:
            print('Model successfully loaded.')
            print('Path: %s'%str(FileName))
        
    def Report(self): ### is it possibe to check if the model is in double?
        print('\nModel Report:')
        print('Preprocess Initialized: ' + str(hasattr(self, 'Shift')))
        print('Architecture: ' + str(self.Architecture))
        print('Loss Function: ' + 'Quadratic')
        print('Activation: ' + str(self.ActivationFunction))
        
    def cuda(self):
        nn.Module.cuda(self)
        self.Shift = self.Shift.cuda()
        self.Scaling = self.Scaling.cuda()
        self.ParameterScaling = self.ParameterScaling.cuda()
        
    def cpu(self):
        self.Shift = self.Shift.cpu()
        self.Scaling = self.Scaling.cpu()
        self.ParameterScaling = self.ParameterScaling.cpu()
        return nn.Module.cpu(self)



### ===============================================================================================
### ===============================================================================================
### ===============================================================================================


####### Loss function(s), with "input" in (0,1) interval
class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
            
class WeightedSELoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedSELoss, self).__init__(size_average, reduce, reduction)
    def forward(self, input, weight, reweights, scaling = 1.):
        x = (1./(1.+input))
        y = torch.mul(x**2, reweights)
        z = torch.einsum('ij,i->ij',(1. - x)**2, weight)
        return torch.sum(y + z)/scaling

class WeightedRolrLoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedRolrLoss, self).__init__(size_average, reduce, reduction)
    def forward(self, input, weight, reweights, scaling = 1.):
        r = torch.einsum('ij,i->ij', reweights, (1./weight))
        x = torch.einsum('i,ij->ij',weight,(input - r)**2)
        y = torch.einsum('ij,ij->ij',reweights,(1./input - 1./r)**2)
        return torch.sum(x + y)/scaling

####### Timing
def report_ETA(beginning, start, epochs, e, loss, VLOSS=None):
    time_elapsed = time.time() - start
    time_left    = str(datetime.timedelta(
        seconds=((time.time() - beginning)/(e+1)*(epochs-(e+1)))))
    if VLOSS == None:
        if e == epochs:
            print('Training epoch %s (took %.2f sec) 4*(loss-0.25) %.8E'%(
                e, time_elapsed, loss))
        else:
            print('Training epoch %s (took %.2f sec, time left %s sec) 4*(loss-0.25) %.8E'%(
                e, time_elapsed, time_left, loss))
    else:
        if e == epochs:
            print('Training epoch %s (took %.2f sec) 4*(loss-0.25) %.8E  4*(valid.loss-0.25) %.8E'%(
                e, time_elapsed, loss, VLOSS))
        else:
            print('Training epoch %s (took %.2f sec, time left %s sec) 4*(loss-0.25) %.8E  4*(valid.loss-0.25) %.8E'%(
                e, time_elapsed, time_left, loss, VLOSS))
    return time.time()

####### Timing with likelihood and NP test reach
def report_ETA_detailed(beginning, start, epochs, e, loss, vloss, lkh, npreach):
    time_elapsed = time.time() - start
    time_left    = str(datetime.timedelta(
        seconds=((time.time() - beginning)/(e+1)*(epochs-(e+1)))))
    if vloss == None:
        if e == epochs:
            print('Training epoch %s (took %.2f sec) 4*(loss-0.25) %.8E'%(
                e, time_elapsed, loss))
        else:
            print('Training epoch %s (took %.2f sec, time left %s sec) 4*(loss-0.25) %.8E'%(
                e, time_elapsed, time_left, loss))
    else:
        if e == epochs:
            print('Training epoch %s (took %.2f sec) 4*(loss-0.25) %.8E  4*(valid.loss-0.25) %.8E'%(
            e, time_elapsed, loss, vloss))
        else:
            print('Training epoch %s (took %.2f sec, time left %s sec) 4*(loss-0.25) %.8E  4*(valid.loss-0.25) %.8E'%(
            e, time_elapsed, time_left, loss, vloss))
        print('=== Likelihood %.8f +/- %.8f'%(lkh[0], lkh[1]))
        print('=== NP pvalue  %.8f +/- %.8f'%(npreach[0], npreach[1]))
    return time.time()




### ===============================================================================================
### ===============================================================================================
### ===============================================================================================

import copy
def OurCudaTensor(input):
    output = copy.deepcopy(input)
    output = output.cuda()
    return output


class OurTrainer(nn.Module):
### Contains all parameters for training: Loss Function, Optimiser, NumberOfEpochs, InitialLearningRate, SaveAfterEpoch 
    def __init__(self, LearningRate = 1e-3, LossFunction = 'Quadratic', Optimiser = 'Adam', NumEpochs = 100):
        super(OurTrainer, self).__init__() 
        self.NumberOfEpochs = NumEpochs
        self.SaveAfterEpoch = lambda :[self.NumberOfEpochs,]
        self.PrintAfterEpoch = lambda :[self.NumberOfEpochs,]
        self.InitialLearningRate = LearningRate
        ValidCriteria = {'Quadratic': WeightedSELoss(), 'Rolr': WeightedRolrLoss()}
        try:
            self.Criterion = ValidCriteria[LossFunction]
        except KeyError:
            print('The loss function specified is not valid. Allowed losses are %s.'
                 %str(list(ValidCriteria)))
            print('Will use Quadratic Loss.') 
        ValidOptimizers = {'Adam': torch.optim.Adam}
        try:
            self.Optimiser =  ValidOptimizers[Optimiser]
        except KeyError:
            print('The specified optimiser is not valid. Allowed optimisers are %s.'
                 %str(list(ValidOptimisers)))
            print('Will use Adam.')          
        
    def Train(self, model, TrainingDataFile, bs = 100000, L1perUnit=None, UseGPU=True, Name="", Folder=os.getcwd(), WeightClipping=False, L1Max=1, ValidationDataFile = None, scaleloss = 1., lossplot = False, testparameters=None,
             TrainingReweightsFile=None):
        
        Data = TrainingDataFile.Data
               
        Weights = TrainingDataFile.Weights
        if model.PDFLearning == 'No':
            Parameters = TrainingDataFile.ReweightValues
            Reweights = TrainingDataFile.Reweights
        else:
            #Parameters = torch.eye(TrainingDataFile.PDFWeights.size()[-1])
            Reweights = TrainingDataFile.PDFWeights
            if TrainingDataFile.PDFWeights.size()[-1] != model.Architecture[-1]:
                print('Number of PDF replicas not compatible with NN architecture.')
                raise ValueError

        tempmodel = copy.deepcopy(model)
        tempmodel.cuda()
        tempData = OurCudaTensor(Data)
        if model.PDFLearning == 'No':
            tempParameters = OurCudaTensor(Parameters)
        tempWeights = OurCudaTensor(Weights)
        tempReweights = OurCudaTensor(Reweights)
        
        if lossplot:
            self.figLoss, self.axLoss  = plt.subplots()
            plt.xscale('log')
            plt.yscale('log')

            self.axLoss.plot([],[])
            #print('a')
            #plt.pause(0.01)
            #print('a')
        
        if ValidationDataFile != None:
            
            ESdic = {}
            VData = ValidationDataFile.Data
            VWeights = ValidationDataFile.Weights
            if model.PDFLearning == 'No':
                VParameters = ValidationDataFile.ReweightValues
                VReweights = ValidationDataFile.Reweights
            else:
                #VParameters = torch.eye(TrainingDataFile.PDFWeights.size()[-1])
                VReweights = ValidationDataFile.PDFWeights
                
            tempVData = OurCudaTensor(VData)
            if model.PDFLearning == 'No':
                tempVParameters = OurCudaTensor(VParameters)
            tempVWeights = OurCudaTensor(VWeights)
            tempVReweights = OurCudaTensor(VReweights)
                
            if lossplot:
                self.axLoss.plot([],[])
        
        if lossplot:
            plt.pause(0.01)
        
        
        Optimiser = self.Optimiser(tempmodel.parameters(), self.InitialLearningRate)
        mini_batch_size = bs
        
        self.losshistory = []
        self.cvlosshistory = []
        self.epochshistory = []
        self.lkhhistory = []
        self.NPpvalhistory = []
        
        beginning = start = time.time()
        
        if WeightClipping:
            tempmodel.GetL1Bound(L1Max)
        
        for e in range(self.NumberOfEpochs):
            total_loss  = 0.

            Optimiser.zero_grad()
            for b in range(0, Data.size(0), mini_batch_size):
                torch.cuda.empty_cache()
                
                if model.PDFLearning == 'No':
                    if len(tempParameters.size()) == 3:
                        # that reweighting values are customised for each data point
                        output = tempmodel.Forward(tempData[b:b+mini_batch_size], tempParameters[b:b+mini_batch_size])
                    else:
                        # that reweighting values are the same for all data points
                        output = tempmodel.Forward(tempData[b:b+mini_batch_size], tempParameters)
                    scale = TrainingDataFile.scale
                else:
                    output = tempmodel.ForwardPDF(tempData[b:b+mini_batch_size])
                    scale = TrainingDataFile.PDFscale
                
                loss = self.Criterion(output, \
                                     tempWeights[b:b+mini_batch_size], \
                                     tempReweights[b:b+mini_batch_size,:], scaling = scale/scaleloss)
                total_loss += loss
                loss.backward()
            Optimiser.step()
            
            if lossplot:
                plt.pause(0.001)
            
            if WeightClipping:
                tempmodel.ClipL1Norm()
            
            if (e+1) in self.PrintAfterEpoch():
                
                # plot loss #
                with torch.no_grad():
                    if lossplot:
                        temp = copy.deepcopy(-4.*(total_loss/scaleloss-0.25))
                        currentplot = self.axLoss.lines
                        currentplotLossx = currentplot[0].get_xdata()
                        currentplotLossy = currentplot[0].get_ydata()
                        self.axLoss.lines = []
                        self.axLoss.plot(np.append(currentplotLossx,int(e)),np.append(currentplotLossy,\
                                                                         temp.cpu()),'b-')
                        plt.pause(0.01)
                
                # compute cross validation loss
                if ValidationDataFile != None:
                    validation_loss  = 0.
                    with torch.no_grad(): 
                        for b in range(0, tempVData.size(0), mini_batch_size):
                            torch.cuda.empty_cache()
                            
                            if model.PDFLearning == 'No':
                                validation_output = tempmodel.Forward(tempVData[b:b+mini_batch_size], tempVParameters[b:b+mini_batch_size])
                                scale = ValidationDataFile.scale
                            else:
                                validation_output = tempmodel.ForwardPDF(tempVData[b:b+mini_batch_size], tempVParameters)
                                scale = ValidationDataFile.PDFscale
                                
                            batchVloss            = self.Criterion(validation_output, \
                                                     tempVWeights[b:b+mini_batch_size], \
                                                     tempVReweights[b:b+mini_batch_size,:], \
                                                                   scaling = scale/scaleloss)
                            validation_loss += batchVloss
                        
                        
                        # compute likelihood & np reach
                        if testparameters:
                            pars, luminosity = testparameters.pars, testparameters.luminosity
                            coeff  = testparameters.coeff
                            lkh    = ComputeLkelihood(tempmodel, NNrho, tempVData, tempVWeights, pars, luminosity)
                            weights= tempVWeights.reshape(-1,1)
                            NPpval = NeymanPearsonTest(tempmodel, NNrho, tempVData, weights, coeff, pars, luminosity)
                        
                        if model.PDFLearning != 'No':
                            eigensystem = FisherPDF(tempmodel, tempVData, tempVWeights)
                        
                        if lossplot:
                            currentplotVLossx = currentplot[1].get_xdata()
                            currentplotVLossy = currentplot[1].get_ydata()
                            tempV = copy.deepcopy(-4.*(validation_loss/scaleloss-0.25))
                            self.axLoss.plot(np.append(currentplotVLossx,int(e)),np.append(currentplotVLossy,\
                                                                         tempV.cpu()),'r-')
                            plt.pause(0.01)
                                                         
                    
                    if testparameters:
                        start       = report_ETA_detailed(beginning, start, self.NumberOfEpochs, e+1, 
                                                          4.*(total_loss/scaleloss-0.25), 
                                                          4.*(validation_loss/scaleloss-0.25), 
                                                          lkh, NPpval)
                    else:
                        start       = report_ETA(beginning, start, self.NumberOfEpochs, e+1, 4.*(total_loss/scaleloss-0.25), \
                                                     VLOSS = 4.*(validation_loss/scaleloss-0.25))
                else:
                    start       = report_ETA(beginning, start, self.NumberOfEpochs, e+1, 4.*(total_loss/scaleloss-0.25))
           
        
            if (e+1) in self.SaveAfterEpoch():
                tempmodel.Save(Name + "%d epoch"%(e+1), Folder, csvFormat=True)
                
                self.losshistory.append(4.*(total_loss/scaleloss-0.25))
                
                if ValidationDataFile != None:
                    self.cvlosshistory.append(4.*(validation_loss/scaleloss-0.25))
                    
                if testparameters: 
                    self.lkhhistory.append(lkh)
                    self.NPpvalhistory.append(NPpval)
                    
                self.epochshistory.append(e+1)
                
                np.savetxt(Folder + Name + ' (loss).csv', np.array([l.detach().tolist() for l in self.losshistory]), '%s')
                if ValidationDataFile != None:
                    np.savetxt(Folder + Name + ' (cvloss).csv', np.array([l.detach().tolist() for l in self.cvlosshistory]), '%s')
                np.savetxt(Folder + Name + ' (lkh).csv', np.array([l.detach().tolist() for l in self.lkhhistory]), '%s')
                np.savetxt(Folder + Name + ' (npval).csv', np.array([l.detach().tolist() for l in self.NPpvalhistory]), '%s')
                np.savetxt(Folder + Name + ' (epochs).csv', np.array(self.epochshistory), '%s')
           
        tempmodel.Save(Name + 'Final', Folder, csvFormat=True)
        
        return tempmodel.cpu()
    
    def SetNumberOfEpochs(self, NE):
        self.NumberOfEpochs = NE
        
    def SetInitialLearningRate(self,ILR):
        self.InitialLearningRate = ILR
        
    def SetSaveAfterEpochs(self,SAE):
        SAE.sort()
        self.SaveAfterEpoch = lambda : SAE
        
    def SetPrintAfterEpochs(self,PAE):
        PAE.sort()
        self.PrintAfterEpoch = lambda : PAE

### ===============================================================================================
### ===============================================================================================
### ===============================================================================================

class FisherPDF():
    def __init__(self, trainedmodel, data, weights):
        with torch.no_grad():
            ## nndata is to be interptered as the detivative of the pdf ratio with respect to the PDF nuisance, at nuisance =0
            nndata = torch.log(trainedmodel.ForwardPDF(data))
            #print(nndata[0])
            temp = torch.einsum('ij,i->ij', nndata, weights)
            self.FisherMatrix = torch.einsum('ij,ik->jk',nndata, temp)
            #print(matr)
            self.Eigensystem = torch.linalg.eigh(self.FisherMatrix)
    
    
    def Print(self):
        print(self.Eigensystem)
        

    def Save(self, Filename, epoch , dic = {}):
    ### Saves the model in Filename
        dic[epoch] =  {'eigenvalues': self.Eigensystem.eigenvalues, 'eigenvectors': self.Eigensystem.eigenvectors}
        torch.save(dic, Filename)
 

            
### ===============================================================================================
### ===============================================================================================
### ===============================================================================================



def ComputeFisherPDF(trainedmodel, data, weights):
    with torch.no_grad():
        ## nndata is to be interptered as the detivative of the pdf ratio with respect to the PDF nuisance, at nuisance =0
        nndata = torch.log(trainedmodel.ForwardPDF(data))
        #print(nndata[0])
        temp = torch.einsum('ij,i->ij', nndata, weights)
        matr = torch.einsum('ij,ik->jk',nndata, temp)
        #print(matr)
        return torch.linalg.eigh(matr)


    def SaveFisherPDF(self, Name, Folder, csvFormat=False):
### Saves the model in Folder/Name
        FileName = Folder + Name + '.pth'
        torch.save({'StateDict': self.state_dict(), 
                   'Scaling': self.Scaling,
                   'Shift': self.Shift,
                   'ParameterScaling': self.ParameterScaling}, 
                   FileName)
        print('Model successfully saved.')
        print('Path: %s'%str(FileName))
        
        if csvFormat:
            modelparams = [w.detach().tolist() for w in self.parameters()]
            np.savetxt(Folder + Name + ' (StateDict).csv', modelparams, '%s')
            statistics = [self.Shift.detach().tolist(), self.Scaling.detach().tolist(),
                         self.ParameterScaling.detach().tolist()]
            np.savetxt(Folder + Name + ' (Statistics).csv', statistics, '%s')




### ===============================================================================================
### ===============================================================================================
### ===============================================================================================




def ComputeReweights(coeffs, wc, weights):
    coeffs_reweight = torch.cat([weights.reshape(-1, 1), coeffs], dim=1)
    wc_reweight = torch.cat([torch.ones((wc.size(0), wc.size(1), 1), device=wc.device
                                       ), wc, wc**2, wc.prod(dim=2, keepdim=True)], dim=2)
    return torch.einsum('bi,bji->bj', coeffs_reweight, wc_reweight)

def ComputeSmartWCgw(Data, Model, col_idx):
    Data = Data[:, col_idx]
    
    n2 = ComputeNN(2, Data, Model)
    n4 = ComputeNN(4, Data, Model)
    n5 = ComputeNN(5, Data, Model)

    Factor = n2**2 + n4**2 + n5**2
    Grid = (torch.Tensor([[0., -2.],[0., -1.], [0., 1.], [0., 2.]]))
    SmartWCgw = (1./torch.sqrt(Factor)).reshape(-1, 1, 1)*Grid

    SmartWCgw = SmartWCgw.detach()
    Factor = Factor.detach()

    return SmartWCgw, Factor

def ComputeNN(nnIndex, Data, Model, WCIndex=1):
    """
    nnIdex should be 1 for n1, 2 for n2, ... 5 for n5.
    Data should not be scaled, but taken on the proper column indices.
    Both Data and Model need to be on cuda/ cpu simultaneously.
    WCIndex should be 0 for Gphi and 1 for GW.
    """
    # scale data
    Data = (Data - Model.Shift)/Model.Scaling
    
    # get the correct list of layers
    nLayers = int(len(Model.LinearLayers)/5)
    nnList = Model.LinearLayers[(nnIndex-1)*(nLayers):(nnIndex)*(nLayers)]
    
    # forward
    x = Data
    for Layer in nnList[:-1]:
        x = Model.ActivationFunction(Layer(x))
    # the last layer is not relued
    x = nnList[-1](x).squeeze()
    
    # adjust for parameter scaling
    x = x/(Model.ParameterScaling[WCIndex])
    
    return x

def CreateDataSet(CoeffFileName, wc, factor, DataFileName, NewFileName, DataSetList):
    # read coefficients and weights
    with h5py.File(CoeffFileName, 'r') as CoeffFile:
        coeffs = torch.Tensor(CoeffFile['ReweightCoeffs'][()], device=wc.device)
        weights = torch.Tensor(CoeffFile['Weights'][()], device=wc.device).reshape(-1, 1)
    
    # broadcasting wc if it is the same for all points
    if len(wc.size()) != 3:
        wc = torch.ones(coeffs.size(0), wc.size(-2), wc.size(-1), device=wc.device)*wc
    
    coeffs  = coeffs[:wc.size(0)]
    weights = weights[:wc.size(0)]
    
    # compute reweights
    Reweights = ComputeReweights(coeffs, wc, weights)
    ReweightValues = wc
    
    # rescaling
    Reweights = Reweights*factor.reshape(-1, 1)
    Weights = (weights.reshape(-1))*(factor.reshape(-1))

    # create new datafile wight computed reweights
    with h5py.File(NewFileName, 'w') as newfile:
        oldfile = h5py.File(DataFileName, 'r')
        newfile.create_dataset('Weights', data=np.array(Weights.cpu()))
        newfile.create_dataset('Reweights', data=np.array(Reweights.cpu()))
        newfile.create_dataset('ReweightValues', data=np.array(ReweightValues.cpu()))
        for key in DataSetList:
            newfile.create_dataset(key, data=oldfile[key][()])
        oldfile.close()
        
        print("New data file created: %s"%(NewFileName))

    return


### ===============================================================================================
### ===============================================================================================
### ===============================================================================================


def cot(x):
    return 1./torch.tan(x)

def MEleading(gphi, gw, s, theta, thetaZ, thetaW, ch, sinZ, sinW, cosZ, cosW):
    q = (1+ch)/2
    fp = q*(1.1326166790371462e-14*gw**2*s**2*(1 + torch.cos(thetaW))**2*torch.sin(theta)**2*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4) + 0.00010011037277227128*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(1 + torch.cos(thetaW))**2*cot(theta/2.)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 4.530466716148585e-14*gw**2*s**2*torch.sin(theta)**2*torch.sin(thetaW/2.)**4*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 2.1296636161281494e-9*gw*s*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*cot(theta/2.)*(cosW**2 - sinW**2)*torch.sin(theta)*torch.sin(thetaW)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) - 0.0141499380049717*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**3*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*cot(theta/2.)*sinW*sinZ*(-0.1486399876234132*torch.sin(theta) - (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)*torch.sin(thetaZ) - 1.5050692203597452e-7*gw*s*torch.cos(thetaW/2.)**3*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*(-0.1486399876234132*torch.sin(theta) - (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)*torch.sin(thetaZ) + 7.525346101798726e-8*gw*s*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*(-0.1486399876234132*torch.sin(theta) - (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ) + 5.866844466093753e-10*gw*s*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**4*cot(theta/2.)*(cosZ**2 - sinZ**2)*torch.sin(theta)*torch.sin(thetaZ)**2 - 0.000013789313534063791*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 1.5600787479516013e-15*gw**2*s**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(theta)**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.002152204791604019*(-0.1486399876234132*torch.sin(theta) - (gphi*s*torch.sin(theta))/(500000.*2**0.5))**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 - 2.1296636161281494e-9*gw*s*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(cosW**2 - sinW**2)*torch.sin(theta)*torch.sin(thetaW)**2*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.) - 0.00707496900248585*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*(-0.1486399876234132*torch.sin(theta) - (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ)*torch.tan(theta/2.) - 5.866844466093753e-10*gw*s*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(cosZ**2 - sinZ**2)*torch.sin(theta)*torch.sin(thetaW/2.)**4*torch.sin(thetaZ)**2*torch.tan(theta/2.) + 0.0004004414910890851*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*torch.sin(thetaW/2.)**4*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.)**2)/(0. + 0.00010011037277227128*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(1 + torch.cos(thetaW))**2*cot(theta/2.)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 0.0021032466099310575*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**3*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*cot(theta/2.)*sinW*sinZ*torch.sin(theta)*torch.sin(thetaW/2.)*torch.sin(thetaZ) - 0.000013789313534063791*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.00004755048105546655*torch.sin(theta)**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.0010516233049655288*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ)*torch.tan(theta/2.) + 0.0004004414910890851*(-0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*torch.sin(thetaW/2.)**4*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.)**2)
    fm = (1-q)*(1.1326166790371462e-14*gw**2*s**2*(1 + torch.cos(thetaW))**2*torch.sin(theta)**2*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4) + 0.00010011037277227128*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(1 + torch.cos(thetaW))**2*cot(theta/2.)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 4.530466716148585e-14*gw**2*s**2*torch.sin(theta)**2*torch.sin(thetaW/2.)**4*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 2.1296636161281494e-9*gw*s*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*cot(theta/2.)*(cosW**2 - sinW**2)*torch.sin(theta)*torch.sin(thetaW)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 0.0141499380049717*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**3*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*cot(theta/2.)*sinW*sinZ*(0.1486399876234132*torch.sin(theta) + (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)*torch.sin(thetaZ) + 1.5050692203597452e-7*gw*s*torch.cos(thetaW/2.)**3*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*(0.1486399876234132*torch.sin(theta) + (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)*torch.sin(thetaZ) - 7.525346101798726e-8*gw*s*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*(0.1486399876234132*torch.sin(theta) + (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ) + 5.866844466093753e-10*gw*s*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**4*cot(theta/2.)*(cosZ**2 - sinZ**2)*torch.sin(theta)*torch.sin(thetaZ)**2 - 0.000013789313534063791*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 1.5600787479516013e-15*gw**2*s**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(theta)**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.002152204791604019*(0.1486399876234132*torch.sin(theta) + (gphi*s*torch.sin(theta))/(500000.*2**0.5))**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 - 2.1296636161281494e-9*gw*s*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(cosW**2 - sinW**2)*torch.sin(theta)*torch.sin(thetaW)**2*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.) + 0.00707496900248585*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*(0.1486399876234132*torch.sin(theta) + (gphi*s*torch.sin(theta))/(500000.*2**0.5))*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ)*torch.tan(theta/2.) - 5.866844466093753e-10*gw*s*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(cosZ**2 - sinZ**2)*torch.sin(theta)*torch.sin(thetaW/2.)**4*torch.sin(thetaZ)**2*torch.tan(theta/2.) + 0.0004004414910890851*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*torch.sin(thetaW/2.)**4*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.)**2)/(0. + 0.00010011037277227128*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(1 + torch.cos(thetaW))**2*cot(theta/2.)**2*(0.029963478569123254*(1 + torch.cos(thetaZ))**2 + 0.1556282990488214*torch.sin(thetaZ/2.)**4) + 0.0021032466099310575*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*torch.cos(thetaW/2.)**3*(-0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*cot(theta/2.)*sinW*sinZ*torch.sin(theta)*torch.sin(thetaW/2.)*torch.sin(thetaZ) - 0.000013789313534063791*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*(cosW**2 - sinW**2)*(cosZ**2 - sinZ**2)*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.00004755048105546655*torch.sin(theta)**2*torch.sin(thetaW)**2*torch.sin(thetaZ)**2 + 0.0010516233049655288*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))*(0.008943596193082094 + 0.0688705533313286*torch.cos(thetaZ))*sinW*sinZ*torch.sin(theta)*torch.sin(thetaW/2.)**2*torch.sin(thetaW)*torch.sin(thetaZ)*torch.tan(theta/2.) + 0.0004004414910890851*(0.23369836056172633 + 2.298904918314821*torch.cos(theta))**2*torch.sin(thetaW/2.)**4*(0.03890707476220535*(1 + torch.cos(thetaZ))**2 + 0.11985391427649301*torch.sin(thetaZ/2.)**4)*torch.tan(theta/2.)**2)
    
    return fp + fm

def MErho(Model, Data, pars):
    (s, th, thz, thw, pt, ch, sz, sw, cz, cw) = Data.transpose(1,0)
    (gphi, gw) = pars

    return MEleading(gphi, gw, s, th, thz, thw, ch, sz, sw, cz, cw)

def NNrho(Model, Data, pars):
    with torch.no_grad():
        return Model.Forward(Data, pars.reshape(1,2)).squeeze()
    
def ComputeLkelihood(Model, rhoFcn, Data, Weights, Parameters, Luminosity):
        
    # compute likelihood point by point       
        
    rho = rhoFcn(Model, Data, Parameters)
        
    l = torch.log(rho) - rho + 1
        
    # split in 10 subsamples
        
    n_divs = 10
    RandIndex = torch.randperm(Data.size(0))
    shuffled_l = l[RandIndex].reshape(n_divs, -1)
    shuffled_Weights = Weights[RandIndex].reshape(n_divs, -1)
        
    # compute full likelihood in each sample
        
    likelihoods = -2 * Luminosity * torch.einsum('ij,ij->i', shuffled_l, shuffled_Weights)
        
    # return total likelihood and standard deviation
        
    return torch.cuda.FloatTensor([likelihoods.sum(), torch.sqrt(n_divs * torch.var(likelihoods))])

from scipy.stats import skewnorm
import math


class TestParameters():
    def __init__(self, pars, luminosity, coeff):
        self.pars = pars
        self.luminosity = luminosity
        self.coeff = coeff

def CubeRoot(x):
    if x >= 0:
        return x**(1/3)
    elif x < 0:
        return -(abs(x)**(1/3))

def SkewNormal(moments, n):
    
    # convert 1st, 2nd and 3rd moments into
    # SkewNormal distribution parameters
    
    mom1 = moments[0].item()
    mom2 = moments[1].item()
    mom3 = moments[2].item()
    
    s = mom3/mom2**(3./2)
    
    mu = mom1 * n
    sigma = (mom2 * n)**0.5
    skew = s/(n**0.5)
      
    if abs(skew) > (4-math.pi)*(2**0.5)/(math.pi-2)**(3./2):
        raise Exception
    
    a = (CubeRoot(2*skew) * math.sqrt(math.pi))/ math.sqrt(2*CubeRoot(4-math.pi)**2 + (2-math.pi)*CubeRoot(2*skew)**2)
    scale = sigma * (1-(2*a**2)/(math.pi*(1+a**2)))**(-0.5)
    loc = mu - (a*scale*(2/math.pi)**0.5)/(1+a**2)**0.5
        
    return a, loc, scale

def ComputeMomentsWithCovariance(logrho, Weights):
    
    # split sample in 100
    
    n_divs = 100
         
    RandIndex = torch.randperm(logrho.size(0))
    shuffled_logrho = logrho[RandIndex].reshape(n_divs, -1)
    shuffled_Weights = Weights[RandIndex].reshape(n_divs, -1)   
    shuffled_Weights = shuffled_Weights/shuffled_Weights.sum(1).reshape(-1,1)
    
    # compute 1st, 2nd and 3rd moments for each sample
    
    moments = torch.stack([
        torch.einsum('ij,ij->i', shuffled_logrho, shuffled_Weights),
        torch.einsum('ij,ij->i', shuffled_logrho**2, shuffled_Weights),
        torch.einsum('ij,ij->i', shuffled_logrho**3, shuffled_Weights)
    ])
    
    # return mean and covariance matrix
    
    return moments.mean(1), moments.cov()/n_divs

def pValueFromMoments(SMmoments, Nsm, BSMmoments, Nbsm):
    
    # compute skew normal parameters
    
    aSM, locSM, scaleSM  = SkewNormal(SMmoments, Nsm.item())
    aBSM, locBSM, scaleBSM = SkewNormal(BSMmoments, Nbsm.item())
    
    # compute p-value

    return skewnorm.cdf(skewnorm.median(aSM, locSM, scaleSM), aBSM, locBSM, scaleBSM)

def NeymanPearsonTest(Model, rhoFCN, Data, Weights, Coeffs, Parameters, Luminosity, compute_error = True):
        
    # compute gphi, gw, gphi^2, gw^2 and gphi*gw
        
    pars = torch.cuda.FloatTensor([Parameters[0], Parameters[1], Parameters[0]**2, Parameters[1]**2, Parameters[0]*Parameters[1]])
        
    # compute logrho
        
    logrho = torch.log(rhoFCN(Model, Data, Parameters))
        
    # compute moments for SM and BSM
        
    SMmoments, SMmomentsCov = ComputeMomentsWithCovariance(logrho, Weights)
    BSMweights = torch.einsum('ij,j->i', Coeffs, pars).reshape(-1,1)
    BSMmoments, BSMmomentsCov = ComputeMomentsWithCovariance(logrho, Weights + BSMweights)
        
    # compute Nsm and Nbsm with their errors
        
    Nsm = Luminosity * Weights.sum(0)
    DNbsm = Luminosity * BSMweights.sum(0)
    Ncov = torch.cuda.FloatTensor([[Nsm, 0.],[0., abs(DNbsm)]]) * Weights.size(0)**(-0.5)
        
    # compute central value for the p-value
              
    try:
        pValue = pValueFromMoments(SMmoments, Nsm, BSMmoments, Nsm + DNbsm)
    except Exception:
        return torch.cuda.FloatTensor([0.5, -1.0])
        
    if not compute_error:
        return torch.cuda.FloatTensor([pValue, -1.0])
        
    # randomly generate many toys using the central values and covariances
    # was 1000 on Mathematica, but it's slow
    n_tests = 50

    Nloc = torch.cuda.FloatTensor([Nsm, DNbsm])
    NDist = torch.distributions.multivariate_normal.MultivariateNormal(Nloc, Ncov).sample((n_tests,))
    SMmomentsDist = torch.distributions.multivariate_normal.MultivariateNormal(SMmoments, SMmomentsCov).sample((n_tests,))
    BSMmomentsDist = torch.distributions.multivariate_normal.MultivariateNormal(BSMmoments, BSMmomentsCov).sample((n_tests,))

    # compute p-value in each toy
    # this is a bit slow
    pList = torch.empty(n_tests).cuda()
    for i in range(n_tests):
        try:
            pList[i] = pValueFromMoments(SMmomentsDist[i], NDist[i,0], BSMmomentsDist[i], NDist[i,0] + NDist[i,1])
        except Exception:
            pList[i] = -1.0

    # return pvalue and standard deviation of toys
       
    return torch.cuda.FloatTensor([pValue, pList[pList > 0].std()])