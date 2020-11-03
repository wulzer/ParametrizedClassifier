import h5py, torch, time, datetime, os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.modules import Module
from torch.nn import CrossEntropyLoss
from tabulate import tabulate

random_seed = torch.randint(0, 1339, (1,)).item()
torch.manual_seed(random_seed)
print('=========== Random Seed: %d ==========='%(random_seed))

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
### Weights are normalized to have sum = 1 on the entire training sample
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
            BSMNLimits = [min(BSMNLimits, NF.ND) for NF in self.BSMDataFiles]
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
        self.BSMXSList = [DF.XS for DF in self.BSMDataFiles]
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
            SMNLimits = [min(SMNLimits, DF.ND) for DF in self.SMDataFiles]
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
        self.SMXSList = [DF.XS for DF in self.SMDataFiles]
        idx_random = torch.randperm(self.SMND)
        self.SMData = self.SMData[idx_random, :]
        self.SMWeights = self.SMWeights[idx_random]

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
        s = self.Weights.sum()
        self.Weights = self.Weights.div(s)

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
        
####### Convert Angles
    def CurateAngles(self, AnglePos):
        Angles = self.Data[:, AnglePos]
        CuratedAngles = torch.cat([torch.sin(Angles), torch.cos(Angles)], dim=1)
        OtherPos = list(set(range(self.Data.size(1)))-set(AnglePos))
        self.Data = torch.cat([self.Data[:, OtherPos], CuratedAngles], dim=1)
        print('####\nAnlges at position %s have been converted to Sin and Cos and put at the last columns of the Data.'%(AnglePos))
        print('####')
        
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
    def forward(self, input, target, weight):
        return torch.sum(torch.mul(weight, (input - target)**2))

class WeightedCELoss(_Loss):
    __constants__ = ['reduction']
        
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedCELoss, self).__init__(size_average, reduce, reduction)
    def forward(self, input, target, weight):
        return torch.sum(torch.mul(weight, (1 - target)*torch.log(1./(1.-input))+target*torch.log(1./input)))
    
####### Loss function(s), with "input" in (0,1) interval
def report_ETA(beginning, start, epochs, e, loss):
    time_elapsed = time.time() - start
    time_left    = str(datetime.timedelta(
        seconds=((time.time() - beginning)/(e+1)*(epochs-(e+1)))))
    print('Training epoch %s (took %.2f sec, time left %s sec) loss %.8f'%(
        e, time_elapsed, time_left, loss))
    return time.time()

class OurModel(nn.Module):
### Defines the  model with parametrized discriminant. Only quadratic dependence on a single parameter is implemented.
### Input is the architecture (list of integers, the last one being equal to 1) and the activation type ('ReLU' or 'Sigmoid')
    def __init__(self, AR = [1, 3, 3, 1] , AF = 'ReLU' ):               
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
            if( ( all(isinstance(n, int) for n in AR)) and ( AR[-1] == 1) ):
                self.Architecture = AR
            else:
                print('Architecture should be a list of integers, the last one should be 1.')
                raise ValueError             
        else:
            print('Architecture should be a list !')
            raise ValueError

### Define Layers
        #self.LinearLayerList1  = nn.ModuleList([nn.Linear(self.Architecture[i], 
        #    self.Architecture[i+1], bias=False) for i in range(len(self.Architecture)-2)])
        self.LinearLayerList1  = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer1 = nn.Linear(self.Architecture[-2], 1)       
        #self.LinearLayerList2 = nn.ModuleList([nn.Linear(self.Architecture[i], 
        #    self.Architecture[i+1], bias=False) for i in range(len(self.Architecture)-2)])
        self.LinearLayerList2 = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer2 = nn.Linear(self.Architecture[-2], 1)
        
        #self.Optimiser = torch.optim.Adam(self.parameters(), self.InitialLearningRate)
        #self.Criterion = WeightedMSELoss()

    def Forward(self, Data, Parameters):
### Forward Function. Performs Preprocessing, returns F = rho/(1+rho) in [0,1], where rho is quadratically parametrized.
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
        
        x1 = x2 = Data
        
        for i, Layer in enumerate(self.LinearLayerList1):
            x1 = self.ActivationFunction(Layer(x1)) 
        x1 = self.OutputLayer1(x1).squeeze()
        
        for i, Layer in enumerate(self.LinearLayerList2):
            x2 = self.ActivationFunction(Layer(x2))
        #x2 = torch.exp(self.OutputLayer2(x2)).squeeze()
        #x2 = torch.abs(self.OutputLayer2(x2)).squeeze()
        x2 = self.OutputLayer2(x2).squeeze()
        #x2 = self.OutputLayer2(x2).squeeze()
        
        
        rho = (1 + torch.mul(x1, Parameters))**2 + (torch.mul(x2, Parameters))**2  
        return (rho.div(1.+rho)).view(-1, 1)
    
    def GetL1Bound(self, L1perUnit):
        self.L1perUnit = L1perUnit
    
    def ClipL1Norm(self):
### Clip the weights      
        def ClipL1NormLayer(DesignatedL1Max, Layer, Counter):
            if Counter == 1:
                ### this avoids clipping the first layer
                return
            L1 = Layer.weight.abs().sum()
            Layer.weight.masked_scatter_(L1 > DesignatedL1Max, 
                                        Layer.weight*(DesignatedL1Max/L1))
            return
        
        Counter = 0
        for m in self.children():
            if isinstance(m, nn.Linear):
                Counter += 1
                with torch.no_grad():
                    DesignatedL1Max = m.weight.size(0)*m.weight.size(1)*self.L1perUnit
                    ClipL1NormLayer(DesignatedL1Max, m, Counter)
            else:
                for mm in m.children():
                    Counter +=1
                    with torch.no_grad():
                        DesignatedL1Max = mm.weight.size(0)*mm.weight.size(1)*self.L1perUnit
                        ClipL1NormLayer(DesignatedL1Max, mm, Counter)
        return 
    
    def DistributionRatio(self, points):
### This is rho. I.e., after training, the estimator of the distribution ratio.
        with torch.no_grad():
            F = self(points)
        return F/(1-F)

    def InitPreprocess(self, Data, Parameters):
### This can be run only ONCE to initialize the preprocess (shift and scaling) parameters
### Takes as input the training Data and the training Parameters as Torch tensors.
        if not hasattr(self, 'Scaling'):
            print('Initializing Preprocesses Variables')
            self.Scaling = Data.std(0)
            self.Shift = Data.mean(0)
            self.ParameterScaling = Parameters.std(0)  
        else: print('Preprocess can be initialized only once. Parameters unchanged.')
            
    def Preprocess(self, Data, Parameters):
### Returns scaled/shifted data and parameters
### Takes as input Data and Parameters as Torch tensors.
        if  not hasattr(self, 'Scaling'): print('Preprocess parameters are not initialized.')
        Data = (Data - self.Shift)/self.Scaling
        Parameters = Parameters/self.ParameterScaling
        return Data, Parameters
    
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
            modelparams = [w.detach().tolist() for w in self.parameters()]
            np.savetxt(Folder + Name + ' (StateDict).csv', modelparams, '%s')
            statistics = [self.Shift.detach().tolist(), self.Scaling.detach().tolist(),
                         self.ParameterScaling.detach().tolist()]
            np.savetxt(Folder + Name + ' (Statistics).csv', statistics, '%s')
    
    def Load(self, Name, Folder):
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
        self.InitialLearningRate = LearningRate
        ValidCriteria = {'Quadratic': WeightedSELoss(), 'CE':WeightedCELoss(), 'BCE':CrossEntropyLoss()}
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
    
    def EstimateRequiredGPUMemory(self, model, Data, Parameters):
        if next(model.parameters()).is_cuda:
            print('Model is on cuda. No estimate possible anymore.')
            return None
        else:
            before = torch.cuda.memory_allocated()
            print(before)
            ### Always make deep copy of objects before sending them to cuda. Delete when done
            ModelCuda = copy.deepcopy(model)
            ModelCuda.cuda()
            DataCuda = OurCudaTensor(Data[:10000])
            ParametersCuda = OurCudaTensor(Parameters[:10000])
            print(torch.cuda.memory_allocated())
            MF = ModelCuda.Forward(DataCuda, ParametersCuda)
            after = torch.cuda.memory_allocated()
            print(after)
            del ModelCuda, DataCuda, ParametersCuda, MF
            torch.cuda.empty_cache()        
            estimate = float(Data.size()[0])/1e4*float(after-before)*1e-9
            print(str(estimate) + ' GB')
            return estimate
        
    def Train(self, model, Data, Parameters, Labels, Weights, bs = 100000, L1perUnit=None, UseGPU=True, Name="", Folder=os.getcwd(), WeightClipping=False, L1Max=1):
        
        tempmodel = copy.deepcopy(model)
        tempmodel.cuda()
        tempData = OurCudaTensor(Data)
        tempParameters = OurCudaTensor(Parameters)
        tempLabels = OurCudaTensor(Labels)
        tempWeights = OurCudaTensor(Weights)
        
        Optimiser = self.Optimiser(tempmodel.parameters(), self.InitialLearningRate)
        mini_batch_size = bs
        beginning = start = time.time()
        
        if WeightClipping:
            tempmodel.GetL1Bound(L1Max)
        
        for e in range(self.NumberOfEpochs):
            total_loss  = 0
            #print("epoch")
            Optimiser.zero_grad()
            for b in range(0, Data.size(0), mini_batch_size):
                torch.cuda.empty_cache()
                output          = tempmodel.Forward(tempData[b:b+mini_batch_size], tempParameters[b:b+mini_batch_size])
                loss            = self.Criterion(output, tempLabels[b:b+mini_batch_size].reshape(-1,1), 
                                                 tempWeights[b:b+mini_batch_size].reshape(-1, 1))
                total_loss += loss
                loss.backward()
            Optimiser.step()
            
            if WeightClipping:
                tempmodel.ClipL1Norm()
            
            if (e+1) in self.SaveAfterEpoch():
                start       = report_ETA(beginning, start, self.NumberOfEpochs, e+1, total_loss)
                tempmodel.Save(Name + "%d epoch"%(e+1), Folder, csvFormat=True)
        
        tempmodel.Save(Name + 'Final', Folder, csvFormat=True)
        
        return tempmodel.cpu()
    
    def SetNumberOfEpochs(self, NE):
        self.NumberOfEpochs = NE
        
    def SetInitialLearningRate(self,ILR):
        self.InitialLearningRate = ILR
        
    def SetSaveAfterEpochs(self,SAE):
        SAE.sort()
        self.SaveAfterEpoch = lambda : SAE