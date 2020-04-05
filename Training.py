import h5py, torch, time, datetime, os
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.modules import Module
from tabulate import tabulate

class OurModel(nn.Module):
   
    def __init__(self, AR = [1, 3, 3, 1] , AF = 'ReLU' ):               
        super(OurModel, self).__init__() 
        
        self.Architecture = [1, 3, 3, 1]
        self.ActivationFunction = 'ReLU'
        self.NumberOfEpochs = 100
        self.InitialLearningRate = 1e3
        self.SaveAfterEpoch = lambda :[self.NumberOfEpochs]  
        
        if AF in ['ReLU', 'Sigmoid']:
            self.ActivationFunction = AF
        else:
            print('Valid Activations are: [\'ReLU\', \'Sigmoid\'].')
            # avoid hard coding in multiple lines
            #self.ActivationFunction = 'ReLU'
            # # # # # # # # # # # # # # # # # # 
            
        if type(AR) == list:
            if all(isinstance(n, int) for n in AR):
                self.Architecture = AR
            else:
                print('Architecture should be a list of integers !')
                # avoid hard coding in multiple lines
                #self.Architecture = [1, 3, 3, 1]
                # # # # # # # # # # # # # # # # # #      
            
        else:
            print('Architecture should be a list !')
            # avoid hard coding in multiple lines
            #self.Architecture = [1, 3, 3, 1]
            # # # # # # # # # # # # # # # # # #           
            
        self.LinearLayerList1  = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer1 = nn.Linear(self.Architecture[-2], 1)       
        self.LinearLayerList2 = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer2 = nn.Linear(self.Architecture[-2], 1)
        
        self.Optimiser = torch.optim.Adam(self.parameters(), self.InitialLearningRate)
        self.Criterion = WeightedMSELoss()
        
    def Preprocess(self, Data, DataParameters):
        if not hasattr(self, 'Scaling'):
            print('Setting up preprocess statistics.')
            self.Scaling = Data.std(0)
            self.Shift = Data.mean(0)
            self.ParameterScaling = DataParameters.std(0)
        
        Data = (Data - self.Shift)/self.Scaling
        DataParameters = DataParameters/self.ParameterScaling
        
        return Data, DataParameters

    def Forward(self, Data, DataParameters):
        ValidActivationFunctions = {'ReLU': torch.relu, 'Sigmoid': torch.sigmoid}
        try:
            ActivationFunction = ValidActivationFunctions[self.ActivationFunction]
        except KeyError:
            print('The activation function specified is not valid. Allowed activations include %s.'
                 %str(list(ValidActivationFunctions.keys())))
            print('Will use ReLU.')
            ActivationFunction = torch.relu
        
        # Checking that data has the right input dimension
        InputDimension = self.Architecture[0]
        if Data.size(1) != InputDimension:
            print('Dimensions of the data and the network mismatch: data: %d, model: %d'
                  %(Data.size(1), InputDimension))

        # Checking that preprocess has been done
        if not hasattr(self, 'Shift'):
            print('Please preprocess!')
        
        x1 = x2 = Data
        
        for i, Layer in enumerate(self.LinearLayerList1):
            x1 = ActivationFunction(Layer(x1))
        x1 = self.OutputLayer1(x1).squeeze()
        
        for i, Layer in enumerate(self.LinearLayerList2):
            x2 = ActivationFunction(Layer(x2))
        x2 = self.OutputLayer2(x2).squeeze()
        
        F = torch.log(
            ((1 + torch.mul(x1, DataParameters))**2 + (torch.mul(x2, DataParameters))**2))        
        return torch.sigmoid(F).view(-1, 1)
    
    def GetL1Max(self, L1perUnit):
        self.L1perUnit = L1perUnit
        L1MaxList = []
        for m in model_plus.children():
            if isinstance(m, nn.Linear):
                L1MaxList.append(m.weight.size(0)*m.weight.size(1) \
                                  *self.L1perUnit)
            else:
                for mm in m:
                    L1MaxList.append(mm.weight.size(0)*mm.weight.size(1)\
                                      *self.L1perUnit)
        self.L1MaxList = L1MaxList
        print('L1MaxList created.')
    
    def ClipL1Norm(self):
        
        def ClipL1NormLayer(DesignatedL1Max, Layer):
            if DesignatedL1 == self.Architecture[0]*self.Architecture[1]*self.L1perUnit:
                # means this is an input layer, we skip
                return
            else:
                L1 = Layer.weight.abs().sum()
                Layer.weight.masked_scatter_(L1 > DesignatedL1Max, 
                                             Layer.weight*(DesignatedL1Max/L1))
                return
        
        Counter = 0
        for m in self.children():
            if isinstance(m, nn.Linear):
                Counter += 1
                with torch.no_grad():
                    DesignatedL1Max = self.L1MaxList[counter-1]
                    ClipL1NormLayer(DesignatedL1Max, m)
            else:
                for mm in m:
                    Counter +=1
                    with torch.no_grad():
                        DesignatedL1Max = self.L1MaxList[counter-1]
                        ClipL1NormLayer(DesignatedL1Max, mm)
        return 
    
    def calculate_ratio(self, points):
        with torch.no_grad():
            y = self(points)
        return y/(1-y)    
    
    def SetOptimiser(self, OP):
        self.Optimiser = OP
        
    def SetCriterion(self, CR):
        self.Criterion = CR
        
    def SetNumberOfEpochs(self,NE):
        self.NumberOfEpochs = NE
        #self.SaveAfterEpoch = [self.NumberOfEpochs
        
    def SetInitialLearningRate(self,ILR):
        self.InitialLearningRate = ILR
        self.Optimiser = torch.optim.Adam(self.parameters(), self.InitialLearningRate)
        
    def SetSaveAfterEpoch(self,SAE):
        SAE.sort()
        self.SaveAfterEpoch = lambda : SAE
        
    def Report(self):
        print(
        'Architecture = ' + str(self.Architecture) + 
        '\nActivation function = ' + self.ActivationFunction +
        '\nInitial learning rate = ' + str(self.InitialLearningRate) +
        '\nNumber of epochs = ' + str(self.NumberOfEpochs) +
        '\nSaving network after epoch(s): ' + str(self.SaveAfterEpoch())
        )
        
    def Train(self, Data, DataParameters, Labels, L1perUnit=None, UseGPU=True, Name="", Folder=os.getcwd()):
        if L1perUnit:
            self.GetL1Max(L1perUnit)

        if UseGPU:
            self.cuda()
            Data, Parameters      = Data.cuda(), DataParameters.cuda()
            Labels = Labels.cuda()
        print(" =================== Memory: %s ==================== "%str(torch.cuda.memory_allocated()))
        print(" =================== BEGINNING TRAIN ==================== ")
        beginning = start = time.time()

        for e in range(self.NumberOfEpochs):
            print(" Epoch %d: Computing Output ======== Memory: %s ==================== "%(
                e+1, str(torch.cuda.memory_allocated())))
            output          = self.Forward(Data, DataParameters)
            print(" Epoch %d: Computing Loss ========== Memory: %s ==================== "%(
                e+1, str(torch.cuda.memory_allocated())))
            loss            = self.Criterion(output, Labels, Data)
        
            if (e+1) in self.SaveAfterEpoch():
                start       = report_ETA(beginning, start, self.NumberOfEpochs, e+1, loss)
                self.Save(Name, Folder)
                print(" Epoch %d: Saving Model ============ Memory: %s ==================== "%(
                    e+1, str(torch.cuda.memory_allocated())))
                
            optimiser.zero_grad()
            print(" Epoch %d: Backward ================= Memory: %s ==================== "%(
                e+1, str(torch.cuda.memory_allocated())))
            loss.backward()
            optimiser.step()
            if L1perUnit:
                self.ClipL1Norm()

        print(" ===================   END OF TRAIN   =================== ")

        return model
    
    def Save(self, Name, Folder):
        FileName = Folder + Name + '.pth'
        torch.save({'StateDict': self.state_dict(), 
                   'Scaling': self.Scaling,
                   'Shift': self.Shift,
                   'ParameterScaling': self.ParameterScaling}, 
                   FileName)
        print('Model successfully saved.')
        print('Path: %s'%str(FileName))
    
    def Load(self, Name, Folder):
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
        
        self.Scaling = torch.load(ModelPath + Name + '.pth')['Scaling']
        self.Shift = torch.load(ModelPath + Name + '.pth')['Shift']
        self.ParameterScaling = torch.load(ModelPath + Name + '.pth')['ParameterScaling']
        
        print('Model successfully loaded.')
        print('Path: %s'%str(FileName))

        
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

class DataFile():
### Reads sample file Info (string), Parameters (list), Values (torch array), Data (torch array) and Weights (torch array)
### FilePath is the path of the input file
### Computes cross-section XS (average weight) and total number of data ND in file
### Checks that files are in correct format (correct Keys)
### and that the length of Parameters and Data equals the one of Values and Weights respectively
    def __init__(self, FilePath, verbose=True):
        print('\nReading file ...' + FilePath)
        file = h5py.File(FilePath, 'r')
        if list(file.keys()) == ['Data', 'Info', 'Parameters', 'Values', 'Weights']:
            if( (len(file['Parameters'][()]) == len(file['Values'][()])) and (len(file['Data'][()]) == len(file['Weights'][()])) ):
                if verbose: print('##### File Info:\n' + file['Info'][()][0] + '\n#####')
                self.FilePath = FilePath
                self.Info = file['Info'][()][0]
                self.Parameters = file['Parameters'][()]
                #print(file['Values'][()].dtype)
                #print('%.15f' % (file['Data'][()][0][0]))
                self.Values = torch.DoubleTensor(file['Values'][()])
                self.Data = torch.DoubleTensor(file['Data'][()])
                self.Weights = torch.DoubleTensor(file['Weights'][()])
                self.XS = self.Weights.mean()
                self.ND = len(self.Weights)
            else: print('--> File not valid:\nunequal lenght of Values and Parameters or of Data and Weights')
        else:
            print('--> File format not valid:\nKeys: ' + print(list(file.keys())) + 
                  'should be ' + print(['Data', 'Info', 'Parameters', 'Values', 'Weights']))

class OurTrainingData():
    def __init__(self, SMfilepathlist, BSMfilepathlist, parameters, SMNLimits="NA", BSMNLimits="NA", verbose=True): 
        self.Parameters = parameters
        print('Loading Datasets with Parameters: ' + str(parameters) ) 
        if len(self.Parameters)!= 1: print('Only 1D Implemented in Training !')        
####### Load SM data
        if type(SMfilepathlist) == list:
            if all(isinstance(n, str) for n in SMfilepathlist):
                self.SMFilePathList = SMfilepathlist
                self.SMNumFiles = len(self.SMFilePathList)
                self.SMDataFiles = [DataFile(SMFilePath, verbose=verbose
                                            ) for SMFilePath in self.SMFilePathList]
            else:
                print('--> SMfilepathlist input should be a list of strings !')
                raise FileNotFoundError
        else:
            print('--> SMfilepathlist input should be a list !')
            raise FileNotFoundError
####### Join SM data and SM weigths
        if type(SMNLimits) == int:
            SMNLimits = [SMNLimits for DF in self.SMDataFiles]
        elif type(SMNLimits) == list and all(isinstance(n, int) for n in SMNLimits):
            if len(SMNLimits) != len(self.SMDataFiles):
                print("--> Please input %d integers to chop each SM file."%(
                    len(self.SMDataFiles)))
                raise ValueError
            elif sum([DF.ND >= Limit for (DF, Limit) in zip(self.SMDataFiles, self.SMNLimits)]
                    ) != len(self.SMDataFiles):
                print("--> Some chop limit larger than available data in the corresponding file.")
                print("--> Lengths of the files: "+str(
                    [DF.ND for DF in self.SMDataFiles]))
                raise ValueError
        else:
            SMNLimits = [len(DF.Data) for DF in self.SMDataFiles]          
        self.SMNDataList = SMNLimits
        self.SMDataList = [DF.Data[:N] for (DF, N) in zip(
            self.SMDataFiles, self.SMNDataList)]
        self.SMNData = sum(self.SMNDataList)
        self.SMWeightsList = [DF.Weights[:N] for (DF, N) in zip(
            self.SMDataFiles, self.SMNDataList)]      
        self.SMValuesList = [DF.Values for DF in self.SMDataFiles]
        self.SMSigmaList = [DF.XS for DF in self.SMDataFiles]

####### Join SM Data
        self.SMData = torch.cat(self.SMDataList, 0)  
        self.SMWeights = torch.cat(self.SMWeightsList, 0)
        self.SMNData = sum(self.SMNDataList)
####### Load BSM data
        if type(BSMfilepathlist) == list:
            if all(isinstance(n, str) for n in BSMfilepathlist):
                self.BSMFilePathList = BSMfilepathlist
                self.BSMNumFiles = len(self.BSMFilePathList)
                self.BSMDataFiles = [DataFile(BSMFilePath, verbose=verbose
                                            ) for BSMFilePath in self.BSMFilePathList]
                #if not None in ImportedFiles:
                #    self.BSMInfoList, self.BSMValuesList, self.BSMDataList, \
                #        self.BSMWeightsList = list(map(list, zip(*ImportedFiles)))
                #    self.BSMNDataList = [len(data) for data in self.BSMDataList]
                #    self.BSMSigmaList = [w.mean() for w in self.BSMWeightsList]
            else:
                print('--> BSMfilepathlist input should be a list of strings !')
                raise FileNotFoundError
        else:
            print('--> BSMfileathlist input should be a list !')
            raise FileNotFoundError
            
        if type(BSMNLimits) == int:
            BSMNLimits = [BSMNLimits for DF in self.BSMDataFiles]
        elif type(BSMNLimits) == list and all(isinstance(n, int) for n in BSMNLimits):
            if len(BSMNLimits) != len(self.BSMDataFiles):
                print("--> Please input %d integers to chop each SM file."%(
                    len(self.BSMDataList)))
                raise ValueError
            elif sum([DF.ND >= Limit for (DF, Limit) in zip(self.BSMDataFiles, self.BSMNLimits)]
                    ) != len(self.SMDataFiles):
                print("--> Some chop limit larger than available data in the corresponding file.")
                print("--> Lengths of the files: "+str([DF.ND for DF in self.BSMNLimits]))
                raise ValueError
        else:
            BSMNLimits =[DF.ND for DF in self.BSMDataFiles]   
        self.BSMNDataList = BSMNLimits
        self.BSMDataList = [DF.Data[:N] for (DF, N) in zip(
            self.BSMDataFiles, self.BSMNDataList)]
        self.BSMNData = sum(self.BSMNDataList)
        self.BSMWeightsList = [DF.Weights[:N] for (DF, N) in zip(
            self.BSMDataFiles, self.BSMNDataList)]      
        self.BSMValuesList = [DF.Values for DF in self.BSMDataFiles]
        self.BSMSigmaList = [DF.XS for DF in self.BSMDataFiles]
        
####### Break SM data in blocks to be paired with BSM data 
        BSMNRatioDataList = [torch.tensor(1., dtype=torch.double)*n/sum(self.BSMNDataList
                                                                       ) for n in self.BSMNDataList]
        #print("BSMNRatioDataList: "+str(BSMNRatioDataList))
        self.SMNSampleList = [int(self.SMNData*BSMNRatioData) for BSMNRatioData in BSMNRatioDataList] 
        #print("self.SMNSampleList: "+str(self.SMNSampleList))
        #print("self.BSMNDataList: "+str(self.BSMNDataList))
        self.SMNSample = sum(self.SMNSampleList)
        #print("self.SMNSample: "+str(self.SMNSample))
        self.SMSample = self.SMData[: self.SMNSample]
        self.SMSampleList = self.SMSample.split(self.SMNSampleList)
        #print("Check ratio: "+str([torch.tensor(self.BSMNDataList[i], dtype=torch.double).div(
        #    torch.tensor(self.SMNSampleList[i], dtype=torch.double)) for i in range(len(BSMNRatioDataList))]))
        ReWeighting = torch.cat([torch.ones(self.SMNSampleList[i], dtype=torch.double).mul(self.BSMNDataList[i]
                                            ).div(self.SMNSampleList[i]) for i in range(len(BSMNRatioDataList))])
        #print("self.SMWeights: "+str(self.SMWeights))
        #print("ReWeighting: "+str(ReWeighting))
        self.SMWeights = self.SMWeights[:self.SMNSample].mul(ReWeighting)
        #print("self.SMWeights: "+str(self.SMWeights))
        #print((self.SMWeights.sum())/np.average(self.SMWeights))
        #print((self.SMWeights.sum())/self.SMWeights.mean())
        #print(len(self.SMWeights))
        #print('Number of points in SM samples: '+str(self.SMNSampleList))
        
        #print('Number of points in BSM samples: '+str(self.BSMNDataList))
####### Create training labels
        self.SMLabels = torch.zeros(self.SMNSample, dtype=torch.double).split(self.SMNSampleList)
        self.BSMLabels = torch.ones(sum(self.BSMNDataList), dtype=torch.double).split(self.BSMNDataList)
####### Create training parameter values
        self.ParameterValuesList = [torch.ones(self.SMNSampleList[i] + self.BSMNDataList[i], dtype=torch.double)*\
                                    (self.BSMValuesList[i]) for i in range(len(self.BSMDataList))]
####### Join SM with BSM data
        self.SMSampleList = torch.split(self.SMSample, self.SMNSampleList, 0)
        self.SMWeightsList = torch.split(self.SMWeights, self.SMNSampleList, 0)
        self.DataList = [torch.cat([self.SMSampleList[i], self.BSMDataList[i]]
                                  ) for i in range(len(self.BSMDataList))]
        self.LabelList = [torch.cat([self.SMLabels[i], self.BSMLabels[i]]
                                  ) for i in range(len(self.BSMDataList))]
        self.WeightsList = [torch.cat([self.SMWeightsList[i], self.BSMWeightsList[i]]
                                  ) for i in range(len(self.BSMNDataList))]
        self.Data = torch.cat(self.DataList)
        self.Labels = torch.cat(self.LabelList)
        self.Weights = torch.cat(self.WeightsList)
        self.TrainingParameters = torch.cat(self.ParameterValuesList)
####### Output Tranining Data
    def ReturnData(self):
        return [self.Data, self.Labels, self.Weights, self.TrainingParameters]

    def Report(self):
        #from tabulate import tabulate
        #print([torch.Tensor(w).sum().div(sigma) for (w, sigma) in zip(self.BSMWeightsList, self.BSMSigmaList)])
        #print([torch.Tensor(w).sum().div(sigma) for (w, sigma) in zip(self.SMFilesWeightsList, self.SMSigmaList)])
        #print([torch.Tensor(w).sum().div(torch.tensor(self.SMSigmaList[0])) for (w) in (self.SMWeightsList)])
        
        print('\nLoaded Files:\n')
        print(tabulate({str(self.Parameters): self.SMValuesList, 
                        "#Events": self.SMNDataList, "XS[pb](avg.w)": self.SMSigmaList}, headers="keys"))
        print(' ')
        print(tabulate({ str(self.Parameters): self.BSMValuesList, 
                        "#Events": self.BSMNDataList, "XS[pb](avg.w)": self.BSMSigmaList}, headers="keys"))
        print('\nPaired BSM/SM Datasets:\n')
        print(tabulate({str(self.Parameters): self.BSMValuesList, "#Ev.BSM": self.BSMNDataList
                        , "#Ev.SM": self.SMNSampleList, 
                        "sum.w BSM\/XSBSM": [(self.BSMWeightsList[i].sum())/self.BSMSigmaList[i] for i in range(len(self.BSMWeightsList))],
                        "sum.w SM\/XSSM": [(self.SMWeightsList[i].sum())/(torch.Tensor(self.SMSigmaList).mean()) for i in range(len(self.SMWeightsList))]
                       }, headers="keys"))