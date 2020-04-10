import torch, os, time, datetime
from torch import nn
from torch.nn.modules import Module

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
        self.NumberOfEpochs = 100
        self.InitialLearningRate = 1e3
        self.SaveAfterEpoch = lambda :[self.NumberOfEpochs] 
### Define Layers
        self.LinearLayerList1  = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer1 = nn.Linear(self.Architecture[-2], 1)       
        self.LinearLayerList2 = nn.ModuleList([nn.Linear(self.Architecture[i], 
            self.Architecture[i+1]) for i in range(len(self.Architecture)-2)])
        self.OutputLayer2 = nn.Linear(self.Architecture[-2], 1)
        
        self.Optimiser = torch.optim.Adam(self.parameters(), self.InitialLearningRate)
        self.Criterion = WeightedMSELoss()

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
        x2 = self.OutputLayer2(x2).squeeze()
        
        rho = (1 + torch.mul(x1, Parameters))**2 + (torch.mul(x2, Parameters))**2      
        return (rho.div(1.+rho)).view(-1, 1)
    
    def GetL1Bound(self, L1perUnit):
### Max L1 norm of weights at each layer. What about bias? Excluding the first layer?
        self.L1perUnit = L1perUnit
        L1MaxList = []
        for m in self.children():
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
                for mm in m:
                    Counter +=1
                    with torch.no_grad():
                        DesignatedL1Max = mm.weight.size(0)*m.weight.size(1)*self.L1perUnit
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
    
    def Save(self, Name, Folder):
### Saves the model in Folder/Name
        FileName = Folder + Name + '.pth'
        torch.save({'StateDict': self.state_dict(), 
                   'Scaling': self.Scaling,
                   'Shift': self.Shift,
                   'ParameterScaling': self.ParameterScaling}, 
                   FileName)
        print('Model successfully saved.')
        print('Path: %s'%str(FileName))
    
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
        
    def SetOptimiser(self, OP):
        self.Optimiser = OP(self.parameters(), self.InitialLearningRate)
        
    def SetCriterion(self, CR):
        self.Criterion = CR
        
    def SetNumberOfEpochs(self, NE):
        self.NumberOfEpochs = NE
        self.SaveAfterEpoch = lambda : list(set(self.SaveAfterEpoch + [NE,]))
        
    def SetInitialLearningRate(self,ILR):
        self.InitialLearningRate = ILR
        self.Optimiser = torch.optim.Adam(self.parameters(), self.InitialLearningRate)
        
    def SetSaveAfterEpoch(self,SAE):
        SAE.sort()
        self.SaveAfterEpoch = lambda : SAE

    def Train(self, Data, DataParameters, Labels, Weights, L1perUnit=None, UseGPU=True, Name="", Folder=os.getcwd()):
        if L1perUnit:
            self.GetL1Bound(L1perUnit)

        if UseGPU:
            self.cuda()
            Data, Parameters      = Data.cuda(), DataParameters.cuda()
            Labels = Labels.cuda()
            Weights = Weights.cuda()
        print(" =================== BEGINNING TRAIN ==================== ")
        beginning = start = time.time()

        for e in range(self.NumberOfEpochs):
            output          = self.Forward(Data, DataParameters)
            loss            = self.Criterion(output, Labels.reshape(-1,1), Weights.reshape(-1, 1))
        
            if (e+1) in self.SaveAfterEpoch():
                start       = report_ETA(beginning, start, self.NumberOfEpochs, e+1, loss)
                self.Save(Name+", %d epoch"%(e+1), Folder)
                
            self.Optimiser.zero_grad()
            loss.backward()
            self.Optimiser.step()
            if L1perUnit:
                self.ClipL1Norm()

        print(" ===================   END OF TRAIN   =================== ")


