import torch
import torch.nn as nn
import numpy as np
from btk import btk
from file import *

class Neural_Network(nn.Module):
    def __init__(self, lab = 'Foot_Off_GS', cont='Left', type='ITW', hidden = 5, tolerance = 0, lr = 0.05):
        super(Neural_Network, self).__init__()
        # parameters
        self.label = lab
        self.cont = cont
        self.capteurs = []
        self.tol = tolerance
        self.n_in, n_h, n_out = 1, hidden, 1
        self.reg = 0.0001
        self.patch = True                                      #flag to indicate if we add 0 when we require an unseen max or min

        #For each type of Data, the capteurs used are different,
        #as well as the number of neuron in the first layer
        if(type == 'ITW'):
            self.capteurs = ['RFHD','LBHD','RSHO','RASI','RPSI','STRN','CLAV', 'RBHD', 'RELB', 'LELB', 'RTHI', 'RWRB', 'LKNE', 'RWRA', 'LWRA', 'RANK', 'LANK', 'LSHO', 'LASI', 'LTHI', 'RKNE']
            self.n_in = len(self.capteurs)*2
        elif(type == 'FD'):
            self.capteurs = ['RFHD', 'LFHD', 'RBHD', 'LBHD', 'RSHO', 'C7', 'T10', 'CLAV', 'RWRA', 'RASI', 'LASI', 'STRN', ]
            self.n_in = len(self.capteurs)*2
            #self.n_in = 52
        elif(type == 'CP'):
            self.capteurs = ['LSHO', 'RFHD', 'RBHD', 'LBHD', 'RASI', 'LASI', 'RPSI', 'LPSI', 'C7', 'T10', 'STRN', 'CLAV']

            self.n_in = len(self.capteurs)*2
        else:
            print('Error with the type of Data. The Type of data should be \'ITW\', \'FD\' or \'CP\'')

        self.model = nn.Sequential(nn.Linear(self.n_in, n_h, bias=True),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out, bias=True),
                             nn.Sigmoid()
                             )
        self.criterion = torch.nn.MSELoss()     #we use a MSELoss function on normalized data (0<y<1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)


    def setPatch(self, p):
        self.patch = p

    def train(self, train, nbrIt = 1000):

        # the training data are shaped and normalize to enter the network
        Xtrain, Ytrain, start_step, end_step = self.shapeTrainingAndTestingData(train) # the data are shaped to fit the neural network
        x = torch.from_numpy(Xtrain).float()
        y = torch.from_numpy(Ytrain.reshape(Ytrain.shape[0],1)).float()


        y_pred = torch.from_numpy(np.array([])).float()

        for epoch in range(nbrIt):
            # as the shaping of data ignore some data if they are unshapable,
            # we check if the training set is not empty
            if(x.shape[0]==0):
                print('empty array given for training, abort')
                break

            # Forward Propagation


            # compute predicted data
            y_pred = self.model(x)


            # Compute (and print) loss
            loss = self.criterion(y_pred, y)
            l1 = 0
            for p in self.model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + self.reg * l1
            #if epoch % 20 == 0 : print('epoch: ', epoch,' loss: ', loss.item())


            # Zero the gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            self.optimizer.step()

        # Denormalize the Datas
        result = self.denormalize(y_pred.detach().numpy().transpose(),start_step,end_step)

        #Y_denormalized = self.denormalize(Ytrain,start_step,end_step)
        #print('targeted ouptup: ', Y_denormalized)
        #print('obtained output: ', result)
        #print( 'difference between actual and predicted output after training: ', Y_denormalized- np.vectorize(round)(result))
        return result

    def test(self, test):
        # the testing data are shaped and normalize to enter the network
        Xtest, Ytest, start_step, end_step = self.shapeTrainingAndTestingData(test)
        x = torch.from_numpy(Xtest).float()
        y = torch.from_numpy(Ytest.reshape(Ytest.shape[0],1)).float()

        # as the shaping of data ignore some data if they are unshapable,
        # we check if the tresting set is not empty
        if(x.shape[0]==0):
            print('empty array given for testing, abort')
            return 0

        # predicte the result
        y_pred = self.model(x)
        # denormalize the datas
        result = np.vectorize(round)(self.denormalize(y_pred.detach().numpy().transpose(),start_step,end_step))
        Y_denormalized = self.denormalize(Ytest,start_step,end_step)

        # return the error
        return result-Y_denormalized

    # This function return all the frame where an event is predicted
    # by the NN for this acquisition
    def predictEvent(self, acq):
        # we we find all the steps in the acquisition
        start_steps, end_steps = selectAllSteps(acq,self.cont)
        predEventFrames = []

        # for each step, we predict one event
        for i in range(len(start_steps)):
            if(start_steps[i] - end_steps[i]>0): print('trouble with the selectAllSteps function (boundaries in inverse order)')
            shapedData = self.shapeStepData(acq, start_steps[i], end_steps[i])
            if shapedData.shape[0]==self.n_in :
                x = torch.from_numpy(np.array(shapedData)).float()
                y_pred = self.model(x)
                result = np.vectorize(round)(self.denormalize(y_pred.detach().numpy().transpose(),start_steps[i],end_steps[i]))
                predEventFrames.append(result)
        #we return all the predicted event (one by step)
        return predEventFrames

    # this function add to the aquisitions all the predicted events
    # so that they can be saved
    def addPredictedEvent(self, File):
        for acq in file:
            predEventFrames = self.predictEvent(acq)
            for numevent in range(acq.GetEventNumber()):
                event = acq.GetEvent(numevent)
                if event.GetLabel() == self.label and event.GetContext() == self.cont:
                    for frame in predEventFrames:
                        if(abs(event.GetFrame()-frame)<20) : predEventFrames.remove(frame)

            for frame in predEventFrames:
                addEvent(acq, self.label, self.cont, int(frame))
                print('event added : ', int(frame))


    # Calcule les frames de départ et de fin du "pas" dans lequel se trouve events
    # un "pas" et définit comme la durée entre deux max du talon du coté correpondant à l'event
    def selectStepWithEvent(self, acq, event):
        if self.cont == 'Left':
            capteur = 'LHEE'
            #capteur = 'LANK'
        else:
            capteur = 'RHEE'
            #capteur = 'RANK'
        data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
        indMax = np.sort(np.ravel(maxLocal(data)))
        indMin = np.sort(np.ravel(minLocal(data)))
        indMin, indMax = cleanMinMax(indMin, indMax)
        event_frame = event.GetFrame()
        end_step = len(data)
        start_step = indMax[-1]
        for i in range(len(indMax)):
            if indMax[i]>event_frame:
                start_step = indMax[i-1]+1
                if(i==0): start_step = 0    #if i==0 the start of the step has been corrupted
                end_step = indMax[i]+1
                break
        if(end_step-start_step<0) :
            print('error with the step boundaries in selectStepWithEvent: eventframe: ', event_frame, ' start_step: ', start_step,' end_step: ', end_step, ' indMax: ', indMax)

        return start_step, end_step

    # create the shaped data corresponding to a step to have an array of size n_in
    def shapeStepData(self, acq, start_step, end_step):
        if(end_step-start_step<=0) :
            print('error with the step boundaries')
            return(np.array([]))
        step = []       #variable used to see how the data are shaped
        #type = []
        resume ={}
        f= False
        prob=[]

        # for each capteur used by the NN, we try to find 1 Max and 1 Min in the step
        # if we can't find one Max/Min, we add a zero to ensure that the size of the result is constant
        for capteur in self.capteurs:
            #print(capteur)
            data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
            indMax = np.ravel(maxLocal(data))
            indMin = np.ravel(minLocal(data))
            indMin, indMax = cleanMinMax(indMin, indMax)
            cnt = 0
            locCount = 0
            for i in indMax:
                if start_step - self.tol <= i and  i-self.tol <= end_step and locCount <1 :
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' max')
                    cnt += 1
                    locCount +=1
            while(locCount <1 and self.patch):
                step.append(0)
                f=True
                locCount +=1
                prob.append(capteur+' max ' + str(locCount))
            resume[capteur + ' max'] = cnt
            cnt = 0

            locCount = 0
            for i in indMin:
                if start_step-self.tol <= i and  i-self.tol<= end_step and locCount <1:
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' min')
                    cnt+=1
                    locCount+=1
            while(locCount <1 and self.patch):
                step.append(0)
                f= True
                locCount +=1
                prob.append(capteur+' min ' + str(locCount))
            resume[capteur + ' min'] = cnt
            #print(type)
            #print(resume)
        if(len(step)!=self.n_in):       # this should never happpen with the add of the zeros for unfound extremum
            print("some data weren't shaped correctly", len(step), resume)
            #print(indMin)
            #print(indMax)
            #print(start_step)
            #print(end_step)
        if(f):
            if len(prob) > self.n_in/4 :    # if more than 1/4 of the shaped data, are added 0, we don't use this data, because it's too corrupted
                print('Some data were too corrupted to be usefully patched and were droped')
                return np.array([])
            print('Some data have been patched ', prob)
        return np.array(step)



    #find the steps with event and shape them to be used by the neural Neural_Network
    #start_step and end_step are used to denormalize the data at the end.
    def shapeTrainingAndTestingData(self,files):
        Xt = []
        Yt = []
        start_step = []
        end_step =[]
        for acq in files:
            n_events = acq.GetEventNumber()             # On récupère le nombre d'évènements, pour les parcourirs
            for numevent in range(n_events):            # On parcours les indices des évènements
                event = acq.GetEvent(numevent)          # On récupère un évènement, grâce à son indice correspondant
                if event.GetLabel() == self.label and event.GetContext() == self.cont:
                    event_frame = event.GetFrame()
                    start, end = self.selectStepWithEvent(acq, event)
                    shapedData = self.shapeStepData(acq, start, end)
                    if shapedData.shape[0]==self.n_in :
                        Xt.append(shapedData)
                        Yt.append((event_frame-start)/(end-start))
                        start_step.append(start)
                        end_step.append(end)
        Xt = np.array(Xt)
        Yt = np.array(Yt)
        start_step = np.array(start_step)
        end_step = np.array(end_step)

        return Xt, Yt, start_step, end_step


    # the output are normalized ( between 0 and 1 ) in the shapeTrainingAndTestingData function,
    # we need to be able to denormalize the predicted results
    def denormalize(self, Y,start,end):
        return Y*(end-start)+start
