import torch
import torch.nn as nn
import numpy as np
from btk import btk
from file import *

class Neural_Network(nn.Module):
    def __init__(self, lab = 'Foot_Off_GS', cont='Left', type='ITW', hidden = 5, tolerance = 3, lr = 0.05):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.label = lab
        self.cont = cont
        self.capteurs = []
        self.tol = tolerance
        self.n_in, n_h, n_out = 1, hidden, 1
        self.patch = True                                      #flag to indicate if we add 0 when we require an anseen max or min
        if(type == 'ITW'):
            self.capteurs = ['STRN', 'CLAV','T10']
            self.n_in = 12
            #self.capteurs = ['RFHD','LBHD','RSHO','RASI','RPSI','STRN','CLAV']
            #self.n_in = 28


        self.model = nn.Sequential(nn.Linear(self.n_in, n_h, bias=True),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out, bias=True),
                             nn.Sigmoid()
                             )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)


    def setPatch(self, p):
        self.patch = p

    def train(self, train, nbrIt = 600):

        Xtrain, Ytrain, start_step, end_step = self.shapeTrainingAndTestingData(train) # the data are shaped to fit the neural network
        x = torch.from_numpy(Xtrain).float()
        y = torch.from_numpy(Ytrain.reshape(Ytrain.shape[0],1)).float()
        y_pred=[]
        for epoch in range(nbrIt):
            # Forward Propagation
            y_pred = self.model(x)

            # Compute and print loss
            loss = self.criterion(y_pred, y)

            #if epoch % 20 == 0 : print('epoch: ', epoch,' loss: ', loss.item())

            # Zero the gradients
            self.optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            self.optimizer.step()

        result = self.denormalize(y_pred.detach().numpy().transpose(),start_step,end_step)
        Y_denormalized = self.denormalize(Ytrain,start_step,end_step)
        #print('targeted ouptup: ', Y_denormalized)
        #print('obtained output: ', result)
        print( 'difference between actual and predicted output after training: ', Y_denormalized- np.vectorize(round)(result))

        return result

    def test(self, test):
        Xtest, Ytest, start_step, end_step = self.shapeTrainingAndTestingData(test)
        x = torch.from_numpy(Xtest).float()
        y = torch.from_numpy(Ytest.reshape(Ytest.shape[0],1)).float()

        y_pred = self.model(x)

        result = np.vectorize(round)(self.denormalize(y_pred.detach().numpy().transpose(),start_step,end_step))
        Y_denormalized = self.denormalize(Ytest,start_step,end_step)


        return result-Y_denormalized

    def predict(self,X):



        return o

    # Calcule les frames de départ et de fin du "pas" dans lequel se trouve events
    # un "pas" et définit comme la durée entre deux max du talon du coté correpondant à l'event
    def selectStepWithEvent(self, acq, event):
        if self.cont == 'Left':
            capteur = 'LHEE'
        else:
            capteur = 'RHEE'
        data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
        indMax = np.sort(np.ravel(maxLocal(data)))
        event_frame = event.GetFrame()
        for i in range(len(indMax)):
            if indMax[i]>event_frame:
                start_step = indMax[i-1]+1
                if(i==0): start_step = 0    #if i==0 the start of the step has been corrupted
                end_step = indMax[i]+1
                break
        if(end_step-start_step<0) :
            print('error with the step boundaries in selectStepWithEvent: eventframe: ', event_frame, ' start_step: ', start_step,' end_step: ', end_step, ' indMax: ', indMax)

        return start_step, end_step

    def shapeStepData(self, acq, start_step, end_step):
        if(end_step-start_step<0) :
            print('error with the step boundaries')
            return(np.array([]))
        step = []
        #type = []
        resume ={}


        for capteur in self.capteurs:
            data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
            indMax = np.ravel(maxLocal(data))
            cnt = 0
            locCount = 0
            f= False
            for i in indMax:
                if start_step - self.tol <= i and  i-self.tol <= end_step and locCount <2 :
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' max')
                    cnt += 1
                    locCount +=1
            while(locCount !=2 and self.patch):
                step.append(0)
                f=True
                locCount +=1
            resume[capteur + ' max'] = cnt
            cnt = 0
            indMin = np.ravel(minLocal(data))
            locCount = 0
            for i in indMin:
                if start_step-self.tol <= i and  i-self.tol<= end_step and locCount <2:
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' min')
                    cnt+=1
                    locCount+=1
            while(locCount !=2 and self.patch):
                step.append(0)
                f= True
                locCount +=1
            resume[capteur + ' min'] = cnt
            #print(type)
            #print(resume)
        if(len(step)!=self.n_in):
            print("some data weren't shaped correctly", len(step), resume)
            #print(indMin)
            #print(indMax)
            #print(start_step)
            #print(end_step)
        if(f):print('Some data have been patched')
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

    def denormalize(self, Y,start,end):
        return Y*(end-start)+start


    def predictEvent(self, acq):
        start_steps, end_steps = selectAllSteps(acq,self.cont)
        predEventFrames = []
        for i in range(len(start_steps)):
            if(start_steps[i] - end_steps[i]>0): print('trouble with the selectAllSteps function (boundaries in inverse order)')
            shapedData = self.shapeStepData(acq, start_steps[i], end_steps[i])
            if shapedData.shape[0]==self.n_in :
                x = torch.from_numpy(np.array(shapedData)).float()
                y_pred = self.model(x)
                result = np.vectorize(round)(self.denormalize(y_pred.detach().numpy().transpose(),start_steps[i],end_steps[i]))
                predEventFrames.append(result)
        return predEventFrames

    def addPredictedEvent(self, acq):
        predEventFrames = self.predictEvent(acq)
        for numevent in range(acq.GetEventNumber()):
            event = acq.GetEvent(numevent)
            if event.GetLabel() == self.label and event.GetContext() == self.cont:
                for frame in predEventFrames:
                    if(abs(event.GetFrame()-frame)<20) : predEventFrames.remove(frame)

        for frame in predEventFrames:
            addEvent(acq, self.label, self.cont, int(frame))
            print('event added : ', int(frame))
