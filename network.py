import torch
import torch.nn as nn
import numpy as np
from btk import btk
from file import *

class Neural_Network(nn.Module):
    def __init__(self, lab, cont, type):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.label = lab
        self.cont = cont
        self.capteurs = []
        n_in, n_h, n_out = 1, 40, 1
        if(type == 'ITW'):
            self.capteurs = ['STRN', 'CLAV','T10']
            n_in = 12


        self.model = nn.Sequential(nn.Linear(n_in, n_h, bias=True),
                             nn.ReLU(),
                             nn.Linear(n_h, n_out, bias=True),
                             nn.Sigmoid()
                             )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)



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
        print('targeted ouptup: ', Y_denormalized)
        print('obtained output: ', result)
        print( 'diff: ', Y_denormalized- np.vectorize(round)(result))

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
        indMax = np.ravel(maxLocal(data))
        event_frame = event.GetFrame()
        for i in range(len(indMax)):
            if indMax[i]>event_frame:
                start_step = indMax[i-1]+1
                end_step = indMax[i]+1
                break
        return start_step, end_step

    def shapeStepData(self, acq, start_step, end_step):
        step = []
        #type = []
        resume ={}

        for capteur in self.capteurs:
            data = np.array(acq.GetPoint(capteur).GetValues()[:, 2])
            indMax = np.ravel(maxLocal(data))
            cnt = 0
            for i in indMax:
                if start_step <= i and  i<= end_step:
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' max')
                    cnt += 1
            resume[capteur + ' max'] = cnt
            cnt = 0
            indMin = np.ravel(minLocal(data))
            for i in indMin:
                if start_step <= i and  i<= end_step:
                    step.append((i-start_step)/(end_step-start_step))
                    #type.append(capteur + ' min')
                    cnt+=1
            resume[capteur + ' min'] = cnt
            #print(type)
            #print(resume)
        if(len(step)!=12):print('ARRETER TOUT !!!!! ', len(step), resume)
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
                    start_step.append(start)
                    end_step.append(end)
                    Yt.append((event_frame-start)/(end-start))
                    shapedData = self.shapeStepData(acq, start, end)
                    Xt.append(shapedData)
        Xt = np.array(Xt)
        Yt = np.array(Yt)
        start_step = np.array(start_step)
        end_step = np.array(end_step)

        return Xt, Yt, start_step, end_step

    def denormalize(self, Y,start,end):
        return Y*(end-start)+start
