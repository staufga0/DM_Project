# In[0]
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from array import *



def dicoCapteurGenre():

    # color='green', label='Left - Off', linestyle='-.'
    # capteurSet_Off_Right = ['RANK', 'RHEE', 'C7',
    #                         'LHEE', 'RPSI', 'RKNE', 'T10', 'LFHD', 'LPSI']
    capteurSet_Off_Right = ['RANK', 'RHEE', 'C7', 'RKNE']
    # capteurSet_Off_Right = ['RANK', 'RPSI', 'RHEE', 'C7', 'CLAV', 'RKNE', 'T10']
    genre_Off_Right = {'RANK': 'max', 'RHEE': 'max', 'C7': 'min', 'RTIB': 'max', 'RKNE': 'min',
                       'LANK': 'min', 'LHEE': 'min', 'RFHD': 'min', 'LFHD': 'min', 'RBHD': 'min', 'RELB': 'min',
                       'RSHO': 'min', 'RPSI': 'min', 'RKNE': 'min', 'CLAV': 'min', 'LPSI': 'min', 'T10' : 'min',
                       'RTOE': 'min'
                       }

    # color='red', label='Left - Off', linestyle='-.'
    capteurSet_Off_Left = ['LANK', 'LHEE', 'RHEE', 'LSHO', 'LKNE']
    genre_Off_Left = {'LANK': 'max', 'LHEE': 'max', 'C7': 'min', 'LTIB': 'max', 'LKNE': 'min',
                      'RANK': 'max', 'RHEE': 'min', 'LFHD': 'min', 'RFHD': 'min', 'LBHD': 'min', 'LELB': 'max',
                      'LSHO': 'min', 'LPSI': 'min', 'LKNE': 'min', 'CLAV': 'min'}

    # color='green', label='Left - Strike', linestyle='--'
    capteurSet_Strike_Right = ['LANK', 'LHEE', 'RANK', 'CLAV', 'T10']
    genre_Strike_Right = {'LANK': 'max', 'LHEE': 'max', 'C7': 'min', 'LTIB': 'max', 'LKNE': 'min',
                          'RANK': 'min', 'RHEE': 'min', 'LFHD': 'min', 'RFHD': 'min', 'LBHD': 'min', 'LELB': 'max',
                          'LSHO': 'min', 'LPSI': 'min', 'LKNE': 'min', 'CLAV': 'min', 'T10' : 'min'}

    # color='red', label='Left - Strike', linestyle='--'
    capteurSet_Strike_Left = ['RANK', 'RHEE', 'C7',
                               'RFHD', 'RBHD', 'RPSI', 'RKNE', 'CLAV']
    genre_Strike_Left = {'RANK': 'max', 'RHEE': 'max', 'C7': 'min', 'RTIB': 'max', 'RKNE': 'min',
                         'LANK': 'min', 'LHEE': 'min', 'RFHD': 'min', 'LFHD': 'min', 'RBHD': 'min', 'RELB': 'max',
                         'RSHO': 'min', 'RPSI': 'min', 'RKNE': 'min', 'CLAV': 'min'}


    # Left   Right
    # Strike  Off
    # In[4]
    capteurDico = {}
    capteurDico['Foot_Off_GS'] = {}
    capteurDico['Foot_Strike_GS'] = {}
    capteurDico['Foot_Off_GS'] = {
        'Left': capteurSet_Off_Left, 'Right': capteurSet_Off_Right}
    capteurDico['Foot_Strike_GS'] = {
        'Left': capteurSet_Strike_Left, 'Right': capteurSet_Strike_Right}

    genreDico = {}
    genreDico['Foot_Off_GS'] = {}
    genreDico['Foot_Strike_GS'] = {}
    genreDico['Foot_Off_GS'] = {'Left': genre_Off_Left, 'Right': genre_Off_Right}
    genreDico['Foot_Strike_GS'] = {
        'Left': genre_Strike_Left, 'Right': genre_Strike_Right}

    return capteurDico, genreDico
