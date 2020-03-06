import numpy as np
import collections
import copy
import sys
from scipy import io, sparse, stats
#import networkx as nx
import math
import glob
from typing import NamedTuple
import shutil
import os

#import sys
#sys.path.append('D:/USERS/lvantonov/Develop/nsearchdpd')


from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from python_modules.GraphStructures import checkAMOnCyclic

## class for kepping parent info
class Parent(NamedTuple):
    """
    Class Parent
    """
    path: str
    GenAM: np.array = 0
    WeigGenAM: np.array = 0
    AM: np.array = 0
    minWaightOfCon: int = 0
    countOfConnections: int = 0
    start: np.array = 0
    finish: int = 0

def generateChildrens(path, n_child):
    filesParentsGenAM = [f for f in glob.glob(path + "\parents\**\GenAM*.mat", recursive=True)]

    dictParents = {}
    i = 0
    for f in filesParentsGenAM:
        GenAM = io.loadmat(f.replace('\\', '/', 1))['GenAM']
        S = np.sum(np.sum(GenAM))
        WeigGenAM = io.loadmat(f.replace('GenAM', 'WeigGenAM', 1))['WeigGenAM']
        minWaightOfCon = io.loadmat(f.replace('GenAM', 'minWaightOfCon', 1))['minWaightOfCon'][0][0]
        countOfConnections = io.loadmat(f.replace('GenAM', 'countOfConnections', 1))['countOfConnections'][0][0]
        calculated = io.loadmat(f.replace('GenAM', 'calculated', 1))['calculated'][0]
        AM = io.loadmat(f.replace('GenAM', 'AM', 1))['AM']
        start = io.loadmat(f.replace('GenAM', 'start', 1))['start'][0]
        finish = io.loadmat(f.replace('GenAM', 'finish', 1))['finish'][0]

        if calculated == 'True':
            par = Parent(f, GenAM, WeigGenAM, AM, minWaightOfCon, countOfConnections, start, finish)
            x = {i: par}
            dictParents.update(x)
            i += 1

    n_parents = len(dictParents)
    #n_child = 0.5 * n_parents # 50% from all parents is count of new children

    #create folder for children
    path = os.path.normpath(path)
    directory = 'children'
    path_children = os.path.join(path, directory)
    try:
        os.stat(path_children)
    except:
        os.mkdir(path_children)

    # Breeding
    for i in np.arange(n_child):

        # count on parents
        n = np.arange(2, 5, 1)
        count_parents = np.random.choice(n, 1)
        pr = np.arange(0, n_parents)
        # random chousen parents
        parents = np.random.choice(pr, replace=False, size=count_parents)

        new_kid = dictParents[0].WeigGenAM * 0
        countOfConnections = 0

        # cycle for the each parent and creation new kid
        parents_txt = ""
        for j in np.arange(count_parents):
            new_kid = new_kid + dictParents[parents[j]].WeigGenAM
            countOfConnections += np.count_nonzero(dictParents[parents[j]].WeigGenAM)
            parents_txt += (os.path.normpath(dictParents[parents[j]].path) + '\n')

            start = dictParents[parents[j]].start
            finish = dictParents[parents[j]].finish

        new_kid = new_kid / count_parents
        countOfConnections = int(countOfConnections / count_parents)

        # genAM for new kid
        GenAM = new_kid / np.amax(new_kid)

        # max powerfull connections in matrix
        valuesCon = sorted(GenAM.ravel(), reverse=True)[0:countOfConnections]
        mask = np.array(np.isin(GenAM, valuesCon))
        # AM for new kid
        AM = mask * 1

        #checking AM correction
        minWaightOfCon = min(valuesCon)

        [AM, GenAM] = checkAMOnCyclic(AM, GenAM, minWaightOfCon, start, finish)

        #path_to_curr_kid = os.path.normpath(path_children)
        directory = str(int(i+1))
        path_to_curr_kid = os.path.join(path_children, directory)

        try:
            os.stat(path_to_curr_kid)
        except:
            os.mkdir(path_to_curr_kid)

        # create data and save it for new kid
        path_to_first_parent = os.path.normpath(dictParents[parents[0]].path)
        shutil.copy(path_to_first_parent.replace('GenAM', 'inputDsc', 1), (path_to_curr_kid + '\inputDsc.mat'))
        shutil.copy(path_to_first_parent.replace('GenAM', 'strateges', 1), (path_to_curr_kid + '\strateges.mat'))
        shutil.copy(path_to_first_parent.replace('GenAM', 'start', 1), (path_to_curr_kid + '\start.mat'))
        shutil.copy(path_to_first_parent.replace('GenAM', 'finish', 1), (path_to_curr_kid + '\\finish.mat'))

        countOfConnections = int(np.sum(np.sum(AM)))
        io.savemat(path_to_curr_kid + '\AM.mat', mdict={'AM': AM})
        io.savemat(path_to_curr_kid + '\GenAM.mat', mdict={'GenAM': GenAM})
        io.savemat(path_to_curr_kid + '\minWaightOfCon.mat', mdict={'minWaightOfCon': minWaightOfCon})
        io.savemat(path_to_curr_kid + '\countOfConnections.mat', mdict={'countOfConnections': countOfConnections})

        calculated = 'False'
        io.savemat(path_to_curr_kid + '\calculated.mat', mdict={'calculated': calculated})

        # how long hi live
        age = 0
        io.savemat(path_to_curr_kid + '/age.mat', mdict={'age': age})

        textfile = open(path_to_curr_kid + '\parents.txt', 'w')
        textfile.write(parents_txt)
        textfile.close()


        print(path)

def generateMutants(path, n_mutants, percent_mutations=0.03):
    filesParentsGenAM = [f for f in glob.glob(path + "\parents\**\GenAM*.mat", recursive=True)]

    dictParents = {}
    i = 0
    for f in filesParentsGenAM:
        GenAM = io.loadmat(f.replace('\\', '/', 1))['GenAM']
        S = np.sum(np.sum(GenAM))
        WeigGenAM = io.loadmat(f.replace('GenAM', 'WeigGenAM', 1))['WeigGenAM']
        minWaightOfCon = io.loadmat(f.replace('GenAM', 'minWaightOfCon', 1))['minWaightOfCon'][0][0]
        countOfConnections = io.loadmat(f.replace('GenAM', 'countOfConnections', 1))['countOfConnections'][0][0]
        calculated = io.loadmat(f.replace('GenAM', 'calculated', 1))['calculated'][0]
        AM = io.loadmat(f.replace('GenAM', 'AM', 1))['AM']
        start = io.loadmat(f.replace('GenAM', 'start', 1))['start'][0]
        finish = io.loadmat(f.replace('GenAM', 'finish', 1))['finish'][0]

        if calculated == 'True':
            par = Parent(f, GenAM, WeigGenAM, AM, minWaightOfCon, countOfConnections, start, finish)
            x = {i: par}
            dictParents.update(x)
            i += 1

    # create folder for the mutants
    path = os.path.normpath(path)
    directory = 'mutants'
    path_mutants = os.path.join(path, directory)
    try:
        os.stat(path_mutants)
    except:
        os.mkdir(path_mutants)

    # n_mutants is count of mutants
    # Breeding
    i = 0
    #for i in np.arange(0, n_mutants, n):
    while i != n_mutants:

        # chouisen parent for the mutation
        num_ch_parent = np.random.choice(len(dictParents))
        ch_parent = dictParents[num_ch_parent]
        # random chousen parent

        # specify probability distribution
        rvs = stats.norm(loc=0, scale=ch_parent.minWaightOfCon).rvs
        # create sparse random matrix with specific probability distribution/random numbers.
        sparseMart = sparse.random(ch_parent.GenAM.shape[0], ch_parent.GenAM.shape[1], density=percent_mutations, data_rvs=rvs).toarray()
        sparseMart[:, 0] = 0
        sparseMart[1, :] = 0

        new_mutant = ch_parent.GenAM + sparseMart
        new_mutant[new_mutant < 0] = 0
        new_mutant[new_mutant > 1] = 1

        # genAM for new MUTANT
        GenAM = new_mutant

        # mutation of count connections
        mut_range = np.arange(-3, 7)
        mut_con = np.random.choice(mut_range)

        # max powerfull connections in matrix
        valuesCon = sorted(GenAM.ravel(), reverse=True)[0:ch_parent.countOfConnections+mut_con]
        mask = np.array(np.isin(GenAM, valuesCon))
        # AM for new kid
        AM = mask * 1

        # checking AM correction
        minWaightOfCon = min(valuesCon)

        start = ch_parent.start
        finish = ch_parent.finish
        [AM, GenAM] = checkAMOnCyclic(AM, GenAM, minWaightOfCon, start, finish)

        # IF ch_parent.AM != new_mutant_AM
        if(np.sum(np.sum(ch_parent.AM - AM)) != 0):
            # path_to_curr_kid = os.path.normpath(path_children)
            directory = str(int(i + 1))
            path_to_curr_mut = os.path.join(path_mutants, directory)

            try:
                os.stat(path_to_curr_mut)
            except:
                os.mkdir(path_to_curr_mut)

            # create data and save it for new kid
            path_to_parent = os.path.normpath(ch_parent.path)
            shutil.copy(path_to_parent.replace('GenAM', 'inputDsc', 1), (path_to_curr_mut + '\inputDsc.mat'))
            shutil.copy(path_to_parent.replace('GenAM', 'strateges', 1), (path_to_curr_mut + '\strateges.mat'))
            shutil.copy(path_to_parent.replace('GenAM', 'start', 1), (path_to_curr_mut + '\start.mat'))
            shutil.copy(path_to_parent.replace('GenAM', 'finish', 1), (path_to_curr_mut + '\\finish.mat'))

            countOfConnections = int(np.sum(np.sum(AM)))
            io.savemat(path_to_curr_mut + '\AM.mat', mdict={'AM': AM})
            io.savemat(path_to_curr_mut + '\GenAM.mat', mdict={'GenAM': GenAM})
            io.savemat(path_to_curr_mut + '\minWaightOfCon.mat', mdict={'minWaightOfCon': minWaightOfCon})
            io.savemat(path_to_curr_mut + '\countOfConnections.mat', mdict={'countOfConnections': countOfConnections})


            calculated = 'False'
            io.savemat(path_to_curr_mut + '\calculated.mat', mdict={'calculated': calculated})

            # how long hi live
            age = 0
            io.savemat(path_to_curr_mut + '/age.mat', mdict={'age': age})

            parents_txt = ch_parent.path
            textfile = open(path_to_curr_mut + '\parents.txt', 'w')
            textfile.write(parents_txt)
            textfile.close()
            i += 1
            print(path_to_curr_mut)
        #else:


path = '../Experiments/GeneticAlgorithmExp3_Gasnikov_15Luts_16_30Firs_15_nmse_Wide_Signal/Slots/3/Generations/4'
path = '../Experiments/test_multiinput_GeneticAlgorithmExp3_Gasnikov_17Luts2D_17x17_39Firsx5coeff_nmse_Wide_Signal_2/Slots/1/Generations/1'
generateMutants(path=path, n_mutants=10, percent_mutations=0.05)
#generateChildrens(path, 1)

""".imshg = Graph(16, 3, 0, 1, 2)
g.calcLayers()
g.generateStruct()
AM = g.getAdjacencyMatrix()
pltow(AM)
plt.show()

t = 1"""




