import numpy as np
from scipy import ndimage
import os
import cPickle as pickle

def ReshapeDataset(dataset):
    #d= dataset.reshape((dataset.shape[1]*dataset.shape[2],dataset.shape[0]))
    #print d.shape
    return dataset

def LoadLetter(filename):
    colordepth = 255.0
    try:
        im = (ndimage.imread(filename)-colordepth/2)/colordepth
        image = im.reshape((im.shape[0]*im.shape[1],1))
    except:
        print "error reading image: " + filename + " skipping."
        return None
    
    return image

def LoadFolder(amountToLoad,folder):
    files = os.listdir(folder)

    letterset = []

    counter = 0
    for fn in files:
        if counter==amountToLoad: break

        fpath = os.path.join(folder,fn)
        if not os.path.isfile(fpath): continue

        ldata = LoadLetter(fpath)
        if ldata==None: continue

        letterset.append(ldata)

        counter +=1

    ppath = os.path.join(os.path.dirname(folder), os.path.basename(folder) + ".pickle")
    print "Saving to: "+ppath
    with open(ppath,"wb") as f:
        pickle.dump(letterset,f)
    return letterset

def LoadPickle(ppath):
    with open(ppath,"rb") as f:
        return pickle.load(f)
    

def LoadDataset(datasetPath, amountToLoad, force=False):
    dataset = []
    datalabels = []

    files = sorted(os.listdir(datasetPath))

    ind = 0
    startind = 0
    for fn in files:
        fpath = os.path.join(datasetPath,fn)
        if os.path.isfile(fpath): continue

        prevSize = len(dataset)

        ppath = os.path.join(datasetPath, fn+".pickle")
        
        if os.path.exists(ppath) and not force:
            print "Found file: " + ppath
            dataset[startind:startind+amountToLoad] = LoadPickle(ppath)
        else:
            dataset[startind:startind+amountToLoad] = LoadFolder( amountToLoad, fpath)

        datalabels[startind: startind+amountToLoad] = np.ones((len(dataset)-prevSize,)) * ind
        #datalabels = np.append(datalabels,ind*np.ones((len(dataset)-prevSize,))) 
        ind = ind + 1
        startind += amountToLoad

    return (ReshapeDataset(np.array(dataset)), np.array(datalabels))

def Randomize(dataset, datalabels):
    permutation = np.random.permutation(dataset.shape[0])
    print dataset.shape, datalabels.shape

    randomizedDataset = dataset[permutation,:]
    randomizedDatalabels = datalabels[permutation]
    return (randomizedDataset, randomizedDatalabels)

def LoadRandomizedDataset(datasetPath, amountToLoad, force=False):
    return Randomize(*LoadDataset(datasetPath, amountToLoad, force))
