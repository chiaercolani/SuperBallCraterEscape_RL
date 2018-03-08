import sys
import matplotlib.pyplot as plt
import numpy as np


def plotResults():
    f= open('MultiBruteForceResults/coordinates.txt','r')
    coord=[]
    for line in f:
        coord=np.append(coord,float(line))

    f.close()
    print(np.amax(coord))
    best=np.argmax(coord)
    print(best)
    x=np.arange(0,len(coord))

    fig= plt.figure(figsize=(20,10))
    ax1=plt.subplot(211)
    plt.scatter(x,coord)

    plt.title('Maximum Z coordinate of the center of mass for each Neural Network run')

    plt.show()

def findBestRuns():
    f= open('MultiBruteForceResults/coordinates.txt','r')
    coord=[]
    position=[]
    i=0
    minHeight=0.85
    maxHeight=2.5
    for line in f:
        if float(line)>minHeight and float(line)<maxHeight:
            coord=np.append(coord,float(line))
            position=np.append(position,i)
        i=i+1
    f.close()
    print(coord)
    print(position)
    print(len(position))

######  MAIN  ########

findBestRuns()
