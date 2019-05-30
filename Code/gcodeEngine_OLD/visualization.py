#Imports
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from printer3D.gcode import cleanupPath
from matplotlib.ticker import LinearLocator, FormatStrFormatter

π = np.pi

def visualize(objH, objR, Θres, fx, pSettings, is3D=True, dspPercent=0.5, thinning=1, multicolor = True, title=None, **kwargs):
    
    layers = np.arange(0, objH-pSettings.fLayerH, pSettings.layerH * thinning) 
    Θ = (np.linspace(0, dspPercent*2*π%(2*π), Θres) - π/2)
    r = objR
    
    if('tsnLayers' in kwargs):
        kwargs['tsnLayers'] = int(kwargs['tsnLayers']/thinning) 
        
    x = r*np.cos(Θ) + pSettings.offsets[0]
    if(is3D):
        y = r*np.sin(Θ) + pSettings.offsets[1]
    z = fx(layers,Θ, **kwargs)
    
    cleanupPath(z)

    
    #Prep
    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    if(is3D):
        ax = fig.gca(projection='3d')
    else:
        ax = fig.subplots()
    
    if(is3D):
        if 'tsnLayers' in kwargs and multicolor:
            for layerNum in range(len(layers)):
                if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                    ax.plot(x, y, z[:, layerNum], color='red', alpha=0.5)
                else:
                    ax.plot(x, y, z[:, layerNum], color='black', alpha=0.5)
        else:
            for layerNum in range(len(layers)):
                ax.plot(x, y, z[:, layerNum], color='black', alpha=0.5)
    else:
        if 'tsnLayers' in kwargs and multicolor:
            for layerNum in range(len(layers)):
                if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                    ax.plot(Θ, z[:, layerNum], color='red', alpha=0.5)
                else:
                    ax.plot(Θ, z[:, layerNum], color='black', alpha=0.5)
        else:
            ax.plot(Θ, z, color='black', alpha=0.5)
    
    if title:
        plt.title(title)

    #Beautify plot
    if(is3D):
        ax.set_xlim(-r + pSettings.offsets[0], r + pSettings.offsets[0])
        ax.set_ylim(-r + pSettings.offsets[1], r + pSettings.offsets[1])
        ax.set_zlim(0, layers[-1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_axis_off()
    ax.view_init(elev=12.5, azim=-12.5)
    #Show plot
    plt.show()
