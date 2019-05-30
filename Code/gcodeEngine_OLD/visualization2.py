#Imports
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from Code.printer3D.gcode import cleanupPath
from Code.printer3D import pSettings
from Code.gcodeEngine.gcodeEngineV2_5 import fxCosPhase
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def visualize(objH, objR, Θres, fx, pSettings, is3D=True, hidePercent=0.5, thinning=1, multicolor=True, title=None, showAxes=False, toolbar=False, maximized=True, view=None, saveAs=False, w=500, h='auto', linewidth=1, cleanup=True, crop=True, showLayers=False, bColor=None, **kwargs):
    π = np.pi
    print("Visualization Started.")
    
    print("Initilizing Variables...", end='')
    dpi = 166
    
    winw = w #=2*π*objR
    if(not(h=='auto') or is3D):
        winh = h #=
        if(not(h)):
            winh=winw
    else:
        winh = winw*objH/(2*π*objR)

    layers = np.arange(0, objH, pSettings.layerH * thinning)+pSettings.fLayerH 
    Θ = np.linspace(0, 2*π-hidePercent*2*π%(2*π), Θres)
    if(showLayers):
        showLayers = (int(showLayers[0] / thinning),int(showLayers[1] / thinning))
    if(is3D):
        Θ -= π/2 
    r = objR
    
    if('tsnLayers' in kwargs):
        kwargs['tsnLayers'] = int(kwargs['tsnLayers']/thinning) 
        
    x = r*np.cos(Θ) + pSettings.offsets[0]
    
    if(is3D):
        y = r*np.sin(Θ) + pSettings.offsets[1]
    print("Done")
    
    print("Finding Path...", end='')
    z = fx(layers,Θ, **kwargs)
    print("Done")
    
    if(cleanup):
        print("Cleaning Path...", end='')
        cleanupPath(z)
    
    print("Done")

    if(not(toolbar)):
        plt.rcParams['toolbar'] = 'None'
        print("Removed Toolbar.")
    
    fig = plt.figure(figsize=(winw/dpi,winh/dpi), dpi=dpi)
    
    if(is3D):
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.subplots()

    if(maximized):
        plt.subplots_adjust(left=0,bottom=0,right=1,top=1)
        print("Maximized Plot.")    

    print("Ploting...", end='')
    if(showLayers):
        if(is3D):
            if 'tsnLayers' in kwargs and multicolor:
                for layerNum in range(showLayers[0], showLayers[1]):
                    if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                        ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='red', alpha=0.5)
                    else:
                        ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='black', alpha=0.5)
            else:
                for layerNum in range(showLayers[0], showLayers[1]):
                    ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='black', alpha=0.5)
        else:
            if 'tsnLayers' in kwargs and multicolor:
                for layerNum in range(showLayers[0], showLayers[1]):
                    if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                        ax.plot(Θ*objR, z[:, layerNum], linewidth=linewidth, color='red', alpha=1)
                    else:
                        ax.plot(Θ*objR, z[:, layerNum], linewidth=linewidth, color='black', alpha=1)
            else:
                for layerNum in range(showLayers[0], showLayers[1]):
                    ax.plot(Θ*objR, z[:, layerNum], linewidth=linewidth, color='black', alpha=1)
    else:
        if(is3D):
            if 'tsnLayers' in kwargs and multicolor:
                for layerNum in range(len(layers)):
                    if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                        ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='red', alpha=0.5)
                    else:
                        ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='black', alpha=0.5)
            else:
                for layerNum in range(len(layers)):
                    ax.plot(x, y, z[:, layerNum], linewidth=linewidth, color='black', alpha=0.5)
        else:
            if 'tsnLayers' in kwargs and multicolor:
                for layerNum in range(len(layers)):
                    if(kwargs['tsnLayers'] < layerNum < layers.size-kwargs['tsnLayers']):
                        ax.plot(Θ*objR, z[:, layerNum], linewidth=linewidth, color='red', alpha=1)
                    else:
                        ax.plot(Θ*objR, z[:, layerNum], linewidth=linewidth, color='black', alpha=1)
            else:
                ax.plot(Θ*objR, z, linewidth=linewidth, color='black', alpha=1)
    print("Done")

    print("Applying Plot Settings.")
    #Beautify plot
    if(is3D):
        ax.set_xlim(-r + pSettings.offsets[0], r + pSettings.offsets[0])
        ax.set_ylim(-r + pSettings.offsets[1], r + pSettings.offsets[1])
        ax.set_zlim(0, layers[-1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(elev=12.5, azim=-12.5)
    else:
        if(crop):
            ax.set_ylim(0, objH+pSettings.fLayerH)
            ax.set_xlim(0, 2*π*objR+0.2)
        #plt.xscale('')
    #ax.set_axis_off()
    
    
    if(view):
        ax.set_xlim(view[0])
        ax.set_ylim(view[1])
        if(is3D):
            ax.set_zlim(view[2])
    
    if(title):
        plt.title(title)
        print("Title Set.")

    if(not(showAxes)):
        #plt.axis('off')
        ax.set_axis_off()
        print("Axes hidden.")
    
    if(bColor):
        #plt.set_facecolor(bColor)
        fig.patch.set_facecolor(bColor)
        
    if(saveAs):
        plt.savefig(saveAs)
        print(saveAs + " saved.")
        
    else:
        plt.show()
        print("Plot shown.")
    
    print("dataPoints:")
    print(f"  x={x.shape}")
    print(f"  z={z.shape}")
    if(is3D):
        print(f"  y={y.shape}")
    
    print("Visualization done.")

if __name__ == '__main__':
    
    printerSettings = pSettings(printerName="FT5-R2", eMod=1)
        
    sin = lambda layers, Θ, amp, freq, **kwargs: amp*np.sin(freq*Θ)[:, np.newaxis] + layers
        
    visualize(fx=fxCosPhase,
                  amp=1, freq=8, tsnLayers=0,
                  objH=20.2, objR=10, Θres=1000, pSettings=printerSettings,
                  is3D=False, hidePercent=0, thinning=4, multicolor = False,
                  title=None, showAxes=False, toolbar=False, maximized=False,
                  view=None, w=1000, h='auto', linewidth=1, saveAs=None,#'1.png',
                  crop=True, showLayers = False, bColor = '#37474F',
                  cleanup = 0)#(50,52), cleanup=1)
#
    
    
    
    
    
    
    
    
    
    
    
    
    
