'''
                                                GCODE FILE VERSIONS
V4 2/9/19
  -Settings (Program V1): 
    -Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=500, pFeed=1200, tFeed=5000, tERetract=0, tEComp=0, 
                 fx=pFunct, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -After 1st or 2nd layer printer tryed to travel off into oblivion(-x direction).
      it would have hit the endstop if not stopped. Cause currently unknown
#
V5 2/10/19
  -Settings (Program V1):
    -Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=100, pFeed=1200, tFeed=5000, tERetract=0, tEComp=0, 
                fx=pFunct, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -No first layer adhesion
#
V6 2/10/19
  -Settings (Program V1):
    -Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=100, pFeed=1200, tFeed=5000, tERetract=3, tEComp=0.2, 
                fx=pFunct, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -No first layer adhesion
#
V7-8 2/10/19
  -Settings (Program V1):
    -Slight variations of V5 & V6
  -Notes:
    -No first layer adhesion
#
V9 2/10/19
  -Settings (Program V1):
    -Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=100, pFeed=1200, tFeed=5000, tERetract=0.2, tEComp=0, 
                fx=pFunct, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -Did not extrude for most of the 1st layer, but did stick
  -Minor Program Changes:
    -Modified compensation algorythm to only account for eComp variable
#
V10 2/11/19
  -Settings (Program V2):
    -Polar2GCode(objH=25, objR=10, Θres=100, layerH=0.2, pFeed=500, tFeed=5000, tERetract=0.0457, tEComp=0, 
                fx=fx, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -
  -Minor Program Changes:
    -
#
V11 2/11/19
  -Settings (Program V2):
    -Polar2GCode(objH=25, objR=10, Θres=100, layerH=0.2, pFeed=500, tFeed=5000, tERetract=0.0914, tEComp=0.0457, 
                fx=fx, amp=0.4, freq=8, zPeriod=25)
  -Notes:
    -
  -Minor Program Changes:
    -Fixed Polar2Gcode and G1 functions to give more constant extrusion
    -Added global precision value so gcode has fewer decimal places
      hopefully this can fix the printer trying to run away because that behavior
      is seen mostly with a high Θres (~500) where the coordinates do not change
      as much between steps
#
V11F 2/11/19
  -Settings (Program V2):
    -Polar2GCode(objH=25, objR=10, Θres=100, layerH=0.2, pFeed=500, tFeed=5000, tERetract=0.0914, tEComp=0.0457, 
                fx=fx, amp=0, freq=8, zPeriod=25)
  -Notes:
    -Same as V11 but with 0 amplitude
  -Minor Program Changes:
    -
#


                                                PROGRAM VERSIONS
V0:
  -Version working
  -Issues to fix:
    *Travel moves are too slow
    -Initial travel path crosses first layer causing poor adheasion
    -layer transition is too quick causing noticible seam
  -Planned Changes
    *Add parameters for complete phase in / out of pFunct amplitude
    -Add rectangular function support
    *Add support for more advanced gcode functions
    *Improve parsing algorythm
    *Improve documentation
    *Improve speed using profiling
#
V1:
  -Version working
  -Changelog:
    -Added gcode file log
    -Improved file management functions
    -Improved documentation
    -Append file function now recognises G92 command
#
V2:
  -

V2.1:
  -Version working
  -Changelog:
    -Added pSettings class to better define the printer settings
  -Planned Changes
    -Cleanup gcode.G1 function
    -Add parameters for complete phase in / out of pFunct amplitude 

V2.2:
  -Version working
  -Changelog:
    -
  -Planned Changes
    -
    -

V2.3:
  -Version working
  -Changelog:
    -
  -Planned Changes
    -
    - 

V2.4:
  -Version working
  -Changelog:
    -Added genEProf class
  -Planned Changes
    -
    -
'''

#Imports
import numpy as np
import os
#from printer3D import pSettings, gcode
from Code.printer3D import *  # @UnusedImport
#from visualization2 import visualize
#from Code.gcodeEngine import visualization2

#===========================================================================================================#


def fx(layers, Θ, amp:float, freq:int, zPeriod:float = None, **kwargs):  # @UnusedVariable
    
    '''
    Description:
    
    Keyword Args:
    '''
    
    zPeriod = zPeriod if zPeriod else layers[-1]
    
    layerAmpScalar = np.sin(π*layers/zPeriod)
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #

def fxPhase(layers, Θ, amp:float, freq:float, tsnLayers:int, **kwargs):  # @UnusedVariable
    
    '''
    Description:
    
    Keyword Args:
    '''
    if(tsnLayers>layers.shape[0]/2):
        print(f"\n-------------------------\n"
              f"Warning:\n"
              f"tsnLayers ({tsnLayers}) can be no larger\n"
              f"than ½ of towers total Layers ({layers.shape[0]/2})\n"
              f"value Defaulting to {layers.shape[0]}"
              f"\n-------------------------\n")
        tsnLayers = layers.shape[0]/2
    
    #Sin wave with A = tsnLayers
    layerAmpScalar = np.sin(np.linspace(0,np.pi/2,tsnLayers, endpoint=False))
    #Set middle portion to have constant value             
    layerAmpScalar = np.append(layerAmpScalar, np.append(np.ones(layers.shape[0]-2*tsnLayers),np.flip(layerAmpScalar)))
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #

def fxCosPhase(layers, Θ, amp:float, freq:float, tsnLayers:int, **kwargs):  # @UnusedVariable
    
    '''
    Description:
        Same as fxPhase but rate of change of
        amplitude is continuous
    
    Keyword Args:
    '''
    if(tsnLayers>layers.shape[0]/2):
        print(f"\n-------------------------\n"
              f"Warning:\n"
              f"tsnLayers ({tsnLayers:.2f}) can be no larger\n"
              f"than ½ of towers total Layers ({layers.shape[0]/2:.2f})\n"
              f"value Defaulting to {layers.shape[0]/2:.2f}"
              f"\n-------------------------\n")
        tsnLayers = int(np.floor(layers.shape[0]/2))
    
    layerAmpScalar = (-np.cos(np.linspace(0,np.pi,tsnLayers, endpoint=False))+1)/2
    #Set middle portion to have constant value             
    layerAmpScalar = np.append(layerAmpScalar, np.append(np.ones(layers.shape[0]-2*tsnLayers),np.flip(layerAmpScalar)))
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #
'''
def render():
    #visualize(title = None, is3D=False, thinning=4, multicolor=False, amp=0, fx=fxPhase, objH=25, objR=10, Θres=1000, pSettings=printerSettings, freq=8, tsnLayers=31)
    visualize(fx=lambda layers, Θ, amp, freq: np.sin(freq*Θ)[:, np.newaxis] + layers,
              amp=5, freq=8,# tsnLayers=31,
              objH=25, objR=10, Θres=100, pSettings=printerSettings,
              is3D=False, dspPercent=0.5, thinning=2, multicolor = True,
              title=None, showAxes=False, toolbar=True, maximized=True,
              view=False, size=(1000,1000), linewidth=1, saveAs=False, cleanup=0)
'''
def makeGcode(progress:bool = False):
    
    a = 2
    gcodeFile = gcode(fName=os.path.join(dirname, f"wigglyTube_{int(a*10)}.GCODE"), pSettings = printerSettings, precision = 8)
    
    gcodeFile.appendFile(os.path.join(dirname,"startGcode.txt"))
    
    gcodeFile.Polar2GCode(objH=25, objR=10, Θres=100, progress=progress, eProfMode="hybrid", fx=fxCosPhase, amp=a, freq=8, tsnLayers=31)
    
    gcodeFile.appendFile(os.path.join(dirname,"endGcode.txt"))
    
    gcodeFile.finish()
    gcodeFile.close()
    
    if(progress):
        print(f"Done Slicing {gcodeFile.fNameRaw}!\n")

    if(progress):
        print("Gcode deneration done.\n")

if __name__ == '__main__':
    π = np.pi
    dirname = os.path.dirname(__file__) 
    printerSettings = pSettings(printerName="FT5-R2", eMod=1)
    makeGcode(progress=True)
    #render()
    
    print("Program execution complete,")
    print("Terminating.")
