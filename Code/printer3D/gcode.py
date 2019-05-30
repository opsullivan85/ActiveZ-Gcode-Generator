'''
Created on Apr 29, 2019

@author: Owen
'''

'''
TODO:
  refactor    
    move Polar2Gcode to gcodeEngine?
'''


import numpy as np
from scipy.spatial import distance as dst
import re
#import os
import pendulum
from typing import Optional, Callable
from Code.printer3D import pSettings  # @UnusedImport
π = np.pi

def cleanupPath(z):
    numAngles = z.shape[0]
    numLayers = z.shape[1]
    
    minZ = np.zeros(numAngles)
    maxZ = np.full(numAngles, z[:,numLayers-1])
    
    #Make sure printer doesnt try to break through bed
    #Or other places it has already been
    for i in np.ndindex(z.shape):
        a, l = i
        if(l < numLayers-1):
            if z[i] >= minZ[a]:
                minZ[a] = z[i]
            else:
                z[i] = minZ[a]
                
    for i in np.ndindex(z.shape):
        a, l = i
        if(l):
            if z[a,numLayers-l] <= maxZ[a]:
                maxZ[a] = z[a,numLayers-l]
            else:
                z[a,numLayers-l] = maxZ[a]
#

class gcode:
    
    regex = (
            r"^G(1|92|0)\s*?"# For 1. G1 | G92 Command:
            r"(?:"# Group all the following
            r"(?:X(-?\d*(?:\.\d*)?)\s*)|"# 2. X value
            r"(?:Y(-?\d*(?:\.\d*)?)\s*)|"# 3. Y value
            r"(?:Z(-?\d*(?:\.\d*)?)\s*)|"# 4. Z value
            r"(?:F(-?\d*(?:\.\d*)?)\s*)|"# 5. F value
            r"(?:E(-?\d*(?:\.\d*)?)\s*)"#  6. E value
            r")+"
            )
    #
    
    def __init__(self, pSettings:pSettings, fName:str="gcode.GCODE", openfile:bool=True, fileMode:str="w", initfname:Optional[str]=None, header:bool=True, precision:int=5) -> None:  # @UnusedVariable
        self.pSettings = pSettings
        self.precision = precision
        #self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.fName = fName#os.path.join(self.dirname, fName)
        self.fNameRaw = fName
        self.fMode = fileMode
        self.coords = [0,0,0,0,0] # 0X 1Y 2Z 3F 4E
        self.isOpen = openfile
        
        if(openfile):
            self.open(fileMode)
        
        if(initfname):
            self.appendFile(initfname)
            
        if(header):
            self.writeHeader()
    #
    
    def open(self, mode:str="a") -> None:
        '''
        Description:
            Opens self.fName, declared in __init__, as self.f
        
        Keyword Args:
            mode (str): Mode the file is to be opened
        '''
        
        self.f = open(self.fName, mode)
        self.isOpen = True
    #
    
    def close(self) -> None:
        '''
        Description:
            Trys to close self.f
        '''
        
        try:
            self.f.flush()
            self.f.close()
            self.isOpen = False
        except AttributeError:
            print(f"\n-------------------------\n"
                  f"Attribute Error:\n"
                  f"File is not currently open\n"
                  f"Continuing execution"
                  f"\n-------------------------\n")
    #
    
    def clr(self, closefile:bool=False) -> None:
        '''
        Description:
            Clears self.f
        
        Keyword Args:
            closeFile (bool): Determines if file should remain open after clearing
        '''
        
        self.open("w")
        if(closefile):
            self.f.close()
    #
    
    def getFLen(self) -> None:
        '''
        Description:
            Gets the length of self.f
        '''
        
        i = 0
        
        with open(self.fName) as f:
            if f:
                for i, l in enumerate(f):  # @UnusedVariable
                    pass
                
        self.flen = i + 1
    #
    
    def appendFile(self, file2add:str) -> None:
        '''
        Description:
            Adds contents of 'file2add' to the end of self.f
        
        Keyword Args:
            file2add (str): The file which is to be appended
        '''
        
        with open(file2add) as f2a:
            #Scan through file2add and populates matches with the matches as floats
            matches = [[float(g) if g else None for g in s] for s in re.findall(gcode.regex, f2a.read(), re.MULTILINE)]  # @UndefinedVariable
            
            #Udpates the coords
            for match in matches:
                for i, dim in enumerate(match[1:]):
                    if dim != None:
                        self.coords[i] = dim
            
            #Writes the file
            f2a.seek(0)
            self.f.write(f2a.read())
            self.f.write("\n")
    #
    
    def writeHeader(self) -> None:
        '''
        Description:
            Adds a header to the gcode file including the date and
            printer settings
        '''
        
        self.f.write(f"Sliced {pendulum.now().format('ddd, YYYY-M-D h:mm:ss A')}\n\n")
        self.f.write("Printer Settings:\n")
        self.f.write(self.pSettings.__str__())
        self.f.write("\n\n\n")
    #
    
    def Polar2GCode(self, objH:float, objR:float, Θres:int, fx:Callable, eProfMode:str="radialConstant", progress:bool=True, **kwargs) -> None:
        '''
        Description:
            Generates gcode for the polar function given and writes it to self.f
        
        Keyword Args:
            objH (float): Height of object
            objR (float): Raduis of object
            fx (function): Function to generate z values of the object 
              based on the layer and angle values. Should return a 2D npArray
              with z values indexed by Θ then layers.
                Keyword Args:
                    layers (float[]): npArray of all layer heights
                    Θ (float []): npArray of all angle values
                    kwargs: Function specific
            progress (bool): Whether or not show generation progress
        '''

        Θ = np.linspace(0, 2*π, Θres) - π/2
        r = objR
        layers = np.arange(0, objH-self.pSettings.fLayerH, self.pSettings.layerH)
        numLayers = layers.size
        
        #x[Θ]
        x = r*np.cos(Θ) + self.pSettings.offsets[0]
        #y[Θ]
        y = r*np.sin(Θ) + self.pSettings.offsets[1]
        #z[Θ,layer]
        z = fx(layers,Θ,**kwargs)+self.pSettings.fLayerH-self.pSettings.fLayerZSquish
        
        cleanupPath(z)
        
        eProf = self.genEProf((x,y,z), eProfMode, **kwargs)
        
        #Set feed
        self.f.write("\n;Setting Feed\n")
        self.coords[3] = self.pSettings.pFeed
        self.f.write(f"G1 F{self.pSettings.pFeed:.{self.precision}f}\n")
        
        for l, layer in enumerate(layers):  # @UnusedVariable
            
            #Comment Gcode and possible print progress to console
            self.f.write(f"\n;Layer {l+1}\n")
            self.f.write(f"M117 {l+1}/{numLayers}\n")
            if(progress):
                print(f"Layer: {l+1}/{numLayers}")
            
            #Write all moves in layer
            for a, angle in enumerate(Θ):  # @UnusedVariable
                self.G1((x[a],y[a],z[a,l]), extrusion=eProf[a,l])
    #

    def genEProf(self, coords, mode:str="radialConstant", **kwargs):
        '''
        Description:
            Calculates the ammount of material which needs to be extruded
            along each point of the path
        
        Keyword Args:
            coords (x float[Θ], y float[Θ], z float[Θ][l]): The movement
              path the printer takes
            mode (String):
                radialConstant: Extrusion is constant and based on x/y distance
                avgZGap: Extrusion is determined by the ammount of open space
                  below the current and next positions of the nozzle
                hybrid: avgZGap for tsnLayers and radialConstant
                  for normal layers
        '''
        x, y, z = coords
        eProf = np.empty(z.shape)
        dstPerMove = dst.euclidean((x[0], y[0]), (x[1], y[1]))

        '''
        Note: eprof[a,l] is the ammount of material which needs
          to be extruded along the path to eprof[a,l]
        '''
        
        if(mode == "radialConstant"):
            #Disable extrusion to first position of layer
            eProf[0,:] = 0
            
            #Extrusion for first layer
            eProf[1:, 0] = dstPerMove * self.pSettings.ePerDst * self.pSettings.fEMod
            
            #Extrusion for normal layers
            eProf[1:, 1:] = dstPerMove * self.pSettings.ePerDst * self.pSettings.eMod
            
        elif(mode == "avgZGap"):
            #Multiply the height of the desired extrusion rectangle by this to find the need extrusion
            ePerZGap = dstPerMove * self.pSettings.nozzleWidth * self.pSettings.eConst
            
            #Disable extrusion for first position of all layers
            eProf[0,:] = 0
            
            #Normal extrusion for first layer
            eProf[1:, 0] = dstPerMove * self.pSettings.ePerDst * self.pSettings.fEMod
            
            #Generates the bulk of the eProf
            for i, val in np.ndenumerate(eProf[1:, 1:]):  # @UnusedVariable
                a, l = i
                #1 is added to compensate for starting at [1,1]
                a += 1
                l += 1
                h = z[a, l] #Here
                bh = z[a, l-1] #Below Here
                p = z[a-1, l] #Previous
                bp = z[a-1, l-1] #Below Previous
                eProf[i[0]+1, i[1]+1] = ePerZGap * (h-bh+p-bp)/2 * self.pSettings.eMod
            
            #Prevents negative extrusion
            eProf = (eProf>0)*eProf
        
        elif(mode == "hybrid"):
            #Multiply the height of the desired extrusion rectangle by this to find the need extrusion
            ePerZGap = dstPerMove * self.pSettings.nozzleWidth * self.pSettings.eConst
            
            #Disable extrusion for first position of all layers
            eProf[0,:] = 0
            
            #Normal extrusion for first layer
            eProf[1:, 0] = dstPerMove * self.pSettings.ePerDst * self.pSettings.fEMod
            
            #Generates the bulk of the eProf
            for i, val in np.ndenumerate(eProf[1:, 1:]):  # @UnusedVariable
                a, l = i
                #1 is added to compensate for starting at [1,1]
                a += 1
                l += 1
                
                #radialConstant
                if(l >= kwargs['tsnLayers'] and l < z.shape[1]-kwargs['tsnLayers']):
                    eProf[a,l]=dstPerMove * self.pSettings.ePerDst * self.pSettings.eMod
                    
                #avgZGap
                else:
                    h = z[a, l] #Here
                    bh = z[a, l-1] #Below Here
                    p = z[a-1, l] #Previous
                    bp = z[a-1, l-1] #Below Previous
                    eProf[i[0]+1, i[1]+1] = ePerZGap * (h-bh+p-bp)/2 * self.pSettings.eMod
            
            #Prevents negative extrusion
            eProf = (eProf>0)*eProf
            
        return(eProf)
    #

    def G1Auto(self, coords, setFeed:Optional[float]=None, ePerDst:float=0.0333, ePercent:float=1) -> None:
        '''
        Description:
            Writes a G1 command to self.f
            Automaticly determines ammount to extrude
            
        Keyword Args:
            coords (float[3]): Coords to move to
            setFeed (float): Feed to change to 
            ePerDst (float): Scalar applied to distance traveled
              to determine the ammount to be extruded
            ePercent (float): Extra ammount as percent to extrude
        '''
        
        #Calculate ammount to be extruded
        self.coords[4] += ePercent*dst.euclidean(coords, (self.coords[0],self.coords[1],self.coords[2]))*ePerDst
        if(self.coords[2] != coords[2]):
            if(setFeed):
                self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{coords[2]:.{self.precision}f} F{setFeed:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
                self.coords[3] = setFeed
            else:
                self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{coords[2]:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
            self.coords[2] = coords[2]
        else:
            if(setFeed):
                self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} F{setFeed:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
                self.coords[3] = setFeed
            else:
                self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
        #Update coords
        self.coords[0] = coords[0]
        self.coords[1] = coords[1]
    #

    def G1(self, coords, extrusion:float, setFeed:Optional[float]=None, ePercent:float=1) -> None:
        '''
        Description:
            Writes a G1 command to self.f
            Automaticly determines ammount to extrude
        
        Keyword Args:
            coords (float[3]): Coords to move to
            extrusion (float): Ammount to extrude
            setFeed (float): Feed to change to 
            ePercent (float): Extra ammount as percent to extrude
        '''
        
        #Calculate ammount to be extruded
        self.coords[4] += ePercent*extrusion
        if(setFeed):
            self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{coords[2]:.{self.precision}f} F{setFeed:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
            self.coords[3] = setFeed
        else:
            self.f.write(f"G1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{coords[2]:.{self.precision}f} E{self.coords[4]:.{self.precision}f}\n")
        #Update coords
        self.coords[0] = coords[0]
        self.coords[1] = coords[1]
        self.coords[2] = coords[2]
    #

    def travelMove(self, coords) -> None:   # @UnusedVariable
        '''
        Description:
            Writes a travel move to self.f
        
        Keyword Args:
            coords (float[3]): Coords to move to
        '''
        
        #Set feed
        self.f.write(f"\n;Travel Move\n")
        self.f.write(f"G0 F{self.pSettings.tFeed:.{self.precision}f}\n")
        #Raise z and retract
        self.f.write(f"G1 Z{(self.coords[2]+self.pSettings.tZGap):.{self.precision}f} E{(self.coords[4]-self.pSettings.tRetr):.{self.precision}f}\n")
        #Goto coords
        self.f.write(f"G0 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{(coords[2]+self.pSettings.tZGap):.{self.precision}f}\n")
        #Lower z
        self.f.write(f"G0 Z{coords[2]:.{self.precision}f}\n")
        #Compensate and unretract
        self.f.write(f"G1 E{(self.coords[4]+self.pSettings.tComp):.{self.precision}f}\n")
        #Reset feed
        self.f.write(f"G0 F{self.coords[3]:.{self.precision}f}\n")
        #Update Coords
        self.coords[0] = coords[0]
        self.coords[1] = coords[1]
        self.coords[2] = coords[2]
        self.coords[4] += self.pSettings.tComp
    #
    
    def finish(self) -> None:
        '''
        Description:
            Writes last commands to self.f
        '''
        
        self.f.write(f"\n;Finished Print\n")
        self.f.write(f"G0 F{self.pSettings.tFeed:.{self.precision}f}\n")
        #Raise z and retract
        self.f.write(f"G1 Z{(self.coords[2]+self.pSettings.tZGap):.{self.precision}f} E{(self.coords[4]-self.pSettings.tRetr):.{self.precision}f}\n")
        #Goto coords
        self.f.write(f"G0 X{self.pSettings.eOffsets[0]:.{self.precision}f} Y{self.pSettings.eOffsets[1]:.{self.precision}f}\n")
        #Update Coords
        self.coords[0] = self.pSettings.eOffsets[0]
        self.coords[1] = self.pSettings.eOffsets[1]
        self.coords[2] += self.pSettings.tZGap
        self.coords[4] -= self.pSettings.tRetr
    #