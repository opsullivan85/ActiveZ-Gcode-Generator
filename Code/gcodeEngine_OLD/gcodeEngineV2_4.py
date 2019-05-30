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
from scipy.spatial import distance as dst
import re
import os
import pendulum
from typing import Tuple, Optional, Callable

π = np.pi

#===========================================================================================================#

class pSettings(object):
    
    """
    Description:
        This is a class to store the settings
        to use for the 3D printer
    """
    
    __slots__ = ['printerName', 'fillamentDia', 'nozzleWidth',
                 'pFeed', 'layerH', 'eMod', 'offsets',
                 'fpFeed', 'fLayerH', 'fEMod', 'fLayerZSquish',
                 'tFeed', 'tRetr', 'tComp', 'tZGap',
                 'eOffsets',
                 'eConst', 'ePerDst']
    #

    settings = {}
    settings['printerName'] =   'Printer name____________________'
    settings['fillamentDia'] =  'Fillament diameter______________'
    settings['nozzleWidth'] =   'Nozzle width____________________'
    settings['pFeed'] =         'Printing feed rate______________'
    settings['layerH'] =        'Layer height____________________'
    settings['eMod'] =          'Extrusion modifier______________'
    settings['offsets'] =       'Offsets_________________________' 
    settings['fpFeed'] =        'First layer printing feed rate__'
    settings['fLayerH'] =       'First layer height______________'
    settings['fEMod'] =         'First layer extrusion modifier__'
    settings['fLayerZSquish'] = 'First layer Z Squish____________'
    settings['tFeed'] =         'Travel feed rate________________'
    settings['tRetr'] =         'Travel retraction_______________'
    settings['tComp'] =         'Travel compensation_____________'
    settings['tZGap'] =         'Travel Z-Gap____________________'
    settings['eOffsets'] =      'End offsets_____________________'
    settings['eConst'] =        '-Extrusion constant_____________'
    settings['ePerDst'] =       '-Extrusion per distance_________'

    def __init__(self, 
                 printerName:str="3D Printer", fillamentDia:float=1.75, nozzleWidth:float=0.4,
                 pFeed:float=500, layerH:float=0.2, eMod:float=1, offsets:Tuple[float,float]=(150,150),
                 fpFeed:float=250, fLayerH:float=0.2, fEMod:float=1.1, fLayerZSquish:float=0,
                 tFeed:float=5000, tRetr:float=0, tComp:float=0.0457, tZGap:float=1,
                 eOffsets:Tuple[float,float]=(10,10)
                 ) -> None:
        
        #Initilize slots to 0
        [self.__setattr__(slot, 0) for slot in self.__slots__]
        
        self.printerName = printerName
        self.fillamentDia = fillamentDia
        self.nozzleWidth = nozzleWidth
        self.pFeed = pFeed
        self.layerH = layerH
        self.eMod = eMod
        self.offsets = offsets
        self.fpFeed = fpFeed
        self.fLayerH = fLayerH
        self.fEMod = fEMod
        self.fLayerZSquish = fLayerZSquish  
        self.tFeed = tFeed
        self.tRetr = tRetr
        self.tComp = tComp
        self.tZGap = tZGap
        self.eOffsets = eOffsets
        
        #Multiply desired extrusion volume by this to get distance of fillament to extrude
        self.eConst = 4/(π*self.fillamentDia**2)
        #Multiply desired extrusion distance by this to get distance of fillament to extrude
        self.ePerDst = self.layerH*self.nozzleWidth*self.eConst
    #
    
    def __str__(self) -> str:
        """
        Description:
            This returns the printer settings as a string
        """
        return "".join(f"{pSettings.settings[setting]}{self.__getattribute__(setting)}\n" for setting in pSettings.settings)
    
#===========================================================================================================#

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
    
    def __init__(self, pSettings:pSettings, fName:str="gcode.GCODE", openfile:bool=True, fileMode:str="a", initfname:Optional[str]=None, header:bool=True, precision:int=5) -> None:  # @UnusedVariable
        self.pSettings = pSettings
        self.precision = precision
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.fName = os.path.join(self.dirname, fName)
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
        """
        Description:
            Opens self.fName, declared in __init__, as self.f
        
        Keyword Args:
            mode (str): Mode the file is to be opened
        """
        
        self.f = open(self.fName, mode)
        self.isOpen = True
    #
    
    def close(self) -> None:
        """
        Description:
            Trys to close self.f
        """
        
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
        """
        Description:
            Clears self.f
        
        Keyword Args:
            closeFile (bool): Determines if file should remain open after clearing
        """
        
        self.open("w")
        if(closefile):
            self.f.close()
    #
    
    def getFLen(self) -> None:
        """
        Description:
            Gets the length of self.f
        """
        
        i = 0
        
        with open(self.fName) as f:
            if f:
                for i, l in enumerate(f):  # @UnusedVariable
                    pass
                
        self.flen = i + 1
    #
    
    def appendFile(self, file2add:str) -> None:
        """
        Description:
            Adds contents of 'file2add' to the end of self.f
        
        Keyword Args:
            file2add (str): The file which is to be appended
        """
        
        with open(os.path.join(self.dirname, file2add)) as f2a:
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
        """
        Description:
            Adds a header to the gcode file including the date and
            printer settings
        """
        
        self.f.write(f"Sliced {pendulum.now().format('ddd, YYYY-M-D h:mm:ss A')}\n\n")
        self.f.write("Printer Settings:\n")
        self.f.write(self.pSettings.__str__())
        self.f.write("\n\n\n")
    #
    
    def Polar2GCode(self, objH:float, objR:float, Θres:int, fx:Callable, eProfMode:str="radialConstant", progress:bool=True, **kwargs) -> None:
        """
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
        """

        Θ = np.linspace(0, 2*π, Θres)
        r = objR
        layers = np.arange(0, objH-self.pSettings.fLayerH, self.pSettings.layerH)
        numLayers = layers.size
        
        #x[Θ]
        x = r*np.cos(Θ) + self.pSettings.offsets[0]
        #y[Θ]
        y = r*np.sin(Θ) + self.pSettings.offsets[1]
        #z[Θ,layer]
        z = fx(layers,Θ,**kwargs)+self.pSettings.fLayerH-self.pSettings.fLayerZSquish
        
        minZ = np.zeros(Θ.shape)
        maxZ = np.full(Θ.shape, z[:,numLayers-1])
        
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

        
        eProf = self.genEProf((x,y,z), eProfMode)
        
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
    
    def genEProf(self, coords, mode:str="radialConstant"):
        """
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
        """
        
        x, y, z = coords
        eProf = np.empty(z.shape)
        dstPerMove = dst.euclidean((x[0], y[0]), (x[1], y[1]))
        
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
                eProf[i[0]+1, i[1]+1] = ePerZGap * (z[i[0],i[1]+1]-z[i[0],i[1]]+z[i[0]+1,i[1]+1]-z[i[0]+1,i[1]])/2 * self.pSettings.eMod
            
            #Prevents negative extrusion
            eProf = (eProf>0)*eProf
        return(eProf)
    #

    def G1Auto(self, coords, setFeed:Optional[float]=None, ePerDst:float=0.0333, ePercent:float=1) -> None:
        """
        Description:
            Writes a G1 command to self.f
            Automaticly determines ammount to extrude
            
        Keyword Args:
            coords (float[3]): Coords to move to
            setFeed (float): Feed to change to 
            ePerDst (float): Scalar applied to distance traveled
              to determine the ammount to be extruded
            ePercent (float): Extra ammount as percent to extrude
        """
        
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
        """
        Description:
            Writes a G1 command to self.f
            Automaticly determines ammount to extrude
        
        Keyword Args:
            coords (float[3]): Coords to move to
            extrusion (float): Ammount to extrude
            setFeed (float): Feed to change to 
            ePercent (float): Extra ammount as percent to extrude
        """
        
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
        """
        Description:
            Writes a travel move to self.f
        
        Keyword Args:
            coords (float[3]): Coords to move to
        """
        
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
        """
        Description:
            Writes last commands to self.f
        """
        
        self.f.write(f"\n;Finished Print\n")
        self.f.write(f"G0 F{self.pSettings.tFeed:.{self.precision}f}\n")
        #Raise z and retract
        self.f.write(f"G1 Z{(self.coords[2]+self.pSettings.zGap):.{self.precision}f} E{(self.coords[4]-self.pSettings.tRetr):.{self.precision}f}\n")
        #Goto coords
        self.f.write(f"G0 X{self.pSettings.eOffsets[0]:.{self.precision}f} Y{self.pSettings.eOffsets[1]:.{self.precision}f}\n")
        #Update Coords
        self.coords[0] = self.pSettings.eOffsets[0]
        self.coords[1] = self.pSettings.eOffsets[1]
        self.coords[2] += self.pSettings.zGap
        self.coords[4] -= self.pSettings.tRetr
    #

#===========================================================================================================#


def fx(layers, Θ, amp:float, freq:int, zPeriod:float, **kwargs):  # @UnusedVariable
    
    '''
    Description:
    
    Keyword Args:
    '''
    
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
              f"Runtime Error\n"
              f"\"tsnLayers\" can be no larger\n"
              f"than ½ of towers total Layers ({layers.shape[0]/2:.2f})\n"
              f"Terminating Program..."
              f"\n-------------------------\n")
        quit()
    layerAmpScalar = np.sin(np.linspace(0,np.pi/2,tsnLayers, endpoint=False))
    layerAmpScalar = np.append(layerAmpScalar, np.append(np.ones(layers.shape[0]-2*tsnLayers),np.flip(layerAmpScalar)))
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #
    

def main(output:bool = False):
    printerSettings = pSettings(printerName="FT5-R2", eMod=1)
    
    a = 6
    tmp = gcode(fName=f"wigglyTube_{int(a*10)}.GCODE", pSettings = printerSettings, fileMode="w", precision = 8)
    tmp.appendFile("startGcode.txt")
    tmp.Polar2GCode(objH=25, objR=10, Θres=1000, progress=output, eProfMode="avgZGap",
                    fx=fxPhase, amp=a, freq=8, tsnLayers=31)
    tmp.appendFile("endGcode.txt")
    tmp.close()
    if(output):
        print(f"Done Slicing {tmp.fNameRaw}!\n")

    if(output):
        print("Gcode Done!\n")

main(output=True)

