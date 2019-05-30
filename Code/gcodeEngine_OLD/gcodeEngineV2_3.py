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
'''

#Imports
import numpy as np
from scipy.spatial import distance as dst
import re
import os
import pendulum

#Θ
π = np.pi

#===========================================================================================================#

class pSettings(object):
    __slots__ = ['printerName', 'fillamentDia', 'nozzleWidth',
                 'pFeed', 'layerH', 'eMod', 'offsets',
                 'fpFeed', 'fLayerH', 'fEMod', 'fLayerZSquish',
                 'tFeed', 'tRetr', 'tComp', 'tZGap',
                 'eOffsets',
                 'eConst', 'ePerDst']

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
                 printerName="3D Printer", fillamentDia=1.75, nozzleWidth=0.4,
                 pFeed=500, layerH=0.2, eMod=1, offsets=(150,150),
                 fpFeed=250, fLayerH=0.2, fEMod=1.1, fLayerZSquish = 0,
                 tFeed=5000, tRetr=0, tComp=0.0457, tZGap=1,
                 eOffsets=(10,10)
                 ):
        
        #Set slots to 0
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
        
        self.eConst = 4/(π*self.fillamentDia**2)
        #Multiply desired extrusion volume by this to get distance of fillament to extrude
        self.ePerDst = self.layerH*self.nozzleWidth*self.eConst
        #Multiply desired extrusion distance by this to get distance of fillament to extrude
    
    def __str__(self):
        #return "".join(f"{name} = {self.__getattribute__(name)}\n" for name in self.__slots__)
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
    def __init__(self, pSettings, fName="gcode.GCODE", openfile=True, fileMode="a", initfname=None, header=True, precision = 5):  # @UnusedVariable
        self.pSettings = pSettings
        self.precision = precision
        self.dirname = os.path.dirname(os.path.realpath(__file__))
        self.fName = os.path.join(self.dirname, fName)
        self.fmode = fileMode
        self.coords = [0,0,0,0,0] # 0X 1Y 2Z 3F 4E
        self.isOpen = openfile
        
        if(openfile):
            self.open(fileMode)
        
        if(initfname):
            self.appendFile(initfname)
            
        if(header):
            self.writeHeader()
    #
    
    def open(self, mode = "a"):
        """
        Description:
            Opens self.fName, declared in __init__, as self.f
        
        Keyword Args:
            mode (str): Mode the file is to be opened
        """
        
        self.f = open(self.fName, mode)
        self.isOpen = True
    #
    
    def close(self):
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
    
    def clr(self, closefile=False):
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
    
    def getFLen(self):
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
    
    def appendFile(self, file2add):
        """
        Description:
            Adds contents of 'file2add' to the end of self.f
        
        Keyword Args:
            file2add (str): The file which is to be appended
        """
        
        with open(os.path.join(self.dirname, file2add)) as f2a:
            #TODO explain this garbage
            matches = [[float(g) if g else None for g in s] for s in re.findall(gcode.regex, f2a.read(), re.MULTILINE)]  # @UndefinedVariable
            
            for match in matches:
                for i, dim in enumerate(match[1:]):
                    if dim != None:
                        self.coords[i] = dim
            f2a.seek(0)
            self.f.write(f2a.read())
            self.f.write("\n")
    #
    
    def writeHeader(self):
        self.f.write(f"Sliced {pendulum.now().format('ddd, YYYY-M-D h:mm:ss A')}\n\n")
        self.f.write("Printer Settings:\n")
        self.f.write(self.pSettings.__str__())
        self.f.write("\n\n\n")
    
    def Polar2GCode(self, objH, objR, Θres, fx, progress=True, experimental = False, **kwargs):

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
        
        ##fLayerH = layerH if fLayerH == None else fLayerH
        Θ = np.linspace(0, 2*π, Θres)
        r = objR
        layers = np.arange(0, objH-self.pSettings.fLayerH, self.pSettings.layerH)
        numLayers = layers.size
        
        if(experimental):
            #Experimental fillament extrusion profile
            #This is 'ePerMove' with the height component removed. Multiply this by the height
            #    required to be extruded to find ePerMove
            #areaPerMove is multiplied by the average height of trapezoid formed between 
            #    the current and previous z coordinates of the current and previous layers
            #    this determines the volume of fillament to be extruded for each move based
            #    on the space immedietly below it
            areaPerMove = 2*π*r*self.pSettings.eConst*self.pSettings.nozzleWidth/Θ.size
        else:
            #Uncomment this when done testing
            ePerMove = 2*π*r*self.pSettings.ePerDst/Θ.size
        
        x = r*np.cos(Θ) + self.pSettings.offsets[0]
        #x[Θ]
        y = r*np.sin(Θ) + self.pSettings.offsets[1]
        #y[Θ]
        z = fx(layers,Θ,**kwargs)+self.pSettings.fLayerH-self.pSettings.fLayerZSquish
        #z[Θ,layer]
        
        #Set feed
        self.f.write("\n;Setting Feed\n")
        self.coords[3] = self.pSettings.pFeed
        self.f.write(f"G1 F{self.pSettings.pFeed:.{self.precision}f}\n")
        
        self.travelMove((x[0],y[0],z[0,0]))
        
        if(experimental):
            if 'tsnLayers' in kwargs and kwargs['tsnLayers'] < numLayers/2:
                
                #Generate eProf
                eProf = np.full(z.shape, self.pSettings.layerH)
                for i, zVal in np.ndenumerate(z):  # @UnusedVariable
                    eProf[i] = areaPerMove * (z[i[0]-1,i[1]]-z[i[0]-1,i[1]-1]+z[i[0],i[1]]-z[i[0],i[1]-1])/2
                eProf[:,0] = areaPerMove * self.pSettings.fLayerH
                eProf = (eProf>0)*eProf
                
                #First Layer
                for a, angle in enumerate(Θ):  # @UnusedVariable
                    self.f.write(f"G1 X{x[a]} Y{y[a]} Z{[z[a,0]]}")
                    #self.G1((x[a],y[a],z[a,0]), extrusion=eProf[a,0], ePercent=self.pSettings.fEMod)
                
                #Other Layers
                for l, layer in enumerate(layers[1:], start=1):  # @UnusedVariable
                    
                    #Comment Gcode and possible print progress to console
                    self.f.write(f"\n;Layer {l+1}\n")
                    self.f.write(f"M117 {l+1}/{numLayers}\n")
                    if(progress):
                        print(f"Layer: {l+1}/{numLayers}")
                    
                    #Travel move to next layer
                    self.f.write(f"G0 Z{z[0,l]:.{self.precision}f}\n")
                    self.coords[2]=z[0,l]
                    
                    #Moves within layer
                    for a, angle in enumerate(Θ[1:], start=1):  # @UnusedVariable
                        self.G1((x[a], y[a], z[a,l]), extrusion=eProf[a,l], ePercent=self.pSettings.eMod)
                        
            else:
                print(f"\n-------------------------\n"
                      f"Runtime Error\n"
                      f"\"tsnLayers\" either does not exist"
                      f"or it is more than ½ the print's layers"
                      f"Terminating Program..."
                      f"\n-------------------------\n")
                quit()

        else:
            #First Layer
            for a, angle in enumerate(Θ):  # @UnusedVariable
                self.G1((x[a],y[a],z[a,0]), extrusion=ePerMove, ePercent=self.pSettings.fEMod)
            
            #Other Layers
            for l, layer in enumerate(layers[1:], start=1):  # @UnusedVariable
                
                #Comment Gcode and possible print progress to console
                self.f.write(f"\n;Layer {l+1}\n")
                self.f.write(f"M117 {l+1}/{numLayers}\n")
                if(progress):
                    print(f"Layer: {l+1}/{numLayers}")
                
                #Write travel Move to next layer
                self.f.write(f"G0 Z{z[0,l]:.{self.precision}f}\n")
                self.coords[2]=z[0,l]
                
                #Write all moves in layer
                for a, angle in enumerate(Θ[1:], start=1):  # @UnusedVariable
                    self.G1((x[a],y[a],z[a,l]), extrusion=ePerMove, ePercent=self.pSettings.eMod)
    #
    
    #eConst 4/(π*self.fillamentDia**2) = 0.4158
    #ePerDst 1*0.4*0.2 * eConst = 0.0333
        #Distance to extrude / Distance of material to print
    
    def G1Auto(self, coords, setFeed=None, ePerDst = 0.0333, ePercent=1):
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

    def G1(self, coords, extrusion, setFeed=None, ePercent=1):
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
            self.f.write(f"Ge1 X{coords[0]:.{self.precision}f} Y{coords[1]:.{self.precision}f} Z{coords[2]:.{self.precision}f} E{self.coords[4]:.{self.precision}f} ;{ePercent*extrusion:.4f}\n")
        #Update coords
        self.coords[0] = coords[0]
        self.coords[1] = coords[1]
        self.coords[2] = coords[2]
    #

    def travelMove(self, coords):   # @UnusedVariable
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
    
    def finish(self):  # @UnusedVariable
        """
        Description:
            Writes last commands to self.f
        
        Keyword Args:
            zGap (float): Ammount to raise Z
            retr (float): Ammount to retract
            endOffsets (float[2]): Position to return to
            self.pSettings.tFeed (float): Feed to move at
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


def fx(layers, Θ, amp, freq, zPeriod, **kwargs):  # @UnusedVariable
    '''
        0,1,2,3,... Layer
    0  [z,z,z,z,...]
    1  [z,z,z,z,...]
    2  [z,z,z,z,...]
    3  [z,z,z,z,...]
    ...[...........]
    Θ
    '''
    layerAmpScalar = np.sin(π*layers/zPeriod)
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #

def fxPhase(layers, Θ, amp, freq, tsnLayers, **kwargs):
    if(tsnLayers>layers.shape[0]/2):
        print(f"\n-------------------------\n"
              f"Runtime Error\n"
              f"\"tsnLayers\" can be no larger\n"
              f"than ½ of towers total Layers ({layers.shape[0]/2:.2f})\n"
              f"Terminating Program..."
              f"\n-------------------------\n")
        quit()
    '''
        0,1,2,3,... Layer
    0  [z,z,z,z,...]
    1  [z,z,z,z,...]
    2  [z,z,z,z,...]
    3  [z,z,z,z,...]
    ...[...........]
    Θ
    '''
    #np.append(a,np.append(np.full(5,5),np.flip(a)))
    layerAmpScalar = np.sin(np.linspace(0,np.pi/2,tsnLayers, endpoint=False))
    layerAmpScalar = np.append(layerAmpScalar, np.append(np.ones(layers.shape[0]-2*tsnLayers),np.flip(layerAmpScalar)))
    baseSineWave = np.sin(freq*Θ)[:, np.newaxis]
    return(amp*layerAmpScalar*baseSineWave+layers)
    #
    

def main(output = False):
    printerSettings = pSettings(printerName="FT5-R2")
    a=4
    tmp = gcode(fName=f"wigglyTube2_{int(a*10)}.GCODE", pSettings = printerSettings, fileMode="w", precision = 8)
    tmp.appendFile("startGcode.txt")
    tmp.Polar2GCode(objH=25, objR=10, Θres=1000, progress=output, experimental=True,
                    fx=fxPhase, amp=a, freq=8, tsnLayers=31)
                    #fx=fx, amp=0, freq=8, zPeriod=25, phaseIO=5)
    tmp.appendFile("endGcode.txt")
    tmp.close()
    #if(output):
    print(f"Done {int(a*10)}!\n")
    #

    #print("File Length: ", tmp.flen)
    print("Gcode Done!\n")
    #

main(False)

