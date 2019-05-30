'''  GCODE FILE VERSIONS
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
'''
#===========================================================================================================#
'''  PROGRAM VERSIONS
V0:
  -Version working
  -Issues to fix:
    *Travel moves are too slow
    -Initial travel path crosses first layer causing poor adheasion
    -layer transition is too quick causing noticible seam
  -Planned Changes
    -Add parameters for complete phase in / out of pFunct amplitude
    -Add rectangular function support
    *Add support for more advanced gcode functions
    *Improve parsing algorythm
    *Improve documentation
    -Improve speed using profiling
#
V1:
  -Version working
  -Changelog:
    -Added gcode file log
    -Improved file management functions
    -Improved documentation
    -Append file function now recognises G92 command
'''

#Error Codes
#https://www.random.org/integers/?num=100&min=1000&max=65535&col=10&base=16&format=html&rnd=id.ErrorCodes

#Imports
import numpy as np
from scipy.spatial import distance
import re
from _overlapped import NULL

π = np.pi
#Θ

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
#     regex = r"^G1\s*?(?:(?:X(-?\d*(?:\.\d*)?)\s*)|(?:Y(-?\d*(?:\.\d*)?)\s*)|(?:Z(-?\d*(?:\.\d*)?)\s*)|(?:F(-?\d*(?:\.\d*)?)\s*)|(?:E(-?\d*(?:\.\d*)?)\s*))+"
    
    def __init__(self, fname = "gcode.GCODE", openfile = True, fileMode = "a", initfname = False):
        self.fname = fname
        self.fmode = fileMode
        self.coords = [0,0,0,0,0] # 0X 1Y 2Z 3F 4E
        self.isOpen = openfile
        
        if(openfile):
            self.open(fileMode)
        
        if(initfname):
            self.appendFile(initfname)
    
    def open(self, mode = "a"):
        self.f = open(self.fname, mode)
    
    def close(self):
        self.f.flush()
        self.f.close()
    
    def clr(self, closefile = False):
        self.open("w")
        self.f.close()
        if(not(closefile)):
            self.open(self.fmode)
    
    def getFLen(self):
        i = 0
        
        with open(self.fname) as f:
            if f:
                for i, l in enumerate(f):  # @UnusedVariable
                    pass
                
        self.flen = i + 1

    def appendFile(self, file2add):
        with open(file2add) as f2a:
            #TODO explain this garbage
            matches = [[float(g) if g else NULL for g in s] for s in re.findall(gcode.regex, f2a.read(), re.MULTILINE)]  # @UndefinedVariable
            
            for match in matches:
                for i, dim in enumerate(match[1:]):
                    if dim:
                        self.coords[i-1] = dim
            
            f2a.seek(0)
            self.f.write(f2a.read())

    def Polar2GCode(self, layerH, objH, objR, Θres, pFeed, tFeed, fx, flayerH=NULL, tZDst=10, tERetract=3, tEComp=0.25, offset=(150,150), **kwargs):
        self.f.write(f"\n\n;Set first layer height\n")
        if(flayerH):
            self.f.write(f"G0 Z{flayerH}\n")
            self.f.write(f"G92 Z0\n")
        else:
            self.f.write(f"G0 Z{layerH}\n")
            self.f.write(f"G92 Z0\n")
            
        layers = np.arange(0, objH, layerH)
        numLayers = layers.size
        Θ = np.linspace(0, 2*π, Θres)
        x = objR * np.sin(Θ)+offset[0]
        y = objR * np.cos(Θ)+offset[1]
        extrusion = True
        
        #Travel to beginning of print
        self.travelMove((x[0],y[0], fx(layers[0], Θ[0], **kwargs)), tFeed, tZDst, tERetract, tEComp)
        
        #Make sure feedrate is correctly set
        if(self.coords[3] != pFeed):
            self.f.write(f"\n\nG0 F{pFeed}\n")
            self.coords[3] = pFeed
            
        for l, layer in enumerate(layers):
            #Comment layer # in gcode and push to LCD
            self.f.write(f"\n;Layer: {l+1}/{numLayers}\n")
            self.f.write(f"M117 {l+1}/{numLayers}\n")
            #Disable extrusion for first move of each layer
            if(l > 0):
                extrusion = False
            
            for a, angle in enumerate(Θ):
                #Grab z val
                z = fx(layer, angle, **kwargs)
                if(extrusion):
                    #Set extrusion coords
                    self.coords[4] += distance.euclidean((x[a],y[a],z), (self.coords[0],self.coords[1],self.coords[2]))*0.03
                    #Write gcode
                    self.f.write(f"G1 X{x[a]} Y{y[a]} Z{z} E{self.coords[4]}\n")
                else:
                    self.f.write(f"G0 X{x[a]} Y{y[a]} Z{z}\n")
                    extrusion = True
                #Update coords
                self.coords[0] = x[a]
                self.coords[1] = y[a]
                self.coords[2] = z
            
            print(f"Layer: {l+1}/{numLayers}")
    
    def travelMove(self, coords, tFeed=3000, zDst=3, eRetract=1, eComp=0.25):
        self.f.write("\n;Travel Move\n")
        #Raise head and retract
        self.f.write(f"G1 Z{coords[2]+zDst} F{tFeed} E{self.coords[4]-eRetract}\n")
        #Move X / Y
        self.f.write(f"G1 X{coords[0]} Y{coords[1]}\n")
        #Lower head, unretract, compensate fillament loss, and reset feed
        self.f.write(f"G1 Z{coords[2]} E{self.coords[4]+eComp}\n")
        self.f.write(f"G1 F{self.coords[3]}\n")
        #Update self coords
        self.coords[0] = coords[0]
        self.coords[1] = coords[1]
        self.coords[2] = coords[2]
        #self.coords[4] += eComp-eRetract
        

#===========================================================================================================#
#Origional function used:
#z = [AMP * np.sin(FREQ*Θ) * np.sin(π*layer/H) + layer for layer in layers]
def pFunct(layer, Θ, amp, freq, zPeriod):
    try:
        return(amp * np.sin(freq*Θ) * np.sin(π*layer/zPeriod) + layer)
    except KeyError as err:
        print(f"\n-------------------------\n"
              f"Runtime Error #f7fc\n"
              f"{err} Argument Missing\n"
              f"Terminating Program..."
              f"\n-------------------------\n")
        quit()

#===========================================================================================================#

def main():
    tmp = gcode("wigglyTube.GCODE", fileMode="w", initfname="startGcode.txt",)
    tmp.Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=100, pFeed=1200, tFeed=5000, tERetract=0.2, tEComp=0, 
                    fx=pFunct, amp=0.4, freq=8, zPeriod=25)
    tmp.getFLen()
    print(tmp.flen)
    tmp.close()