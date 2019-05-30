'''
V0:
  -Version working
  -Issues to fix:
    -Travel moves are too slow
    -Initial travel path crosses first layer causing poor adheasion
    -layer transition is too quick causing noticible seam
  -Planned Changes
    -Add parameters for complete phase in / out of pFunct amplitude
    -Add rectangular function support
    -Add support for more advanced gcode functions
    -Improve parsing algorythm
    -Improve documentation
    -Improve speed using profiling
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
    
    def __init__(self, fname = "gcode.GCODE", initfname = False):
        self.fname = fname
        self.coords = [0,0,0,0,0] # 0X 1Y 2Z 3F 4E
        
        if(initfname):
            self.appendFile(initfname)

    def getFLen(self):
        i = 0
        
        with open(self.fname) as f:
            if f:
                for i, l in enumerate(f):  # @UnusedVariable
                    pass
                
        self.flen = i + 1

    def appendFile(self, file2add):
        with open(file2add) as f2a:
            with open(self.fname, "a") as f:
                #TODO explain this garbage
                matches = [[float(g) if g else NULL for g in s] for s in re.findall(gcode.regex, f2a.read(), re.MULTILINE)]  # @UndefinedVariable
                
                for match in matches:
                    for i, dim in enumerate(match[1:]):
                        if dim:
                            self.coords[i-1] = dim
#                     elif(match[0]=92):
#                         for i, dim in enumerate(match[1:]):
#                             if dim:
#                                 self.coords[i] = dim
                f2a.seek(0)
                f.write(f2a.read())

    def clr(self):
        f = open(self.fname, "w")
        f.close()

    def Polar2GCode(self, layerH, objH, objR, Θres, feed, fx, **kwargs):
        
        layers = np.arange(0, objH, layerH)
        numLayers = layers.size
        Θ = np.linspace(0, 2*π, Θres)
        x = objR * np.sin(Θ)
        y = objR * np.cos(Θ)
        

        with open(self.fname, "a") as f:
            #Make sure feedrate is correctly set
            if(self.coords[3] != feed):
                f.write(f"\n\nG1 F{feed}\n")
                self.coords[3] = feed
                
            for l, layer in enumerate(layers):
                #Comment layer # in gcode
                f.write(f"\n;Layer: {l+1}/{numLayers}\n")
                f.write(f"M117 {l+1}/{numLayers}\n")
                #Disable extrusion for first move of each layer
                travelMove = True
                
                for a, angle in enumerate(Θ):
                    #Grab z val
                    z = fx(layer, angle, **kwargs)
                    
                    if(not(travelMove)):
                        #Set extrusion coords
                        self.coords[4] += distance.euclidean((x[a],y[a],z), (self.coords[0],self.coords[1],self.coords[2]))*0.03
                        #Write gcode
                        f.write(f"G1 X{x[a]+150} Y{y[a]+150} Z{z} E{self.coords[4]}\n")
                    else:
                        f.write(f"G0 X{x[a]+150} Y{y[a]+150} Z{z}\n")
                        travelMove = False
                    #Update coords
                    self.coords[0] = x[a]
                    self.coords[1] = y[a]
                    self.coords[2] = z
                
                print(f"Layer: {l+1}/{numLayers}")

#===========================================================================================================#
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

tmp = gcode("wigglyTube.GCODE")
tmp.clr()
tmp.appendFile("startGcode.txt")
tmp.getFLen()
print(tmp.flen)
tmp.Polar2GCode(layerH=0.2, objH=25, objR=10, Θres=100, feed=300, fx=pFunct, amp=0.2, freq=10, zPeriod=25)
#tmp.getFLen()
#print(tmp.flen)