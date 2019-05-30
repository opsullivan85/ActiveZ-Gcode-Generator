
#Imports
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import distance
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from _overlapped import NULL

#Prep
plt.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')

π = np.pi
#Θ

#===========================================================================================================#

class gcode:
    regex = (
            r"^G1\s*?"# For G1 Command:
            r"(?:"# Group all the following
            r"(?:X(-?\d*(?:\.\d*)?)\s*)|"# 1. X value
            r"(?:Y(-?\d*(?:\.\d*)?)\s*)|"# 2. Y value
            r"(?:Z(-?\d*(?:\.\d*)?)\s*)|"# 3. Z value
            r"(?:F(-?\d*(?:\.\d*)?)\s*)|"# 4. F value
            r"(?:E(-?\d*(?:\.\d*)?)\s*)"#  5. E value
            r")+"
            )
#     regex = r"^G1\s*?(?:(?:X(-?\d*(?:\.\d*)?)\s*)|(?:Y(-?\d*(?:\.\d*)?)\s*)|(?:Z(-?\d*(?:\.\d*)?)\s*)|(?:F(-?\d*(?:\.\d*)?)\s*)|(?:E(-?\d*(?:\.\d*)?)\s*))+"
    
    def __init__(self, fname = "gcode.GCODE", initfname = False, **kwargs):
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
            with open(self.fname, "w") as f:
                #TODO explain this garbage
                matches = [[float(g) if g else NULL for g in s] for s in re.findall(gcode.regex, f2a.read(), re.MULTILINE)]  # @UndefinedVariable
                for match in matches:
                    for i, dim in enumerate(match):
                        if dim:
                            self.coords[i] = dim
                f2a.seek(0)
                f.write(f2a.read())

    def clr(self):
        f = open(self.fname, "w")
        f.close()

    def Polar2GCode(self, fx, layerH, objH, res, feed = 1200, **kwargs):
        layers = np.arange(0, objH + layerH, layerH)
        numLayers = layers.size
        Θ = np.linspace(0, π/2, res)
        R = 1
        x = R * np.sin(Θ)
        y = R * np.cos(Θ)
        with open(self.fname, "a") as f:
            if(self.coords[3] != f):
                f.write(f"G1 F{feed}\n")
                self.coords[3] = f
            for l, layer in enumerate(layers):
                f.write(f"\n;Layer: {l+1}/{numLayers}\n")
                for a, angle in enumerate(Θ):
                    z = fx(layer, angle, *kwargs)
                    self.coords[4] += distance.euclidean((x[a],y[a],z), (self.coords[0],self.coords[1],self.coords[2]))* 0.03
                    f.write(f"G1 X{x[a]} Y{y[a]} Z{z} E{self.coords[4]}\n")
                    self.coords[0] = x[a]
                    self.coords[1] = y[a]
                    self.coords[2] = z
                print(f"Layer: {l+1}/{numLayers}")

#===========================================================================================================#

def pFunct(layer, angle, **kwargs):
    return(layer+angle)

#===========================================================================================================#

tmp = gcode("a.GCODE")
tmp.appendFile("gcode.txt")
#print(tmp.flen)
tmp.clr()
tmp.Polar2GCode(pFunct, 2, 1, 10)
tmp.getFLen()
#print(tmp.flen)