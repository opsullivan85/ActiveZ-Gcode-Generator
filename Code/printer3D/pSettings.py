'''
Created on Apr 29, 2019

@author: Owen
'''

from typing import Tuple
import numpy as np
π = np.pi

class pSettings(object):
    
    '''
    Description:
        This is a class to store the settings
        to use for the 3D printer
    '''
    
    __slots__ = ['printerName', 'fillamentDia', 'nozzleWidth',
                 'pFeed', 'layerH', 'eMod', 'offsets',
                 'fpFeed', 'fLayerH', 'fEMod', 'fLayerZSquish',
                 'tFeed', 'tRetr', 'tComp', 'tZGap',
                 'eOffsets',
                 'eConst', 'ePerDst']
    #

    description = {}
    description['printerName'] =   'Printer name____________________'
    description['fillamentDia'] =  'Fillament diameter______________'
    description['nozzleWidth'] =   'Nozzle width____________________'
    description['pFeed'] =         'Printing feed rate______________'
    description['layerH'] =        'Layer height____________________'
    description['eMod'] =          'Extrusion modifier______________'
    description['offsets'] =       'Offsets_________________________' 
    description['fpFeed'] =        'First layer printing feed rate__'
    description['fLayerH'] =       'First layer height______________'
    description['fEMod'] =         'First layer extrusion modifier__'
    description['fLayerZSquish'] = 'First layer Z Squish____________'
    description['tFeed'] =         'Travel feed rate________________'
    description['tRetr'] =         'Travel retraction_______________'
    description['tComp'] =         'Travel compensation_____________'
    description['tZGap'] =         'Travel Z-Gap____________________'
    description['eOffsets'] =      'End offsets_____________________'
    description['eConst'] =        '-Extrusion constant_____________'
    description['ePerDst'] =       '-Extrusion per distance_________'

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
        
        
        self.eConst = 4/(π*self.fillamentDia**2)
        #Multiply desired extrusion volume by this to get distance of fillament to extrude
        self.ePerDst = self.layerH*self.nozzleWidth*self.eConst
        #Multiply desired extrusion distance by this to get distance of fillament to extrude
    #
    
    def __str__(self) -> str:
        """
        Description:
            This returns the printer settings as a string
        """
        return "".join(f"{pSettings.description[setting]}{self.__getattribute__(setting)}\n" for setting in pSettings.description)
