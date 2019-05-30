# ActiveZ-Gcode-Generator
The files for my Junior Project about Non-Planar 3D printing

General File Contents
Jr_Proj:
  Cad
    Contains various Solidworks files for my test fixture.
    Drawings
      Contains mechanical drawings of the parts in my test fixture. This will be completed next year when my design is finalized.
  Code
    gcodeEngine
      Contains various media files created by my program for use in my presentation.
      gcodeEngineV2_5.py - Master program file. Uses other files to finally create the gcode file.
      visualization3.py - The program used to generate the images of graphs used in my presentation
      Also contains the output gcode file, and files used at the beginning and end of the gcode file.
    gcodeEngine_OLD
      Contains legacy versions of my program
    printer3D
      gcode.py - backend class which manages everything related to gcode
      pSettings.py - stores all settings for the 3D printer
      __init__.py - setup for this printer3D package
  LaTeX
    Contains all latex files for the equations used in my presentation. I also included plain text versions of all of these so they can be opened from google drive
    Contains both pdf and png images of all equations
  Presentation
    Confused Robot.png - the image of a robot which I modified in my presentation
    Functions
      Contains other images for my presentation

Specific File Structure
Jr_Proj:.
├───CAD
│   │   0_500_Hex_ID_x_1_125_OD_x_0_313_WD_Flanged_Bearing_v2_217-3875.SLDPRT
│   │   1_2_Hex_Bore_Aluminum_VersaHub_217-2592.SLDPRT
│   │   Assembly.SLDASM
│   │   Base.SLDPRT
│   │   Clamp Side ASM.SLDASM
│   │   Clamp Side.SLDPRT
│   │   Clamping_Shaft_Collar_-_1_2_Hex_ID_217-2737.SLDPRT
│   │   Fixed Side.SLDPRT
│   │   Hex_Shaft.SLDPRT
│   │   High_Strength_Clamping_Shaft_Collar_-_1_2_Hex_ID_217-4106.SLDASM
│   │   Hose Clamp.SLDPRT
│   │   Test Piece.SLDPRT
│   │
│   └───Drawings
│       │   Clamp Side.SLDDRW
│
├───Code
│   ├───gcodeEngine
│   │   │   0.png
│   │   │   1.png
│   │   │   basic_animation.mp4
│   │   │   endGcode.txt
│   │   │   gcodeEngineV2_5.py
│   │   │   startGcode.txt
│   │   │   visualization3.py
│   │   │   wigglyTube_20.GCODE
│   │
│   ├───gcodeEngine_OLD
│   │   │   endGcode.txt
│   │   │   gcodeEngineV0.py
│   │   │   gcodeEngineV1.py
│   │   │   gcodeEngineV2.py
│   │   │   gcodeEngineV2_1.py
│   │   │   gcodeEngineV2_2.py
│   │   │   gcodeEngineV2_3.py
│   │   │   gcodeEngineV2_4.py
│   │   │   gcodeEngine_V-1.py
│   │   │   gcodeEngine_V-2.py
│   │   │   startGcode.txt
│   │   │   visualization.py
│   │   │   visualization2.py
│   │   │   wigglyTube2_40.GCODE
│   │
│   └───printer3D
│       │   gcode.py
│       │   pSettings.py
│       │   __init__.py
│
├───LaTeX
│   │   0f.pdf
│   │   0f.png
│   │   0f.synctex.gz
│   │   0f.txt
│   │   0f.tex
│   │   1f.pdf
│   │   1f.png
│   │   1f.synctex.gz
│   │   1f.txt
│   │   1f.tex
│   │   2f.pdf
│   │   2f.png
│   │   2f.synctex.gz
│   │   2f.txt
│   │   2f.tex
│   │   3f.pdf
│   │   3f.png
│   │   3f.synctex.gz
│   │   3f.txt
│   │   3f.tex
│   │   4f.pdf
│   │   4f.png
│   │   4f.synctex.gz
│   │   4f.txt
│   │   4f.tex
│   │   Equations.txt
│   │   Equations.tex
│
└───Presentation
    │   Confused Robot.png
    │
    └───Functions
        │   0f.png
        │   0g.png
        │   1f.png
        │   1g.png
        │   2f.png
        │   2g.png
        │   3f.png
        │   3g.png
        │   4f.png
        │   4g.gif
        │   gcode0.png
        │   gcode1.png


