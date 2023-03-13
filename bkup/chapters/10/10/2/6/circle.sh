#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#cd /sdcard/Download/math 
#texfot pdflatex circle1.tex
#termux-open circle1.pdf


#Test Python Installation
#Uncomment only the following line
python3 assignment2.py
python3 proof.py
texfot pdflatex circle1.tex
termux-open circle1.pdf


#Test Python Installation
#Uncomment only the following line
python3 assignment2.py

