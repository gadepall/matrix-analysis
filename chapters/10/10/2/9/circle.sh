#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#cd /sdcard/Download/math 
texfot pdflatex circle.tex
python3 c.py
termux-open circle.pdf


#Test Python Installation
#Uncomment only the following line

