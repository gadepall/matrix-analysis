#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

cd /sdcard/matrices/circle 
texfot pdflatex assign4.tex
termux-open assign4.pdf


#Test Python Installation
#Uncomment only the following line
python3 circle1.py
