#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#Test Python Installation
#Uncomment only the following line
python3 /sdcard/dinesh/conics/conics3.py


texfot pdflatex conics.tex
termux-open conics.pdf


