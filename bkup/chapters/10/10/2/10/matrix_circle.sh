#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line


python3 /home/bhavani/Documents/matrix/matrix_circle/circle.py

cd /home/bhavani/Documents/matrix/matrix_circle
pdflatex circle.tex
xdg-open circle.pdf


#Test Python Installation
#Uncomment only the following line
