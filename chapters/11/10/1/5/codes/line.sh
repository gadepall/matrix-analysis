#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#to run matrixlines

#cd /sdcard/download/radhika/IDE
#pio run

python3 /sdcard/download/codes/line_assignment/codes/lines.py

cd /sdcard/download/codes/line_assignment/document
pdflatex line.tex
termux-open line.pdf


