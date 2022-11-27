#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#
python3 /home/apiiit-rkv/Desktop/fwc_matrix/matrix_conics/ellipse.py
cd /home/apiiit-rkv/Desktop/fwc_matrix/matrix_conics
pdflatex conic.tex
xdg-open conic.pdf

#Test Python Installation
#Uncomment only the following line

#python3 /sdcard/Download/anusha1/python1/asgn1.py
