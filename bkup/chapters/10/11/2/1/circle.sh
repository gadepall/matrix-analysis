#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#
python3 /sdcard/Download/fwc_matrix/matrix_line/circle.py

cd /sdcard/Download/fwc_matrix/matrix_line
pdflatex line.tex
termux-open line.pdf

#Test Python Installation
#Uncomment only the following line

#python3 /sdcard/Download/anusha1/python1/asgn1.py
