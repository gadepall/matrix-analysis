#!/bin/bash


#Download python and latex templates


#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#cd /sdcard/Download/FWC/trunk/IDE/co


cd /sdcard/Download/FWC/trunk/matrix
python3 mat.py

texfot pdflatex line.tex
termux-open line.pdf


#Test Python Installation
#Uncomment only the following line
#python3 /data/data/com.termux/files/home/storage/shared/training/math/codes/tri_sss.py
#pio run
