#!/bin/bash


#Download python and latex templates

#svn co https://github.com/gadepall/training/trunk/math  /sdcard/Download/math

#Test Latex Installation
#Uncomment only the following lines and comment the above line

#cd /sdcard/Download/math
#line
#texfot pdflatex line.tex
#python3 l.py
#termux-open line.pdf
#cd ../circle_assignment
#texfot pdflatex circle.tex
#python3 c.py
#termux-open circle.pdf
cd ../conics_assignment
texfot pdflatex conics.tex
python3 co.py
termux-open conics.pdf

