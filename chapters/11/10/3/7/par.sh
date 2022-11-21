
 #!/bin/bash
cd /sdcard/Download/matrices
texfot pdflatex mat.tex
termux-open mat.pdf
cd /sdcard/Download/matrices
python3 par.py
