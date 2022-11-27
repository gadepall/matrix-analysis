!/bin/bash
cd ..
cd docs
texfot pdflatex conic.tex
termux-open conic.pdf
cd ..
cd code_conic
python3 conic.py


#cd ..
#cd ..
#cd assignment_optimization_1
#cd docs
#
#texfot pdflatex opt_1.tex
#termux-open opt_1.pdf
