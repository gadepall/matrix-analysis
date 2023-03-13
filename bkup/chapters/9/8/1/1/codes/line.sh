#!/bin/bash

python3 line.py
cd ..

pdflatex line.tex

zathura line.pdf


