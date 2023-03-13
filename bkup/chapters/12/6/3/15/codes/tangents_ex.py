#Code by GVV Sharma
#July 16, 2022
#released under GNU GPL
#Using sympy to find the tangents given the normal vector


import sys                                          #for path to external scripts
#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts
sys.path.insert(0, '/sdcard/github/cbse-papers/CoordGeo')        #path to my scripts
#sys.path += ['/data/data/com.termux/files/home/storage/shared/github/training/math/codes/CoordGeo','/data/data/com.termux/files/home/arch/home/user/miniforge3/envs/my-env']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.linalg as LA
import sympy as smp
from sympy.abc import x,y


#local imports
from line.funcs import *

from triangle.funcs import *
from conics.funcs import circ_gen


#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts

#if using termux
import subprocess
import shlex
#end if

#Ellipse parameters

#Sympy version

#Inputs
usym = -smp.Rational(1,2)*smp.Matrix(([2,1]))
Vsym = smp.Matrix(([1, 0],[0,0]))
fsym = smp.Matrix(([7]))

#n = smp.Matrix(([2,-1]))
n = smp.Matrix(([1,3]))


#Eigenvalues
Psym, Dsym = Vsym.diagonalize()
Psym=Psym/smp.sqrt((Psym.T@Psym)[0,0])
psym1 = Psym.col(0)
psym2 = Psym.col(1)

#print(Vsym.det())
eta = usym.T@psym1
a = smp.Matrix(((usym.T + eta*psym1.T), Vsym))
b = smp.Matrix(((-fsym), (eta*psym1.T-usym.T).T)) 
c =a.LDLsolve(b)

kap = (usym.T@psym1)/(n.T@psym1)[0,0]
a = smp.Matrix(((usym.T + kap*n.T), Vsym))
b = smp.Matrix(((-fsym), (kap*n.T-usym.T).T)) 
q =a.LDLsolve(b)
#print(eta.T*psym1.T)
#print(usym,psym1)
#print(usym.T)
#print(eta)
print(kap,q,n.T@q)

flag = 0
if flag == 1:
    #intermediate
    Vinv = Vsym.inv()
    f0sym = (usym.T@Vinv@usym- fsym)
    #Vertex
    csym = -smp.Matrix(Vinv@usym)
    
    #Tangent equation
    kap = smp.sqrt(f0sym/(n.T@Vinv@n)[0,0])[0,0]
    
    #axes lengths
    a = smp.sqrt(f0sym/Dsym[0,0])
    b = smp.sqrt(f0sym/Dsym[1,1])
    #print(kap)
    #print(f0sym)
    q1 = -kap*Vinv@n-Vinv@usym
    q2 = kap*Vinv@n-Vinv@usym
    fn =  n.T@Vinv@n
    #print(a,b)
    #print(smp.latex(psym1))
    #print(smp.latex(psym2))
    #print(smp.latex(Vsym),smp.latex(Vinv),smp.latex(usym))
    #print(smp.latex(csym),smp.latex(f0sym),smp.latex(Vsym.det()),smp.latex(Dsym))
    #print(smp.latex(f0sym))
    #print(smp.latex(cisym))
    #print(f0sym,cisym)
    #print(f0sym,fn)kap
    cisym =  n.T@(-kap*Vinv@n-Vinv@usym)
    fn =  n.T@Vinv@n
    #print(a,b)
    #print(smp.latex(psym1))
    #print(smp.latex(psym2))
    #print(smp.latex(Vsym),smp.latex(Vinv),smp.latex(usym))
    #print(smp.latex(csym),smp.latex(f0sym),smp.latex(Vsym.det()),smp.latex(Dsym))
    #print(smp.latex(f0sym))
    #print(smp.latex(cisym))
    #print(f0sym,cisym)
    #print(f0sym,fn)
    
    
    
    
    
    
    
    
    
