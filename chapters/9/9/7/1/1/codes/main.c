#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"lib.h"
int main()
{
	double **a,**b,**c,**d,**e1,**m1,**m2,**n1,**n2,**t1,**t2,**k1,**k2; //initializing the  variables as matrices
	double d1,d2,d3,d4,h1,h2,ang1,ang2;
	double theta=radians(30);
	int m=2,n=1;
	int i=5,j=9; //initializing assumptions 
	a=loadtxt("a.dat",2,1); //loading the  point A from the text file
	e1=loadtxt("e1.dat",2,1); //loading the e1(1,0) from the .dat file
	b=mult_int(j,e1,m,n);
	c=np_array(i*cos(theta),i*sin(theta));
	d=np_array(i*cos(theta),i*sin(-theta));

	m1=linalg_sub(b,c,m,n);
	m2=linalg_sub(b,a,m,n);
	n1=linalg_sub(d,b,m,n);
	n2=linalg_sub(a,b,m,n);

	d1=linalg_norm(m1,m);
	d2=linalg_norm(m2,m);
	d3=linalg_norm(n1,m);
	d4=linalg_norm(n2,m);

	t1=transpose(m1,m,n);
	t2=transpose(n1,m,n);

	k1=matmul(t1,m2,m,n,2);
	k2=matmul(t2,n2,m,n,2);

	h1=d1*d2;
	h2=d3*d4;

	ang1=a*cos(k1/h1);
	ang2=a*cos(k2/h2);
	

	if(round(ang1)==round(ang2)){
	printf("∠ CBA = ∠ ABD");
	}
}
