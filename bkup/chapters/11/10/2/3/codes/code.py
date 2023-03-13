import matplotlib.pyplot as plt
import numpy as np

def plot_line(m):
    # Set x and y values for the line
    x=np.array(([0,0]))
    X = np.linspace(-5,5,100)
    Y = [m*i for i in X]
    
    # Plot the line
    plt.plot(X, Y,label='y=mx')
    tri_coords = x.T
    plt.scatter(tri_coords[0], tri_coords[1])
    vert_labels = ['x(0,0)']
    for i, txt in enumerate(vert_labels):
      plt.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

    # Add labels and show the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Equation of line passing through (0,0) with slope {m}')
    plt.legend(loc='best')
    plt.grid()
    plt.axis('equal')
    plt.show()
    #if using termux
    plt.savefig('../figs/fig.pdf')

m = 2
plot_line(m)

