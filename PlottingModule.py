import math
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



from scipy.special import logit
from scipy.interpolate import interp1d




class DataPlotter:
    def __init__(self, range, x_name="x", y_name="y"):

        self.range = range
        self.incriment = (range[1] - range[0]) -1 #to set incriment, set attribute after instance init


        self.x_name = x_name
        self.y_name = y_name

        # Generate the x values to plot
        self.x = np.linspace(self.range[0], self.range[1], self.incriment)

    def plot_function(self, function):
        self.y = function(self.x)
        plt.plot(self.x,self.y)

        plt.xlabel(str(self.y_name))
        plt.ylabel(str(self.x_name))

        plt.show()

    def interpolate_shape(self,data):
        x = data[:, 0]
        y = data[:, 1]

        # Use linear interpolation
        f = interp1d(x, y, kind='linear')

        # Interpolate over a new set of x-values with a higher resolution
        x_interp = np.linspace(x[0], x[-1], num=1000, endpoint=True)
        y_interp = f(x_interp)

        return np.column_stack((x_interp, y_interp))

    def plot_array(self, data):
        # Get the number of columns in the data
        num_cols = data.shape[0]
        print("shape", num_cols)

        # Plot the data
        for i in range(1, num_cols):
            plt.plot(data[:, 0], data[:, i])

        # Add axis labels and a title
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Plot of data")

        # Show the plot
        plt.show()

    def plot_point_onx(self,x,y ):

        plt.plot(x, y)
    def show(self):
        plt.show()







def foo(x):


    y = x**2
    return y
'''#Plot function
Plotter = DataPlotter( [-100, 100] )
#Plotter.plot_function(foo)


#Make random np matrix
rand_mtx = np.random.random([3,2])
#interpolate matrix
interpolated_data = Plotter.interpolate_shape(rand_mtx)
#Plot interpolation
Plotter.plot_array(rand_mtx)


print(interpolated_data, rand_mtx.shape)'''