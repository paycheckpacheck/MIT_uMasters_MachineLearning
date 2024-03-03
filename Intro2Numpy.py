import matplotlib.pyplot as plt

import numpy as np

'''Creating arrays'''
print("Creating arrays:\n")

a = np.array([1, 2, 3]) # 1-D array
print("1-D array: ", a, "\n")

b = np.array([[1, 2, 3], [4, 5, 6]]) # 2-D array
print("2-D array: \n", b, "\n")

c = np.zeros((3, 3)) # 3-D array filled with zeros
print("3-D array filled with zeros: \n", c, "\n")

d = np.ones((3, 3)) # 3-D array filled with ones
print("3-D array filled with ones: \n", d, "\n")

e = np.eye(3) # 3-D identity matrix
print("3-D identity matrix: \n", e, "\n")
'''============================================================='''




'''Product counting arange array or linspace evenly spaced array'''
f = np.arange(10) # 1-D array with values 0 to 9
print("1-D array with values 0 to 9: ", f, "\n")
g = np.linspace(0, 1, 5) # 1-D array with 5 evenly spaced values between 0 and 1
print("1-D array with 5 evenly spaced values between 0 and 1: ", g, "\n")
'''============================================================='''


# Array shape and reshaping
print("Array shape and reshaping:\n")

h = np.array([[1, 2, 3], [4, 5, 6]])
print("Original array:\n", h, "\n")

print("Array shape: ", h.shape, "\n")

i = h.reshape((3, 2)) # Reshaping the array
print("Reshaped array:\n", i, "\n")

# Array indexing and slicing
print("Array indexing and slicing:\n")

j = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Original array: ", j, "\n")

print("First element: ", j[0], "\n")
print("Last element: ", j[-1], "\n")
print("Elements from 3rd to 6th: ", j[2:6], "\n")

k = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original array:\n", k, "\n")

print("First row: ", k[0], "\n")
print("Last row: ", k[-1], "\n")
print("First two rows:\n", k[:2], "\n")

# Array arithmetic and statistical functions
print("Array arithmetic and statistical functions:\n")

l = np.array([1, 2, 3, 4, 5])
m = np.array([6, 7, 8, 9, 10])

print("Array l: ", l, "\n")
print("Array m: ", m, "\n")

print("Array l + m: ", l + m, "\n")
print("Array l - m: ", l - m, "\n")
print("Array l * m: ", l * m, "\n")

#random matrix
rand_mtx = np.random.rand(10, 10)
print(f"Random Matrix {rand_mtx}, SHAPE:, {rand_mtx.shape}")


#Typical Functions: ELEMENT-WISE
exp_rand_mtx = np.exp(rand_mtx)
sin_rand_mtx = np.sin(rand_mtx)

cos_rand_mtx = np.cos(rand_mtx)
tanh_rand_mtx = np.tanh(rand_mtx)

#MAX and MIN in array
np.min(rand_mtx)
rand_mtx.min()
np.max(rand_mtx)
rand_mtx.max()


#norm of matrix
np.linalg.norm(rand_mtx)



#Plot a function
x = np.linspace(-2*np.pi, 2*np.pi, 400)
y = 2*np.tanh(.5*x)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()

#Make multipule plots
x = np.linspace(0, 2*np.pi, 400)
y1 = np.tanh(x)
y2 = np.cos(x**2)
fig, axes = plt.subplots(1, 2, sharey=True)
axes[1].plot(x, y1)
axes[1].plot(x, -y1)
axes[0].plot(x, y2)



import numpy as np

# Define a 1-dimensional numpy array
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Slice from the second element to the seventh element
print(a[1:7])
# Output: [2 3 4 5 6 7]

# Slice every second element starting from the first
print(a[::2])
# Output: [1 3 5 7 9]

# Slice every second element starting from the second
print(a[1::2])
# Output: [2 4 6 8 10]

# Reverse the numpy array
print(a[::-1])
# Output: [10 9 8 7 6 5 4 3 2 1]

# Define a 2-dimensional numpy array
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Slice the first two rows and the first two columns
print(b[:2, :2])
# Output:
# [[1 2]
#  [4 5]]

# Slice the last two columns of the last row
print(b[-1, -2:])
# Output: [8 9]




