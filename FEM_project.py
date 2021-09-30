'''
# problem description in latex code:
\begin{align}
\text{Use a custom FEM im} & \text{plementation to get a } \\
\text{numerical solution to } & \text{the following problem:} \\
- \Delta u + a_0 u &= f \text{ in } \Omega  \\
\partial_\nu u &= g \text{ on } \partial \Omega \\
\text{Where } \Omega = (0,1)^2 \text{, } a_0 \in \mathbb{R}& \text{ arbitrary and } \partial_\nu u = \nu \cdot \nabla u \\
\text{ is the derivative with resp} & \text{ect to the normal vector } \nu \text{.}
\end{align} 

'''

#############################################################################################################
# first define a function from scratch that solves the corresponding FEM problem (FEM = finite element method)
import numpy as np
# import some math functions a user might want to use in the input form
from math import sqrt,exp,log,log2,log10,sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,asinh,acosh,atanh
def generate_FEM_solution(n,a_0,f,g):

    # amount of nodes and triangles for given integer n > 0
    num_inner_nodes = (n-1)**2
    num_outer_nodes = 4*n
    num_nodes = (n+1)**2        # number of inner + outer nodes
    num_triangles = 2*(n**2)

    # to save the x,y data of nodes, indices of outer nodes, 
    # all pairs of indices that make an outer edge, and three node indices that make up a triangle
    outer_nodes = set()
    node = [None]*(num_outer_nodes + num_inner_nodes)
    outer_edges = []
    triangles = []

    # x,y coordinates of nodes/vertices
    for row in range(0,n+1):
        for col in range(0,n+1):
            node[col + row*(n+1)] = np.array((col/n,row/n))

    # triangle stores the indices of vertices of all triangles, e.g. if (0,1,4) is in triangle, than the x,y coordinates of node[0],node[1] and node[4] correspond to the nodes/vertices of a triangle
    # generate a Friedrichs Keller triangulation of the domain \Omega
    for row in range(n):
        for col in range(n):
            triangles.append((col+row*(n+1),col+1+row*(n+1),col+1 +(row+1)*(n+1)))
            triangles.append((col+row*(n+1),col+(row+1)*(n+1),col+1 +(row+1)*(n+1)))

    # the indices of all outer nodes
    # nodes with y == 0
    for col in range(n+1):
        outer_nodes.add(col)
    # nodes with x == 0 or x == 1
    for row in range(1,n):
        outer_nodes.add(row*(n+1))
        outer_nodes.add(row*(n+1)+n)
    # nodes with y == 1
    for col in range(n+1):
        outer_nodes.add((n+1)*n+col)

    # allocate space for matrix entries for the linear System of equations, resulting from a Galerkin method (triangle wise)
    A = np.zeros((num_nodes,num_nodes))
    b = np.zeros((num_nodes,1))

    # matrices to transform an individual triangle to the reference triangle
    S_1 = 1/2*np.array([[1,-1,0],[-1,1,0],[0,0,0]])
    S_2 = 1/2*np.array([[2,-1,-1],[-1,0,1],[-1,1,0]])
    S_3 = 1/2*np.array([[1,0,-1],[0,0,0],[-1,0,1]])
    S_0 = 1/24*np.array([[2,1,1],[1,2,1],[1,1,2]])

    # loop over ever triangle / trianglewise calculation of LSE (linear system of equations)
    for triangle in triangles:
        # indices of the 3 nodes which are the vertices of tri
        index_1, index_2, index_3 = triangle
        # get coordinates of the 3 nodes with the same index
        node_1, node_2, node_3 = node[index_1], node[index_2], node[index_3]

        # append indices that make up an outher edge
        if (index_1 in outer_nodes and index_2 in outer_nodes) and (node_1[0] == node_2[0] or node_1[1] == node_2[1]):
            outer_edges.append((index_1,index_2))
        if (index_2 in outer_nodes and index_3 in outer_nodes) and (node_2[0] == node_3[0] or node_2[1] == node_3[1]):
            outer_edges.append((index_2,index_3))
        if (index_1 in outer_nodes and index_3 in outer_nodes) and (node_1[0] == node_3[0] or node_1[1] == node_3[1]):
            outer_edges.append((index_1,index_3))

        # 2 vectors that describe triangle geometrically
        b_1 = node_2 - node_1
        b_2 = node_3 - node_1

        # absolute value of determinant of matrix B = np.array([b_1,b_2])
        detB = abs(b_1[0]*b_2[1] - b_1[1]*b_2[0])
        detB_inv = 1/detB

        # gamma values (with respect to current triangle)
        gamma_1 = detB_inv * np.dot(b_2,b_2)
        gamma_2 = - detB_inv * np.dot(b_2,b_1)
        gamma_3 = detB_inv * np.dot(b_1,b_1)

        # calculate A_loc in local coordinates, corresponding to index_1,  index_2, index_3
        A_loc = gamma_1 * S_1 + gamma_2 * S_2 + gamma_3 * S_3 + a_0 * detB * S_0

        # add contribution of triangle (quantified in A_loc) to matrix A
        index = [index_1, index_2, index_3]
        for row in range(3):
            for col in range(3):
                A[index[row],index[col]] += A_loc[row,col]

        # add values (with respect to f) to b vector
        for index in [index_1, index_2, index_3]:
            b[index] += detB/6 * f(node[index][0],node[index][1])

    # take Neumann boundary condition into account, consider all outer edges
    for edge in outer_edges:
        # get x,y values of nodes that make up edge, and calculate edge length
        index_1, index_2 = edge
        node_1, node_2 = node[index_1], node[index_2]
        edge_length = sqrt( (node_2[0]-node_1[0])**2 + (node_2[1]-node_1[1])**2 )

        # add contribution of Neumann boundary values to vector b
        b[index_1] += edge_length * 1/2 * g(node_1[0],node_1[1])
        b[index_2] += edge_length * 1/2 * g(node_2[0],node_2[1])

    # vector of x and y values of nodes as a numpy array (such that we can plot solution)
    X = np.array([x for x,_ in node])
    Y = np.array([y for _,y in node])
    # calculate values of numerical solution on nodes, this will be a numpy array with the same size as x and y
    U = np.linalg.solve(A, b).transpose()[0]

    # return the calculated/generated values
    return X,Y,U,node,triangles







#############################################################################################################
#############################################################################################################
#############################################################################################################







# Graphical user interface (GUI) for FEM Project, using Tkinter

from tkinter import *
from tkinter.filedialog import asksaveasfilename
from PIL import ImageTk,Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

#############################################################################################################
##### main:
# to get a window with a title
window = Tk()
window.title("Finite Element Method example project")
window.resizable(0, 0)  # such that window is not resizable
BackgroundColour = "#2e2e2e"
window.configure(background=BackgroundColour)
text_entry_box_width = 88
button_width = 40

#############################################################################################################
# 1 - Image of problem we solve numerical

# load and display image
image_PDE = Image.open("FEM_project_image_of_problem_description.png")
image_PDE = image_PDE.resize((700,300))
photoimage_PDE = ImageTk.PhotoImage(image_PDE)
Label(window, image=photoimage_PDE, bg=BackgroundColour).grid(row=0, column=0, sticky=NW, padx=10, pady=10)


#############################################################################################################
# 2 - Slider for mesh elements n

# possible slider values for n are 1 and all values 5,10,15,...,100
valuelist = [1] + [k for k in range(5,101,5)]
# returns closest value on list
def valuecheck(value):
    newvalue = min(valuelist, key=lambda x:abs(x-float(value)))
    n_slider.set(newvalue)
# create slider for values of n
Label(window, text="meshparameter n = ", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=1,column=0,sticky=SW, padx=10)
n = IntVar()
n_slider = Scale(window, orient=HORIZONTAL, variable=n, from_=min(valuelist), to=max(valuelist), command=valuecheck, resolution=1, length=700, bg=BackgroundColour, fg="white", troughcolor="grey")
n_slider.grid(row=2,column=0,sticky=NW, padx=10)
n_slider.set(25)    # initial value n=25

#############################################################################################################
# 3 - Input for function f, g and parameter a_0

# text entry box for function f
Label(window, text="function f(x,y) =", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=3,column=0,sticky=SW, padx=10)
f_entry = Entry(window, width=text_entry_box_width, bg="white")
f_entry.grid(row=4,column=0,sticky=NW, padx=10)
f_entry.insert(0,"sin(10*x)*exp((2*y)**2)")

# text entry box for function g
Label(window, text="function g(x,y) =", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=5,column=0,sticky=SW, padx=10)
g_entry = Entry(window, width=text_entry_box_width, bg="white")
g_entry.grid(row=6,column=0,sticky=NW, padx=10)
g_entry.insert(0,"x+y")

# text entry box for a_0
Label(window, text="constant a_0 =", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=7,column=0,sticky=SW, padx=10)
a_0_entry = Entry(window, width=text_entry_box_width, bg="white")
a_0_entry.grid(row=8,column=0,sticky=NW, padx=10)
a_0_entry.insert(0,"1")

#############################################################################################################
# 4- initialize figure and canvas to display the figure (empty at start of application) and an empty dataframe

# figure and canvas 
fig = plt.figure(figsize=(5,2), dpi=200)
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().grid(row=0,column=2,rowspan=9,sticky=E+W+S+N, padx=10, pady=10)

# empty dataframe, to save numerical data later
df = pd.DataFrame()
df["x"], df["y"], df["u"] = np.array([]), np.array([]), np.array([])

#############################################################################################################
# 5 - button to carry out FEM calculations, create a dataframe with solution and plot
error_box = Text(window, font="none 12 bold", height=3, width=95)
error_box.grid(row=9,column=2,rowspan=2,sticky=E+W+S+N, padx=10, pady=10)
error_box.insert(END, " ")

# generate FEM solution function
def gen_FEM_solution():
    # empty text in error box
    error_box.delete('1.0',END)
    # try to use input from entry boxes
    try:    
        # get values from sliders and convert to needed datatype
        n = n_slider.get()
        f = eval("lambda x,y: " + f_entry.get())
        g = eval("lambda x,y: " + g_entry.get())
        a_0 = float(a_0_entry.get())

        error_box.insert(END, f"Input parameters: \nn={n}    f(x,y) = {f_entry.get()}    g(x,y) = {g_entry.get()}    a_0 = {a_0_entry.get()} \n")

        # empty current dataframe
        df.drop(columns=["x","y","u"],inplace=True)
        df.drop(index=np.arange(0, len(df)),inplace=True)
    
        # use function to generate FEM solution and save in dataframe (node indices are not needed for the plots)
        df["x"], df["y"], df["u"], _, triangles = generate_FEM_solution(n,a_0,f,g)

        # plot solution on triangular mesh
        fig.clf()   # to clear current figure
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(df["x"], df["y"], df["u"], triangles=triangles, cmap=cm.coolwarm, linewidth=0.2, antialiased=True)
        u_min, u_max = df["u"].min(), df["u"].max() 
        epsilon = max(0.00001, 0.001*(u_max-u_min))
        ax.set_zlim(u_min - epsilon, u_max + epsilon)

        # draw into GUI
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.get_tk_widget().grid(row=0,column=2,rowspan=9,sticky=E+W+S+N, padx=10, pady=10)
    
        # desplay that a valid solution could be created
        error_box.insert(END, "Successfully calculated a numerical solution")

    # if error occurs, display in error_box
    except:
        error_box.delete('1.0',END)
        error_box.insert(END, "Warning!!! \nAn error occurred, check that a_0 is a valid float value and f,g are valid functions defined for 0 <= x,y <= 1, examples for valid definitions: x**2+y, x*sin(y), log(x+1)*exp(sin(y)), atan(x**y)")
    

# generate FEM solution label
# Label(window, text="click to generate and plot numerical solution", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=9,column=0,sticky=W, padx=10, pady=10)
# generate FEM; solution button
Button(window, text="Generate and plot solution", width=button_width, command=gen_FEM_solution).grid(row=9,column=0,sticky=W, padx=10, pady=10)

#############################################################################################################
# 6 - button to save data frame of solutin as csv file

def save_csv_file():
    """Save the current data frame as a csv file."""
    filepath = asksaveasfilename(
        defaultextension="csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, "w") as output_file:
        csv = df.to_csv(index=False)
        output_file.write(csv)
    window.title(f"Finite Element Method example project - numerical data of solution saved in: - {filepath}")
# save button
Button(window, text="Save numerical solution as csv file", width=button_width, command=save_csv_file).grid(row=9,column=0,sticky=E, padx=10, pady=10)


#############################################################################################################
# 7 - button to reset values to initial state

# function that resets values
def reset_to_initial_values():
    n_slider.set(25)
    f_entry.delete(0,END)
    f_entry.insert(0,"sin(10*x)*exp((2*y)**2)")
    g_entry.delete(0,END)
    g_entry.insert(0,"x+y")
    a_0_entry.delete(0,END)
    a_0_entry.insert(0,"1")

# reset button
Button(window, text="Reset to initial values", width=button_width, command=reset_to_initial_values).grid(row=10,column=0,sticky=W, padx=10, pady=10)


#############################################################################################################
# 8 - button to exit application

# exit function
def close_window():
    fig.clf()
    window.destroy()
    exit()

# exit label
# Label(window, text="click to close window", bg=BackgroundColour, fg="white", font="none 12 bold").grid(row=10,column=0,sticky=W, padx=10, pady=10)
# exit button
Button(window, text="Exit", width=button_width, command=close_window).grid(row=10,column=0,sticky=E, padx=10, pady=10)

##### run the main loop
window.mainloop()


