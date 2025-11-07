#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: NV_machine_learning.py
Description: This script simulates a one qubit cluster sorter by means of a NV center on diamond.
Authors: Asier Mongelos and Miguel Lopez Varga
Creation date: 2025-05-20
"""

import numpy as np
from scipy.linalg import expm, sqrtm
import math
import matplotlib.pyplot as plt
##Operators##
Sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
Sx = np.array([[0,1,0],[1,0,1],[0,1,0]])*(1/np.sqrt(2))
Sy = np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])*(1/np.sqrt(2))

#We define the rotation operations over the N Vcenter
##Each function retunrs th eunitary operation representing the rotation over
#states 0 and 1 of the NV. The pulses are then considered to be in tune
# with the transition 0--1. 
#Functions tae the arguments:
    # alpha    --> Rotation angle in Radians
    # Omega    --> MW amplitude in MHz
    # B        --> magnetic field value in T (tesla)
    # detuning --> error in targetin gthe resonance line (default =0) in MHz.
#
def Rx(alpha, Omega, B, detuning=0):
    #Physical parameters
    D = 2870        #MHz, Zer-field splitting.
    gamma_e = 28024 #MHz/T electron gyromagnetic ratio
    
    H = D*np.matmul(Sz,Sz) + gamma_e*B*Sz + (Omega/2)*Sx -(D+gamma_e*B+detuning)*np.matmul(Sz,Sz)
    
    t = alpha/(np.sqrt(2)*np.pi*Omega)
    
    U = expm(1j*(2*np.pi)*H*t)
    return U

def Ry(alpha, Omega, B, detuning=0):
    #Physical parameters
    D = 2870        #MHz, Zer-field splitting.
    gamma_e = 28024 #MHz/T electron gyromagnetic ratio
    
    H = D*np.matmul(Sz,Sz) + gamma_e*B*Sz + (Omega/2)*Sy -(D+gamma_e*B+detuning)*np.matmul(Sz,Sz)
    
    t = alpha/(np.sqrt(2)*np.pi*Omega)
    
    U = expm(1j*(2*np.pi)*H*t)
    return U

# These are the projectors over the X, Y, and Z axes of the Bloch sphere.
# They are used to calculate the Bloch coordinates of a density matrix.
def ProjX(rho):
    ProjX = np.array([[1/2,1/2,0],[1/2,1/2,0],[0,0,0]])
    
    return np.real(np.trace(np.matmul(rho,ProjX)))

def ProjY(rho):
    ProjY = np.array([[1/2,1j/2,0],[-1j/2,1/2,0],[0,0,0]])
    
    return np.real(np.trace(np.matmul(rho,ProjY)))

def ProjZ(rho):
    return np.real(rho[1,1])

def BlochCoordinates(rho):
    x = ProjX(rho)
    y = ProjY(rho)
    z = ProjZ(rho)
    
    return np.array([x,y,z])

# Here we calculate the fidelity between two density matrices.
def Fidelity(rho, target, pure=True):
    
    if pure:
        f = np.trace(np.matmul(rho,target))
    else:
        f = np.matmul(sqrtm(target),rho)
        f = sqrtm(np.matmul(f,sqrtm(target)))
        f = (np.trace(f))**2
    
    return f

# This function calculates the distance between two points in thhe coordinate plane
def Distance(x_1, x_2):
    return float(np.sqrt((float(x_1[0])-float(x_2[0]))**2+(float(x_1[1])-float(x_2[1]))**2+(float(x_1[2])-float(x_2[2]))**2))

# This function calculates the centroid of a set of coordinates.
def Centroide (x):
    centroid = []
    centroid= sum(x)
    centroid= centroid /len(x)

    return np.array(centroid)

# This function converts Cartesian coordinates to coefficients for the quantum state.
def CartToCoef(points_x, points_y, points_z):
    phi_once = np.arctan2(points_y, points_x)
    theta_once = np.arccos(points_z)
    c1_once = np.cos(theta_once / 2)
    c2_once = np.exp(complex(0, 1) * phi_once) * np.sin(theta_once / 2)

    return np.array(c1_once, dtype=complex), np.array(c2_once, dtype=complex)

# This function generates points on a sphere using the Fibonacci spiral method.
def FibonacciSphere(samples):
    points_x = []
    points_y = []
    points_z = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points_x.append(x)
        points_y.append(y)
        points_z.append(z)
    return points_x, points_y, points_z

def StateLabels(number_labels):
    labels = []
    points_x, points_y, points_z = FibonacciSphere(number_labels)
    ket_1 = np.array([[1],[0],[0]])
    ket_0 = np.array([[0],[1],[0]])
    
    for i in range(len(points_x)):
        #Rescale points to be in range 0,1
        points_x[i] = (points_x[i]+1)/2
        points_y[i] = (points_y[i]+1)/2
        points_z[i] = (points_z[i]+1)/2
        #state = points_x[i]*(ket_0+ket_1)/np.sqrt(2) + points_y[i]*(ket_0+1j*ket_1)/np.sqrt(2) + points_z[i]*ket_0
        if points_z[i]!=0:
            state = np.sqrt(points_z[i])*ket_0 + ((2* points_x[i]-1) + 1j * (2* points_y[i]-1))/(2 * np.sqrt(points_z[i])) * ket_1
        else:
            state = ket_1
        labels.append(np.matmul(state, np.matrix.conjugate(state).T))
    state_labels = np.array(labels, dtype=complex)
    return state_labels

# This function tests the quantum states against the labels and returns the maximum fidelities and their corresponding indices.
def Test(quantum_states, labels):
    #print('shape of quantum states = ' + str(np.shape(quantum_states)))
    #print('shape of labels = ' + str(np.shape(labels)))
    fidelities = np.zeros((len(quantum_states), len(labels)), dtype=complex)
    for i in range(len(quantum_states)):
        for j in range(len(labels)):
            fidelities[i][j] = Fidelity(quantum_states[i], labels[j], pure=True)
    fidelities = np.array(fidelities, dtype=complex)
    max_fidelities = np.max(fidelities, axis=1)
    arg_fidelities = np.argmax(fidelities, axis=1)
   # print('max fidelities = ' + str(max_fidelities))
   # print('arg fidelities = ' + str(arg_fidelities))
   # print('shape of max fidelities = ' + str(np.shape(max_fidelities)))
   # print('shape of arg fidelities = ' + str(np.shape(arg_fidelities)))
   # print('shape of fidelities = ' + str(np.shape(fidelities)))

    return max_fidelities, arg_fidelities

# This function computes the cost function for a batch of quantum states, coordinates, and labels.
def CostFunction(quantum_states, coordinates, labels,_lambda):
    # Compute prediction for each input in data batch
    loss = 0  # initialize loss
    fidelities, arg_fidelities = Test(quantum_states,labels)
    coordinates = np.array(coordinates)
    arg_fidelities = np.array(arg_fidelities)
    for i in range(len(coordinates)):
        f_i = fidelities[i]
        for j in range(i + 1, (len(coordinates))):
            f_j = fidelities[j]
            if arg_fidelities[i] == arg_fidelities[j]:
                delta_y = 1
            else:
                delta_y = 0
            loss = loss + delta_y * (Distance(coordinates[i], coordinates[j]) + _lambda*Distance(coordinates[i], Centroide(coordinates[np.where(arg_fidelities == arg_fidelities[i])]))) * ((1 - f_i) * (1 - f_j))
    
    return np.real(loss / len(coordinates))

############### Simulation of the cluster sorter ################
############### Data for the simulation ################
# Number of labels
number_labels = 3
# Initialize the labels
labels = StateLabels(number_labels)
# Choose the number of times the whole set of gates is applied
number_iterations = 300
# Choose the step for calculate the gradient
st = 0.1
# Choose the value of the learning rate
lr = 0.005
# Choose the value of lambda for the cost function
_lambda = 1
# Choose the magnetic field value in Tesla
B = 0.1
# Choose the detuning value in MHz
detuning = 0
# Choose the amplitude of the microwave pulse in MHz
Omega = 100


### Test of rotations: ###
angle_sweep = np.linspace(0,2*np.pi, 100)
initial_state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)
for i in range(len(angle_sweep)):
    rho = np.matmul(np.matmul(Rx(angle_sweep[i], Omega, B, detuning), initial_state),np.matrix.conjugate(Rx(angle_sweep[i], Omega, B, detuning).T))
    rotx = ProjZ(rho)
    prY = ProjY(rho)
    prX = ProjX(rho) 
    plt.scatter(angle_sweep[i], rotx)
    plt.scatter(angle_sweep[i], prX, marker='v')
    plt.scatter(angle_sweep[i], prY, marker = 's')
    plt.scatter(angle_sweep[i], np.trace(rho), marker = '.')
    print(np.trace(rho))
plt.title('Rotation in X')
plt.xlabel('rotation angle(rad)')
plt.ylabel('Projections')
plt.show()

for i in range(len(angle_sweep)):
    rho = np.matmul(np.matmul(Ry(angle_sweep[i], Omega, B, detuning), initial_state),np.matrix.conjugate(Ry(angle_sweep[i], Omega, B, detuning).T))
    roty = ProjZ(rho)
    prY = ProjY(rho)
    prX = ProjX(rho)
    plt.scatter(angle_sweep[i], roty)
    plt.scatter(angle_sweep[i], prX, marker='v')
    plt.scatter(angle_sweep[i], prY, marker = 's')
    print(np.trace(rho))
plt.title('Rotation in Y')
plt.xlabel('rotation angle(rad)')
plt.ylabel('Projections')
plt.show()

###############################################################
# Initialize data set
np.random.seed(123)
# Set up cluster parameters
number_clusters = 3
points_per_cluster = 10
N_points = number_clusters * points_per_cluster
#centers = [(-0.29*np.pi, 0*np.pi), (0*np.pi, 0.12*np.pi), (0.29*np.pi, 0*np.pi), (0*np.pi, -0.12*np.pi)]
centers = [(0.29*np.pi, 0.1*np.pi), (0.5*np.pi, 1*np.pi), (1*np.pi, 0.3*np.pi), (0.1*np.pi, 2*0.12*np.pi)]
width = 0.15
# Initialize arrays for coordinates
coordinates = []

# Generate points within clusters
for i in range(number_clusters):
    # Generate points within current cluster
    for j in range(points_per_cluster):
        # Generate point with Gaussian distribution
        point = np.random.normal(loc=centers[i], scale=width)
        print(point)
        coordinates.append([point[0], point[1], 0])
        plt.scatter(point[0], point[1])
plt.show()
# Convert coordinates to numpy array
coordinates = np.array(coordinates)

# Convert coordinates to quantum states
# The idea is to start with a quantum state in the zero state of the bloch sphere
# That is the state |0> = [1, 0, 0] and the we have to apply the gates to rotate the state to the desired point in the bloch sphere
# The gates are the Rx and Ry functions defined above
# So we start with the state |0>
initial_state = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=complex)

# Initialize the quantum states list
quantum_states_list = []

# Generate the quantum states by applying the gates to the initial state, we apply the gates in the same order for all the points
quantum_states = []
for i in range(N_points):
    # Start with the initial state
    quantum_state = initial_state
    # Apply the gates in the same order for all the points
    # Apply the Rx gate
    quantum_state = np.matmul(np.matmul(Rx(coordinates[i][0], Omega, B, detuning), quantum_state),np.matrix.conjugate(Rx(coordinates[i][0], Omega, B, detuning).T))
    #print(np.trace(quantum_state))
    # Apply the Ry gate
    quantum_state = np.matmul(np.matmul(Ry(coordinates[i][1], Omega, B, detuning), quantum_state),np.matrix.conjugate(Ry(coordinates[i][1], Omega, B, detuning).T))
    # Append the quantum state to the list
    quantum_states.append(quantum_state)
    #print(np.trace(quantum_state))

# We save the quantum states in a list of quantum states
quantum_states_list.append(quantum_states)

# Save the first data
cost_value = CostFunction(quantum_states, coordinates, labels, _lambda)
print(f"Initial cost value: {cost_value}")

# Initialize a cost list to store the cost values
cost_list = []

# Calculate the first fidelities and arg_fidelities
fidelities, arg_fidelities = Test(quantum_states, labels)

# Initialize the parameters for the optimization
# Randomly initialize the parameters for the gates
params = np.random.uniform(low=0.0, high=0.1 ,size=2) * 2 * np.pi

# Initialize the gradient
gradient = np.zeros_like(params)

# Optimization loop
# =============================================================================
# for iteration in range(number_iterations):
#     # Calculate the cost function
#     cost_value = CostFunction(quantum_states, coordinates, labels, _lambda)
#     cost_list.append(cost_value)
#     
#     # Calculate the gradient
#     for i in range(len(params)):
#         # Perturb the parameter
#         params[i] += st
#         quantum_states_perturbed = []
#         for j in range(N_points):
#             #quantum_state = initial_state.copy()
#             #quantum_state = quantum_states[j]
#             quantum_state = quantum_states_list[0][j]            
#             for k in range(number_labels):
#                 quantum_state = np.matmul(np.matmul(Rx(params[0], Omega, B, detuning), quantum_state), np.matrix.conjugate(Rx(params[0], Omega, B, detuning).T))
#                 quantum_state = np.matmul(np.matmul(Ry(params[1], Omega, B, detuning), quantum_state), np.matrix.conjugate(Ry(params[1], Omega, B, detuning).T))
#             quantum_states_perturbed.append(quantum_state)
#             quantum_states[j] = quantum_state
#         perturbed_cost = CostFunction(quantum_states_perturbed, coordinates, labels, _lambda)
#         print('Perturbed cost:', perturbed_cost)
#         # Calculate the gradient
#         gradient[i] = (perturbed_cost - cost_value) / st
#         print('Gradient:', gradient)
#         # Reset the parameter
#         params[i] -= st
#     
#     # Update the parameters using the gradient descent step
#     params -= lr * gradient
#     
#     print(f"Iteration {iteration + 1}, Cost: {cost_value}, Params: {params}")

#New optimization loop for testing---Asier#
for iteration in range(number_iterations):
    
    quantum_states_unperturbed = []
    for j in range(N_points):
        point_state = quantum_states[j]
        quantum_state = np.matmul(np.matmul(Rx(params[0], Omega, B, detuning), point_state), np.matrix.conjugate(Rx(params[0], Omega, B, detuning).T))
        quantum_state = np.matmul(np.matmul(Ry(params[1], Omega, B, detuning), quantum_state), np.matrix.conjugate(Ry(params[1], Omega, B, detuning).T))
        quantum_states_unperturbed.append(quantum_state)
        #print(np.trace(quantum_state))
    cost_value = CostFunction(quantum_states_unperturbed, coordinates, labels, _lambda)
    cost_list.append(cost_value)
    
    for i in range(len(params)):
        params[i] = params[i] + st
        quantum_states_perturbed = []
        for j in range(N_points):
            point_state = quantum_states[j]
            quantum_state = np.matmul(np.matmul(Rx(params[0], Omega, B, detuning), point_state), np.matrix.conjugate(Rx(params[0], Omega, B, detuning).T))
            quantum_state = np.matmul(np.matmul(Ry(params[1], Omega, B, detuning), quantum_state), np.matrix.conjugate(Ry(params[1], Omega, B, detuning).T))
            quantum_states_perturbed.append(quantum_state)
            #print(np.trace(quantum_state))
        perturbed_cost = CostFunction(quantum_states_perturbed, coordinates, labels, _lambda)
        # Calculate the gradient
        gradient[i] = (perturbed_cost - cost_value) / st
        # Reset the parameter
        params[i] -= st
    params -= lr*gradient
    print(f"Iteration {iteration + 1}, Cost: {cost_value}, Params: {params}")
# =============================================================================

# After the optimization loop, we can print the final parameters and cost value
print(f"Final parameters: {params}")
print(f"Final cost value: {cost_value}")

# Plot the cost function values over iterations

plt.plot(cost_list)
plt.xlabel('Iteration')
plt.ylabel('Cost Value')
plt.title('Cost Function Value Over Iterations')
plt.grid()
plt.show()


# Now, plot the quantum states on the Bloch sphere with the arg_fidelities with different colors
from mpl_toolkits.mplot3d import Axes3D
def plot_bloch_sphere(quantum_states, arg_fidelities):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for each label
    colors = ['r', 'g', 'b', 'y']
    
    for i, state in enumerate(quantum_states):
        x, y, z = BlochCoordinates(state)
        ax.scatter(x, y, z, color=colors[arg_fidelities[i]], label=f'Point {i+1}')
    
    # Create a Bloch sphere centered at (0.5, 0.5, 0.5) with radius 0.5
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_sphere = 0.5 + 0.5 * np.cos(u) * np.sin(v)
    y_sphere = 0.5 + 0.5 * np.sin(u) * np.sin(v)
    z_sphere = 0.5 + 0.5 * np.cos(v)
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2, linewidth=0)

    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere Representation of Quantum States')
    
    plt.show()
    
m, arg_fidelities_labels = Test(labels, labels)
m, arg_fidelities = Test(quantum_states_unperturbed, labels)  
plot_bloch_sphere(labels, arg_fidelities_labels)    
plot_bloch_sphere(quantum_states_unperturbed, arg_fidelities)
