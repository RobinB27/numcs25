import numpy as np
import matplotlib.pyplot as plt

"""
    Shows a vector y in both Standard & Trigonometric Basis (via discrete fourier transform)
"""

def root_of_unity(N: int):
    return np.exp( -2.0j*np.pi / N )


def fourier_matrix(N: int):
    F = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            F[i][j] = root_of_unity(N)**(i*j)
    return F


def test_fourier():
    """
        Compare with Numpy's DFT.
    """
    y = np.array([1, 2, 3 + 1.0j, 4, 5, 6, 7, 8], dtype=complex) 
    c_np = np.fft.fft(y)
    
    F = fourier_matrix(y.size)
    c_fm = F @ y
    
    print("Numpy:\n", c_np)
    print("Fourier Matrix:\n", c_fm)
    print("Error:\n", np.abs( c_np-c_fm))
    print("Max Error:\n", np.max( np.abs( c_np-c_fm ) ))
    

def plot_bargraph(ax, y: np.ndarray):
    """
        Draws a bar graph on 'ax' for y, showing all components real & imag values
    """
    pos = np.array([0, 0.25])       # Distance between bars
    pos -= 0.125                    # Center correctly
    width = 0.25; multiplier = 0

    for i in range(n):
        offset = width * multiplier
        measurement = ([np.real(y)[i], np.imag(y)[i]])
        rects = ax.bar(pos + offset, measurement, width, color=["lightsteelblue", "slategrey"])
        # axs[0].hlines(0, 0, n)        # Value numbers next to each bar
        multiplier += 4

# Parameters to set: the vector to display

y = np.array([1 + 8j, 2 + 7j, 3 + 6j, 4 + 5j, 5 + 4j, 6 + 3j, 7 + 2j, 8 + 1j], dtype=complex)  

# Experiment Runner

test_fourier()

n = y.size
F = fourier_matrix(n)
y_trig = F @ y
y_trig = y_trig / n     # Scale correctly

# Plotting

N = np.arange(1, n+1, 1)

figs, axs = plt.subplots(1, 2)

plot_bargraph(axs[0], y)
axs[0].hlines(0, 0, n)
    
axs[0].set_title("y: Standard Basis")
axs[0].set_xticks(np.arange(n), np.arange(n))
axs[0].set_xlabel("Vector Components")

plot_bargraph(axs[1], y_trig)
axs[1].hlines(0, 0, n)

axs[1].set_title("y: Trig. Basis")
axs[1].set_xticks(np.arange(n), np.arange(n))
axs[1].set_xlabel("Vector Components")

plt.show()