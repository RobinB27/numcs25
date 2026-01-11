import numpy as np
import matplotlib.pyplot as plt

"""
    Uses FFT to remove noise from a signal, extracting only the original frequencies
"""

def power_spectrum(coeffs: np.ndarray):
    return np.abs( coeffs )**2 / coeffs.size

def eval_freq_sin(x: np.ndarray, freqs: np.ndarray, add_noise=True, noise_mult=2):
    """
        Evaluate a function made of sin waves at the frequencies given
    """
    y = np.zeros(x.size)
    for fq in freqs:
        y += np.sin( fq * 2*np.pi * x )
    if (add_noise): y += noise_mult*np.random.randn(N)
    return y


def high_pass(coeffs: np.ndarray, tolerance: int):
    """"
        Removes all coefficients with a power spectrum weaker than tolerance given
    """
    n = coeffs.size
    coeffs_filtered = np.zeros(n, dtype=complex)
    py = power_spectrum(coeffs)
    
    for i in range(n):
        if (py[i] >= tolerance): 
            coeffs_filtered[i] = coeffs[i] 
    
    return coeffs_filtered

# Parameters to set

freqs = [10, 20, 30]    # Frequencies for the function
N = 2 ** 7              # Points to evaluate on
add_noise = True    
noise_strength = 2      # Multiplier for noise
tolerance = 25          # Filter out all coeffs with p-spectrum less than this

# Experiment runner

x = np.arange(N) / N
y_clean = eval_freq_sin(x, freqs, False)
y_noise = eval_freq_sin(x, freqs, add_noise, noise_strength)

coeffs = np.fft.fft(y_noise)
py = power_spectrum(coeffs)  

coeffs_filtered = high_pass(coeffs, tolerance)
py_filtered = power_spectrum(coeffs_filtered)
y_filtered = np.fft.ifft(coeffs_filtered)

print("Remaining coefficients after filtering:\n")
count = 0
for i in range(coeffs.size):
    if (coeffs_filtered[i] == 0): continue
    print(coeffs[i]); count+= 1
print(f"\nFrequencies:\t {N}\nFiltered:\t {count}")

# Plotting

figs, axs = plt.subplots(2, 2)

axs[0][0].plot(x, y_noise, color="lightsteelblue")
# axs[0][0].plot(x, y_clean)
axs[1][0].plot(x, y_filtered, color="slategrey")

axs[0][0].set_title("Function")
axs[0][0].grid()
axs[1][0].set_title("Filtered")
axs[1][0].grid()

axs[0][1].bar(np.arange(0, N//2), py[:N//2], color="lightsteelblue")
axs[1][1].bar(np.arange(0, N//2), py_filtered[:N//2], color="slategrey")

axs[0][1].set_title("Power Spectrum")
axs[0][1].grid()
axs[1][1].set_title("Filtered")
axs[1][1].grid()

plt.show()