# GWMemoryModel
Time- and frequency-domain waveform models for the gravitational wave memory effect for nonspinning black-hole binaries.

# Overview
This code implements waveform models for the gravitational-wave memory effect, based on the work developed in the following three papers:

Paper I: Arwa Elhashash and David A. Nichols. "Waveform models for the gravitational-wave memory effect: Extreme mass-ratio limit and final memory offset". [Phys. Rev. D **111**, 044052 (2025)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.044052). [ArXiv:2407.19017.](https://arxiv.org/abs/2407.19017)  
Paper II: Arwa Elhashash and David A. Nichols. "Waveform models for the gravitational-wave memory effect: II. Time-domain and frequency-domain models for nonspinning binaries". [Phys. Rev. D **112**, 064014 (2025)](https://journals.aps.org/prd/abstract/10.1103/wzqk-62wc). [ArXiv:2504.18635.](https://arxiv.org/abs/2504.18635)
Paper III: Arwa Elhashash and David A. Nichols. "Waveform models for the gravitational-wave memory effect: III. Phenomenological frequency-domain model for nonspinning binaries". [Arxiv:2609.04340.](https://arxiv.org/abs/2609.04340)

The model provides time-domain and frequency-domain waveforms for the gravitational-wave memory effect for nonspinning binary black hole systems, covering the inspiral, intermediate, and ringdown phases.

# Features

Time-domain Inspiral Memory Model: Post-Newtonian waveform computation for the inspiral phase  
Time-domain Ringdown Memory Model: Quasi-normal mode (QNM) based model for the ringdown phase  
Time-domain Intermediate Memory Model: Smooth transition phenomenological model between inspiral and ringdown phases  
Time-domain Full Memory Model: Complete waveform covering all three phases  
Frequency-domain Full Memory Model: Phenomenological model 
Mass Ratio Range: Supports binary systems with mass ratios q ∈ [1, 8]  
Nonspinning Systems: Calibrated for binary black holes with zero initial spins

# Dependencies
This code requires the following Python packages that can be installed using pip or conda:
- [numpy](https://numpy.org/doc/stable/index.html)
- [qnm](https://pypi.org/project/qnm/) 
- [surfinBH](https://pypi.org/project/surfinBH/)

# Installation and Usage

1. Clone or download the repository
2. Import the module in your Python script
## Example

```python
import GWMemoryModel as gwm

# Define mass ratio (m1/m2 where m1 >= m2)
q = 2.0  # Mass ratio between 1 and 8

# Compute full time-domain memory waveform
result = gwm.compute_TD_memory_model(q=q, outputs='full')

# Extract time and memory arrays
time = result['time']
memory = result['memory']

# Plot the result
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(time, memory)
plt.xlabel(r'$t/M$')
plt.ylabel(r'$r h_{20}/M$')
plt.title(r'Memory Strain ($q=2$)')
plt.show()
```
![memory_q2.pdf](https://github.com/user-attachments/files/21182690/memory_q2.pdf)


```python

# Define mass ratio (m1/m2 where m1 >= m2) and frequency (between 1e-6 and 5e-2)
q = 2.0  # Mass ratio between 1 and 8
freq = np.linspace(1e-6, 5e-2, 2000)

# Compute frequency-domain memory waveform
result = gwm.compute_FD_memory_model(q=q, freq=freq)

# Plot the amplitude
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(freq, np.abs(result))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M f$')
plt.ylabel(r'$r|\tilde{h}_{20}|/M^2$')
plt.title(r'Frequency-domain Memory Amplitude ($q=2$)')
plt.show()
```
![memory_amp_q2.pdf](https://github.com/user-attachments/files/31879752/memory_amp_q2.pdf)
