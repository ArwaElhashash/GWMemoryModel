# GWMemoryModel
Time-domain model for the gravitational wave memory effect for nonspinning black-hole binaries.

# Overview
This code implements waveform models for the gravitational-wave memory effect, based on the work developed in the following two papers:

Paper I: Arwa Elhashash and David A. Nichols. "Waveform models for the gravitational-wave memory effect: Extreme mass-ratio limit and final memory offset". [Phys. Rev. D **111**, 044052 (2025)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.044052). [ArXiv:2407.19017.](https://arxiv.org/abs/2407.19017)  
Paper II: Arwa Elhashash and David A. Nichols. "Waveform models for the gravitational-wave memory effect: II. Time-domain and frequency-domain models for nonspinning binaries". [ArXiv:2504.18635.](https://arxiv.org/abs/2504.18635)

The model provides time-domain waveforms for the gravitational-wave memory effect for nonspinning binary black hole systems, covering the inspiral, intermediate, and ringdown phases.

# Features

Inspiral Memory Model: Post-Newtonian waveform computation for the inspiral phase  
Ringdown Memory Model: Quasi-normal mode (QNM) based model for the ringdown phase  
Intermediate Memory Model: Smooth transition phenomenological model between inspiral and ringdown phases  
Full Memory Model: Complete waveform covering all three phases  
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

# Compute full memory waveform
result = gwm.compute_memory_model(q=q, outputs='full')

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



