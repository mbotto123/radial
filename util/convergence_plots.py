import matplotlib.pyplot as plt

dim = '2D'
ppr_type = 'Discrete'

#------------------------------------------------------------------------------#
# P1 L^2 error
#------------------------------------------------------------------------------#

file_name = f'../build/Uniform_Poisson_{dim}_P1_{ppr_type}PPR.txt'

h = []
error_L2 = []
error_L2_enriched = []

with open(file_name, 'r') as file:
  next(file)

  for line in file:
    vals = line.split()

    h.append(float(vals[2]))
    error_L2.append(float(vals[3]))
    error_L2_enriched.append(float(vals[5]))

plt.figure(1)

plt.loglog(h, error_L2, 'o-', color='red', label='P1 Solution')
plt.loglog(h, error_L2_enriched, '^--', color='orange', label='P2 Enriched Solution')

#------------------------------------------------------------------------------#
# P2 L^2 error
#------------------------------------------------------------------------------#

file_name = f'../build/Uniform_Poisson_{dim}_P2_{ppr_type}PPR.txt'

h = []
error_L2 = []
error_L2_enriched = []

with open(file_name, 'r') as file:
  next(file)

  for line in file:
    vals = line.split()

    h.append(float(vals[2]))
    error_L2.append(float(vals[3]))
    error_L2_enriched.append(float(vals[5]))

plt.loglog(h, error_L2, 'o-', color='blue', label='P2 Solution')
plt.loglog(h, error_L2_enriched, '^--', color='purple', label='P3 Enriched Solution')

plt.xlabel(r'$h_E = \mathrm{1/(Elements)}^{1/2}$')
plt.ylabel(r'$L^2$ Error')

plt.grid(True, which='minor', linestyle=':')
plt.legend()

#------------------------------------------------------------------------------#
# P1 H^1 semi-norm error
#------------------------------------------------------------------------------#

file_name = f'../build/Uniform_Poisson_{dim}_P1_{ppr_type}PPR.txt'

h = []
error_H1semi = []
error_H1semi_enriched = []

with open(file_name, 'r') as file:
  next(file)

  for line in file:
    vals = line.split()

    h.append(float(vals[2]))
    error_H1semi.append(float(vals[7]))
    error_H1semi_enriched.append(float(vals[9]))

plt.figure(2)

plt.loglog(h, error_H1semi, 'o-', color='red', label='P1 Solution')
plt.loglog(h, error_H1semi_enriched, '^--', color='orange', label='P2 Enriched Solution')

#------------------------------------------------------------------------------#
# P2 H^1 semi-norm error
#------------------------------------------------------------------------------#

file_name = f'../build/Uniform_Poisson_{dim}_P2_{ppr_type}PPR.txt'

h = []
error_H1semi = []
error_H1semi_enriched = []

with open(file_name, 'r') as file:
  next(file)

  for line in file:
    vals = line.split()

    h.append(float(vals[2]))
    error_H1semi.append(float(vals[7]))
    error_H1semi_enriched.append(float(vals[9]))

plt.loglog(h, error_H1semi, 'o-', color='blue', label='P2 Solution')
plt.loglog(h, error_H1semi_enriched, '^--', color='purple', label='P3 Enriched Solution')

plt.xlabel(r'$h_E = \mathrm{1/(Elements)}^{1/2}$')
plt.ylabel(r'$H^1$ Semi-norm Error')

plt.grid(True, which='minor', linestyle=':')
plt.legend()

plt.show()