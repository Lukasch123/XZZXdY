import matplotlib.pyplot as plt
import numpy as np
import modified_app
from colour import Color

data = modified_app.read_data() #insert own filepath here

code_to_xysPf = {}      #Total logical failure rate
code_to_xysPfZ = {}     #Bit-flip failure rate

for data_run in data:
    xysPf = code_to_xysPf.setdefault(data_run['code'], [])
    xysPfy = code_to_xysPfZ.setdefault(data_run['code'], [])
    #Can also choose 'data_run['n_k_d'][2]' as x-axis inorder to plot code distance
    xysPf.append((data_run['error_probability'], data_run['logical_failure_rate']))
    #Can choose ['n_logical_commutations'][0] for P_{fX}, [1] for P_{fY} or [2] for P_{fZ} 
    xysPfy.append((data_run['error_probability'], data_run['n_logical_commutations'][2]/data_run['n_run']))

# Analytical results with phase-bias eta and code distance d
Pfz = lambda eta, d: 1/2-1/2*np.exp(-2*(2*d-1)*np.arctanh(1/(2*eta)))
Pf = lambda eta, d: 3/4-1/4*np.exp(-2*(2*d-1)*np.arctanh(1/(2*eta)))

#Phase-bias parameter
eta = 300

#The simulated distances
distances = np.array([9, 15, 21, 27, 33, 39, 45, 51])

orange = Color("orange")
colors_or = list(orange.range_to(Color("red"), 8))
color_strings_or = [str(color) for color in colors_or]

for code, xys in code_to_xysPf.items():
    plt.plot(*zip(*xys), 'x-', label=f'$d$ = {code[18:]}', color=color_strings_or[int((int(code[18:])-9)/6)])

index = 0
for distance in distances:
    plt.scatter((1+1/eta)/(2+1/eta), Pf(eta, distance), color=color_strings_or[index])
    index += 1

for code, xys in code_to_xysPfZ.items():
    plt.plot(*zip(*xys), 'x', linestyle='dashed', color=color_strings_or[int((int(code[18:])-9)/6)])

index = 0
for distance in distances:
    plt.scatter((1+1/eta)/(2+1/eta), Pfz(eta, distance), color=color_strings_or[index])
    index += 1

handles, labels = plt.gca().get_legend_handles_labels()
custom_lines = [plt.Line2D([], [], color=color_strings_yr[0], linestyle='-'), plt.Line2D([], [], color=color_strings_yr[0], linestyle='--')]

custom_tick_position = (1+1/eta)/(2+1/eta)
custom_tick_label = r'$p_s$'
existing_ticks = plt.xticks()[0]
existing_ticks = [tick for tick in existing_ticks if abs(tick - 0.5) > 0.01]
existing_labels = plt.xticks()[1]
existing_labels = [label for label, tick in zip(existing_labels, existing_ticks) if abs(tick - 0.5) > 0.01]

custom_ticks = list(existing_ticks) + [custom_tick_position]
custom_labels = list(existing_labels) + [custom_tick_label]
plt.xticks(custom_ticks, custom_labels)

plt.xlabel(r'Fysisk felsannolikhet, $p$')
plt.ylabel(f'Logisk felsannolikhet')
plt.title(f'XZZXdY, $\eta =$ {eta}')
plt.ylim(0, 0.9)
plt.xlim(0, 0.52)
plt.vlines((1+1/eta)/(2+1/eta), 0, 1, color='red', alpha=0.5, linestyles='dotted')
plt.legend(handles + custom_lines, labels + [r'$P_f$', r'$P_{fZ}$'])
plt.show()
