import matplotlib.pyplot as plt
import numpy as np
import modified_app
from colour import Color

data = modified_app.read_data(r"C:\Users\adamu\OneDrive\Dokument\Skola\Kandidatarbete\Data\chi_test2_eta_300_d5.json")
#data_sim = modified_app.read_data(r"C:\Users\adamu\OneDrive\Dokument\Skola\Kandidatarbete\Data\chi_test_eta_300_d5-9.json")
print(data)

code_to_xysPf = {}
code_to_xysPfy = {}

for data_run in data:
    xysPf = code_to_xysPf.setdefault(data_run['code'], [])
    xysPfy = code_to_xysPfy.setdefault(data_run['code'], [])
    xysPf.append((data_run['error_probability'], data_run['logical_failure_rate']))
    xysPfy.append((data_run['error_probability'], data_run['n_logical_commutations'][1]/data_run['n_run']))
    #xys.append((data_run['error_probability'], data_run['n_logical_commutations'][1]/data_run['n_run']))

Pfz = lambda eta, d: 1/2-1/2*np.exp(-2*(2*d-1)*np.arctanh(1/(2*eta)))
Pf = lambda eta, d: 3/4-1/4*np.exp(-2*(2*d-1)*np.arctanh(1/(2*eta)))

distances = np.array([9, 15, 21, 27, 33, 39, 45, 51])

blue = Color("blue")
colors_bg = list(blue.range_to(Color("cyan"), 8))
color_strings_bg = [str(color) for color in colors_bg]

orange = Color("orange")
colors_yr = list(orange.range_to(Color("red"), 8))
color_strings_yr = [str(color) for color in colors_yr]

for code, xys in code_to_xysPf.items():
    for index, list in enumerate(zip(*xys)):
        if index == 0:
            x_list = list
        else:
            y_list = list
    plt.plot(x_list[::2], y_list[::2], 'x-', label=f'$\chi = 40$ and $d$ = {code[19:]}', color=color_strings_yr[int((int(code[19:])-9)/6)])
    plt.plot(x_list[1::2], y_list[1::2], 'o-', label=f'$\chi = 10$ and $d$ = {code[19:]}', color=color_strings_yr[int((int(code[19:])-9)/6)])

eta = 300

#for code, xys in code_to_xysPf.items():
    #plt.plot(*zip(*xys), 'x-', color=color_strings_yr[int((int(code[18:])-9)/6)], label=f'$d$ = {code[18:]}')

index = 0
for distance in distances:
    plt.scatter((1+1/eta)/(2+1/eta), Pf(eta, distance), color=color_strings_yr[index])
    index += 1

for code, xys in code_to_xysPfy.items():
    for index, list in enumerate(zip(*xys)):
        if index == 0:
            x_list = list
        else:
            y_list = list
    plt.plot(x_list[::2], y_list[::2], 'x-', color=color_strings_yr[int((int(code[19:])-9)/6)])
    plt.plot(x_list[1::2], y_list[1::2], 'o-', color=color_strings_yr[int((int(code[19:])-9)/6)])

#for code, xys in code_to_xysPfy.items():
    #plt.plot(*zip(*xys), 'x', linestyle='dashed', color=color_strings_yr[int((int(code[18:])-9)/6)])

index = 0
for distance in distances:
    plt.scatter((1+1/eta)/(2+1/eta), Pfz(eta, distance), color=color_strings_yr[index])
    index += 1
'''
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
'''
plt.xlabel(r'Fysisk felsannolikhet, $p$')
plt.ylabel(f'Felsannolikhet, $\eta$ = {eta}')
plt.title('XY')
plt.ylim(0, 0.9)
plt.xlim(0, 0.52)
plt.vlines((1+1/eta)/(2+1/eta), 0, 1, color='red', alpha=0.5, linestyles='dotted')
#plt.legend(handles + custom_lines, labels + [r'$P_f$', r'$P_{fZ}$'])
plt.legend()
plt.show()