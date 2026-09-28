import numpy as np
import matplotlib.pyplot as plt
import os

def erlang_b(c, a):
    if a == 0:
        return 0.0
    inv_b = 1.0
    for i in range(1, c + 1):
        inv_b = 1.0 + inv_b * (i / a)
    return 1.0 / inv_b

a_vals = np.linspace(0.1, 150, 500)
c_vals = [20, 50, 100]

# Slight widening of the plot to make room for the legend on the side
plt.figure(figsize=(9, 5))
for c in c_vals:
    p_b = [erlang_b(c, a) * 100 for a in a_vals]  # in percentages
    plt.plot(a_vals, p_b, label=f'$m = {c}$')

plt.axhline(y=1.0, color='r', linestyle='--', label=r'$P_B^{\max} = 1\%$')

plt.xlabel('Offered traffic A [Erlang]')
plt.ylabel('Blocking probability $P_B$ [%]')
plt.ylim(-0.5, 20)
plt.xlim(0, 150)
plt.grid(True, linestyle=':', alpha=0.7)

# Move the legend outside the plot (top right corner)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../latex/tex/img/krzywa_erlanga.pdf'))
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f"Plot saved to {out_path}")
