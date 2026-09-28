import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath("."))
from minizinc.reference_topology import CITY_COORDS, DEFAULT_CENTRAL, DEFAULT_EDGE

# Extract all cities
cities = list(CITY_COORDS.keys())
lats = [CITY_COORDS[c][0] for c in cities]
lons = [CITY_COORDS[c][1] for c in cities]

# Assign "demand" (simplified visualization of asymmetry)
# We give higher weight to Silesian cities and main hubs
silesia = ["Katowice", "Gliwice", "Zabrze", "Bytom", "Rybnik", "Tychy", "DabrowaGornicza", 
           "Chorzow", "Jaworzno", "JastrzebieZdroj", "Myslowice", "SiemianowiceSlaskie", 
           "TarnowskieGory", "PiekarySlaskie", "Bedzin", "Swietochlowice"]
demand = []
for c in cities:
    if c in silesia:
        demand.append(100) # High demand Silesia
    elif c in DEFAULT_CENTRAL:
        demand.append(60) # Medium demand central cities
    else:
        demand.append(20) # Low demand

plt.figure(figsize=(9, 8))

# Draw heatmap/scatter
scatter = plt.scatter(lons, lats, s=[d * 10 for d in demand], c=demand, cmap='Reds', alpha=0.7, edgecolors='k')
plt.colorbar(scatter, label='Traffic density (demand size)')

# Label central hubs
for c in DEFAULT_CENTRAL:
    plt.text(CITY_COORDS[c][1] + 0.1, CITY_COORDS[c][0] + 0.1, c, fontsize=9, fontweight='bold')

plt.title('Visualization of spatial traffic asymmetry (REFERENCE Topology)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../latex/tex/img/przestrzenna_mapa_ciepla.pdf'))
plt.savefig(out_path, format='pdf')
print(f"Plot saved to {out_path}")
