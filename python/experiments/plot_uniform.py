import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

# Add path to import uniform variant
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from minizinc.reference_topology_uniform import CITY_COORDS, DEFAULT_CENTRAL, DEFAULT_EDGE

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
})

fig, ax = plt.subplots(figsize=(10, 9))

# Loading map background
map_img_path = "mapa.png" # TODO: Provide path to your map image
if os.path.exists(map_img_path):
    img = mpimg.imread(map_img_path)
    ax.imshow(img, extent=[13.8, 24.4, 48.8, 55.4], alpha=0.5, aspect='auto')

# Drawing cities with differentiation
central_lats, central_lons = [], []
edge_lats, edge_lons = [], []
pop_lats, pop_lons = [], []

for city, (lat, lon) in CITY_COORDS.items():
    if city in DEFAULT_CENTRAL:
        central_lats.append(lat)
        central_lons.append(lon)
    elif city in DEFAULT_EDGE:
        edge_lats.append(lat)
        edge_lons.append(lon)
    else:
        pop_lats.append(lat)
        pop_lons.append(lon)

# Scatter plots for 3 types
ax.scatter(central_lons, central_lats, c='blue', s=120, marker='s', edgecolors='black', label='Central cluster', zorder=5)
ax.scatter(edge_lons, edge_lats, c='orange', s=80, marker='^', edgecolors='black', label='Edge cluster', zorder=4)
ax.scatter(pop_lons, pop_lats, c='gray', s=40, marker='o', edgecolors='black', label='Point of Presence (PoP)', zorder=3)

import math
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    return 2 * r * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))

# Logical connections to nearest neighbors
for pop_lat, pop_lon in zip(pop_lats, pop_lons):
    if edge_lats: # Safeguard in case of missing Edge
        dists = [(haversine_km(pop_lat, pop_lon, elat, elon), elat, elon) for elat, elon in zip(edge_lats, edge_lons)]
        dists.sort()
        _, best_lat, best_lon = dists[0]
        ax.plot([pop_lon, best_lon], [pop_lat, best_lat], c='gray', linestyle=':', alpha=0.4, zorder=1)

for edge_lat, edge_lon in zip(edge_lats, edge_lons):
    if central_lats:
        dists = [(haversine_km(edge_lat, edge_lon, clat, clon), clat, clon) for clat, clon in zip(central_lats, central_lons)]
        dists.sort()
        _, best_lat, best_lon = dists[0]
        ax.plot([edge_lon, best_lon], [edge_lat, best_lat], c='black', linestyle='-', alpha=0.6, zorder=2)

ax.set_title("Balanced Topology (Uniform): Node Placement", pad=15, fontweight='bold')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Move legend outside the map
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), fancybox=True, shadow=True, ncol=3)

plt.tight_layout()

out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../latex/tex/img/topologia_rownomierna.pdf'))
fig.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Saved uniform topology to {out_path}")
