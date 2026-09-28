"""REFERENCE topology helper for UNIFORM scenario generation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

@dataclass(frozen=True)
class Node:
    name: str
    lat: float
    lon: float

@dataclass(frozen=True)
class ClusterSpec:
    name: str
    lat: float
    lon: float
    kind: str  # "central" or "edge"

# The 20 MAN / Clusters are identical to the base topology
# to allow for 1:1 cross-evaluation of ML models.
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    # 20 MAN / Clusters
    "Gdansk": (54.3520, 18.6466), "Olsztyn": (53.7784, 20.4801), "Bialystok": (53.1325, 23.1688),
    "Warszawa": (52.2297, 21.0122), "Lodz": (51.7592, 19.4560), "Poznan": (52.4064, 16.9252),
    "Bydgoszcz": (53.1235, 18.0084), "Torun": (53.0138, 18.5984), "Szczecin": (53.4285, 14.5528),
    "Koszalin": (54.1944, 16.1722), "ZielonaGora": (51.9356, 15.5062), "Wroclaw": (51.1079, 17.0385),
    "Opole": (50.6751, 17.9213), "Czestochowa": (50.8118, 19.1203), "Kielce": (50.8661, 20.6286),
    "Radom": (51.4027, 21.1471), "Lublin": (51.2465, 22.5684), "Rzeszow": (50.0412, 21.9991),
    "Krakow": (50.0647, 19.9450), "Gliwice": (50.2945, 18.6714),
    
    # 30 UNIFORM Additional PoPs (Geographically balanced)
    # Reduced Silesia to 3, added 10 new PoPs to isolated regions
    "Katowice": (50.2649, 19.0238), "Gdynia": (54.5189, 18.5305), "Rybnik": (50.0971, 18.5417),
    "BielskoBiala": (49.8225, 19.0444), "Elblag": (54.1561, 19.4045), "Plock": (52.5461, 19.7064),
    "Walbrzych": (50.7671, 16.2814), "Wloclawek": (52.6484, 19.0678), "Tarnow": (50.0125, 20.9858),
    "Kalisz": (51.7611, 18.0911), "Legnica": (51.2071, 16.1611), "Grudziadz": (53.4841, 18.7536),
    "Slupsk": (54.4641, 17.0284), "JeleniaGora": (50.9044, 15.7189), "NowySacz": (49.6211, 20.6961),
    "Konin": (52.2233, 18.2511), "PiotrkowTryb": (51.4056, 19.7033), "Siedlce": (52.1669, 22.2908),
    "Inowroclaw": (52.7984, 18.2611), "Lubin": (51.4011, 16.2011),
    
    # 10 NEW Uniform PoPs
    "Stargard": (53.3380, 15.0400), "Swinoujscie": (53.9100, 14.2470), "Kolobrzeg": (54.1750, 15.5830),
    "Szczecinek": (53.7080, 16.6990), "Suwalki": (54.0990, 22.9270), "Lomza": (53.1780, 22.0590),
    "Chelm": (51.1410, 23.4720), "Zamosc": (50.7230, 23.2510), "Przemysl": (49.7830, 22.7670),
    "Gniezno": (52.5340, 17.5820),
}

DEFAULT_CENTRAL = ["Gdansk", "Poznan", "Warszawa", "Krakow", "Wroclaw"]
DEFAULT_EDGE = [
    "Szczecin",
    "Koszalin",
    "Bydgoszcz",
    "Torun",
    "Olsztyn",
    "Bialystok",
    "Lodz",
    "Opole",
    "Kielce",
    "Lublin",
    "Rzeszow",
    "ZielonaGora",
    "Czestochowa",
    "Radom",
    "Gliwice",
]

def haversine_km(a: Node, b: Node) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    lat1 = math.radians(a.lat)
    lon1 = math.radians(a.lon)
    lat2 = math.radians(b.lat)
    lon2 = math.radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(s))

def build_clusters(
    central_names: Iterable[str],
    edge_names: Iterable[str],
) -> List[ClusterSpec]:
    clusters: List[ClusterSpec] = []
    for name in central_names:
        lat, lon = CITY_COORDS[name]
        clusters.append(ClusterSpec(name=name, lat=lat, lon=lon, kind="central"))
    for name in edge_names:
        lat, lon = CITY_COORDS[name]
        clusters.append(ClusterSpec(name=name, lat=lat, lon=lon, kind="edge"))
    return clusters

def build_pops(
    count: int,
    seed: int,
    candidate_names: Iterable[str] | None = None,
    jitter_deg: float = 0.15,
) -> List[Node]:
    rng = random.Random(seed)
    names = list(candidate_names) if candidate_names else list(CITY_COORDS.keys())
    pops: List[Node] = []
    for i in range(count):
        base_name = rng.choice(names)
        lat, lon = CITY_COORDS[base_name]
        lat += rng.uniform(-jitter_deg, jitter_deg)
        lon += rng.uniform(-jitter_deg, jitter_deg)
        pops.append(Node(name=f"Pop{i+1}", lat=lat, lon=lon))
    return pops

def distance_matrix(pops: List[Node], clusters: List[ClusterSpec]) -> List[List[int]]:
    matrix: List[List[int]] = []
    for pop in pops:
        row: List[int] = []
        for cluster in clusters:
            dist = haversine_km(pop, Node(cluster.name, cluster.lat, cluster.lon))
            row.append(int(round(dist)))
        matrix.append(row)
    return matrix
