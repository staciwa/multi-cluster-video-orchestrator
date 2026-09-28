% input data
include "data.dzn";


%% 1. PARAMETRY

% resource types: vCPU, RAM, ...
enum ResourceTypes;

% capacity of one SRS instance
int: SRSCapacity;

% maximum allowable latency in [ms]
int: MaxLatency;

% transmission cost per 1 km
int: TransmissionCost;

% average latency per 10km transmission in [ms]
% per 10 km, to be able to provide value in int
int: LatencyPer10km;

% koszt migracji
int: MigrationCost;

% penalty for unserved demand
int: UnservedPenalty;

% number of clusters
int: Clusters_;

% number of demands
int: Demands_;

% cluster set
set of int: Clusters = 1..Clusters_;

% demand set
set of int: Demands = 1..Demands_;

% resources in clusters
array [Clusters, ResourceTypes] of int: Resources;

% required amount of resource $r$ for one SRS instance
array [ResourceTypes] of int: SRSResources;

% distance of demand location from cluster in [km]
array [Demands, Clusters] of int: Distances;

% initial demand allocation ("from time t-1")
array [Demands, Clusters] of int: InitialPlacement;

% SRS instance deployment cost in cluster c
array [Clusters] of int: DeploymentCost;

% Weight for Cost
float: Alpha_Cost;

% Weight for Latency
float: Beta_Latency;   



%% 2. ZMIENNE

% Placement[d,c] = 1 means that demand d is handled by cluster c
array [Demands, Clusters] of var 0..1: Placement;

% Static instance limit for each cluster
array[Clusters] of int: MaxPhysicalInstances = [
  min([ Resources[c,rt] div SRSResources[rt] | rt in ResourceTypes ])
  | c in Clusters
];

% Number of running SRS instances in cluster c
array[Clusters] of var int: Instances;

% liczba migracji
var int: Migrations = sum(d in Demands, c in Clusters) (
  bool2int(InitialPlacement[d,c] == 1 /\ Placement[d,c] == 0)
);

% 1 if demand served, 0 otherwise
array[Demands] of var 0..1: IsUnserved;

% Resource utilization in each cluster
array[Clusters, ResourceTypes] of var int: UsedResources = 
  array2d(Clusters, ResourceTypes, [ Instances[c] * SRSResources[rt] 
                                        | c in Clusters, rt in ResourceTypes ]);

%% 3. OGRANICZENIA

% input data validation
constraint card(ResourceTypes) > 0 /\ SRSCapacity > 0 /\ MaxLatency > 0 /\ TransmissionCost > 0
            /\ LatencyPer10km > 0 /\ Clusters_ > 0 /\ Demands_ > 0 /\ MigrationCost >= 0 /\ UnservedPenalty >= 0
            /\ forall(c in Clusters) (DeploymentCost[c] > 0)
            /\ forall(c in Clusters, rt in ResourceTypes) (Resources[c, rt] > 0)
            /\ forall(rt in ResourceTypes) (SRSResources[rt] > 0);
            
% in the InitialPlacement variable, each demand must be handled by max one cluster
constraint forall(d in Demands) (sum(c in Clusters) (InitialPlacement[d,c]) <= 1);

% each demand must be assigned to exactly one cluster, or be unallocated
constraint forall(d in Demands) (sum(c in Clusters) (Placement[d,c]) + IsUnserved[d] = 1);

% Limit on the number of instances - upper bound is help for the solver, domain restriction
constraint forall(c in Clusters) (
    Instances[c] >= 0 /\ Instances[c] <= MaxPhysicalInstances[c]
);

% The number of running instances must be sufficient to handle demands
% Mathematical trick implementing ceil(x/y) = (x + y - 1) div y; allows solver to immediately reject too small number of instances
constraint forall(c in Clusters) (
%     Instances[c] * SRSCapacity >= sum(d in Demands) (Placement[d,c])
      Instances[c] >= (sum(d in Demands)(Placement[d,c]) + SRSCapacity - 1) div SRSCapacity
);

% resources used by all SRS instances deployed in the cluster cannot exceed its capacity
constraint forall(c in Clusters, rt in ResourceTypes) (Instances[c] * SRSResources[rt] <= Resources[c,rt]);

% Latency cannot exceed the maximum
% Distances in km, LatencyPer10km is ms per 10km, so we divide by 10
constraint forall(d in Demands, c in Clusters) (
    Placement[d,c] == 1 -> ((Distances[d,c] * LatencyPer10km) div 10 <= MaxLatency)
);


%% 4. FUNKCJA CELU

% ---------- Cost components ----------

% unserved demands
var int: UnservedDemands = sum(d in Demands) (IsUnserved[d]);
var int: TotalUnservedCost = UnservedDemands * UnservedPenalty;

% total resource cost in clusters
var int: TotalDeploymentCost = sum(c in Clusters) (Instances[c] * DeploymentCost[c]);

% total transmission cost
var int: TotalTransmissionCost = sum(c in Clusters, d in Demands) (Placement[d,c] * Distances[d,c] * TransmissionCost);

% total migration cost
var int: TotalMigrationCost = Migrations * MigrationCost;

% total cost
var int: TotalCost =  TotalDeploymentCost + TotalTransmissionCost + TotalMigrationCost + TotalUnservedCost;

% ---------- Latency component ----------

var int: TotalLatency = sum(d in Demands, c in Clusters) (
    (Placement[d,c] * Distances[d,c] * LatencyPer10km) div 10
);

% ---------- Normalizacja ----------

% To add Cost (large numbers) to Latency (small numbers), they must be brought to a common denominator
% Theoretical maximum values (worst-case) used as divisors for normalization.

int: MaxPossibleLatency = Demands_ * MaxLatency;
% Max cost is a pessimistic sum of all possible costs (approximately)
int: MaxPossibleCost = sum(DeploymentCost) 
                       + Demands_ * max([Distances[d,c] | d in Demands, c in Clusters]) * TransmissionCost 
                       + Demands_ * MigrationCost
                       + Demands_ * UnservedPenalty;

% Normalized values (range 0..1)
var float: NormCost = TotalCost / MaxPossibleCost;
var float: NormLatency = TotalLatency / MaxPossibleLatency;

% Objective function: Weighted sum of normalized components
var float: GlobalScore = (Alpha_Cost * NormCost) + (Beta_Latency * NormLatency);
                        
%% 5. SOLVE

solve minimize GlobalScore;


%% 6. OUTPUT
output [
  "{\n",
  "  \"weights\": { \"alphaCost\": ", showJSON(Alpha_Cost), ", \"betaLatency\": ", showJSON(Beta_Latency), " },\n",
  "  \"scores\": {\n",
  "      \"normalizedCost\": ", showJSON(NormCost), ",\n",
  "      \"normalizedLatency\": ", showJSON(NormLatency), ",\n",
  "      \"globalScore\": ", showJSON(GlobalScore), "\n",
  "  },\n",
  "  \"totalLatency\": ", showJSON(TotalLatency), ",\n",
  "  \"totalCost\": ", showJSON(TotalCost), ",\n",
  "  \"placement\": ", showJSON(Placement), ",\n",
  "  \"unserved\": ", showJSON(UnservedDemands), ",\n",
  "  \"migrations\": ", showJSON(Migrations), ",\n",
  "  \"usedResources\": ", showJSON(UsedResources), ",\n",
  "  \"instances\": ", showJSON(Instances), ",\n",
  "  \"costBreakdown\": {\n",
  "    \"deployment\": ", showJSON(TotalDeploymentCost), ",\n",
  "    \"transmission\": ", showJSON(TotalTransmissionCost), ",\n",
  "    \"migration\": ", showJSON(TotalMigrationCost), ",\n",
  "    \"unservedPenalty\": ", showJSON(TotalUnservedCost), "\n",
  "  }\n",
  "}\n"
];



