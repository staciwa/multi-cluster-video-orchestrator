% SRS orchestration model with Erlang M/M/c/c queue theory (Aggregated Version - PoPs)
% 
% At time t, the system sees Pops_ access points (PoP). From each PoP p 
% PopDemands[p] demands arrive (cohorts of N_users_per_demand sessions). 
% The model decides how many demands from a given PoP to assign to a given Cluster (AssignedDemands).

%% 1. PARAMETRY

% Parametry teorio-kolejkowe wymiarowania Erlanga B (wykorzystywane przy generowaniu danych):
% int: N_users_per_demand = 100;     % Number of sessions in one cohort (macro-demand)
% float: a_session = 1.0;            % Offered traffic intensity per session [Erl]
% float: BlockageProbability = 0.01; % Target maximum blocking probability (P_B <= 1%)

enum ResourceTypes;

float: BlockageProbability;

% Lookup table: (note: in MiniZinc arrays are 1-indexed, so ErlangTable[1] is 0 demands, ErlangTable[n+1] is n demands)
% Precomputed in Python (erlang_utils.py) based on Erlang B formula (M/M/c/c):
% ErlangTable[n+1] = minimum number of server instances for A = n * N_users_per_demand * a_session with P_B <= BlockageProbability
int: TotalDemands;
array[1..TotalDemands+1] of int: ErlangTable;

int: SRSCapacity; % Informational parameter from base scheme (SRS instance capacity). In Erlang model, traffic dimensioning is fully incorporated in precomputed ErlangTable.
int: MaxLatency;
int: TransmissionCost;
int: LatencyPer10km;
int: MigrationCost;
int: UnservedPenalty;
int: Clusters_;
int: Pops_;

set of int: Clusters = 1..Clusters_;
set of int: Pops = 1..Pops_;

array[Pops] of int: PopDemands;

% --- Origin (Video source) ---
array[Clusters] of int: OriginDistances;

array [Clusters, ResourceTypes] of int: Resources;
array [ResourceTypes] of int: SRSResources;
array [Pops, Clusters] of int: Distances;

% How many demands from a given PoP were previously in a given cluster
array [Pops, Clusters] of int: InitialPlacement;
array [Clusters] of int: DeploymentCost;

float: Alpha_Cost;
float: Beta_Latency;


%% 2. ZMIENNE

% Aggregated assignment: how many demands from PoP p go to Cluster c
array [Pops, Clusters] of var 0..max(PopDemands): AssignedDemands;
array [Pops] of var 0..max(PopDemands): UnservedDemands;

% Static instance limit for each cluster
array[Clusters] of int: MaxPhysicalInstances = [
  min([ Resources[c,rt] div SRSResources[rt] | rt in ResourceTypes ])
  | c in Clusters
];

int: MaxGlobalInstances = max(c in Clusters)(MaxPhysicalInstances[c]);
array[Clusters] of var 0..MaxGlobalInstances: Instances;

% Migrations: counts how many demands "decreased" compared to the previous state (InitialPlacement).
% If fewer demands are assigned to a cluster than before, the difference must have migrated or been rejected.
% A migration fee is charged for each demand unit removed from its initial assignment
% (if unserved, pays UnservedPenalty; if migrated, pays MigrationCost).
var 0..sum(p in Pops, c in Clusters)(InitialPlacement[p,c]): Migrations;
constraint Migrations = sum(p in Pops, c in Clusters) (
    max(0, InitialPlacement[p,c] - AssignedDemands[p,c])
);

array[Clusters, ResourceTypes] of var int: UsedResources = 
  array2d(Clusters, ResourceTypes, [ Instances[c] * SRSResources[rt] 
                                        | c in Clusters, rt in ResourceTypes ]);

% Number of demands in each cluster in total from all PoPs
array[Clusters] of var 0..TotalDemands: DemandsPerCluster = [
  sum(p in Pops)(AssignedDemands[p,c]) | c in Clusters
];


%% 3. OGRANICZENIA

% Input data validation
constraint card(ResourceTypes) > 0 /\ SRSCapacity > 0 /\ MaxLatency > 0 /\ TransmissionCost > 0
            /\ LatencyPer10km > 0 /\ Clusters_ > 0 /\ Pops_ > 0 
            /\ MigrationCost >= 0 /\ UnservedPenalty >= 0
            /\ BlockageProbability > 0 /\ BlockageProbability < 1
            /\ forall(c in Clusters) (DeploymentCost[c] > 0)
            /\ forall(c in Clusters, rt in ResourceTypes) (Resources[c, rt] > 0)
            /\ forall(rt in ResourceTypes) (SRSResources[rt] > 0)
            /\ TotalDemands == sum(PopDemands);

            
% The sum of assigned and unserved demands for each PoP must equal the total PoP demand
constraint forall(p in Pops) (
    sum(c in Clusters) (AssignedDemands[p,c]) + UnservedDemands[p] == PopDemands[p]
);

% Instances must fit in the clusters
constraint forall(c in Clusters) (
    Instances[c] >= 0 /\ Instances[c] <= MaxPhysicalInstances[c]
);

% For each cluster, the number of instances must guarantee P_B <= BlockageProbability
constraint forall(c in Clusters) (
    Instances[c] >= ErlangTable[DemandsPerCluster[c] + 1]
);

% Used resources do not exceed available ones
constraint forall(c in Clusters, rt in ResourceTypes) (
    UsedResources[c,rt] <= Resources[c,rt]
);

% Latency constraint: for pairs (PoP, cluster) exceeding MaxLatency, assignment is 0
constraint forall(p in Pops, c in Clusters where (Distances[p, c] + OriginDistances[c]) * LatencyPer10km > MaxLatency)(
    AssignedDemands[p, c] = 0
);


%% 4. FUNKCJA CELU

% --- Cost components ---
var int: TotalUnservedDemands = sum(p in Pops) (UnservedDemands[p]);
var int: TotalPenalty = TotalUnservedDemands * UnservedPenalty;

var int: TotalDeploymentCost = sum(c in Clusters) (Instances[c] * DeploymentCost[c]);

var int: TotalTransmissionCost = sum(c in Clusters) (
    bool2int(Instances[c] > 0) * OriginDistances[c] * TransmissionCost
) + sum(p in Pops, c in Clusters) (
    AssignedDemands[p,c] * Distances[p,c] * TransmissionCost
);

var int: TotalMigrationCost = Migrations * MigrationCost;

var int: TotalCost = TotalDeploymentCost + TotalTransmissionCost 
                   + TotalMigrationCost + TotalPenalty;

% --- Latency component ---
var int: TotalLatency = sum(p in Pops, c in Clusters) (
    AssignedDemands[p,c] * (Distances[p,c] + OriginDistances[c]) * LatencyPer10km
);

% --- Normalizacja ---
int: MaxPossibleLatency = max(1, TotalDemands * MaxLatency);
int: MaxPossibleCost = max(1, sum(c in Clusters where MaxPhysicalInstances[c] > 0) (MaxPhysicalInstances[c] * DeploymentCost[c]) 
                       + sum(c in Clusters where MaxPhysicalInstances[c] > 0) (OriginDistances[c] * TransmissionCost)
                       + TotalDemands * max([Distances[p,c] | p in Pops, c in Clusters]) * TransmissionCost 
                       + TotalDemands * MigrationCost
                       + TotalDemands * UnservedPenalty);

var float: NormCost = int2float(TotalCost) * 1000.0 / int2float(MaxPossibleCost);
var float: NormLatency = int2float(TotalLatency) * 1000.0 / int2float(MaxPossibleLatency);

var float: GlobalScore = Alpha_Cost * NormCost + Beta_Latency * NormLatency;

solve minimize GlobalScore;


%% 6. OUTPUT

output [
  "{\n",
  "  \"erlangParams\": {\n",
  "      \"blockageProbability\": ", showJSON(BlockageProbability), "\n",
  "  },\n",
  "  \"demandsPerCluster\": ", showJSON(DemandsPerCluster), ",\n",
  "  \"weights\": { \"alphaCost\": ", showJSON(Alpha_Cost), ", \"betaLatency\": ", showJSON(Beta_Latency), " },\n",
  "  \"scores\": {\n",
  "      \"normalizedCost\": ", showJSON(NormCost), ",\n",
  "      \"normalizedLatency\": ", showJSON(NormLatency), ",\n",
  "      \"globalScore\": ", showJSON(GlobalScore), "\n",
  "  },\n",
  "  \"totalLatency\": ", showJSON(TotalLatency), ",\n",
  "  \"totalCost\": ", showJSON(TotalCost), ",\n",
  "  \"placement\": ", showJSON(AssignedDemands), ",\n",
  "  \"unserved\": ", showJSON(TotalUnservedDemands), ",\n",
  "  \"migrations\": ", showJSON(Migrations), ",\n",
  "  \"usedResources\": ", showJSON(UsedResources), ",\n",
  "  \"instances\": ", showJSON(Instances), ",\n",
  "  \"costBreakdown\": {\n",
  "    \"deployment\": ", showJSON(TotalDeploymentCost), ",\n",
  "    \"transmission\": ", showJSON(TotalTransmissionCost), ",\n",
  "    \"migration\": ", showJSON(TotalMigrationCost), ",\n",
  "    \"unservedPenalty\": ", showJSON(TotalPenalty), "\n",
  "  }\n",
  "}\n"
];
