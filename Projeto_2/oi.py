from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS_N = FuzzySystem()

# Define fuzzy sets and linguistic variables

L_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
L_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
L_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Latency", LinguisticVariable([L_1, L_2, L_3], concept="Latency", universe_of_discourse=[0, 4]))

O_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
O_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
O_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Output", LinguisticVariable([O_1, O_2, O_3], concept="Output Network Throughput", universe_of_discourse=[0, 100]))

# Define output fuzzy sets and linguistic variable
N_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=0.5), term="low")
N_2 = FuzzySet(function=Triangular_MF(a=0, b=0.5, c=1), term="normal")
N_3 = FuzzySet(function=Triangular_MF(a=0.5, b=1, c=1), term="high")
FS_N.add_linguistic_variable("Network", LinguisticVariable([N_1, N_2], universe_of_discourse=[0, 1]))

# Define fuzzy rules

R_N1 = "IF (Memory IS low) AND (Processor IS low) THEN (Load IS low)"
R_N2 = "IF (Memory IS low) AND (Processor IS normal) THEN (Load IS low)"
R_N3 = "IF (Memory IS low) AND (Processor IS high) THEN (Load IS normal)"
R_N4 = "IF (Memory IS normal) AND (Processor IS low) THEN (Load IS low)"
R_N5 = "IF (Memory IS normal) AND (Processor IS normal) THEN (Load IS normal)"
R_N6 = "IF (Memory IS normal) AND (Processor IS high) THEN (Load IS fast)"
R_N7 = "IF (Memory IS high) AND (Processor IS low) THEN (Load IS normal)"
R_N8 = "IF (Memory IS high) AND (Processor IS normal) THEN (Load IS fast)"
R_N9 = "IF (Memory IS high) AND (Processor IS high) THEN (Load IS fast)"

FS_N.add_rules([R_N1, R_N2, R_N3, R_N4, R_N5, R_N6, R_N7, R_N8, R_N9])

# Set antecedents values
FS_N.set_variable("Core", 20)
FS_N.set_variable("CPU", 3)

# Perform Mamdani inference and print output
print(FS_N.Mamdani_inference(["Fan_speed"]))