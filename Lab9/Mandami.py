from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables

S_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=1.5), term="low")
S_2 = FuzzySet(function=Triangular_MF(a=0.5, b=3.5, c=4), term="normal")
S_3 = FuzzySet(function=Triangular_MF(a=2.5, b=4, c=4), term="turbo")
FS.add_linguistic_variable("CPU", LinguisticVariable([S_1, S_2, S_3], concept="CPU Speed", universe_of_discourse=[0,4]))

F_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=50), term="cold")
F_2 = FuzzySet(function=Triangular_MF(a=30, b=50, c=70), term="warm")
F_3 = FuzzySet(function=Triangular_MF(a=50, b=100, c=100), term="hot")
FS.add_linguistic_variable("Core", LinguisticVariable([F_1, F_2, F_3], concept="Core Temp", universe_of_discourse=[0,100]))

# Define output fuzzy sets and linguistic variable
T_1 = FuzzySet(function=Triangular_MF(a=0, b=0, c=3500), term="slow")
T_2 = FuzzySet(function=Triangular_MF(a=2500, b=6000, c=6000), term="fast")
FS.add_linguistic_variable("Fan_speed", LinguisticVariable([T_1, T_2], universe_of_discourse=[0,6000]))

# Define fuzzy rules


R1 = "IF (Core IS cold) AND (CPU IS low) THEN (Fan_speed IS slow)"
R2 = "IF (Core IS cold) AND (CPU IS normal) THEN (Fan_speed IS slow)"
R3 = "IF (Core IS cold) AND (CPU IS turbo) THEN (Fan_speed IS fast)"
R4 = "IF (Core IS warm) AND (CPU IS low) THEN (Fan_speed IS slow)"
R5 = "IF (Core IS warm) AND (CPU IS normal) THEN (Fan_speed IS slow)"
R6 = "IF (Core IS warm) AND (CPU IS turbo) THEN (Fan_speed IS fast)"
R7 = "IF (Core IS hot) AND (CPU IS low) THEN (Fan_speed IS fast)"
R8 = "IF (Core IS hot) AND (CPU IS normal) THEN (Fan_speed IS fast)"
R9 = "IF (Core IS hot) AND (CPU IS turbo) THEN (Fan_speed IS fast)"

FS.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9])

# Set antecedents values
FS.set_variable("Core", 20)
FS.set_variable("CPU", 3)

# Perform Mamdani inference and print output
print(FS.Mamdani_inference(["Fan_speed"]))