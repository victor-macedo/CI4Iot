from simpful import *

# A simple fuzzy inference system for the tipping problem
# Create a fuzzy system object
FS = FuzzySystem()

# Define fuzzy sets and linguistic variables
S_1 = FuzzySet(points=[[0., 1.],  [50., 0.]], term="Low")
S_2 = FuzzySet(points=[[0., 0.], [30., 0], [50., 1.], [70., 0.]], term="Normal")
S_3 = FuzzySet(points=[[50., 0.],  [100., 1.]], term="Turbo")
FS.add_linguistic_variable("CPU", LinguisticVariable([S_1, S_2, S_3], concept="CPU Speed"))

F_1 = FuzzySet(points=[[0., 1.],  [50., 0.]], term="cold")
F_2 = FuzzySet(points=[[0., 0.], [30., 0], [50., 1.], [70., 0.]], term="warm")
F_3 = FuzzySet(points=[[0., 0.], [50., 0.],  [100., 1.]], term="hot")
FS.add_linguistic_variable("Core", LinguisticVariable([F_1, F_2], concept="Core Temp"))

# Define output crisp values
FS.set_crisp_output_value("slow", 3500)
FS.set_crisp_output_value("fast", 2500)

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

# Perform Sugeno inference and print output
print(FS.Sugeno_inference(["Tip"]))