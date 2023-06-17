import pandas as pd
import pants

df = pd.read_excel("Projeto_3/Project3_DistancesMatrix.xlsx", header =0, index_col=0)

nodes = list(df)

def distance(a, b):
    return df[a][b]

world = pants.World(nodes, distance)

ants = []
ant = pants.Ant(alpha=0.5, beta=2.25)
ant.initialize(world, start=world.nodes[0])
for i in range(100):
   ants.append(ant.clone())


solver = pants.Solver()

solution = solver.aco(ants)

print(solution.distance, "\n")
print(solution.tour,"\n")    # Nodes visited in order

