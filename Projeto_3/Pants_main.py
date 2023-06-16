import pandas as pd
import pants

df = pd.read_excel("Projeto_3/Project3_DistancesMatrix.xlsx", header =0, index_col=0)
#df.set_index('Distance')
nodes = list(df)

def euclidean(a, b):
    return df[a][b]

world = pants.World(nodes, euclidean)
ids = world.nodes

solver = pants.Solver()
solution = solver.solve(world)
# or
solutions = solver.solutions(world)

print(solution.distance, "\n")
print(solution.tour,"\n")    # Nodes visited in order
print(solution.path,"\n")    # Edges taken in order
# or
best = float("inf")
for solution in solutions:
  assert solution.distance < best
  best = solution.distance