import pandas as pd
import pants

df = pd.read_excel("Projeto_3/Project3_DistancesMatrix.xlsx", header =0, index_col=0)
#df.set_index('Distance')
nodes = list(df)

def euclidean(a, b):
    return df[a][b]

world = pants.World(nodes, euclidean)
ids = world.nodes

ants = []
ant = pants.Ant(alpha=0.5, beta=2.25)
ant.initialize(world, start=world.nodes[0])
for i in range(100):
   ants.append(ant.clone())

for a in ants:
  for i in range (100):
    if ant.can_move() == True:
      ant.move()

B_ant = ant
for a in ants:
   if (a.distance < B_ant.distance) & (a.distance != 0):
      B_ant = a
print("Resultados formiga \n")

print(B_ant.distance)
print(B_ant.tour)

solver = pants.Solver()
#solver.create_colony(world)
solver.find_solutions(ants)
solver.global_update(ants)


solution = solver.aco(ants)
# or
solutions = solver.solutions(world)

print("Resultados solver \n")

print(solution.distance, "\n")
print(solution.tour,"\n")    # Nodes visited in order
#print(solution.path,"\n")    # Edges taken in order
# or
best = float("inf")
for solution in solutions:
  assert solution.distance < best
  best = solution.distance