import pandas as pd
import time
import pants

df = pd.read_excel("Projeto_3/Project3_DistancesMatrix.xlsx", header =0, index_col=0)
nodes = list(df)

def distance(a, b):
    return df[a][b]

world = pants.World(nodes, distance)

ants = []
ant = pants.Ant(alpha=1, beta=10)
ant.initialize(world, start=world.nodes[0])
for i in range(1000):
    ants.append(ant.clone())

start = time.time()
solver = pants.Solver()

solution = solver.aco(ants)

caminho = solution.tour
caminho.append("C")

end = time.time()
print("Tempo: ", end - start, "s")
print(solution.distance, "\n")
print(caminho,"\n")    # Nodes visited in order

