import pandas as pd
import time
import pants
import numpy as np

df = pd.read_excel("Projeto_3/Project3_DistancesMatrix.xlsx", header =0, index_col=0)
nodes = list(df)

def distance(a, b):
    return df[a][b]

world = pants.World(nodes, distance)
l_alpha =[0.1,0.5,1,2,3]
l_beta = [0.1, 1,3,5,7,10]
df_final = pd.DataFrame({str(l_alpha[0]): [0],str(l_alpha[1]): [0],str(l_alpha[2]): [0],str(l_alpha[3]): [0],str(l_alpha[4]): [0]})

for a in l_alpha:
    for b in l_beta:
        ants = []
        ant = pants.Ant(alpha=a, beta=b)
        ant.initialize(world, start=world.nodes[0])
        for i in range(1000):
            ants.append(ant.clone())

        start = time.time()
        solver = pants.Solver()

        solution = solver.aco(ants)

        caminho = solution.tour
        caminho.append("C")

        end = time.time()
        #print("Tempo: ", end - start, "s")
        #print('\n Teste com:', a, b)
        #print(solution.distance, "\n")
        distancia = solution.distance
        df_final.at[b,str(a)] = distancia
            #print(caminho,"\n")    # Nodes visited in order
df_final.to_csv('Projeto_3/comparacao.csv')


