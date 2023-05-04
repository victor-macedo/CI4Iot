import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../CI4Iot/Lab3_DataSets/DCOILBRENTEUv2.csv')
plt.plot(df["DCOILBRENTEU"])