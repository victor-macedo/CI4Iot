import matplotlib.pyplot as plt
import numpy as np



plt.figure()
plt.plot([0,0,0.25],[0,1,0],label = "very_low")
plt.plot([0,0.25,0.5],[0,1,0],label = "low")
plt.plot([0.25,0.5,0.75],[0,1,0],label = "normal")
plt.plot([0.5,0.75,1],[0,1,0],label = "high")
plt.plot([0.75,1,1],[0,1,0],label = "very_high")
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()