import os
from matplotlib import pyplot as plt

# change work directory to find the model file
# os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

with open('goals.txt') as f:
    lines = f.readlines()
    data = []
    for line in lines:
        data.append(float(line))
    
plt.title('Goal Function')
plt.plot(data)
plt.show()
