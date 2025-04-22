
 ## Name : PANDEESWARAN N
### Register No.: 212224230191

## Experiment 1
### DATE: 22/04/2025

# Implementation of Bayesian Networks

## Aim:
To create a Bayesian Network for the given dataset in Python.

## Algorithm:
1. Import necessary libraries: `pandas`, `networkx`, `matplotlib.pyplot`, `Bbn`, `Edge`, `EdgeType`, `BbnNode`, `Variable`, `EvidenceBuilder`, `InferenceController`.
2. Set pandas options to display more columns.
3. Read in weather data from a CSV file using pandas.
4. Remove records where the target variable `RainTomorrow` has missing values.
5. Fill in missing values in other columns with the column mean.
6. Create bands for variables that will be used in the model (`Humidity9amCat`, `Humidity3pmCat`, and `WindGustSpeedCat`).
7. Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN).
8. Create `BbnNode` objects for `Humidity9amCat`, `Humidity3pmCat`, `WindGustSpeedCat`, and `RainTomorrow`, using the `probs()` function to calculate their probabilities.
9. Create a `Bbn` object and add the `BbnNode` objects to it, along with edges between the nodes.
10. Convert the BBN to a join tree using the `InferenceController`.
11. Set node positions for the graph.
12. Set options for the graph appearance.
13. Generate the graph using `networkx`.
14. Update margins and display the graph using `matplotlib.pyplot`.

## Program:

```python
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from pybbn.graph.dag import Bbn
from pybbn.graph.dag import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

pd.options.display.max_columns = 50

df = pd.read_csv('weatherAUS.csv', encoding='utf-8')
df = df[pd.isnull(df['RainTomorrow']) == False]
df = df.drop(columns='Date')

numeric_columns = df.select_dtypes(include=['number']).columns
df.loc[:, numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(lambda x: '0.<=40' if x <= 40 else '1.40-50' if 40 < x <= 50 else '2.>50')
df['Humidity9amCat'] = df['Humidity9am'].apply(lambda x: '1.>60' if x > 60 else '0.<=60')
df['Humidity3pmCat'] = df['Humidity3pm'].apply(lambda x: '1.>60' if x > 60 else '0.<=60')

print(df)

def probs(data, child, parent1=None, parent2=None):
    if parent1 == None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1 != None:
        # Check if child node has 1 parent or 2 parents
        if parent2 == None:
            # Calculate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Calculate probabilities
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob

H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))

bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)

pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "pink",
    "edgecolors": "blue",
    "edge_color": "green",
    "linewidths": 5,
    "width": 5,
}

n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()

print(probs(df, child='Humidity9amCat'))
print(probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
print(probs(df, child='WindGustSpeedCat'))
print(probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))
```

## Output:

![image](https://github.com/user-attachments/assets/295f5e58-6984-4ba8-99a6-1ea921bb7303)
![image](https://github.com/user-attachments/assets/fd4662a3-bf03-4b71-b7c5-8dd8f83cda49)
![image](https://github.com/user-attachments/assets/ac0c8ca3-fad5-4f28-b360-1dd96e2e91e9)
![image](https://github.com/user-attachments/assets/141ceedc-d113-4c3f-b888-a02fa471f08d)
![image](https://github.com/user-attachments/assets/b7987469-7c45-4b18-8813-6f08c87c6493)


## Graph:

![image](https://github.com/user-attachments/assets/5c8faa95-ad28-4314-8cfd-95b23c15bec9)



## Result:
   Thus a Bayesian Network is generated using Python
