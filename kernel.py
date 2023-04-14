import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from cvxopt import matrix, solvers

### Loading the data
graphs = pd.read_pickle('training_data.pkl')
# Transforming the labels into {-1, 1} for simplicity
labels = 2 * np.array(pd.read_pickle('training_labels.pkl')) - 1
test = pd.read_pickle('test_data.pkl')

# Weisfeiler-Lehman Subtree Kernel
def wlsk(X, h=5):
    # First convert all node labels to string instead of lists of one integer
    X = deepcopy(X)
    n = len(X)
    # Compute the inverse of the size of each graph for normalizing later
    inv_graph_sizes = np.zeros(n)
    for i, g in enumerate(X):
        labels = nx.get_node_attributes(g, 'labels')
        labels = {j: {'labels': str(lbl[0])} for j, lbl in labels.items()}
        nx.set_node_attributes(g, labels)
        inv_graph_sizes[i] = 1 / g.number_of_nodes()
    K = np.zeros((n, n))
    # We know the set of possible node labels in all graphs before the first label propagation
    ind_labels = {str(i): i for i in range(50)}
    for step in tqdm(range(h)):
        # Remember the labels created with this propagation 
        new_ind_labels = set()
        # Propagate the labels and compute histograms of node labels
        histograms = np.zeros((n, len(ind_labels)))
        for i, g in enumerate(X):
            new_labels = {}
            for node, attrs in g.nodes(data=True):
                label = attrs['labels']
                histograms[i, ind_labels[label]] += 1
                new_label = []
                # Propagate neighbor labels
                for neighbor in g.neighbors(node):
                    new_label.append(g.nodes[neighbor]['labels'])
                # Transform the multiset of labels into a new label
                new_label = ''.join(sorted(new_label))
                new_labels[node] = {'labels': new_label}
                new_ind_labels.add(new_label)
            # Update the labels in the graph
            nx.set_node_attributes(g, new_labels)
        # Add the vertex histogram kernel values corresponding to this propagation
        K += (inv_graph_sizes[:, None] * histograms) @ (inv_graph_sizes[:, None] * histograms).T
        # Create indexing for the new labels
        ind_labels = {lbl: i for i, lbl in enumerate(new_ind_labels)}
    return K / h

# Compute the Gramm matrix of both the train and test graphs
print('Computing the Gram matrix using the Weisfeiler-Lehman Subtree Kernel')
K = wlsk(graphs + test, h=5)
n = len(graphs)
K_train = K[:n, :n]
K_test = K[n:, :n]

# Define the parameters of the convex optimization problem for kernel C-SVM
# Put class weight so that samples with label -1 (10%) have a bigger impact on the learning
class_weights = np.array([0, 1, 10])
C = 1e-3
P = np.outer(labels, labels) * K_train
w = class_weights[labels]
P *= np.outer(w, w)
q = -np.ones(n)
G = np.vstack([-np.eye(n), np.eye(n)])
h = np.hstack([np.zeros(n), C * np.ones(n)])
A = labels.reshape((1, -1))
b = np.zeros(1)
# Solve the optimization problem
print('Solving the optimization problem for kernel C-SVM')
P, q, G, h, A, b = map(lambda x: matrix(x, tc='d'), [P, q, G, h, A, b])
solvers.options['show_progress'] = True
solution = solvers.qp(P, q, G, h, A, b)
alphas = np.array(solution['x']).flatten()
sv_indices = np.where(alphas > 1e-7)[0]
print('Number of support vectors:', len(sv_indices))
sv_alphas = alphas[sv_indices]
sv_labels = labels[sv_indices]
b = np.mean(sv_labels - sv_alphas @ K_train[sv_indices][:, sv_indices])
Y_pred = ((np.sign(K_test[:, sv_indices] @ sv_alphas + b) + 1) / 2).astype(int)
sub = pd.DataFrame(Y_pred).reset_index()
sub.columns = ['Id', 'Predicted']
sub['Id'] += 1
sub.to_csv('test_pred.csv', index=False)
print('Saved the predictions in the test_pred.csv file')