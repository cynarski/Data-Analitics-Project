import networkx as nx
import matplotlib.pyplot as plt

def plot_poisson_dag():
    G = nx.DiGraph()

    labels = {
        'X': 'X (Predictors)',
        'nkill': 'nkill (Observed Counts)',
        'alpha': 'alpha',
        'beta': 'beta',
        'eta': 'eta = alpha + X * beta',
        'nkill_dist': 'Poisson_log(nkill | eta)',
        'nkill_pred': 'nkill_pred'
    }

    G.add_nodes_from(labels)

    G.add_edges_from([
        ('X', 'eta'),
        ('beta', 'eta'),
        ('alpha', 'eta'),
        ('eta', 'nkill_dist'),
        ('nkill_dist', 'nkill'),
        ('eta', 'nkill_pred')
    ])

    pos = {
        'X': (0, 2),
        'alpha': (0, 0),
        'beta': (2, 0),
        'eta': (1, 1),
        'nkill_dist': (3, 1),
        'nkill': (4, 1),
        'nkill_pred': (3, -1),
    }

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', arrowsize=20)

    ax = plt.gca()
    for node, (x, y) in pos.items():
        ax.text(
            x, y + 0.05,
            labels[node],
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(facecolor='black', alpha=0.75, boxstyle='round, pad=0.2')
        )

    plt.title('Poisson Regression Model DAG')
    plt.axis('off')
    plt.show()


def plot_negative_binomial_dag():
    G = nx.DiGraph()

    labels = {
        'X': 'X (Predictors)',
        'nkill': 'nkill (Observed Counts)',
        'alpha': 'alpha',
        'beta': 'beta',
        'phi': 'phi',
        'eta': 'eta = alpha + X * beta',
        'nkill_dist': 'NegBinomial_2_log(nkill | eta, phi)',
        'nkill_pred': 'nkill_pred'
    }
    for node in labels:
        G.add_node(node)

    G.add_edges_from([
        ('X', 'eta'),
        ('beta', 'eta'),
        ('alpha', 'eta'),
        ('phi', 'nkill_dist'),
        ('eta', 'nkill_dist'),
        ('nkill_dist', 'nkill'),
        ('phi', 'nkill_pred'),
        ('eta', 'nkill_pred')
    ])

    pos = {
        'X': (0, 2),
        'alpha': (0, 0),
        'beta': (2, 0),
        'eta': (1, 1),
        'phi': (2, -1),
        'nkill_dist': (3, 1),
        'nkill': (4, 1),
        'nkill_pred': (3, -1),
    }

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=3000, node_color='skyblue', arrowsize=20)

    ax = plt.gca()
    for node, (x, y) in pos.items():
        if node == "nkill":
            y = y - 0.1
        else:
            y = y + 0.05
        ax.text(
            x, y,
            labels[node],
            ha='center',
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(facecolor='black', alpha=0.75, boxstyle='round, pad=0.2')
        )

    plt.title('Negative Binomial Regression Model DAG')
    plt.axis('off')
    plt.show()