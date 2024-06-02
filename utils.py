import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from constants import Constant

import networkx as nx

def euclidean_distance(point1: tuple, point2: tuple) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def plot_graph(PG, minutes, seconds):

    # Extract x and y coordinates
    pos = {node[0]: (node[1]['x'], node[1]['y']) for node in PG.nodes(data=True)}

    # Plot the graph with x and y coordinates
    plt.figure(figsize=(8, 6))
    nx.draw(PG, pos, with_labels=True, node_color=['red']*5 + ['blue'] *5 + ['orange'] + ['green'] * 2, node_size=300, font_size=10)
    plt.title('Graph with X and Y coordinates')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    court = plt.imread("../court.png")
    plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                        Constant.Y_MAX, Constant.Y_MIN])
    plt.text(0.5, 1.05, f'{minutes}:{seconds}', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    # Add a square to the plot

    plt.show()