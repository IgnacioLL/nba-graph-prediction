from itertools import combinations
from utils import euclidean_distance
import networkx as nx
import pandas as pd

from constants import Constant


def create_graph_from_dataset(data):
    saved_graphs = []
    demo_data = data.iloc[0]
    teams_list  = [demo_data['player2_team'],demo_data['player9_team']]
    for idx, moment in data.iterrows():
        seconds = round(moment['game_clock'] % 60,1) ## TODO: How could it be added in the graph?
        minutes = int(moment['game_clock'] // 60) ## TODO: How could it be added?

        # Create an empty graph
        PG = PlayersGraph(input_data=moment, team_list=teams_list)

        PG.create_nodes()

        PG.create_edges_bw_players()

        PG.keep_shortest_path()

        PG.add_ball_node()

        PG.add_closest_player_to_ball_edge()

        PG.add_baskets()

        PG.add_edges_ball_basket()

        saved_graphs.append(nx.Graph(PG))

    return saved_graphs

class PlayersGraph(nx.Graph):

    def __init__(self, input_data: pd.DataFrame, team_list: list):
        super().__init__()
        self.input_data = input_data
        self.team_list = team_list


    def _create_combinations_bw_players(self) -> list:
        # Define the list
        list_edges = []
        for team in self.team_list:
            filtered_nodes = [(n, attr) for n, attr in self.nodes(data=True) if attr.get('team', 0) == team]

            # Generate permutations of different lengths
            combiantions_list = list(combinations(filtered_nodes, 2))
            for edge in combiantions_list:
                ## Extract each player
                player1 = edge[0]
                player2 = edge[1]

                ## Extract player 1 and  info
                player1_index = player1[0]
                player1_point = (player1[1]['x'],player1[1]['y'])
                
                player2_index = player2[0]
                player2_point = (player2[1]['x'],player2[1]['y'])

                ## Extract distances between them
                weight = euclidean_distance(player1_point, player2_point)

                ## Create edge
                list_edges.append([player1_index, player2_index, weight])

        return list_edges

    def _create_edges_in_graph(self, list_edges: list) -> None:
        pd_data = pd.DataFrame(list_edges, columns=["node1", "node2", "weight"])
        for idx, row in pd_data.iterrows():
            self.add_edge(row[0], row[1], weight=row[2])


    def create_edges_bw_players(self) -> None:
        list_edges = self._create_combinations_bw_players()
        self._create_edges_in_graph(list_edges=list_edges)



    def keep_shortest_path(self) -> None:
        # Create a new graph G based on the current graph
        G = nx.Graph(self)

        # Compute the minimum spanning tree of G using Kruskal's algorithm
        mst = nx.minimum_spanning_tree(G=G, algorithm='kruskal')

        # Clear the current graph's edges and add the edges from the MST
        self.clear_edges()
        self.add_edges_from(mst.edges(data=True))


    def create_nodes(self) -> None:
        for player in range(0,10):
            self.add_node(
                player,
                name=self.input_data[f"player{player + 1}_name"], 
                x=self.input_data[f"player{player + 1}_x_coord"], 
                y=self.input_data[f"player{player + 1}_y_coord"],
                team=self.input_data[f"player{player + 1}_team"]
                )


    def add_ball_node(self) -> None:
        self.point_ball = (self.input_data['playerball_x_coord'], self.input_data['playerball_y_coord'])

        self.add_node(
            Constant.INDEX_BALL, 
            name=-1,
            x=self.input_data['playerball_x_coord'],
            y=self.input_data['playerball_y_coord'], 
            team=-1)


    def _create_possible_edges_bw_ball_players(self) -> pd.DataFrame:

        list_edges = []
        for node, attr in self.nodes(data=True):
            if node != Constant.INDEX_BALL:
                point_player = (attr['x'], attr['y'])
                weight = euclidean_distance(self.point_ball, point_player)
                list_edges.append([node, Constant.INDEX_BALL, weight])

        l_ball_players = pd.DataFrame(list_edges, columns=['node', 'ball', 'weight'])

        return l_ball_players

    def add_closest_player_to_ball_edge(self) -> None:

        l_ball_players = self._create_possible_edges_bw_ball_players()

        closest_player = l_ball_players.loc[l_ball_players['weight'] == l_ball_players['weight'].min()]
        l_closest_player = closest_player.iloc[0].tolist()
        self.add_edge(l_closest_player[0], l_closest_player[1], weight=l_closest_player[2])


    def add_baskets(self) -> None:
        self.add_node(Constant.INDEX_BASKET1, name=-2, x=Constant.basket_X_1, y=Constant.basket_Y_1, team=-1)
        self.add_node(Constant.INDEX_BASKET2, name=-3, x=Constant.basket_X_2, y=Constant.basket_Y_2, team=-1)


    def add_edges_ball_basket(self) -> None:
        basket1_point = (Constant.basket_X_1, Constant.basket_Y_1)
        basket2_point = (Constant.basket_X_2, Constant.basket_Y_2)

        weight_basket1 = euclidean_distance(self.point_ball, basket1_point)
        weight_basket2 = euclidean_distance(self.point_ball, basket2_point)

        self.distance_rim = min(weight_basket1, weight_basket2)

        if weight_basket1<weight_basket2:
            self.add_edge(Constant.INDEX_BALL, Constant.INDEX_BASKET1, weight=weight_basket1)
        else:
            self.add_edge(Constant.INDEX_BALL, Constant.INDEX_BASKET2, weight=weight_basket2)


    def set_posession_team(self) -> None:
        self.player_w_ball = [edge[0] for edge in self.edges(data=True) if edge[1] == 10][0]
        self.team_possession = self.input_data[f'player{self.player_w_ball + 1}_team']


    def get_posession_team(self) -> str:
        return self.team_possession



