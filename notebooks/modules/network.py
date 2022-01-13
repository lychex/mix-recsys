from collections import defaultdict
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class NetworkRec():
    def __init__(self, df, user_str, item_str, edge_str):
        self.G = nx.Graph()
        self.G.add_nodes_from(df[user_str], bipartite=user_str)
        self.G.add_nodes_from(df[item_str],  bipartite=item_str)

        for r, d in df.iterrows():
            self.G.add_edge(d[user_str], d[item_str], data=d[edge_str])

        self.item_nodes = [n for n,d in self.G.nodes(data=True) if d['bipartite']==item_str]
        self.user_nodes = [n for n,d in self.G.nodes(data=True) if d['bipartite']==user_str]

    def shared_partition_nodes(self, node1, node2):
        # Check that the nodes belong to the same partition
        assert self.G.nodes[node1]['bipartite'] == self.G.nodes[node2]['bipartite']
        # Get neighbors of node 1: nbrs1
        nbrs1 = self.G.neighbors(node1)
        # Get neighbors of node 2: nbrs2
        nbrs2 = self.G.neighbors(node2)
        # Compute the overlap using set intersections
        overlap = set(nbrs1).intersection(nbrs2)
        return overlap


    def user_similarity(self, node1, node2):
        # Get the set of nodes shared between the two users
        shared_nodes = self.shared_partition_nodes(node1, node2)
        # Return the fraction of nodes in the projects partition
        return len(shared_nodes) / len(self.item_nodes)



    def most_similar_users(self, test_node):
        # Get other nodes from user partition
        unique_user_nodes = set(self.user_nodes)
        # Revomve test node
        unique_user_nodes.remove(test_node)
        # Create the dictionary: similarities
        similarities = defaultdict(list)
        for node in unique_user_nodes:
            similarity = self.user_similarity(test_node, node)
            similarities[similarity].append(node)
        # Compute maximum similarity score: max_similarity
        max_similarity = max(similarities.keys())
        # Return list of users that share maximal similarity
        return similarities[max_similarity]

    def recommend_items(self, from_user, to_user):
        # Get the set of pages that from_user has viewed
        from_items = set(self.G.neighbors(from_user))
        # Get the set of pages that to_user has viewed
        to_items = set(self.G.neighbors(to_user))
        # Identify pages that the from_user is connected to that the to_user is not connected to
        return from_items.difference(to_items)

#--------------------------------------------------------------------
# Visualize network
#--------------------------------------------------------------------

# Drop NA values from node and link columns
def clean_data(df,node,link, node_minimum_nchar, link_minimum_nchar):
    df = df.dropna(subset=[link])
    df[node] = df[node].map(str)
    df[link] = df[link].map(str)
    df = df[(df[node].map(len)>node_minimum_nchar) & (df[link].map(len)>link_minimum_nchar)]
    return(df)

#Remove Node_Link combinations
def remove_duplicate_node_link(df,node,link):
    df['node_link'] = df[node].map(str)+df[link].map(str)
    df = df.drop_duplicates(subset=['node_link'])
    return(df)

# Links per unique node (feature creation)
# Find number of links per unique node
def links_per_node(df,node,link):
    df=df.pivot_table(link, node,aggfunc=len,)
    s = pd.DataFrame(df.index)
    t = pd.DataFrame(df.values)
    lpn = pd.concat([s, t], axis=1)
    lpn.columns=[node, 'NumberOfLinks']
    return(lpn)
# Merge links per node width original dataframe
def add_links_per_node(df,node,link):
    return(df.merge(links_per_node(df,node,link),on=node,how='left'))

# Get duplicated users to make sure they link atleast 2 artists
def get_duplicated_links(df,link):
    ids = df[df[link].duplicated()][link].reset_index()
    new_df =  df[df[link].isin(ids[link])]
    return(new_df)

# Get unique artist_x - user - artist_y rows
def get_unique_node_link_node_grouping(df,node,link,node_x,node_y):
    # Get from - to groupings
    new_df = df.merge(df,on = link)
    # Drop duplicate node_x-link-node_y rows
    new_df['ordered_nodes'] = new_df.apply(lambda x: '-'.join(sorted([x[node_x],x[link],x[node_y]])),axis=1)
    new_df = new_df.drop_duplicates(['ordered_nodes'])
    # Drop node_x == node_y rows
    new_df = new_df[new_df[node_x] != new_df[node_y]]
    return(new_df)


def create_network_DATA(df,node,link,node_minimum_nchar,link_minimum_nchar,node_x,node_y):
    # Assign names for node x and node y
    df = clean_data(df,node,link, node_minimum_nchar, link_minimum_nchar)
    df = remove_duplicate_node_link(df,node,link)
    df = add_links_per_node(df,node,link)
    df = get_duplicated_links(df,link)
    df = get_unique_node_link_node_grouping(df,node,link,node_x,node_y)
    return(df)

# Get Nodes
def get_nodes(df,node,node_x,node_y):
    nodes = pd.concat([df[node_x],df[node_y]],ignore_index=True).unique()
    nodes = pd.DataFrame(nodes)
    nodes['id'] = nodes.index
    nodes.columns = [node, 'id']
    return(nodes)

# Get Links and Weights
def get_weights(df,node_x,node_y):
    df = df[[node_x,node_y]]
    df = df.groupby([node_x,node_y]).size().reset_index()
    df.columns=['from','to','weight']
    return(df)


def create_network(df,from_col, to_col, attr):
    import networkx as nx
    G = nx.Graph
    G = nx.from_pandas_edgelist(df,source=from_col,target=to_col,edge_attr=[attr])
    print(nx.info(G))
    return(G)



def plot_network(G,node_size_multiplier,size_font,node_distance):
    plt.figure(figsize=(16,16))
    plt.axis('off')
    # Spring Layout
    layout = nx.spring_layout(G,k=node_distance)
    # Edge thickness as weights
    weights = [G[u][v]['weight']/200 for u,v in G.edges()]
    # Node size as degree of centrality
    d = nx.degree(G)
    nx.draw_networkx_nodes(G, pos=layout,
                        nodelist=dict(d).keys(),node_size=[x * node_size_multiplier for x in dict(d).values()],
                        alpha=0.9,node_color='coral')
    nx.draw_networkx_edges(G, pos=layout, width=weights,
                        style='solid', edge_color='brown')
    nx.draw_networkx_labels(G, pos=layout, font_size=size_font)
    plt.show()