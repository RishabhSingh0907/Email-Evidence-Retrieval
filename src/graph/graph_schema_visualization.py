import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph for schema visualization
G = nx.DiGraph()

# Define node types
nodes = {
    "Person": {"type": "Node", "color": "lightblue"},
    "Email": {"type": "Node", "color": "lightgreen"},
    "Company": {"type": "Node", "color": "lightyellow"},
    "Link": {"type": "Node", "color": "lightcoral"},
    "PhoneNumber": {"type": "Node", "color": "lightgray"}
}

# Add nodes to graph
for node, attr in nodes.items():
    G.add_node(node, color=attr["color"])

# Define relationships
relations = [
    ("Person", "Email", "SENT"),
    ("Email", "Person", "TO"),
    ("Email", "Person", "CC"),
    ("Email", "Person", "BCC"),
    ("Email", "Email", "REPLIED_TO"),
    ("Email", "Email", "FORWARDED_FROM"),
    ("Person", "Company", "AFFILIATED_WITH"),
    ("Email", "Link", "MENTIONS_LINK"),
    ("Email", "PhoneNumber", "MENTIONS_PHONE")
]

# Add edges with labels
for src, dst, label in relations:
    G.add_edge(src, dst, label=label)

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # layout for consistent placement
colors = [G.nodes[node]['color'] for node in G.nodes]

plt.figure(figsize=(10, 8))
nx.draw(
    G,
    pos,
    with_labels=True,
    node_color=colors,
    node_size=3000,
    font_size=10,
    font_weight='bold',
    arrowsize=20
)

# Draw edge labels
edge_labels = {(src, dst): label for src, dst, label in relations}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Email Data Graph Schema")
plt.axis('off')
plt.show()
