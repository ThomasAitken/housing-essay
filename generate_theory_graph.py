from graphviz import Digraph

dot = Digraph(format='png')
dot.attr(rankdir='TB', dpi='300', ranksep='1.1', nodesep='0.4')

# Compact edges by default
dot.attr('edge', minlen='1')

dot.attr('graph', fontsize='16')

# --- Central spine ---
dot.node('A', 'Initial Urban Density',
         shape='box', style='filled', fillcolor='#FFD580', color='#D97B00', penwidth='2', fontsize='16')
dot.node('B', 'Differences in Long-Run House Price Growth',
         shape='box', style='filled', fillcolor='#D97B00', penwidth='2', fontsize='16')

# Keep the yellow style for B too (missed fill color above)
dot.node('B', 'Differences in Long-Run House Price Growth',
         shape='box', style='filled', fillcolor='#FFD580', color='#D97B00', penwidth='2')

dot.edge('A', 'B', weight='5')  # shorter spine (no extra minlen)

# --- A-row anchors for symmetry ---
dot.node('L1', '', shape='point', width='0', height='0', style='invis')
dot.node('R1', '', shape='point', width='0', height='0', style='invis')
with dot.subgraph(name='row_A') as r:
    r.attr(rank='same')
    r.node('L1'); r.node('A'); r.node('R1')
dot.edge('L1', 'A', style='invis', weight='10')
dot.edge('A', 'R1', style='invis', weight='10')

# --- Left-side waypoint on B's row to force C3->E1 around LEFT of B ---
dot.node('BENDL', '', shape='point', width='0', height='0', style='invis')
with dot.subgraph(name='row_B') as r:
    r.attr(rank='same')
    r.node('BENDL'); r.node('B')  # BENDL sits to the left of B on same rank
dot.edge('BENDL', 'B', style='invis', weight='10')  # bias ordering: BENDL -> B (left to right)

# --- CT1 (left) ---
with dot.subgraph(name='cluster_CT1') as c:
    c.attr(label='Causal Theory 1: Land Value Growth', style='filled', color='#E6F3FF', fontcolor='#005B96')
    c.node('C1', 'City population increases', shape='ellipse')
    c.node('C2', 'Land value grows even without\nsupply-demand mismatch', shape='ellipse')
    c.node('C3', 'Low density means more properties with land')
    c.edge('C1', 'C2')
    c.edge('C2', 'C3')

# --- CT2 (right) ---
with dot.subgraph(name='cluster_CT2') as c:
    c.attr(label='Causal Theory 2: Supply-Demand Mismatch', style='filled', color='#E6F3FF', fontcolor='#005B96')
    c.node('D1', 'Low-density cities face building obstacles', shape='ellipse')
    c.node('D1a', 'Expensive prime land', shape='note')
    c.node('D1b', 'Political resistance', shape='note')
    c.node('D1c', 'Poor public transport', shape='note')
    c.node('D1d', 'Cultural preference for land', shape='note')
    c.edge('D1', 'D1a'); c.edge('D1', 'D1b'); c.edge('D1', 'D1c'); c.edge('D1', 'D1d')
    c.node('D2', 'Supply-demand mismatch', shape='ellipse')
    c.node('D3', 'Mismatch drives short-term spikes', shape='ellipse')
    c.edge('D1', 'D2')
    c.edge('D2', 'D3')

# Connect CTs to spine (shorter arrows)
dot.edge('A', 'C1')
dot.edge('A', 'D1')
dot.edge('C3', 'B', weight='6')
dot.edge('D3', 'B', weight='6')

# Keep CT1 left of A and CT2 right of A
dot.edge('L1', 'C1', style='invis', weight='10')
dot.edge('R1', 'D1', style='invis', weight='10')

# --- CT3 (feedback at bottom) ---
with dot.subgraph(name='cluster_CT3') as c:
    c.attr(label='Causal Theory 3: Positive Feedback', style='filled', color='#E6F3FF', fontcolor='#005B96')
    c.node('E1', 'Price growth anchors expectations', shape='ellipse')
    c.node('E2', 'Willingness to take larger mortgages', shape='ellipse')
    c.node('E3', 'Reinforces further price growth', shape='ellipse')
    c.edge('E1', 'E2')
    c.edge('E2', 'E3')
    c.edge('E3', 'E1', style='dashed', constraint='false')

# Shorter arrow to feedback
dot.edge('B', 'E1')

# Cross-theory dotted links (force LEFT route for C3->E1 via BENDL)
dot.edge('C3', 'BENDL', style='dotted', constraint='false')
dot.edge('BENDL', 'E1', style='dotted', constraint='false')

# Right-side dotted link can remain direct
dot.edge('D3', 'E1', style='dotted', constraint='false')

# Render (optional)
output_path = "urban_density_house_price_growth"
dot.render(output_path)
