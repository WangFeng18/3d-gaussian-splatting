import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Cost', 'Number of Gaussians'])
fig.add_trace(go.Scatter(x=df['iteration'], y=df['loss'], mode='lines'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['iteration'], y=df['n_gaussians'], mode='lines'), row=2, col=1)

# Update layout
fig.update_layout(
    title='3D Gaussian Splatting Optimization',
    xaxis=dict(title='Iterations')
)
fig.write_html("metrics.html")
