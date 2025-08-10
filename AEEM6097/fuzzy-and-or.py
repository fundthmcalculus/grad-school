import numpy as np
import plotly.graph_objects as go

# Number of points in each dimension for plotting
N = 20
# Generate 2D data points on unit square [0, 1] x [0, 1]
x1 = np.linspace(0, 1, N)
x2 = np.linspace(0, 1, N)
x1, x2 = np.meshgrid(x1, x2)
# Plot the AND operation as x1*x2
and_x1_x2 = x1 * x2
# Plot the OR operation as x1 + x2 - x1*x2
or_x1_x2 = x1 + x2 - x1 * x2


# Create an animation of the morphology between the two:
def morph_and_or(c: float):
    """Morph between AND + OR operations."""
    return (1 - c) * and_x1_x2 + c * or_x1_x2


n_frames = 25
t_frame = np.linspace(0, 1, n_frames)

frames = [go.Frame(name=str(t), data=[go.Surface(z=morph_and_or(t),x=x1,y=x2, showscale=True)], layout=dict(title=f'c={t}')) for
          t in t_frame]

fig = go.Figure(data=[go.Surface(z=and_x1_x2,x=x1,y=x2, showscale=True)],
                layout=go.Layout(scene=dict(camera=dict(eye=dict(x=-3, y=-3, z=1)  # Look towards (1,1) from above
                )), sliders=[dict(steps=[dict(label=str(t), method='animate', args=[[str(t)]]) for t in t_frame],
                    transition=dict(duration=100), visible=True)]))

fig.frames = frames
fig.show()
