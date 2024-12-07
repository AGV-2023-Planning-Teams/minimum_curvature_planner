import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from perception_data import *

def matplotlib_visualize_splines(points: np.ndarray, line_label: str, points_label: str, is_loop: bool = False):
    if is_loop:
        points = np.vstack([points, points[0]])

    # Define the parametric points (x and y coordinates)
    t_points = np.arange(points.shape[0])
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])

    # Create parametric cubic splines for x(t) and y(t)
    bc_type = 'periodic' if is_loop else 'not-a-knot'
    spline_x = CubicSpline(t_points, x_points, bc_type=bc_type)
    spline_y = CubicSpline(t_points, y_points, bc_type=bc_type)

    # Generate values of t for plotting the spline
    t_fine = np.linspace(t_points[0], t_points[-1], 100)

    # Evaluate the splines to get the points on the curve
    x_fine = spline_x(t_fine)
    y_fine = spline_y(t_fine)

    # Plot the parametric spline
    plt.plot(x_fine, y_fine, label=line_label)
    plt.plot(x_points, y_points, 'o', label=points_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure the x and y axes have the same scale

def coords(t, a, b, c, d):
    t_floor = np.int32(t)
    t_frac = t - t_floor
    return np.array([ (a[I] + b[I]*f + c[I]*f**2 + d[I]*f**3) for i in range(t.shape[0]) if (I := (t_floor[i])%(a.shape[0]), f := t_frac[i]) ])

def implemented_visualize_splines(points: np.ndarray, line_label: str, points_label: str, show_control_points: bool = False, dashed: bool = False):
    Ainv = matAInv(points.shape[0])
    centreline = Centreline(points.shape[0], points, None, None)
    abcd_x = (Ainv @ q_comp(centreline, 0)).reshape((-1, 4))
    abcd_y = (Ainv @ q_comp(centreline, 1)).reshape((-1, 4))

    # Define the parametric points (x and y coordinates)
    t_points = np.arange(points.shape[0])
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])

    # Generate values of t for plotting the spline
    t_fine = np.linspace(t_points[0], t_points[-1] + 1, 1000)

    # Evaluate the splines to get the points on the curve
    x_fine = coords(t_fine, abcd_x[:, 0], abcd_x[:, 1], abcd_x[:, 2], abcd_x[:, 3])
    y_fine = coords(t_fine, abcd_y[:, 0], abcd_y[:, 1], abcd_y[:, 2], abcd_y[:, 3])

    # Plot the parametric spline
    if dashed: plt.plot(x_fine, y_fine, '--', label=line_label)
    else: plt.plot(x_fine, y_fine, label=line_label)
    if show_control_points: plt.plot(x_points, y_points, 'o', label=points_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure the x and y axes have the same scale

def points_raceline(centreline: Centreline, alpha: np.ndarray):
    return centreline.p + centreline.n * np.repeat(alpha, 2, axis=0).reshape((-1, 2))

def visualize_solution(centreline: Centreline, points_raceline: np.ndarray):
    plt.figure(figsize=(10, 10))
    plt.title('Minimum Curvature Plan')
    points_centreline = centreline.p
    points_boundary_max = centreline.p + centreline.n * np.repeat(centreline.half_w_tr, 2, axis=0).reshape((-1, 2))
    points_boundary_min = centreline.p - centreline.n * np.repeat(centreline.half_w_tr, 2, axis=0).reshape((-1, 2))
    # matplotlib_visualize_splines(points_centreline, True)
    # matplotlib_visualize_splines(points_boundary_max, 'Outer boundary from scipy', 'Control Points: Outer boundary from scipy', True)
    # matplotlib_visualize_splines(points_boundary_min, 'Inner boundary from scipy', 'Control Points: Inner boundary from scipy', True)
    # matplotlib_visualize_splines(points_raceline, 'Raceline from scipy', 'Control Points: Raceline from scipy', True)
    implemented_visualize_splines(points_centreline, 'Centreline from implementation', 'Control Points: Centreline from implementation', dashed=True)
    implemented_visualize_splines(points_boundary_max, 'Outer boundary from implementation', 'Control Points: Outer boundary from implementation')
    implemented_visualize_splines(points_boundary_min, 'Inner boundary from implementation', 'Control Points: Inner boundary from implementation')
    implemented_visualize_splines(points_raceline, 'Raceline from implementation', 'Control Points: Raceline from implementation')
    plt.show()
