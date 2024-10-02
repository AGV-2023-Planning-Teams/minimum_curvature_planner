import numpy as np
import matplotlib.pyplot as plt
import pandas
from matrices import *
from perception_data import Centreline
from solve_qp import solve_for_alpha, solve_for_alpha_deg2, solve_for_alpha_deg2_not_pseudo
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def coords(t, a, b, c, d):
    t_floor = np.int32(t)
    t_frac = t - t_floor
    return np.array([ (a[I] + b[I]*f + c[I]*f**2 + d[I]*f**3) for i in range(t.shape[0]) if (I := (t_floor[i])%(a.shape[0]), f := t_frac[i]) ])

def coords_deg2(t, a, b, c):
    t_floor = np.int32(t)
    t_frac = t - t_floor
    return np.array([ (a[I] + b[I]*f + c[I]*f**2) for i in range(t.shape[0]) if (I := (t_floor[i])%(a.shape[0]), f := t_frac[i]) ])

def coords_not_loop(t, a, b, c, d):
    t_floor = np.int32(t)
    result = a + b * t + c * t**2 + d * t**3
    
    return result

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

def implemented_visualize_splines_deg2(points: np.ndarray, line_label: str, points_label: str, show_control_points: bool = False, dashed: bool = False):
    # Ainv = matAInv_deg2_not_pseudo(points.shape[0])
    # centreline = Centreline(points.shape[0], points, None, None)
    # abc_x = (Ainv @ q_comp_deg2_not_pseudo(centreline, 0)).reshape((-1, 3))
    # abc_y = (Ainv @ q_comp_deg2_not_pseudo(centreline, 1)).reshape((-1, 3))

    Ainv = matAInv_deg2(points.shape[0])
    centreline = Centreline(points.shape[0], points, None, None)
    abc_x = (Ainv @ q_comp_deg2(centreline, 0)).reshape((-1, 3))
    abc_y = (Ainv @ q_comp_deg2(centreline, 1)).reshape((-1, 3))

    # Define the parametric points (x and y coordinates)
    t_points = np.arange(points.shape[0])
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])

    # Generate values of t for plotting the spline
    t_fine = np.linspace(t_points[0], t_points[-1] + 1, 1000)

    # Evaluate the splines to get the points on the curve
    x_fine = coords_deg2(t_fine, abc_x[:, 0], abc_x[:, 1], abc_x[:, 2])
    y_fine = coords_deg2(t_fine, abc_y[:, 0], abc_y[:, 1], abc_y[:, 2])

    # Plot the parametric spline
    if dashed: plt.plot(x_fine, y_fine, '--', label=line_label)
    else: plt.plot(x_fine, y_fine, label=line_label)
    if show_control_points: plt.plot(x_points, y_points, 'o', label=points_label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Ensure the x and y axes have the same scale

def x0xf_visualize_splines(points: np.ndarray, line_label: str, points_label: str, refspline: np.ndarray, show_control_points: bool = False, dashed: bool = False):
    
    point0 = points[0]
    pointf = points[1]
    
    abcd_x = get_from_refspline(x0=point0[0], xf=pointf[0], refspline=refspline)
    abcd_y = get_from_refspline(x0=point0[1], xf=pointf[1], refspline=refspline)

    # Define the parametric points (x and y coordinates)
    t_points = np.arange(points.shape[0])
    x_points = np.array(points[:, 0])
    y_points = np.array(points[:, 1])

    # Generate values of t for plotting the spline
    # t_fine = np.linspace(t_points[0], t_points[-1] + 1, 1000)
    t_fine = np.linspace(0, 1, int(100*(abs(point0[0]-pointf[0]) + abs(point0[1]-pointf[1]))))
    print(t_fine.shape)

    print(abcd_x, abcd_y)

    # Evaluate the splines to get the points on the curve
    x_fine = coords_not_loop(t_fine, abcd_x[:, 0], abcd_x[:, 1], abcd_x[:, 2], abcd_x[:, 3])
    y_fine = coords_not_loop(t_fine, abcd_y[:, 0], abcd_y[:, 1], abcd_y[:, 2], abcd_y[:, 3])

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
    # matplotlib_visualize_splines(points_centreline, 'Centreline from scipy', 'Control Points: Raceline from scipy', True)
    # matplotlib_visualize_splines(points_boundary_max, 'Outer boundary from scipy', 'Control Points: Outer boundary from scipy', True)
    # matplotlib_visualize_splines(points_boundary_min, 'Inner boundary from scipy', 'Control Points: Inner boundary from scipy', True)
    # matplotlib_visualize_splines(points_raceline, 'Raceline from scipy', 'Control Points: Raceline from scipy', True)
    implemented_visualize_splines(points_centreline, 'Centreline from implementation', 'Control Points: Centreline from implementation', dashed=True)
    implemented_visualize_splines(points_boundary_max, 'Outer boundary from implementation', 'Control Points: Outer boundary from implementation')
    implemented_visualize_splines(points_boundary_min, 'Inner boundary from implementation', 'Control Points: Inner boundary from implementation')
    implemented_visualize_splines(points_raceline, 'Raceline from implementation', 'Control Points: Raceline from implementation')
    # implemented_visualize_splines_deg2(points_centreline, 'Centreline from implementation', 'Control Points: Centreline from implementation', dashed=True)
    # implemented_visualize_splines_deg2(points_boundary_max, 'Outer boundary from implementation', 'Control Points: Outer boundary from implementation')
    # implemented_visualize_splines_deg2(points_boundary_min, 'Inner boundary from implementation', 'Control Points: Inner boundary from implementation')
    # implemented_visualize_splines_deg2(points_raceline, 'Raceline from implementation', 'Control Points: Raceline from implementation')
    # implemented_visualize_splines_deg2(points_raceline, 'Raceline deg 2', 'Control Points: Raceline from implementation')
    plt.show()

gate_dimensions = [1.6, 1.6]

class Splined_Track():
    def __init__(self, waypoints, track_width):
        """
        :param waypoints: Array of track center waypoints in the form [[x1, y1], [x2, y2], ...]
        :param track_width: Width of the track
        
        n_gates: Number of gates (waypoints)
        arc_length: Array of arc lengths between the gates. spline maps from arc_length to waypoints
        track: Cubic spline object
        track_centers: Array of track center points extrapolated from the spline
        """

        dists = np.linalg.norm(waypoints[1:, :] - waypoints[:-1, :], axis=1)
        self.arc_length = np.zeros(shape=np.size(waypoints, 0))
        self.arc_length[1:] = np.cumsum(dists)
        self.waypoints = waypoints
        self.track_width = track_width
        self.track = CubicSpline(self.arc_length, waypoints, bc_type='periodic')
        dists = np.linalg.norm(waypoints[1:, :] - waypoints[:-1, :], axis=1)
        taus = np.linspace(0, self.arc_length[-1], 2**12)
        self.track_centers = self.track(taus)
        self.track_tangent = self.track.derivative(nu=1)(taus)
        self.track_tangent /= np.linalg.norm(self.track_tangent, axis=1)[:, np.newaxis]
        self.track_normals = np.zeros_like(self.track_tangent)
        self.track_normals[:, 0],  self.track_normals[:, 1] = -self.track_tangent[:, 1], self.track_tangent[:, 0]
        self.track_normals /= np.linalg.norm(self.track_normals, axis=1)[:, np.newaxis]

        
    def nearest_trackpoint(self, p): 
        """Find closest track frame to a reference point p.
        :param p: Point of reference
        :return: Index of track frame, track center, tangent and normal.

        function s in the paper
        """
        i = np.linalg.norm(self.track_centers - p, axis=1).argmin()
        return i, self.track_centers[i], self.track_tangent[i], self.track_normals[i]
    
    def plot_waypoints_2d(self, ax):
        ax.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'x')

    def plot_track(self, ax, draw_boundaries=False):
            # Plot the spline
            ax.plot(self.track_centers[:, 0], self.track_centers[:, 1], '--')
            # Plot the track width
            if draw_boundaries:
                left_boundary = self.track_centers + self.track_normals*self.track_width
                right_boundary = self.track_centers - self.track_normals*self.track_width
                ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'b-')
                ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b-')

def plot_tangents(ax, track):
    random_samples = np.random.randint(0, track.track_centers.shape[0], 10)
    ax.quiver(track.track_centers[random_samples, 0], track.track_centers[random_samples, 1], track.track_tangent[random_samples, 0], track.track_tangent[random_samples, 1], color='r')
    ax.quiver(track.track_centers[random_samples, 0], track.track_centers[random_samples, 1], track.track_normals[random_samples, 0], track.track_normals[random_samples, 1], color='g')

if __name__ == "__main__":
    plt.title('refpline for DP')
    x0xf_visualize_splines(np.array([[0, 1], [6, -8.78]]), 'Parametric Spline: from implementation', 'Control Points', [-1, 16, -5.8, -2])
    plt.show()
    # waypoints_csv = pandas.read_csv('Hockenheim_waypoints.csv')
    # points = waypoints_csv.values
    # centreline = Centreline(points.shape[0], points, 1.1*np.ones(points.shape[0]), 0.2)

    # Ainv = matAInv(centreline.N)
    # A_ex_c = A_ex_comp(centreline.N, 2)
    # q_x = q_comp(centreline, 0)
    # q_y = q_comp(centreline, 1)
    # x_d = first_derivatives(centreline, Ainv, q_x) # vector containing x_i'
    # y_d = first_derivatives(centreline, Ainv, q_y) # vector containing y_i'

    # centreline.calc_n(x_d, y_d)

    # points_centreline = centreline.p
    # points_boundary_max = centreline.p + centreline.n * np.repeat(centreline.half_w_tr, 2, axis=0).reshape((-1, 2)) 
    # points_boundary_min = centreline.p - centreline.n * np.repeat(centreline.half_w_tr, 2, axis=0).reshape((-1, 2))

    

    

    # alpha = solve_for_alpha(centreline)

    # # alpha = solve_for_alpha_deg2(centreline)
    # # alpha = solve_for_alpha_deg2_not_pseudo(centreline)


    # points_racelines = centreline.p + centreline.n * np.repeat(alpha, 2, axis=0).reshape((-1, 2)) 

    # plt.figure(figsize=(10, 10))
    # plt.title('Minimum Curvature Plan')
    # implemented_visualize_splines(centreline.p, 'Centreline from implementation', 'Control Points: Centreline from implementation', dashed=True)
    # implemented_visualize_splines(points_boundary_max, 'Outer boundary from implementation', 'Control Points: Outer boundary from implementation')
    # implemented_visualize_splines(points_boundary_min, 'Inner boundary from implementation', 'Control Points: Inner boundary from implementation')
    # implemented_visualize_splines(points_racelines, 'Raceline from implementation', 'Control Points: Raceline from implementation')

    # # implemented_visualize_splines_deg2(centreline.p, 'Centreline from implementation', 'Control Points: Centreline from implementation', dashed=True)
    # # implemented_visualize_splines_deg2(points_boundary_max, 'Outer boundary from implementation', 'Control Points: Outer boundary from implementation')
    # # implemented_visualize_splines_deg2(points_boundary_min, 'Inner boundary from implementation', 'Control Points: Inner boundary from implementation')
    # # implemented_visualize_splines_deg2(points_racelines, 'Raceline from implementation', 'Control Points: Raceline from implementation')
    # # track = Splined_Track(points_centreline, 1.1)
    # # points_raceline = Splined_Track(points_centreline, 1.1)
    # plt.show()