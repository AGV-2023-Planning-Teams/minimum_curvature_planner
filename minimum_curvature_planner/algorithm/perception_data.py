from import_cp import cp

class Centreline:
    """
    members:
    int N
    float p[N][2]
    float n[N][2]
    (x_i -> p[i, 0], y_i -> p[i, 1])
    """
    def __init__(self, N, p, half_track_width, vehicle_width):
        self.N = cp.uint32(N)
        self.p = cp.array(p, dtype=cp.float32)
        self.half_w_tr = cp.array(half_track_width, dtype=cp.float32)
        self.w_veh = cp.float32(vehicle_width)

    def calc_n(self, x_derivatives, y_derivatives):
        self.n = cp.array([[y_d, -x_d] for i in range(self.N) if (x_d := x_derivatives[i], y_d := y_derivatives[i])], dtype=cp.float32) # not normalized
        self.n /= cp.repeat(cp.linalg.norm(self.n, axis=1), 2, axis=0).reshape(-1, 2) # normalized here