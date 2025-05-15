#include <math.h>

#include "bb_bem.h"

// Constants from Fortran 
#define PI            3.141592653589793
#define EPSILON_0     (8.854187818e-12)

// 3D cross product: w = u × v 
static void cross_product(const double u[3], const double v[3], double w[3]) {
    w[0] = u[1] * v[2] - u[2] * v[1];
    w[1] = u[2] * v[0] - u[0] * v[2];
    w[2] = u[0] * v[1] - u[1] * v[0];
}

// Face integral function translated from Fortran face_integral 
static double face_integral(const double xs[3],
                            const double ys[3],
                            const double zs[3],
                            double x, double y, double z) {
    double r[3], u[3], v[3], w[3];
    double xi, xj, yi, dx, dy, t, l, m, d, ti, tj;
    double theta, omega, q, g, zp, zpabs;
    double ox, oy, oz;
    double sum = 0.0;

    // Compute distances r[i] = |P - vertex_i| 
    for (int i = 0; i < 3; i++) {
        double dx0 = xs[i] - x;
        double dy0 = ys[i] - y;
        double dz0 = zs[i] - z;
        r[i] = sqrt(dx0 * dx0 + dy0 * dy0 + dz0 * dz0);
    }

    // Compute normalized face normal w = (xs2-xs1)×(xs3-xs2) 
    u[0] = xs[1] - xs[0];
    v[0] = xs[2] - xs[1];
    u[1] = ys[1] - ys[0];
    v[1] = ys[2] - ys[1];
    u[2] = zs[1] - zs[0];
    v[2] = zs[2] - zs[1];
    cross_product(u, v, w);
    {
        double wn = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        w[0] /= wn;
        w[1] /= wn;
        w[2] /= wn;
    }

    // Project P onto plane: zp = (P - vertex1)·w 
    u[0] = x - xs[0];
    u[1] = y - ys[0];
    u[2] = z - zs[0];
    zp = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    zpabs = fabs(zp);

    // Origin of projected plane 
    ox = x - zp * w[0];
    oy = y - zp * w[1];
    oz = z - zp * w[2];

    // Loop over each edge of the triangle 
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;

        // Position of vertex j in projected plane coords 
        u[0] = xs[j] - ox;
        u[1] = ys[j] - oy;
        u[2] = zs[j] - oz;
        xj = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        u[0] /= xj;
        u[1] /= xj;
        u[2] /= xj;

        // y-axis basis via cross(w, u) 
        cross_product(w, u, v);

        // Coordinates of vertex i in (u,v) basis 
        xi = (xs[i] - ox) * u[0] + (ys[i] - oy) * u[1] + (zs[i] - oz) * u[2];
        yi = (xs[i] - ox) * v[0] + (ys[i] - oy) * v[1] + (zs[i] - oz) * v[2];

        dx = xj - xi;
        dy = -yi; // since yj = 0 
        t = sqrt(dx * dx + dy * dy);
        l = dx / t;
        m = dy / t;
        d = l * yi - m * xi;
        ti = l * xi + m * yi;
        tj = l * xj; // since yj = 0 

        theta = atan2(yi, xi);
        omega = theta
            - atan2(r[i] * d, zpabs * ti)
            + atan2(r[j] * d, zpabs * tj);
        q = log((r[j] + tj) / (r[i] + ti));
        g = d * q - zpabs * omega;

        sum += g;
    }

    return fabs(sum) / (4.0 * PI * EPSILON_0);
}

/**
 * element_ij_ - User-defined function to compute the integral
 *
 * Here we assume each face is a triangle (3 nodes per face).
 */
double element_ij_(
    const int* p_i,
    const int* p_j,
    const bb_props_t* props
) {
    int fi = *p_i;
    int fj = *p_j;

    int* face2node = props->face2node;
    vector3_t* np = props->np;

    // Coordinates of face fi for centroid 
    double xf_i[3], yf_i[3], zf_i[3];
    for (int k = 0; k < 3; k++) {
        int node = face2node[fi * 3 + k];
        xf_i[k] = np[node].x;
        yf_i[k] = np[node].y;
        zf_i[k] = np[node].z;
    }
    // Centroid of face fi 
    double xp = (xf_i[0] + xf_i[1] + xf_i[2]) / 3.0;
    double yp = (yf_i[0] + yf_i[1] + yf_i[2]) / 3.0;
    double zp = (zf_i[0] + zf_i[1] + zf_i[2]) / 3.0;

    // Coordinates of face fj 
    double xf_j[3], yf_j[3], zf_j[3];
    for (int k = 0; k < 3; k++) {
        int node = face2node[fj * 3 + k];
        xf_j[k] = np[node].x;
        yf_j[k] = np[node].y;
        zf_j[k] = np[node].z;
    }

    // Compute and return the face integral 
    return face_integral(xf_j, yf_j, zf_j, xp, yp, zp);
}

double rhs_vector_i_(
    const int* p_i,
    const int* p_n,
    const bb_props_t* props
) {
    if (props->ndble_para_fc == 0) {
        return 1.0; // TODO
    }

    return props->dble_para_fc[*p_i * props->ndble_para_fc + 0];
}
