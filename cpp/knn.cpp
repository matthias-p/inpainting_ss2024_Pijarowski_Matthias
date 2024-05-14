#include <array>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#define DEFINED 0

namespace py = pybind11;


float euclidean_distance(const std::tuple<int, int> point1, const std::tuple<int, int> point2) {
    const auto [p1_y, p1_x] = point1;
    const auto [p2_y, p2_x] = point2;

    return std::sqrt(std::pow((p2_y - p1_y), 2) + std::pow((p2_x - p1_x), 2));
}


std::set<std::tuple<int, int>> nn_circular(const py::array_t<double> &mask, std::tuple<int, int> origin, const uint64_t k) {
    py::buffer_info mask_info = mask.request();
    assert(mask_info.ndim == 2);

    std::set<std::tuple<int, int>> neighbors;
    const auto [origin_y, origin_x] = origin;
    int height = mask_info.shape[0], width = mask_info.shape[1];
    double *mask_ptr = static_cast<double *>(mask_info.ptr);

    float d_tl = euclidean_distance(origin, {0, 0});
    float d_bl = euclidean_distance(origin, {height, 0});
    float d_tr = euclidean_distance(origin, {0, width});
    float d_br = euclidean_distance(origin, {height, width});
    int max_distance = std::round(std::max({d_tl, d_bl, d_tr, d_br}));

    std::array<float, 360> thetas;
    const float step_size = static_cast<float>(2) / 360;  // need pi between 0 and 2 for full period
    for (int i = 0; i < 360; i++) {
        thetas[i] = i * step_size * std::numbers::pi;
    }

    for (int radius = 1; radius < max_distance; radius++) {
        for (auto & theta: thetas) {
            int y = std::round(radius * std::cos(theta) + origin_y); 
            int x = std::round(radius * std::sin(theta) + origin_x);
            if (y >= 0 && y < height && x >= 0 && x < width && mask_ptr[y * height + x] == DEFINED) {
                neighbors.insert({y, x});
                if (neighbors.size() == k) {
                    return neighbors;
                }
            }
        }
    }

    return neighbors;
}


bool sort_vector(const std::tuple<int, int, float> &x, const std::tuple<int, int, float> &y) {
    return std::get<2>(x) < std::get<2>(y);
}


void add_elem(std::vector<std::tuple<int, int, float>> &neighbors, int y, int x, float distance, const uint64_t k) {
    neighbors.push_back({y, x, distance});
    if (neighbors.size() > k) {
        std::sort(neighbors.begin(), neighbors.end(), sort_vector);
        neighbors.pop_back();
    }
}


std::vector<std::tuple<int, int, float>> nn_quadratic(const py::array_t<double> &mask, const std::tuple<int, int> origin, const uint64_t k) {
    double *mask_ptr = static_cast<double *>(mask.request().ptr);

    const auto [origin_x, origin_y] = origin;
    int height = mask.shape(0), width = mask.shape(1);
    float worst_distance = 0;

    std::vector<std::tuple<int, int, float>> neighbors;

    for (int distance = 1; distance < height / 2 + 1; distance++) {
        int y_t = origin_y - distance;
        int y_b = origin_y + distance;
        int x_l = origin_x - distance;
        int x_r = origin_x + distance;

        for (int x = x_l; x <= x_r; x++) {
            if (mask_ptr[y_t * width + x] == DEFINED) {
                float d = euclidean_distance(origin, {y_t, x});
                add_elem(neighbors, y_t, x, d, k);
                if (neighbors.size() == k && d > worst_distance)
                    return neighbors;
                worst_distance = std::max({worst_distance, d});
            }
            if (mask_ptr[y_b * width + x] == DEFINED) {
                float d = euclidean_distance(origin, {y_b, x});
                add_elem(neighbors, y_b, x, d, k);
                if (neighbors.size() == k && d > worst_distance)
                    return neighbors;
                worst_distance = std::max({worst_distance, d});
            }
        }

        for (int y = y_t; y <= y_b; y++) {
            if (mask_ptr[y * width + x_l] == DEFINED) {
                float d = euclidean_distance(origin, {y, x_l});
                add_elem(neighbors, y, x_l, d, k);
                if (neighbors.size() == k && d > worst_distance)
                    return neighbors;
                worst_distance = std::max({worst_distance, d});
            }
            if (mask_ptr[y * width + x_r] == DEFINED) {
                float d = euclidean_distance(origin, {y, x_r});
                add_elem(neighbors, y, x_r, d, k);
                if (neighbors.size() == k && d > worst_distance)
                    return neighbors;
                worst_distance = std::max({worst_distance, d});
            }
        }
    }
    return neighbors;
}


std::set<std::tuple<int, int>> get_neighbors(const py::array_t<int64_t> &mask, const py::array_t<int32_t> &Y, const py::array_t<int32_t> &X, const uint64_t k) {
    int64_t* mask_ptr = static_cast<int64_t*>(mask.request().ptr);
    int32_t* Y_ptr = static_cast<int32_t*>(Y.request().ptr);
    int32_t* X_ptr = static_cast<int32_t*>(X.request().ptr);

    std::set<std::tuple<int, int>> neighbors;

    for (int i = 0; i < 360; i++) {
        int y = Y_ptr[i], x = X_ptr[i];
        if (mask_ptr[y * mask.shape(1) + x] == DEFINED && y >= 0 && y < mask.shape(0) && x >= 0 && x < mask.shape(1)) {
            neighbors.insert({y, x});
            if (neighbors.size() == k) {
                break;
            }
        }
    }
    return neighbors;
}



PYBIND11_MODULE(example, m) {
    m.doc() = "example plugin";

    m.def("euclidean_distance", &euclidean_distance, "Compute euclidean distance");
    m.def("nn_circular", &nn_circular);
    m.def("nn_quadratic", &nn_quadratic);
    m.def("get_neighbors", &get_neighbors);
}


// int main() {
//     std::array<float, 360> thetas;
//     float step_size better name than (distance) = static_cast<float>(2) / 360;
//     for (int i = 0; i < 360; i++) {
//         thetas[i] = i * distance * std::numbers::pi;
//     }
    
//     for (auto & values: thetas){
//         std::cout << values << std::endl;
//     }
// }