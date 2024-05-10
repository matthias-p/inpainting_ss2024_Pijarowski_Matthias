#include <array>
#include <vector>
#include <numbers>
#include <cmath>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define DEFINED 0

namespace py = pybind11;


float euclidean_distance(const std::tuple<int, int> point1, const std::tuple<int, int> point2) {
    const auto [p1_y, p1_x] = point1;
    const auto [p2_y, p2_x] = point2;

    return std::sqrt(std::pow((p2_y - p1_y), 2) + std::pow((p2_x - p1_x), 2));
}


std::vector<std::tuple<int, int>> nn_circular(py::array_t<double> mask, std::tuple<int, int> origin, int k) {
    py::buffer_info mask_info = mask.request();
    assert(mask_info.ndim == 2);

    std::vector<std::tuple<int, int>> neighbors;
    std::set<std::tuple<int, int>> processed;
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
                if (processed.contains({y, x})) {
                    continue;
                } else {
                    processed.insert({y, x});
                    neighbors.push_back({y, x});
                }
            }
        }
    }

    return neighbors;
}



PYBIND11_MODULE(example, m) {
    m.doc() = "example plugin";

    m.def("euclidean_distance", &euclidean_distance, "Compute euclidean distance");
    m.def("nn_circular", &nn_circular);
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