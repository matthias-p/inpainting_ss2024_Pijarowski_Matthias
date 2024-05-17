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


std::set<std::pair<int, int>> nn_circular(const py::array_t<int64_t> &mask, const std::pair<int32_t, int32_t> origin, const std::vector<float_t> discretization, int32_t max_distance, u_int8_t k) {
    int height = mask.shape(0), width = mask.shape(1);
    int64_t* mask_ptr = static_cast<int64_t*>(mask.request().ptr);
    std::set<std::pair<int, int>> neighbors;

    const auto [origin_y, origin_x] = origin;
    for (int distance = 1; distance < max_distance; distance++) {
        for (const auto d: discretization) {
            int y = distance * std::cos(d) + origin_y;
            if (y < 0 || y >= height) {
                continue;
            }
            int x = distance * std::sin(d) + origin_x;
            if (x < 0 || x >= width) {
                continue;
            }
            if (mask_ptr[y * width + x] == DEFINED) {
                neighbors.insert({y, x});
                if (neighbors.size() == k)
                    return neighbors;
            }
        }
    }
    return neighbors;
}


std::set<std::pair<int, int>> nn_circular_native(const py::array_t<int64_t> &mask, const std::pair<int32_t, int32_t> origin, u_int8_t k) {
    const int height = mask.shape(0), width = mask.shape(1);
    const auto [origin_y, origin_x] = origin;
    int64_t* mask_ptr = static_cast<int64_t*>(mask.request().ptr);
    std::set<std::pair<int, int>> neighbors;

    const int max_distance = std::max({euclidean_distance(origin, {0, 0}), euclidean_distance(origin, {0, width}), euclidean_distance(origin, {height, 0}), euclidean_distance(origin, {height, width})});
    float discretization_distance = 2.0 / 360.0;
    for (int distance = 1; distance < max_distance; distance++) {
        for (int i = 0; i < 360; i++) {
            float theta = i * discretization_distance * std::numbers::pi;
            int y = distance * std::cos(theta) + origin_y;
            if (y < 0 || y >= height) {
                continue;
            }
            int x = distance * std::sin(theta) + origin_x;
            if (x < 0 || x >= width) {
                continue;
            }
            if (mask_ptr[y * width + x] == DEFINED) {
                neighbors.insert({y, x});
                if (neighbors.size() == k)
                    return neighbors;
            }
        }
    }
    return neighbors;
}


PYBIND11_MODULE(knn, m) {
    m.doc() = "example plugin";

    m.def("euclidean_distance", &euclidean_distance, "Compute euclidean distance");
    m.def("nn_circular", &nn_circular);
    m.def("nn_circular_native", &nn_circular_native);
}