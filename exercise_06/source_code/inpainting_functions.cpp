#include <cmath>
#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#define DEFINED 0
#define UNDEFINED 255

namespace py = pybind11;


std::vector<std::pair<int, int>> generate_candidates(const py::array_t<uint8_t, py::array::c_style> &mask, const std::vector<std::pair<int, int>> neighbors, uint32_t n) {
    uint8_t* mask_ptr = static_cast<uint8_t*>(mask.request().ptr);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> height_distrib(0, mask.shape(0) - 1);
    std::uniform_int_distribution<> width_distrib(0, mask.shape(1) - 1);

    std::vector<std::pair<int, int>> candidates;

    while (candidates.size() < n) {
        retry_gen:;
        int y = height_distrib(gen);
        int x = width_distrib(gen);

        if (mask_ptr[y * mask.shape(1) + x] == UNDEFINED) {
            goto retry_gen;
        }
        for (const auto &neighbor: neighbors) {
            if ((neighbor.first + y < 0) || (neighbor.first + y >= mask.shape(0)) || (neighbor.second + x < 0) || (neighbor.second + x >= mask.shape(1)) || (mask_ptr[neighbor.first + y * mask.shape(0) + neighbor.second + x] == UNDEFINED)) {
                goto retry_gen;
            }
        }
        candidates.push_back({y, x});
    }
    return candidates;
}


float sparse_l2(const py::array_t<uint8_t, py::array::c_style> &image, const std::pair<int, int> &origin, const std::pair<int, int> &candidate, const std::vector<std::pair<int, int>> &neighbors) {
    const uint8_t* image_ptr = static_cast<uint8_t*>(image.request().ptr);
    const int width = image.shape(1);

    float sum = 0;
    for (const auto& neighbor: neighbors) {
        int origin_index = (origin.first + neighbor.first) * width * 3 + (origin.second + neighbor.second) * 3;
        int candiate_index = (candidate.first + neighbor.first) * width * 3 + (candidate.second + neighbor.second) * 3;
        for (int i = 0; i < 3; i++) {
            sum += std::pow(image_ptr[origin_index + i] - image_ptr[candiate_index + i], 2);
        }
    }
    return std::sqrt(sum);
}


std::pair<int, int> choose_candidate(const py::array_t<uint8_t> &image, const std::pair<int, int> origin, const std::vector<std::pair<int, int>> candidates, const std::vector<std::pair<int, int>> neighbors) {

}


PYBIND11_MODULE(inpainting_functions, m) {
    m.doc() = "example plugin";

    m.def("generate_candidates", &generate_candidates);
    m.def("choose_candidate", &choose_candidate);
    m.def("sparse_l2", &sparse_l2);
}