#include <torch/extension.h>

TORCH_LIBRARY(neighbors, m) {
    m.def("get_neighbor_pairs_brute(Tensor positions, Tensor batch, Tensor box_vectors, bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs, Tensor num_pairs)");
    m.def("get_neighbor_pairs_shared(Tensor positions, Tensor batch, Tensor box_vectors, bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs, Tensor num_pairs)");
    m.def("get_neighbor_pairs_cell(Tensor positions, Tensor batch, Tensor box_vectors, bool use_periodic, Scalar cutoff_lower, Scalar cutoff_upper, Scalar max_num_pairs, bool loop, bool include_transpose) -> (Tensor neighbors, Tensor distances, Tensor distance_vecs, Tensor num_pairs)");
}
