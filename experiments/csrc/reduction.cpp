#include <vector>
#include <string>
#include <iostream>
#include <cudnn.h>
#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

// Row-major to scalar: Failed
// Row-major batchwise: Failed
// Row-major across row: Failed
// Row-major across column: Failed
// 3D Column-major to scalar: OK
// 3D Column-major batch-wise: Failed
// 3D Column-major across row: Failed
// 3D Column-major across column: OK
// 2D Row-major to scalar: OK
// 2D Row-major across row: OK
// 2D Row-major across column: Failed
// 2D Column-major to scalar: Failed
// 2D Column-major across row: Failed
// 2D Column-major across column: Failed

bool run_test(
    const std::string& test_name,
    const std::vector<int64_t>& a_dims,
    const std::vector<int64_t>& a_strides,
    const std::vector<int64_t>& c_dims,
    const std::vector<int64_t>& c_strides)
{
    fe::graph::Graph graph{};

    // tensors

    const auto input = graph.tensor(
        fe::graph::Tensor_attributes()
            .set_uid(100)
            .set_dim(a_dims)
            .set_stride(a_strides)
            .set_data_type(fe::DataType_t::FLOAT));
    
    // operations
    
    auto reduction = fe::graph::Reduction_attributes()
         .set_mode(fe::ReductionMode_t::MAX)
         .set_compute_data_type(fe::DataType_t::FLOAT);
    
    auto pointwise = fe::graph::Pointwise_attributes()
         .set_mode(fe::PointwiseMode_t::SUB)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    const auto max = graph.reduction(input, reduction);

    max->set_is_virtual(true)
        .set_dim(c_dims)
        .set_stride(c_strides)
        .set_data_type(fe::DataType_t::FLOAT);

    const auto output = graph.pointwise(input, max, pointwise);

    output->set_is_virtual(false)
        .set_dim(a_dims)
        .set_stride(a_strides)
        .set_data_type(fe::DataType_t::FLOAT);

    bool is_good = true;

    cudnnHandle_t handle;

    is_good &= cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS;
    is_good &= graph.validate().is_good();
    is_good &= graph.build_operation_graph(handle).is_good();
    is_good &= graph.create_execution_plans({fe::HeurMode_t::A}).is_good();
    is_good &= graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good();
    is_good &= cudnnDestroy(handle) == CUDNN_STATUS_SUCCESS;

    std::cout << test_name << ": " << (is_good ? "OK" : "Failed") << std::endl;

    return is_good;
}

int main()
{
    // cudnn_frontend::isLoggingEnabled() = false;

    constexpr int64_t b = 4;
    constexpr int64_t m = 32;
    constexpr int64_t n = 16;

    run_test(
        "3D Row-major to scalar",
        {b,     m, n},
        {n * m, n, 1},
        {1, 1, 1},
        {1, 1, 1});
        
    run_test(
        "3D Row-major batchwise",
        {b,     m, n},
        {n * m, n, 1},
        {b, 1, 1},
        {1, 1, 1});
      
    run_test(
        "3D Row-major across row",
        {b,     m, n},
        {n * m, n, 1},
        {b, m, 1},
        {m, 1, 1});
    
    run_test(
        "3D Row-major across column",
        {b,     m, n},
        {n * m, n, 1},
        {b, 1, n},
        {n, n, 1});


    run_test(
        "3D Column-major to scalar",
        {b,     m, n},
        {n * m, 1, m},
        {1, 1, 1},
        {1, 1, 1});

    run_test(
        "3D Column-major batch-wise",
        {b,     m, n},
        {n * m, 1, m},
        {b, 1, 1},
        {1, 1, 1});
    
    run_test(
        "3D Column-major across row",
        {b,     m, n},
        {n * m, 1, m},
        {b, m, 1},
        {m, 1, 1});
    
    run_test(
        "3D Column-major across column",
        {b,     m, n},
        {n * m, 1, m},
        {b, 1, n},
        {n, 1, 1});

    
    
    run_test(
        "2D Row-major to scalar",
        {m, n},
        {n, 1},
        {1, 1},
        {1, 1});
    
    run_test(
        "2D Row-major across row",
        {m, n},
        {n, 1},
        {m, 1},
        {1, 1});
    
    run_test(
        "2D Row-major across column",
        {m, n},
        {n, 1},
        {1, n},
        {1, 1});


    run_test(
        "2D Column-major to scalar",
        {m, n},
        {1, m},
        {1, 1},
        {1, 1});
    
    run_test(
        "2D Column-major across row",
        {m, n},
        {1, m},
        {m, 1},
        {1, 1});
    
    run_test(
        "2D Column-major across column",
        {m, n},
        {1, m},
        {1, n},
        {1, 1});
}
