using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph.Nodes;

using OpCode = cutensorOperator_t;

internal sealed class Sum<T>(
    CudaNode<T> source,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    CudaReduction<T>(source, axis, OpCode.CUTENSOR_OP_ADD, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

internal sealed class Max<T>(
    CudaNode<T> source,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    CudaReduction<T>(source, axis, OpCode.CUTENSOR_OP_MAX, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

internal sealed class Min<T>(
    CudaNode<T> source,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    CudaReduction<T>(source, axis, OpCode.CUTENSOR_OP_MIN, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;
