using System.Numerics;
using BitTensor.CUDA.Interop;

namespace BitTensor.CUDA.Graph.Nodes;

using OpCode = cutensorOperator_t;

public sealed class Sum<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    AbstractReduction<T>(a, axis, OpCode.CUTENSOR_OP_ADD, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

public sealed class Max<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    AbstractReduction<T>(a, axis, OpCode.CUTENSOR_OP_MAX, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;

public sealed class Min<T>(
    AbstractNode<T> a,
    HashSet<Index> axis,
    float scale = 1,
    bool keepDims = false) : 
    AbstractReduction<T>(a, axis, OpCode.CUTENSOR_OP_MIN, scale, keepDims)
    where T : unmanaged, IFloatingPoint<T>;
