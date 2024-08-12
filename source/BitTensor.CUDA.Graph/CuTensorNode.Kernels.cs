using BitTensor.CUDA.Interop;
using System.Numerics;
using BitTensor.CUDA.Kernels;
using ILGPU;
using ILGPU.Runtime;

namespace BitTensor.CUDA.Graph;

using View = ArrayView<float>;

public static partial class CuTensorNode
{
    public static CuTensorNode<float> Sigmoid(CuTensorNode<float> a)
    {
        var context = GetContext(a);
        var output = context.Allocate<float>(a.Shape);
        var forward = context.LoadUnaryKernel(CuKernels.SigmoidForward);
        return new(
            output,
            children: [a],
            forward: () => forward(a.Size, a.View, output.View),
            backward: (grad, self) => [SigmoidBackward(grad, self)]);
    }

    private static CuTensorNode<float> SigmoidBackward(CuTensorNode<float> grad, CuTensorNode<float> sigmoid)
    {
        var context = GetContext(grad, sigmoid);
        var output = context.Allocate<float>(grad.Shape);
        var backward = context.LoadBinaryKernel(CuKernels.SigmoidBackward);
        return new(
            output,
            children: [grad, sigmoid],
            forward: () => backward(grad.Size, grad.View, sigmoid.View, output.View),
            backward: (_, _) => throw new NotSupportedException());
    }
    
    public static CuTensorNode<float> Tanh(CuTensorNode<float> a)
    {
        var context = GetContext(a);
        var output = context.Allocate<float>(a.Shape);
        var forward = context.LoadUnaryKernel(CuKernels.TanhForward);
        return new(
            output,
            children: [a],
            forward: () => forward(a.Size, a.View, output.View),
            backward: (grad, self) => [TanhBackward(grad, self)]);
    }

    private static CuTensorNode<float> TanhBackward(CuTensorNode<float> grad, CuTensorNode<float> tanh)
    {
        var context = GetContext(grad, tanh);
        var output = context.Allocate<float>(grad.Shape);
        var backward = context.LoadBinaryKernel(CuKernels.TanhBackward);
        return new(
            output,
            children: [grad, tanh],
            forward: () => backward(grad.Size, grad.View, tanh.View, output.View),
            backward: (_, _) => throw new NotSupportedException());
    }
    
    private static Action<Index1D, View, View> LoadUnaryKernel(
        this CuContext context, 
        Action<Index1D, View, View> kernel) => 
        context.Accelerator.LoadAutoGroupedStreamKernel(kernel);
    
    private static Action<Index1D, View, View, View> LoadBinaryKernel(
        this CuContext context, 
        Action<Index1D, View, View, View> kernel) => 
        context.Accelerator.LoadAutoGroupedStreamKernel(kernel);
}