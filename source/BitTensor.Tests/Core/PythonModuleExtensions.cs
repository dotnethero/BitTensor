using BitTensor.Abstractions;
using BitTensor.CUDA;
using BitTensor.CUDA.Graph;
using Python.Runtime;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

public record TensorData(Shape Shape, float[] Values);

static class PythonModuleExtensions
{
    public static void ExecuteJax(this PyModule scope, string code)
    {
        var jaxcode = 
            $"""
            import jax
            import jax.numpy as jnp

            {code}
            """;

        ExecuteScript(scope, jaxcode);
    }
    
    public static void ExecuteScript(this PyModule scope, string code)
    {
        var script = PythonEngine.Compile(code);
        scope.Execute(script);
    }
    
    public static Shape GetShape(this PyModule scope, string jnptensor)
    {
        var extents = scope.Eval<int[]>($"{jnptensor}.shape")!;
        return Shape.Create(extents);
    }
    
    public static TensorData GetTensor(this PyModule scope, string jnptensor)
    {
        var values = scope.Eval<float[]>($"{jnptensor}.flatten().tolist()")!;
        var shape = scope.GetShape(jnptensor);
        return new(shape, values);
    }

    public static TensorData Get1D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[]>($"{jnptensor}.tolist()")!;
        return new([array.Length], array);
    }

    public static TensorData Get2D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][]>($"{jnptensor}.tolist()")!;
        return new([array.Length, array[0].Length], array.Collect2D());
    }
    
    public static TensorData Get3D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][][]>($"{jnptensor}.tolist()")!;
        return new([array.Length, array[0].Length, array[0][0].Length], array.Collect3D());
    }

    public static CuTensor AsTensor(this TensorData tensor, CuContext context) => context.Allocate(tensor.Shape, tensor.Values);

    public static CuTensorNode AsNode(this TensorData tensor, CuContext context) => AsTensor(tensor, context).CreateNode();
}
