using BitTensor.Abstractions;
using BitTensor.CUDA;
using Python.Runtime;

// ReSharper disable once CheckNamespace

namespace BitTensor.Core.Tests;

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
    
    public static int[] GetShape(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<int[]>($"{jnptensor}.shape")!;
        return array;
    }
    
    public static CuTensor GetTensor(this PyModule scope, string jnptensor)
    {
        var values = scope.Eval<float[]>($"{jnptensor}.flatten().tolist()")!;
        var shape = scope.Eval<int[]>($"{jnptensor}.shape")!;
        return new(shape, values);
    }

    public static CuTensor Get1D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[]>($"{jnptensor}.tolist()")!;
        return new([array.Length], array);
    }

    public static CuTensor Get2D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][]>($"{jnptensor}.tolist()")!;
        return new([array.Length, array[0].Length], array.Collect2D());
    }
    
    public static CuTensor Get3D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][][]>($"{jnptensor}.tolist()")!;
        return new([array.Length, array[0].Length, array[0][0].Length], array.Collect3D());
    }
}
