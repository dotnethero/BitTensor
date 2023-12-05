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

    public static Tensor Get1D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[]>($"{jnptensor}.tolist()")!;
        return Tensor.Create(array);
    }

    public static Tensor Get2D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][]>($"{jnptensor}.tolist()")!;
        return Tensor.Create(array);
    }
    
    public static Tensor Get3D(this PyModule scope, string jnptensor)
    {
        var array = scope.Eval<float[][][]>($"{jnptensor}.tolist()")!;
        return Tensor.Create(array);
    }
}
