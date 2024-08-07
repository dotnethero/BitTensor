using System.Diagnostics;
using BitTensor.Core;
using NUnit.Framework;

namespace BitTensor.Tests.Profiling;

[TestFixture]
[Explicit]
class ProfileTests
{
    [Test]
    public static void Profile_matmul()
    {
        var a = Tensor.Random.Normal([100, 1000, 1000]);
        var b = Tensor.Random.Normal([5, 1, 1000, 1000]);
        var sw = Stopwatch.StartNew();
        var c = Tensor.Matmul(a, b);
        c.EnsureHasUpdatedValues();
        Console.WriteLine(sw.Elapsed);
    }

    [Test]
    public static void Profile_mul()
    {
        var a = Tensor.Random.Normal([100, 1000, 1000]);
        var b = Tensor.Random.Normal([5, 1, 1000, 1000]);
        var sw = Stopwatch.StartNew();
        var c = a * b;
        c.EnsureHasUpdatedValues();
        Console.WriteLine(sw.Elapsed);
    }
}