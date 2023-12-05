using NUnit.Framework;

namespace BitTensor.Tests;

[TestFixture]
class PerformanceTests
{
    [Test]
    [Explicit]
    public void Test()
    {
        Program.Module_performace();
    }
}