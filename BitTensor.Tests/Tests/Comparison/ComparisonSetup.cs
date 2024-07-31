using System.Runtime.Serialization;
using NUnit.Framework;
using Python.Runtime;

#pragma warning disable SYSLIB0050
#pragma warning disable SYSLIB0011

namespace BitTensor.Tests.Comparison;

[SetUpFixture]
public sealed class ComparisonSetup
{
    static ComparisonSetup()
    {
        Runtime.PythonDLL = @"C:\Program Files\Python 3\python312.dll";
        RuntimeData.FormatterType = typeof(NoopFormatter);
    }

    [OneTimeSetUp]
    public void Setup()
    {
        PythonEngine.Initialize();
    }

    [OneTimeTearDown]
    public void Shutdown()
    {
        PythonEngine.Shutdown();
    }

    private class NoopFormatter : IFormatter 
    {
        public SerializationBinder? Binder { get; set; }
        public StreamingContext Context { get; set; }
        public ISurrogateSelector? SurrogateSelector { get; set; }
    
        public void Serialize(Stream s, object o) {}
        public object Deserialize(Stream s) => throw new NotImplementedException();
    }
}