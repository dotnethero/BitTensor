using System.Runtime.Serialization;
using NUnit.Framework;
using Python.Included;
using Python.Runtime;

#pragma warning disable SYSLIB0050
#pragma warning disable SYSLIB0011

namespace BitTensor.Tests.Comparison;

[SetUpFixture]
public sealed class ComparisonSetup
{
    static ComparisonSetup() => RuntimeData.FormatterType = typeof(NoopFormatter);

    [OneTimeSetUp]
    public async Task Setup()
    {
        await Installer.SetupPython();

        if (!Installer.IsModuleInstalled("jax[cpu]"))
        {
            await Installer.TryInstallPip();
            await Installer.PipInstallModule("jax[cpu]", "0.4.20");
        }
        
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