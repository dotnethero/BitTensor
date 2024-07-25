namespace BitTensor.Core;

/// <summary>
/// Activation function
/// </summary>
/// <param name="output"></param>
/// <returns></returns>
public delegate T Activation<T>(T output) where T : AbstractTensor;
