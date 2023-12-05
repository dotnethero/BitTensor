namespace BitTensor.Core;

/// <summary>
/// Function that recalculates tensor values based on computation tree
/// </summary>
/// <param name="self">Tensor itself</param>
public delegate void ForwardFunction(Tensor self);

/// <summary>
/// Function that transforms tensor gradient to children gradients
/// </summary>
/// <param name="grad">Parent gradient</param>
/// <param name="self">Tensor itself</param>
public delegate Tensor[] BackwardFunction(Tensor grad, Tensor self);

/// <summary>
/// Activation function
/// </summary>
/// <param name="output"></param>
/// <returns></returns>
public delegate Tensor ActivationFunction(Tensor output);
