# tinytensor

A minimal automatic differentiation library built from scratch in NumPy, written to learn the fundamentals of how machine learning frameworks like PyTorch work under the hood.

This is a learning project, not a production library. The goal was to implement reverse-mode autodiff myself, including the calculus behind it, rather than to build something fast or feature-complete.

## What it does

`tinytensor.py` implements:

- **`Tensor`**: a NumPy array wrapper that records the operations applied to it (`+`, `-`, `*`, `@`) as a computation graph, and supports `.backward()` to compute gradients via reverse-mode autodiff (topological sort + chain rule, same idea as micrograd/PyTorch's autograd).
- **`einsum`**: a differentiable wrapper around `np.einsum`. Most Einstein-summation contractions (dot products, matrix multiplication, batched ops, etc.) can be expressed as a schema string (e.g. `"ij,jk->ik"`), and its gradient is derived automatically.
- **Automatic partial derivatives for arbitrary index schemes**: `partial_jacobian` / `partial_gradient` compute the Jacobian (or vector-Jacobian product) of any einsum contraction with respect to any one of its input tensors, purely from the index schema. This is the core piece I wanted to understand: instead of hand-deriving a backward rule for every operation, one general formula handles addition, multiplication, matmul, and general tensor contractions. The gradient itself is computed by translating the schema into another `np.einsum` call, so no tensor entries are ever computed by hand in a Python loop; the actual number-crunching stays inside NumPy's optimized C implementation.
- **Activations**: `relu`, `tanh`, with their backward passes.

`NeuralNetwork.ipynb` builds a small `Layer`/`MLP` class on top of `tinytensor` and trains a couple of tiny feedforward networks (manual gradient descent, mean-squared-error loss) to demonstrate that the autodiff engine actually works end-to-end.

## Example

```python
import numpy as np
import tinytensor as tt

a = tt.Tensor([1.0, 2.0, 3.0], calculateGrad=True)
b = tt.Tensor([4.0, 5.0, 6.0], calculateGrad=True)

out = tt.einsum([a, b], "i,i->")  # dot product
out.grad = np.array(1.0)
out.backward()

print(a.grad)  # d(out)/da = b.data
print(b.grad)  # d(out)/db = a.data
```

See `NeuralNetwork.ipynb` for a full example: a small MLP built from `tinytensor` tensors, trained on toy data with backprop and gradient descent.

## Project structure

```
tinytensor.py         # core Tensor class, einsum autodiff, activations
NeuralNetwork.ipynb    # Layer/MLP built on tinytensor, training examples
```
