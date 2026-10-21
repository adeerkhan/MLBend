# mlbend-torch

The same linear model, over [bendtorch](https://github.com/aaaxn/bendtorch)'s
tape.

bend-ml's autodiff is functional: the backward pass is a function of the
forward. bendtorch's is a tape: the forward records the nodes, the
backward walks them, and the step reads the gradients off the leaves.

The two hold different tensors -- bend-ml's are lists, bendtorch's are its
own storage with a shape in the type -- so the model is built twice, from
the same numbers, and the two losses are compared.

The nodes of the tape: the weights are leaf 0, the bias leaf 1, the data
leaf 2, the logits node 3 and the loss node 4.

## Where it stands

The package compiles and runs: the data, the weights and the bias are
built, the leaves are on the tape, and the loss is read back from it. The
`linear` node still refuses the operands -- error 106, an inner dimension
mismatch -- so the shape convention of the weights (PyTorch's `[out, in]`
or the mathematical `[in, out]`) and the rank of the bias are not settled
yet. `bendtorch`'s own MNIST demo is the reference for both.
