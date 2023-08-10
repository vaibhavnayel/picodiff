# Picodiff
Picodiff is a minimal implementation of reverese mode automatic differentiation in c++ built on Eigen. It doesn't use a tape but tracks dependencies in a graph, avoiding recomputations by counting children and grad updates on every node.

# Usage


Implementing a simple neural net and performing one training step
```c++
#include "autograd.h"

int main()
{
    Variable x(MatrixXd::Random(100,20));
    Variable y(MatrixXd::Random(100,1));

    Variable layer_1(MatrixXd::Random(20,64));
    Variable layer_2(MatrixXd::Random(64,64));
    Variable layer_3(MatrixXd::Random(64,1));

    Variable x1 = (x*layer_1).sigmoid();
    Variable x2 = (x1*layer_2).sigmoid();
    Variable y_hat = (x2*layer_3);
    Variable loss = (y-y_hat).square();
    Variable loss_r = loss.mean();
    loss.backward();
    x.print();y.print();x1.print();x2.print();y_hat.print(); loss.print();

    layer_1.grad_step(0.0001);
    layer_2.grad_step(0.0001);
    layer_3.grad_step(0.00001);


    return 0;
} 
```

## TODO
- [ ] implement reshape ops
- [ ] support more reduction ops - sum, min, max
- [ ] enable double backward
- [ ] create nn package

