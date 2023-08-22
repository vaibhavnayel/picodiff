# Picodiff
Picodiff is a minimal implementation of reverese mode automatic differentiation in c++ built on Eigen. It doesn't use a tape but tracks dependencies in a graph, avoiding recomputations by counting children and grad updates on every node.

# Usage


Below is a simple implementation of a feed forward neural net with 1000 training steps.
```c++
#include "autograd.h"

int main()
{
    // declare leaf variables 
    Variable x(MatrixXd::Random(100,20));
    Variable y(MatrixXd::Random(100,1));

    Variable layer_1(MatrixXd::Random(20,64));
    Variable layer_2(MatrixXd::Random(64,64));
    Variable layer_3(MatrixXd::Random(64,1));

    double step_size = 0.0001;

    for(int i=0;i<1000;i++)
    {   
        // variables created by operators are added to graph
        Variable x1_ = (x*layer_1);
        Variable x1 = x1_.sigmoid();
        Variable x2_ = (x1*layer_2);
        Variable x2 = x2_.sigmoid();
        Variable y_hat = (x2*layer_3);
        Variable diff = (y-y_hat);
        Variable loss = diff.square();
        Variable loss_r = loss.mean();
        loss_r.backward();

        std::cout<<loss_r.data<<std::endl;

        // take a gradient step
        layer_1.gradient_step(step_size); layer_2.gradient_step(step_size); layer_3.gradient_step(step_size);
        
        // set grads to zero
        layer_1.reset(); layer_2.reset(); layer_3.reset(); x.reset(); y.reset();
        
        // clear the autograd graph and delete intermediate variables
        graph.clear();
    }


    return 0;
} 
```

## TODO
- [ ] implement reshape ops
- [ ] support more reduction ops - sum, min, max
- [ ] enable double backward
- [ ] create nn package

