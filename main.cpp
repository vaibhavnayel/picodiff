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
    x.print();y.print();x1.print();x2.print();y_hat.print(); //loss.print();

    return 0;
} 