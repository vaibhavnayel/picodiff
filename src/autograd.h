#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXXd;

class AutogradFn;

class Variable
{
public:
    MatrixXd data;
    MatrixXd grad;
    AutogradFn* bkwd_fn = nullptr;
    int num_children = 0;
    int grad_updates = 0;

    Variable(MatrixXd mat): data{mat} 
    {
        grad = MatrixXd::Zero(data.rows(),data.cols());
    }

    bool gradients_accumulated()
    {
        return num_children==grad_updates;
    }

    void print()
    {
        std::cout 
        << "children: " << num_children  
        << " grad updates: " << grad_updates 
        // << " data: " << data 
        << " bkwd fn: " << bkwd_fn 
        // << " grad: " << grad 
        << " this: " << this << std::endl;
    }

    void gradient_step(double step_size)
    {
        data -= step_size*grad;
    }

    void reset()
    {
        grad = MatrixXd::Zero(data.rows(),data.cols());
        bkwd_fn = nullptr;
        num_children = 0;
        grad_updates = 0;
    }

    //forward declarations
    Variable operator + (Variable& var2);
    Variable operator - (Variable& var2);
    Variable operator * (Variable& var2);
    Variable operator / (Variable& var2);
    Variable sigmoid();
    Variable mean();
    Variable square();

    void backward(MatrixXd grad);
    void backward();
};


class AutogradFn
{
public:
    virtual Variable forward(Variable* var1, Variable* var2) = 0;
    virtual Variable forward(Variable* var1) = 0;
    virtual void backward(MatrixXd dL_doutput) = 0;
    virtual void print() = 0;
};