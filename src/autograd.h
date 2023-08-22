#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Array;
using Eigen::ArrayXXd;


class Node 
{
    // variables and functions are nodes in the graph
};


class AutogradFn;
class Variable: public Node
{
public:
    MatrixXd data;
    MatrixXd grad;
    AutogradFn* bkwd_fn = nullptr;
    int num_children = 0;
    int grad_updates = 0;

    Variable(){}

    Variable(MatrixXd mat): data{mat} 
    {
        grad = MatrixXd::Zero(data.rows(),data.cols());
    }

    void zero_grad()
    {
        grad = MatrixXd::Zero(data.rows(),data.cols());
    }

    void print()
    {
        std::cout 
        << "children: " << num_children  
        << " grad updates: " << grad_updates 
        << " data: " << data 
        << " bkwd fn: " << bkwd_fn 
        << " grad: " << grad 
        << " this: " << this << std::endl;
    }
    
    bool gradients_accumulated()
    {
        return num_children==grad_updates;
    }

    void gradient_update(MatrixXd grad_update)
    {   
        grad = grad.size()? grad+grad_update : grad_update;
        grad_updates += 1;
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


class AutogradFn: public Node
{
public:
    virtual void forward(Variable& var1, Variable& var2, Variable& output) = 0;
    virtual void forward(Variable& var1, Variable& output) = 0;
    virtual void backward(MatrixXd dL_doutput) = 0;
    virtual void print() = 0;
};

void log(const char* str);
void default_1var_forward();
void default_2var_forward();

class AutogradFn2Var: public AutogradFn
{
public:
    Variable* input1;
    Variable* input2;

    virtual void forward(Variable& var1, Variable& var2, Variable& output) = 0;
    virtual void backward(MatrixXd dL_doutput) = 0;
    void forward(Variable& var1, Variable& output) override {default_1var_forward();}
    void print() override
    {
        std::cout
            <<"input1: " << input1
            <<" input2: " << input2;
    }

};


class AutogradFn1Var: public AutogradFn
{
public:
    Variable* input1;

    virtual void forward(Variable& var1, Variable& output) = 0;
    virtual void backward(MatrixXd dL_doutput) = 0;
    void forward(Variable& var1, Variable& var2, Variable& output) override {default_2var_forward();}
    void print() override
    {
        std::cout
            <<"input1: " << input1;
    }

};


class Graph
{
public:
    std::vector<Node*> nodes;

    void clear()
    {
        for (Node* ptr : nodes)
        {
            delete ptr;
        }
        nodes.clear();
    }
};

extern Graph graph;