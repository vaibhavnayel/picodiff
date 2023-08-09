#include "autograd.h"

class AddFn: public AutogradFn
{
public:
    Variable* input1;
    Variable* input2;

    Variable forward(Variable* var1) override {return AutogradFn::forward(var1);}

    void print()
    {
        std::cout
            <<"input1: " << input1
            <<" input2: " << input2;

    }

    Variable forward(Variable* var1, Variable* var2) override
    {   
        input1 = var1;
        input2 = var2;

        var1->num_children += 1;
        var2->num_children += 1;

        Variable output(var1->data + var2->data);
        output.bkwd_fn = this;
        return output;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += dL_doutput;
        input2->grad += dL_doutput;

        input1->grad_updates += 1;
        input2->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator+(Variable& var2)
{   
    AddFn* add = new AddFn;
    Variable output = add->forward(this, &var2);
    output.bkwd_fn = add;
    return output;
}

class SubtractFn: public AutogradFn
{
public:
    Variable* input1;
    Variable* input2;

    Variable forward(Variable* var1) override {return AutogradFn::forward(var1);}

    void print()
    {
        std::cout
            <<"input1: " << input1
            <<" input2: " << input2;

    }

    Variable forward(Variable* var1, Variable* var2) override
    {   
        input1 = var1;
        input2 = var2;

        var1->num_children += 1;
        var2->num_children += 1;

        Variable output(var1->data - var2->data);
        output.bkwd_fn = this;
        return output;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += dL_doutput;
        input2->grad -= dL_doutput;

        input1->grad_updates += 1;
        input2->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator-(Variable& var2)
{   
    SubtractFn* subtract = new SubtractFn;
    Variable output = subtract->forward(this, &var2);
    output.bkwd_fn = subtract;
    return output;
}

class MatmulFn: public AutogradFn
{
public:
    Variable* input1;
    Variable* input2;

    Variable forward(Variable* var1) override {return AutogradFn::forward(var1);}

    void print()
    {
        std::cout
            <<"input1: " << input1
            <<" input2: " << input2;

    }

    Variable forward(Variable* var1, Variable* var2) override
    {   
        input1 = var1;
        input2 = var2;

        var1->num_children += 1;
        var2->num_children += 1;

        Variable output(var1->data * var2->data);
        output.bkwd_fn = this;
        return output;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += dL_doutput*input2->data.transpose();
        input2->grad += input1->data.transpose()*dL_doutput;

        input1->grad_updates += 1;
        input2->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator*(Variable& var2)
{   
    MatmulFn* matmul = new MatmulFn;
    Variable output = matmul->forward(this, &var2);
    output.bkwd_fn = matmul;
    return output;
}

class DivFn: public AutogradFn
{
public:
    Variable* input1;
    Variable* input2;

    Variable forward(Variable* var1) override {return AutogradFn::forward(var1);}

    void print()
    {
        std::cout
            <<"input1: " << input1
            <<" input2: " << input2;

    }

    Variable forward(Variable* var1, Variable* var2) override
    {   
        input1 = var1;
        input2 = var2;

        var1->num_children += 1;
        var2->num_children += 1;

        Variable output((var1->data.array() / var2->data.array()).matrix());
        output.bkwd_fn = this;
        return output;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += (dL_doutput.array()/input2->data.array()).matrix();
        input2->grad += (dL_doutput.array()*(-input1->data.array()/input2->data.array().square())).matrix();

        input1->grad_updates += 1;
        input2->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator/(Variable& var2)
{   
    DivFn* div = new DivFn;
    Variable output = div->forward(this, &var2);
    output.bkwd_fn = div;
    return output;
}

class SigmoidFn: public AutogradFn
{
public:
    Variable* input1;

    void print()
    {
        std::cout
            <<"input1: " << input1;

    }

    Variable forward(Variable* var1) override 
    {   
        input1 = var1;

        var1->num_children += 1;

        Variable output((1.0/(1.0 + (-(var1->data)).array().exp())).matrix());
        output.bkwd_fn = this;
        return output;
    }

    Variable forward(Variable* var1, Variable* var2) override {return AutogradFn::forward(var1, var2);}

    void backward(MatrixXd dL_doutput) override
    {
        ArrayXXd s = (1.0/(1.0 + (-(input1->data)).array().exp()));
        input1->grad += (dL_doutput.array()*(s*(1-s))).matrix();

        input1->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::sigmoid()
{   
    SigmoidFn* sig = new SigmoidFn;
    Variable output = sig->forward(this);
    output.bkwd_fn = sig;
    return output;
}

class MeanFn: public AutogradFn
{
public:
    Variable* input1;

    void print()
    {
        std::cout
            <<"input1: " << input1;

    }

    Variable forward(Variable* var1) override 
    {   
        input1 = var1;

        var1->num_children += 1;

        Variable output(MatrixXd::Constant(1,1,var1->data.array().mean()));
        output.bkwd_fn = this;
        return output;
    }

    Variable forward(Variable* var1, Variable* var2) override {return AutogradFn::forward(var1, var2);}

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += MatrixXd::Constant(input1->data.rows(),input1->data.cols(),dL_doutput(0)/(double)input1->data.size());
        
        input1->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::mean()
{   
    MeanFn* mean = new MeanFn;
    Variable output = mean->forward(this);
    output.bkwd_fn = mean;
    return output;
}

class SquareFn: public AutogradFn
{
public:
    Variable* input1;

    void print()
    {
        std::cout
            <<"input1: " << input1;
    }

    Variable forward(Variable* var1) override 
    {   
        input1 = var1;

        var1->num_children += 1;

        Variable output(var1->data.array().square().matrix());
        output.bkwd_fn = this;
        return output;
    }

    Variable forward(Variable* var1, Variable* var2) override {return AutogradFn::forward(var1, var2);}

    void backward(MatrixXd dL_doutput) override
    {   
        input1->grad += 2*input1->data;
        
        input1->grad_updates += 1;
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::square()
{   
    SquareFn* sqr = new SquareFn;
    Variable output = sqr->forward(this);
    output.bkwd_fn = sqr;
    return output;
}



void Variable::backward(MatrixXd grad)
{   
    if (bkwd_fn){
        bkwd_fn->backward(grad);
    }
}

void Variable::backward()
{
    if (bkwd_fn){
        bkwd_fn->backward(MatrixXd::Constant(data.rows(),data.cols(),1.0));
    }
}
