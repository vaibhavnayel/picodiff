#include "autograd.h"

// global variable to hold the graph (there can only be one)
Graph graph;

class AddFn: public AutogradFn2Var
{
public:

    void forward(Variable& var1, Variable& var2, Variable& output) override
    {   
        input1 = &var1;
        input2 = &var2;

        var1.num_children += 1;
        var2.num_children += 1;

        output.data = var1.data + var2.data;
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update(dL_doutput);
        input2->gradient_update(dL_doutput);

        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator+(Variable& var2)
{   
    AddFn* add = new AddFn;
    Variable* output = new Variable;
    graph.nodes.push_back(add);
    graph.nodes.push_back(output);
    add->forward(*this, var2, *output);
    output->bkwd_fn = add;
    return *output;
}

class SubtractFn: public AutogradFn2Var
{
public:

    void forward(Variable& var1, Variable& var2, Variable& output) override
    {   
        input1 = &var1;
        input2 = &var2;

        var1.num_children += 1;
        var2.num_children += 1;

        output.data = var1.data - var2.data;
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update(dL_doutput);
        input2->gradient_update(-dL_doutput);
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator-(Variable& var2)
{   
    SubtractFn* subtract = new SubtractFn;
    Variable* output = new Variable;
    graph.nodes.push_back(subtract);
    graph.nodes.push_back(output);
    subtract->forward(*this, var2, *output);
    output->bkwd_fn = subtract;
    return *output;
}

class MatmulFn: public AutogradFn2Var
{
public:

    void forward(Variable& var1, Variable& var2, Variable& output) override
    {   
        input1 = &var1;
        input2 = &var2;

        var1.num_children += 1;
        var2.num_children += 1;

        output.data = var1.data * var2.data;
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update(dL_doutput*input2->data.transpose());
        input2->gradient_update(input1->data.transpose()*dL_doutput);
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator*(Variable& var2)
{   
    MatmulFn* matmul = new MatmulFn;
    Variable* output = new Variable;
    graph.nodes.push_back(matmul);
    graph.nodes.push_back(output);
    matmul->forward(*this, var2, *output);
    output->bkwd_fn = matmul;
    return *output;
}

class DivFn: public AutogradFn2Var
{
public:

    void forward(Variable& var1, Variable& var2, Variable& output) override
    {   
        input1 = &var1;
        input2 = &var2;

        var1.num_children += 1;
        var2.num_children += 1;

        output.data = ((var1.data.array() / var2.data.array()).matrix());
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update((dL_doutput.array()/input2->data.array()).matrix());
        input2->gradient_update((dL_doutput.array()*(-input1->data.array()/input2->data.array().square())).matrix());
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
        if(input2->gradients_accumulated()) input2->backward(input2->grad);
    }
};

Variable Variable::operator/(Variable& var2)
{   
    DivFn* div = new DivFn;
    Variable* output = new Variable;
    graph.nodes.push_back(div);
    graph.nodes.push_back(output);
    div->forward(*this, var2, *output);
    output->bkwd_fn = div;
    return *output;
}

class SigmoidFn: public AutogradFn1Var
{
public:

    void forward(Variable& var1, Variable& output) override
    {   
        input1 = &var1;

        var1.num_children += 1;

        output.data = ((1.0/(1.0 + (-(var1.data)).array().exp())).matrix());
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {
        ArrayXXd s = (1.0/(1.0 + (-(input1->data)).array().exp()));
        input1->gradient_update((dL_doutput.array()*(s*(1-s))).matrix());
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::sigmoid()
{   
    SigmoidFn* sig = new SigmoidFn;
    Variable* output = new Variable;
    graph.nodes.push_back(sig);
    graph.nodes.push_back(output);
    sig->forward(*this, *output);
    output->bkwd_fn = sig;
    return *output;
}

class MeanFn: public AutogradFn1Var
{
public:

    void forward(Variable& var1, Variable& output) override 
    {   
        input1 = &var1;

        var1.num_children += 1;

        output.data = MatrixXd::Constant(1,1,var1.data.array().mean());
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update(MatrixXd::Constant(input1->data.rows(),input1->data.cols(),dL_doutput(0)/(double)input1->data.size()));

        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::mean()
{   
    MeanFn* mean = new MeanFn;
    Variable* output = new Variable;
    graph.nodes.push_back(mean);
    graph.nodes.push_back(output);
    mean->forward(*this, *output);
    output->bkwd_fn = mean;
    return *output;
}

class SquareFn: public AutogradFn1Var
{
public:

    void forward(Variable& var1, Variable& output) override
    {   
        input1 = &var1;

        var1.num_children += 1;

        output.data = var1.data.array().square().matrix();
        output.bkwd_fn = this;
    }

    void backward(MatrixXd dL_doutput) override
    {   
        input1->gradient_update(2*input1->data);
        
        //recurse only if all children have passed gradients
        if(input1->gradients_accumulated()) input1->backward(input1->grad);
    }
};

Variable Variable::square()
{   
    SquareFn* sqr = new SquareFn;
    Variable* output = new Variable;
    graph.nodes.push_back(sqr);
    graph.nodes.push_back(output);
    sqr->forward(*this, *output);
    output->bkwd_fn = sqr;
    return *output;
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
