#include "autograd.h"

void log(const char* str) {
    std::cout << str << "\n";
}

//default implementation for 1 variable forward
Variable AutogradFn::forward(Variable* var1)
{   
    log("this function doesn't work with 1 argument. returning 0");
    Variable output(MatrixXd::Zero(1,1));
    return output;
}

//default implementation for 2 variable forward
Variable AutogradFn::forward(Variable* var1, Variable* var2)
{
    log("this function doesn't work with 2 arguments, returning 0");
    Variable output(MatrixXd::Zero(1,1));
    return output;
}
