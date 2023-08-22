#include "autograd.h"

void log(const char* str) {std::cout << str << "\n";}
//default implementation for 1 variable forward
void default_1var_forward(){log("this function doesn't work with 1 argument");}
//default implementation for 2 variable forward
void default_2var_forward(){log("this function doesn't work with 2 arguments");}
