#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <assert.h>
#include "Matrix.h"
//#include <vector>

#define SHOW_MATRIX_PTR

using namespace std;

Matrix& f_a(){
    Matrix *result = new Matrix(true);
    return *result;
}

void f_b(){
    Matrix a = f_a();
}


int main(){
    Matrix a(5, 5, 0);
    a.set_matrix_1_to_x();
    a.print_matrix();
    a.rotate_180().print_matrix();
    return 0;
}
