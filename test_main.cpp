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
    Matrix a(1, 1, 2);
    Matrix b(4, 3, 2);
    b.set_matrix_1_to_x();
    b.print_matrix();
    (b / a).print_matrix();
    (b + a).print_matrix();
    (b - a).print_matrix();
    (b * a).print_matrix();
    return 0;
}
