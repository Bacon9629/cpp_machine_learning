#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
//#include <vector>

using namespace std;

class Matrix{
private:
    Matrix* calculate_check_need_copy(){
        Matrix *result = this;
        if (!is_cal_result){
            result = copy();
            result->is_cal_result = true;
        }
        return result;
    }

public:
    double *matrix;
    size_t shape[4] = {0, 0, 0, 0};
    size_t index_reflec_1d_[4] = {0, 0, 0, 0};
    size_t size_1d = 1;
    bool is_cal_result = false;  // 此matrix是計算中得出的結果

    static Matrix& get_matrix(Matrix &_matrix, size_t a, size_t b){
        Matrix *result = new Matrix(
                _matrix.matrix + _matrix.index_reflec_1d_[0] * a + _matrix.index_reflec_1d_[0] * b,
                _matrix.shape[2], _matrix.shape[3]
                );
        result->is_cal_result = true;
        return *result;
    }

    inline static double& get(Matrix &_matrix, size_t a, size_t b, size_t c, size_t d){
        size_t *_a = _matrix.index_reflec_1d_;
        return _matrix.matrix[a * _a[0] + b * _a[1] + c * _a[2] + d * _a[3]];
    }

    inline static double& get(Matrix &_matrix, size_t row, size_t col){
        return get(_matrix, 0, 0, row, col);
    }

    // 取 start 到 end - 1 的row
    inline static Matrix getRow(Matrix &_matrix, size_t start_row, size_t end_row){
        if (
                start_row > _matrix.shape[2] || end_row > _matrix.shape[2] ||
                _matrix.shape[0] != 0 || end_row < start_row
                )
        {
            cout << "shape_wrong: Matrix getRow" << endl;
        }
        size_t row_size = end_row - start_row;
        Matrix *result = new Matrix(
                &(Matrix::get(_matrix, start_row, 0)),
                row_size,
                _matrix.shape[3]);

        return *result;
    }

    static Matrix& dot(Matrix &matrix_a, Matrix &matrix_b){
        size_t row_a = matrix_a.shape[2];
        size_t col_a = matrix_a.shape[3];
        size_t row_b = matrix_b.shape[2];
        size_t col_b = matrix_b.shape[3];

        if (col_a != row_b){
            std::cout << "shape wrong" << std::endl;
            assert("matrix error - dot");
        }

        const size_t row_result = row_a;
        const size_t col_result = col_b;
        Matrix *result = new Matrix(row_result, col_result, 0);

        for (int r = 0; r < row_result; r++) {
            for (int c = 0; c < col_result; c++) {
                // 指定在result內的哪個位置
                // 接下來依照指定的result位置取出a、b的值來做計算

                for (int i = 0; i < col_a; i++) {
                    result->get(r, c) += matrix_a.get(r, i) * matrix_b.get(i, c);
                }
            }
        }
        return *result;
    }

    inline static Matrix& transpose(Matrix &_matrix) {
        Matrix *result = new Matrix(_matrix.shape[3], _matrix.shape[2], 0);
        result->is_cal_result = true;

        for (size_t i = 0; i < _matrix.shape[2]; i++){
            for (size_t j = 0; j < _matrix.shape[3]; j++){
                result->get(j, i) = _matrix.get(i, j);
            }
        }

        return *result;
    }

//    inline static Matrix expand_row(Matrix *matrix_a, Matrix *matrix_b){
//        Matrix _temp_b;
////        if(matrix_a->row() != matrix_b->row()){
//        _temp_b = Matrix(matrix_a->row(), matrix_b->col(), 0);
//        for (int i=0;i<matrix_a->row();i++){
//            _temp_b.matrix[i] = matrix_b->matrix[0];
//        }
////        }
//        return _temp_b;
//    }


    Matrix() {
        init((size_t)0, (size_t)0, 0, 0, 0);
    }

    Matrix(size_t row, size_t col) {
        init((size_t)1, (size_t)1, row, col, 0);
    }

    Matrix(size_t row, size_t col, double init_val){
        init((size_t)1, (size_t)1, row, col, init_val);
    }

    Matrix(double* _matrix_point, size_t row, size_t col){
        init(_matrix_point, 1, 1, row, col);
    }

    Matrix(size_t a, size_t b, size_t c, size_t d, double init_val){
        init(a, b, c, d, init_val);
    }

    Matrix(double* _matrix_point, size_t a, size_t b, size_t c, size_t d){
        init(_matrix_point, a, b, c, d);
    }

    void init(double* _matrix_point, size_t a, size_t b, size_t c, size_t d){
//        size_t temp[4] = {a, b, c, d};
        size_1d = a * b * c * d;
//        for (size_t i=0;i<4;i++){
//            size_1d *= temp[i] == 0 ? 1 : temp[i];
//        }
        shape[3] = d;
        shape[2] = c;
        shape[1] = b;
        shape[0] = a;
        index_reflec_1d_[3] = 1;
        index_reflec_1d_[2] = d;
        index_reflec_1d_[1] = d * c;
        index_reflec_1d_[0] = b * d * c;
        matrix = new double [size_1d];
//        matrix = (double*) calloc(size_1d, sizeof(double));
        memcpy(matrix, _matrix_point, sizeof(double) * size_1d);

//        cout << "_matrix_point construct " << this << endl;
    }

    void init(size_t a, size_t b, size_t c, size_t d, double init_val){
//        size_t temp[4] = {a, b, c, d};
        size_1d = a * b * c * d;
//        for (size_t i=0;i<4;i++){
//            size_1d *= temp[i] == 0 ? 1 : temp[i];
//        }
        shape[3] = d;
        shape[2] = c;
        shape[1] = b;
        shape[0] = a;
        index_reflec_1d_[3] = 1;
        index_reflec_1d_[2] = d;
        index_reflec_1d_[1] = d * c;
        index_reflec_1d_[0] = b * d * c;
        matrix = new double [size_1d]();
//        matrix = (double*) calloc(size_1d, sizeof(double));
        if (init_val != 0){
            for (size_t i = 0; i < size_1d; i++){
                matrix[i] = init_val;
            }
        }

//        cout << "init_val construct " << this << endl;
    }

    ~Matrix(){
//        cout << "free " << this << endl;
        if (matrix != NULL){
//            free(matrix);
            delete []matrix;
        }
    }

    inline double& get(size_t a, size_t b, size_t c, size_t d){
        return Matrix::get(*this, a, b, c, d);
    }

    inline double& get(size_t row, size_t col){
        return Matrix::get(*this, row, col);
    }

    inline Matrix dot(Matrix &matrix_b) {
        return Matrix::dot(*this, matrix_b);
    }

    inline void random_matrix(){
        srand(time(NULL));
        double temp = (RAND_MAX + 1.0);
        for (int i=0; i < size_1d; i++){
                matrix[i] = rand() / temp - 0.5;
        }
    }

    void set_matrix_1_to_x(){
        for (size_t i = 0; i < size_1d; i++){
            matrix[i] = double (i);
        }
    }

    inline void print_matrix(){
        cout << "{" << endl;
        for (int k=0;k<shape[0];k++){
            cout << "{" << endl;
            for (int l=0;l<shape[1];l++){
                for (int i=0;i<shape[2];i++){
                    for (int j=0;j<shape[3];j++){
                        cout << get(k, l, i, j) << ", ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
            cout << endl;
            cout << "}" << endl;
        }
        cout << "}" << endl;
    }

    inline void print_shape(){
        cout << "shape: " << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << " " << endl;
    }

    inline Matrix* copy(){
        Matrix *result = new Matrix(matrix, shape[0], shape[1], shape[2], shape[3]);
        cout << "copy " << result << endl;
        return result;
    }

    Matrix& operator+ (double a){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] += a;
        }
        return *result;
    }

    Matrix& operator+ (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] += _matrix.matrix[i];
        }
        return *result;
    }

    Matrix& operator- (double a){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] -= a;
        }
        return *result;
    }

    Matrix& operator- (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] -= _matrix.matrix[i];
        }
        return *result;
    }

    Matrix& operator* (double a){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] *= a;
        }
        return *result;
    }

    Matrix& operator* (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] *= _matrix.matrix[i];
        }
        return *result;
    }

    Matrix& operator/ (double a){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] /= a;
        }
        return *result;
    }

    Matrix& operator/ (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        for (size_t i = 0; i < size_1d; i++){
            result->matrix[i] /= _matrix.matrix[i];
        }
        return *result;
    }

    void operator= (Matrix &_matrix){
        for ( size_t i = 0; i < 4; i++){
            shape[i] = _matrix.shape[i];
            index_reflec_1d_[i] = _matrix.index_reflec_1d_[i];
            size_1d = _matrix.size_1d;
        }
        if (matrix != NULL){
            delete []matrix;
        }

        if (_matrix.is_cal_result){
            matrix = _matrix.matrix;
            _matrix.matrix = NULL;
            delete &_matrix;
        }else{
            matrix = new double [size_1d];
            memcpy(matrix, _matrix.matrix, sizeof(double) * size_1d);
        }
    }
};


int main(){
    Matrix a = Matrix(5, 5, 4);
    a.set_matrix_1_to_x();

    a.get(3, 3) = 333;

    Matrix b;
    b = Matrix::transpose(a);

    cout << "a: " << a.size_1d << endl;

    a.print_shape();
    a.print_matrix();
    cout << "b: " << endl;
    b.print_shape();
    b.print_matrix();

    int aaaaa = 0;
    cin >> aaaaa;


    return 0;
}