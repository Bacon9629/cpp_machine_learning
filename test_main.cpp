#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <assert.h>
//#include <vector>

#define SHOW_MATRIX_PTR

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
    size_t size_1d = -1;
    bool is_cal_result = false;  // 此matrix是計算中得出的結果

    static Matrix& get_matrix(Matrix &_matrix, size_t a, size_t b){
        Matrix *result = new Matrix(
                _matrix.matrix + _matrix.index_reflec_1d_[0] * a + _matrix.index_reflec_1d_[0] * b,
                _matrix.shape[2], _matrix.shape[3], true
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

    inline static Matrix& getPicture_row(Matrix &_matrix, size_t start_picture, size_t end_picture){
        assert(!(start_picture > _matrix.shape[0] || end_picture > _matrix.shape[0] ||
                 _matrix.shape[0] != 0 || end_picture < start_picture));

        size_t row_size = end_picture - start_picture;
        Matrix *result = new Matrix(
                &(Matrix::get(_matrix, start_picture, 0)),
                row_size,
                _matrix.shape[2],
                _matrix.shape[1],
                _matrix.shape[0], true);
        return *result;
    }

    // 取 start 到 end - 1 的row
    inline static Matrix& getPictures(Matrix &_matrix, size_t start_picture, size_t end_picture){
//        if (
//                start_row > _matrix.shape[2] || end_row > _matrix.shape[2] ||
//                _matrix.shape[0] != 0 || end_row < start_row
//                )
//        {
//            cout << "shape_wrong: Matrix getRow" << endl;
//        }

        assert(start_picture < _matrix.shape[0]);
        assert(end_picture <= _matrix.shape[0]);
        assert(end_picture > start_picture);

        size_t pictures_size = end_picture - start_picture;
        Matrix *result = new Matrix(
                &(Matrix::get(_matrix, start_picture, 0, 0, 0)),
                pictures_size,
                _matrix.shape[1],
                _matrix.shape[2],
                _matrix.shape[3], true);

        return *result;
    }

    // 取 start 到 end - 1 的row
    inline static Matrix& getRow(Matrix &_matrix, size_t start_row, size_t end_row){
//        if (
//                start_row > _matrix.shape[2] || end_row > _matrix.shape[2] ||
//                _matrix.shape[0] != 0 || end_row < start_row
//                )
//        {
//            cout << "shape_wrong: Matrix getRow" << endl;
//        }
        assert(start_row < _matrix.shape[2]);
        assert(end_row <= _matrix.shape[2]);
        assert(end_row > start_row);

        size_t row_size = end_row - start_row;
        Matrix *result = new Matrix(
                &(Matrix::get(_matrix, start_row, 0)),
                row_size,
                _matrix.shape[3], true);

        return *result;
    }

    static Matrix& dot(Matrix &matrix_a, Matrix &matrix_b){
        size_t row_a = matrix_a.shape[2];
        size_t col_a = matrix_a.shape[3];
        size_t row_b = matrix_b.shape[2];
        size_t col_b = matrix_b.shape[3];

        assert(col_a == row_b);

        const size_t row_result = row_a;
        const size_t col_result = col_b;
        Matrix *result = new Matrix(row_result, col_result, 0, true);

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
        Matrix *result = new Matrix(_matrix.shape[3], _matrix.shape[2], 0, true);
        result->is_cal_result = true;

        for (size_t i = 0; i < _matrix.shape[2]; i++){
            for (size_t j = 0; j < _matrix.shape[3]; j++){
                result->get(j, i) = _matrix.get(i, j);
            }
        }

        return *result;
    }

    Matrix(Matrix &a, bool is_calculate = false) {
        size_t *_shape = a.shape;
        init(a.matrix, _shape[0], _shape[1], _shape[2], _shape[3], is_calculate);
        if (a.is_cal_result){
            delete &a;
        }
    }

    Matrix(bool is_calculate = false) {
        init((size_t)0, (size_t)0, 0, 0, 0, is_calculate);
    }

    Matrix(size_t row, size_t col, double init_val, bool is_calculate = false){
        init((size_t)1, (size_t)1, row, col, init_val, is_calculate);
    }

    Matrix(double* _matrix_point, size_t row, size_t col, bool is_calculate = false){
        init(_matrix_point, 1, 1, row, col, is_calculate);
    }

    Matrix(size_t a, size_t b, size_t c, size_t d, double init_val, bool is_calculate = false){
        init(a, b, c, d, init_val, is_calculate);
    }

    Matrix(double* _matrix_point, size_t a, size_t b, size_t c, size_t d, bool is_calculate = false){
        init(_matrix_point, a, b, c, d, is_calculate);
    }

    void init(double* _matrix_point, size_t a, size_t b, size_t c, size_t d, bool is_calculate){
        is_cal_result = is_calculate;
        reshape(a, b, c, d);
        matrix = new double [size_1d];
//        matrix = (double*) calloc(size_1d, sizeof(double));
        memcpy(matrix, _matrix_point, sizeof(double) * size_1d);

#ifdef SHOW_MATRIX_PTR
        cout << "_matrix_point construct " << this << endl;
#endif
    }

    void init(size_t a, size_t b, size_t c, size_t d, double init_val, bool is_calculate){
        is_cal_result = is_calculate;
        reshape(a, b, c, d);
        matrix = new double [size_1d]();
        if (init_val != 0){
            for (size_t i = 0; i < size_1d; i++){
                matrix[i] = init_val;
            }
        }

#ifdef SHOW_MATRIX_PTR
        cout << "init_val construct " << this << endl;
#endif
    }

    ~Matrix(){
#ifdef SHOW_MATRIX_PTR
        cout << "free " << this << endl;
#endif
        if (matrix != NULL){
            delete []matrix;
        }
    }

    inline void reshape(size_t a, size_t b, size_t c, size_t d){
        size_t temp = a * b * c * d;
        if (size_1d != -1){
            assert(size_1d == temp);
        }else{
            size_1d = a * b * c * d;
        }
        shape[3] = d;
        shape[2] = c;
        shape[1] = b;
        shape[0] = a;
        index_reflec_1d_[3] = 1;
        index_reflec_1d_[2] = d;
        index_reflec_1d_[1] = d * c;
        index_reflec_1d_[0] = b * d * c;
    }

    inline double& get(size_t a, size_t b, size_t c, size_t d){
        return Matrix::get(*this, a, b, c, d);
    }

    inline double& get(size_t row, size_t col){
        return Matrix::get(*this, row, col);
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

    void transpose(){
        *this = Matrix::transpose(*this);
    }

    Matrix& expand_row(size_t *target_shape){
        return expand_row(target_shape[2], target_shape[3]);
    }

    Matrix& expand_row(size_t row, size_t col){
        Matrix *result = new Matrix(row, col, 0, true);
        for (size_t i=0;i<row;i++)
            for (size_t j=0; j < col; j++)
                result->get(i, j) = get(0, i);
        return *result;
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
        Matrix *result = new Matrix(matrix, shape[0], shape[1], shape[2], shape[3], true);
//        cout << "copy " << result << endl;
        return result;
    }

    Matrix& exp_(){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] = exp(result->matrix[i]);
        }
        return *result;
    }

    Matrix& log_(){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] = log(result->matrix[i]);
        }
        return *result;
    }

    Matrix& log10_(){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] = log10(result->matrix[i]);
        }
        return *result;
    }

    Matrix& operator+ (double a){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] += a;
        }
        return *result;
    }

    Matrix& operator+ (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;

        if (_matrix.shape[0] == 1 && shape[0] == 1 && _matrix.shape[1] == 1 && shape[1] == 1 &&
                _matrix.shape[3] == shape[3]){

            cout << size_1d << endl;
            for (size_t i = 0; i < size_1d; i += shape[3]){
                for (size_t j = 0; j < shape[3]; j++) {
                    temp[i + j] += _matrix.matrix[j];
                }
            }

        }else{
            assert(_matrix.shape[0] == this->shape[0]);
            assert(_matrix.shape[1] == this->shape[1]);
            assert(_matrix.shape[2] == this->shape[2]);
            assert(_matrix.shape[3] == this->shape[3]);

            for (size_t i = 0; i < size_1d; i++){
                temp[i] += _matrix.matrix[i];
            }
        }


        return *result;
    }

    Matrix& operator- (double a){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] -= a;
        }
        return *result;
    }

    Matrix& operator- (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;

        if (_matrix.shape[0] == 1 && shape[0] == 1 && _matrix.shape[1] == 1 && shape[1] == 1 &&
            _matrix.shape[3] == shape[3]){

            cout << size_1d << endl;
            for (size_t i = 0; i < size_1d; i += shape[3]){
                for (size_t j = 0; j < shape[3]; j++) {
                    temp[i + j] -= _matrix.matrix[j];
                }
            }

        }else{
            assert(_matrix.shape[0] == this->shape[0]);
            assert(_matrix.shape[1] == this->shape[1]);
            assert(_matrix.shape[2] == this->shape[2]);
            assert(_matrix.shape[3] == this->shape[3]);

            for (size_t i = 0; i < size_1d; i++){
                temp[i] -= _matrix.matrix[i];
            }
        }
        return *result;
    }

    Matrix& operator* (double a){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] *= a;
        }
        return *result;
    }

    Matrix& operator* (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        if (_matrix.shape[0] == 1 && shape[0] == 1 && _matrix.shape[1] == 1 && shape[1] == 1 &&
            _matrix.shape[3] == shape[3]){

            cout << size_1d << endl;
            for (size_t i = 0; i < size_1d; i += shape[3]){
                for (size_t j = 0; j < shape[3]; j++) {
                    temp[i + j] *= _matrix.matrix[j];
                }
            }

        }else{
            assert(_matrix.shape[0] == this->shape[0]);
            assert(_matrix.shape[1] == this->shape[1]);
            assert(_matrix.shape[2] == this->shape[2]);
            assert(_matrix.shape[3] == this->shape[3]);

            for (size_t i = 0; i < size_1d; i++){
                temp[i] *= _matrix.matrix[i];
            }
        }
        return *result;
    }

    Matrix& operator/ (double a){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        for (size_t i = 0; i < size_1d; i++){
            temp[i] /= a;
        }
        return *result;
    }

    Matrix& operator/ (Matrix &_matrix){
        Matrix *result = calculate_check_need_copy();
        double* temp = result->matrix;
        if (_matrix.shape[0] == 1 && shape[0] == 1 && _matrix.shape[1] == 1 && shape[1] == 1 &&
            _matrix.shape[3] == shape[3]){

            cout << size_1d << endl;
            for (size_t i = 0; i < size_1d; i += shape[3]){
                for (size_t j = 0; j < shape[3]; j++) {
                    temp[i + j] /= _matrix.matrix[j];
                }
            }

        }else{
            assert(_matrix.shape[0] == this->shape[0]);
            assert(_matrix.shape[1] == this->shape[1]);
            assert(_matrix.shape[2] == this->shape[2]);
            assert(_matrix.shape[3] == this->shape[3]);

            for (size_t i = 0; i < size_1d; i++){
                temp[i] /= _matrix.matrix[i];
            }
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

Matrix& f_a(){
    Matrix *result = new Matrix(true);
    return *result;
}

void f_b(){
    Matrix a = f_a();
}


int main(){
    Matrix a(1, 5, 3);
    Matrix b(3, 5, 2);
    b = b + a;
    b.print_matrix();
//    a.set_matrix_1_to_x();
//    a.transpose();
//    a.print_matrix();
    return 0;
}
