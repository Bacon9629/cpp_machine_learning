#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>

using namespace std;

//
// Created by Bacon on 2022/11/29.
//
#ifndef AI_CLASS_MATRIX_H
#define AI_CLASS_MATRIX_H

#include <iostream>
#include <cmath>
#include <cassert>

using namespace std;

//template <typename Type>
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
    double *matrix = 0;
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

    /***
     * for 4 dim matrix
     * @param _matrix input_matrix
     * @param padding padding
     * @return matrix
     */
    static Matrix &padding(Matrix &_matrix, size_t padding){
        Matrix *result = new Matrix(_matrix.shape[0], _matrix.shape[1] + padding * 2, _matrix.shape[2] + padding * 2, _matrix.shape[3], 0, true);
        for (size_t i = 0; i < _matrix.shape[0]; i++){
            for (size_t j = 0; j < _matrix.shape[1]; j++){
                for (size_t k = 0; k < _matrix.shape[2]; k++){
                    for (size_t l = 0; l < _matrix.shape[3]; l++){
                        result->get(i, j + padding, k + padding, l) = _matrix.get(i, j, k, l);
                    }
                }
            }
        }
        return *result;
    }

    // 取 start 到 end - 1 的row
    inline static Matrix& getPictures(Matrix &_matrix, size_t start_picture, size_t end_picture){
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

    static Matrix& get_per_channel(Matrix &x, size_t which_channel){
        assert(x.shape[3] > which_channel);
        Matrix* result = new Matrix(x.shape[0], x.shape[1], x.shape[2], 1, 0, true);
        for (size_t i = 0; i < x.shape[0]; i++){
            for (size_t j = 0; j < x.shape[0]; j++){
                for (size_t k = 0; k < x.shape[0]; k++){
                    result->get(i, j, k, 0) = x.get(i, j, k, which_channel);
                }
            }
        }

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

    Matrix(size_t *_shape, double init_val, bool is_calculate = false){
        init(_shape[0], _shape[1], _shape[2], _shape[3], init_val, is_calculate);
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
        if (matrix != 0){
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

    Matrix& rotate_180(){
//        assert(shape[1] == shape[2]);  // 4D matrix, [img, row, col, channel]
        Matrix *result = new Matrix(shape[0], shape[1], shape[2], shape[3], 0, true);
        Matrix temp(this->matrix, shape[0], shape[1], shape[2], shape[3],  true);
        for (size_t i = 0; i < shape[0]; i++){
            for (size_t row = 0; row < shape[1]; row++){
                for (size_t col = 0; col < shape[2]; col++){
                    double *ori = &(temp.get(i, shape[1] - 1 - row, shape[2] - 1 - col, 0));
                    double *target = &result->get(i, row, col, 0);
                    for (size_t j = 0; j < shape[3]; j++){
                        *(target + j) = *(ori + j);
                    }
                }
            }
        }
        return *result;
    }

    void set_per_channel(Matrix &x, size_t which_channel){
        assert(x.shape[3] == 1);
        for (size_t i = 0; i < shape[0]; i++){
            for (size_t row = 0; row < shape[1]; row++){
                for (size_t col = 0; col < shape[2]; col++){
                    get(i, row, col, which_channel) = x.get(i, row, col, 0);
                }
            }
        }
    }

    double sum(){
        double result = 0;
        for (size_t i = 0; i < size_1d; i++){
            result += matrix[i];
        }
        return result;
    }

    Matrix& row_sum(){
        assert(shape[0] == 1 || shape[1] == 1);
        Matrix *result = new Matrix(matrix, 1, shape[3], true);
        for (size_t i = 1; i < shape[2]; i++){
            for (size_t j = 0; j < shape[3]; j++){
                result->get(0, j) += get(i, j);
            }
        }

        return *result;
    }

    Matrix& row_max(){
        Matrix *result = new Matrix(shape[2], 1, 0, true);
        for (size_t i=0;i<shape[2];i++) {
            double *now_max_ptr = &(result->get(i, 0));
            for (size_t j = 0; j < shape[3]; j++) {
                double *now_x_ptr = &(get(i, j));
                if (*now_x_ptr > *now_max_ptr) {
                    *now_max_ptr = *now_x_ptr;
                }
//                max.get(i, 0) = x.get(i, j) > max.get(i, 0) ? x.get(i, j) : max.get(i, 0);
            }
        }
        return *result;
    }

    /***
     * 將一張照片的shape[3]轉成shape[0]，給de_conv計算用
     * shape[0]必須 = 1
     * @param channel_size 權重or原始照片的channel size
     * @return 假設原始shape(1, 3, 3, 2)轉換成shape(2, 3, 3, channel_size)
     */
    Matrix &per_picture_change_shape_3_to_0_for_conv(){
        assert(shape[0] == 1);
        Matrix *result = new Matrix(shape[3], shape[1], shape[2], 1, 0, true);
        for (size_t i = 0; i < shape[3]; i++){
            for (size_t j = 0; j < shape[1]; j++){
                for (size_t k = 0; k < shape[2]; k++){
                    result->get(i, j, k, 0) = get(0, j, k, i);
                }
            }
        }
        return *result;
    }

    Matrix &shape_3_to_0(){
        Matrix *result = new Matrix(shape[3], shape[1], shape[2], shape[0], 0, true);
        Matrix temp(this->matrix, shape[0], shape[1], shape[2], shape[3]);
        for (size_t i = 0; i < shape[3]; i++){
            for (size_t j = 0; j < shape[1]; j++){
                for (size_t k = 0; k < shape[2]; k++){
                    for (size_t l = 0; l < shape[0]; l++){
                        result->get(i, j, k, l) = temp.get(l, j, k, i);
                    }
                }
            }
        }
        return *result;
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
//        print_shape();
        if (shape[0] != 1 || shape[1] != 1){
            cout << "{" << endl;
            for (int k=0;k<shape[0];k++){
                cout << "  {" << endl;
                for (int l=0;l<shape[1];l++){
                    cout << "    ";
                    for (int i=0; i < shape[2]; i++) {
                        cout << "{ ";
                        for (int j=0; j < shape[3]; j++) {
                            cout << get(k, l, i, j) << ", ";
                        }
                        cout << "}, ";
                    }
                    cout << endl;
                }
                cout << "  }, " << endl;
            }
            cout << "}" << endl;

        }else{
            cout << "{" << endl ;
            for ( size_t row = 0; row < shape[2]; row++){
                cout << "  ";
                for (size_t col = 0; col < shape[3]; col++) {
                    cout << get(row, col) << ", ";
                }
                cout << endl;
            }
            cout << "}" << endl;

        }

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
        double* result_matrix = result->matrix;
        if (_matrix.shape[0] != this->shape[0]){
            cout << endl;
        }
        assert(_matrix.shape[0] == this->shape[0]);
        assert(_matrix.shape[1] == this->shape[1]);

        if (_matrix.shape[2] == shape[2] && _matrix.shape[3] == shape[3]){
            for (size_t i = 0; i < size_1d; i++){
                result_matrix[i] += _matrix.matrix[i];
            }
        }else if(_matrix.shape[2] == 1 && _matrix.shape[3] == shape[3]) {
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] += _matrix.matrix[j];
                }
            }
        }else if(_matrix.shape[3] == 1 && _matrix.shape[2] == shape[2]){
            size_t a = 0;
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] += _matrix.matrix[a];
                }
                a += 1;
            }
        }else{
            cout << "shape error:\n";
            print_shape();
            _matrix.print_shape();
            assert("shape error");
        }

        if (_matrix.is_cal_result){
            delete &_matrix;
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
        double* result_matrix = result->matrix;
        assert(_matrix.shape[0] == this->shape[0]);
        assert(_matrix.shape[1] == this->shape[1]);

        if (_matrix.shape[2] == shape[2] && _matrix.shape[3] == shape[3]){
            for (size_t i = 0; i < size_1d; i++){
                result_matrix[i] -= _matrix.matrix[i];
            }
        }else if(_matrix.shape[2] == 1 && _matrix.shape[3] == shape[3]) {
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] -= _matrix.matrix[j];
                }
            }
        }else if(_matrix.shape[3] == 1 && _matrix.shape[2] == shape[2]){
            size_t a = 0;
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] -= _matrix.matrix[a];
                }
                a += 1;
            }
        }else{
            cout << "shape error:\n";
            print_shape();
            _matrix.print_shape();
            assert("shape error");
        }

        if (_matrix.is_cal_result){
            delete &_matrix;
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
        double* result_matrix = result->matrix;
        if (_matrix.shape[0] != this->shape[0]){
            cout << endl;
        }
        assert(_matrix.shape[0] == this->shape[0]);
        assert(_matrix.shape[1] == this->shape[1]);

        if (_matrix.shape[2] == shape[2] && _matrix.shape[3] == shape[3]){
            for (size_t i = 0; i < size_1d; i++){
                result_matrix[i] *= _matrix.matrix[i];
            }
        }else if(_matrix.shape[2] == 1 && _matrix.shape[3] == shape[3]) {
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] *= _matrix.matrix[j];
                }
            }
        }else if(_matrix.shape[3] == 1 && _matrix.shape[2] == shape[2]){
            size_t a = 0;
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] *= _matrix.matrix[a];
                }
                a += 1;
            }
        }else{
            cout << "shape error:\n";
            print_shape();
            _matrix.print_shape();
            assert("shape error");
        }

        if (_matrix.is_cal_result){
            delete &_matrix;
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
        double* result_matrix = result->matrix;
        assert(_matrix.shape[0] == this->shape[0]);
        assert(_matrix.shape[1] == this->shape[1]);

        if (_matrix.shape[2] == shape[2] && _matrix.shape[3] == shape[3]){
            for (size_t i = 0; i < size_1d; i++){
                result_matrix[i] /= _matrix.matrix[i];
            }
        }else if(_matrix.shape[2] == 1 && _matrix.shape[3] == shape[3]) {
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] /= _matrix.matrix[j];
                }
            }
        }else if(_matrix.shape[3] == 1 && _matrix.shape[2] == shape[2]){
            size_t a = 0;
            for (size_t i = 0; i < size_1d; i += shape[3]) {
                for (size_t j = 0; j < shape[3]; j++) {
                    result_matrix[i + j] /= _matrix.matrix[a];
                }
                a += 1;
            }
        }else{
            cout << "shape error:\n";
            print_shape();
            _matrix.print_shape();
            assert(false);
        }

        if (_matrix.is_cal_result){
            delete &_matrix;
        }

        return *result;
    }

    void operator= (Matrix &_matrix){
        for ( size_t i = 0; i < 4; i++){
            shape[i] = _matrix.shape[i];
            index_reflec_1d_[i] = _matrix.index_reflec_1d_[i];
            size_1d = _matrix.size_1d;
        }
        if (matrix != 0){
            delete []matrix;
        }

        if (_matrix.is_cal_result){
            matrix = _matrix.matrix;
            _matrix.matrix = 0;
            delete &_matrix;
        }else{
            matrix = new double [size_1d];
            memcpy(matrix, _matrix.matrix, sizeof(double) * size_1d);
        }
    }
};

#endif //AI_CLASS_MATRIX_H


// loss function - start

class LossFunc{
public:
    virtual double forward(Matrix &y, Matrix &target) = 0;
    virtual Matrix& backward(Matrix &y, Matrix &target) = 0;
};

class MSE: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override{
        double result = 0;
        Matrix temp = y - target;
        temp = temp * temp;
        result = temp.sum();
        result /= double (temp.shape[2] * temp.shape[3]);
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(true);
        *result = y - target;
        return *result;
    }
};

class CrossEntropy_SoftMax: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                if (target.get(i, j) == 1){
                    result -= log(y.get(i, j));
                    break;
                }else{
                    continue;
                }
            }
        }
        result = result / double (y.shape[2]);
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(y,  true);
        double *result_matrix = result->matrix;
        double *target_matrix = target.matrix;
        for (size_t i = 0; i< result->size_1d; i++){
            if (target_matrix[i] != 0){
                result_matrix[i] -= 1;
            }
        }
        return *result;
    }
};

// loss function - end


// active function - start

class ActiveFunc{
public:
    virtual Matrix& func_forward(Matrix &x) = 0;
    virtual Matrix& func_backward(Matrix &x) = 0;

};

class Relu: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape, 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? temp : 0;
        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape, 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? 1 : 0;
        }
        return *result;
    }
};

class Sigmoid: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape, 0, true);
        double* result_pointer = result->matrix;
        double* x_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_pointer[i];
            result_pointer[i] = 1 / (1 + exp(-x_pointer[i]));
        }


//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                result->get(i, j) = 1 / (1 + exp(-x.get(i, j)));
//
//            }
//        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix a = func_forward(x);
        Matrix *result = new Matrix(true);
        *result = (a - 1) * -1 * a;
        return *result;
    }
};

class SoftMax_CrossEntropy: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(true);
        Matrix x_exp = x.exp_();
        Matrix x_total = x_exp.row_sum();
        *result = x_exp / x_total;
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        return *(new Matrix(x.shape, 1, true));
    }
};

// active function - end


// Optimizer - start

class Optimizer{
public:
    virtual void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) = 0;
};

class SGD: public Optimizer{
public:
    double eta;

    SGD(double _eta){
        eta = _eta;
    }

    void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
        double _eta = eta / double (w.shape[2]);
        w = w - (grad_w * _eta);
        b = b - (grad_b * _eta);
    }

};

class MMT: public Optimizer{
public:
    double eta;
    double beta = 0.2;
    Matrix last_grad_w;
    Matrix last_grad_b;

    MMT(double _eta){
        init(_eta, 0.2);
    }

    MMT(double _eta, double _beta){
        init(_eta, _beta);
    }

    void init(double _eta, double _beta){
        eta = _eta;
        beta = _beta;
    }

    void gradient_descent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
//        last_grad_w =  alpha * grad_w + beta * last_grad_w;
//        w -= last_grad_w;
        if (last_grad_w.size_1d == 0){
            last_grad_w = *(new Matrix(grad_w.shape, 0, true));
            last_grad_b = *(new Matrix(grad_b.shape, 0, true));
        }

        last_grad_w = (grad_w * eta) + (last_grad_w * beta);
        w = w - last_grad_w;
        last_grad_b = (grad_b * eta) + (last_grad_b * beta);
        b = b - last_grad_b;
    }

};

// Optimizer - end


class Layer{
public:
    Matrix x;  // 輸入
    Matrix y;  // y = xw+b
    Matrix u;  // u = active_func(y)；此層輸出(下一層的輸入)
    Matrix w, b;
    Matrix delta;

    Matrix grad_w;
    Matrix grad_b;

    ActiveFunc *active_func;
    Optimizer *optimizer;

    virtual Matrix& forward(Matrix &_x, bool is_train) = 0;
    virtual Matrix& backward(Matrix &_delta, bool is_train) = 0;
    virtual void update() = 0;
};

class DropoutLayer:public Layer{
public:

    static Matrix& construct_random_bool_list(size_t row, size_t col, double probability){
        Matrix *result = new Matrix(row, col, 0, true);
        for(size_t i = 0; i < row; i++){
            for(size_t j = 0; j < col; j++){
                result->get(i, j) = (double(rand()) / RAND_MAX < probability) ? 0 : 1;
            }
        }
        return *result;
    }

    double dropout_probability;
    Matrix dropout_matrix;

    DropoutLayer(double _dropout_probability){
        dropout_probability = _dropout_probability;
    }



    Matrix& forward(Matrix &_x, bool is_train) override {
        x = _x;
        if (!is_train){
            return x * (1 - dropout_probability);
        }

        dropout_matrix = DropoutLayer::construct_random_bool_list(x.shape[2], x.shape[3], dropout_probability);
        x = x * dropout_matrix;
        return x;
    }

    Matrix& backward(Matrix &_delta, bool is_train) override {
        if (!is_train){
            return _delta;
        }
        delta = _delta * dropout_matrix;

        return delta;
    }

    void update() override {
    }
};

class ConvLayer: public Layer{
public:

    /***
     * 捲積
     * @param img shape = (img_account, img_row, img_col, img_channel)
     * @param filter shape = (filter_size, kernel_row, kernel_col, kernel_channel)，kernel size 必須要是奇數，filter 必須是正方形
     * @return feature_img, shape = (img_account, img_row, img_col, img_channel)
     */
    static Matrix &convolution(Matrix &img, Matrix &filter){
        assert(filter.shape[1] < img.shape[1]);
        assert(filter.shape[2] < img.shape[2]);
        if (filter.shape[3] != img.shape[3]){
            cout << endl;
        }
        assert(filter.shape[3] == img.shape[3]);  // filter 與 img 的 channel 要一樣
//        assert(filter.shape[1] == img.shape[2]);  // filter 必須正方形

        size_t channel_size = img.shape[3];
        size_t filter_size = filter.shape[0];
        const size_t kernel_high = filter.shape[1];
        const size_t kernel_width = filter.shape[2];
        Matrix *result = new Matrix(img.shape[0], img.shape[1] - kernel_high + 1, img.shape[2] - kernel_width + 1, filter.shape[0], 0, true);
        size_t kernel_area_size = kernel_high * kernel_width;

        size_t* kernel_reflect_img_pos_idx = new size_t[kernel_high * kernel_width];

        // 建立kernel_reflect_img_pos_idx
        kernel_reflect_img_pos_idx[0] = 0;
        for (size_t col = 1; col < kernel_width; col++){
            kernel_reflect_img_pos_idx[col] = kernel_reflect_img_pos_idx[0] + col * channel_size;
        }

        size_t source_img_width = img.shape[2];
        for (size_t row = 1; row < kernel_high; row++){  // 最左排完成
            kernel_reflect_img_pos_idx[row * kernel_width + 0] = kernel_reflect_img_pos_idx[(row-1) * kernel_width] + channel_size * source_img_width;
            for (size_t col = 1; col < kernel_width; col++){
                kernel_reflect_img_pos_idx[row * kernel_width + col] = kernel_reflect_img_pos_idx[row * kernel_width] + col * channel_size;
            }
        }


        // 捲積
//        size_t col_jump = kernel_width * channel_size;
        for (size_t k = 0; k < img.shape[0]; k++){  // 對一張圖片

            for (size_t j = 0; j < filter_size; j++){  // 一張圖片對一個filter

                for (size_t result_row = 0; result_row < result->shape[1]; result_row++){
                    double *img_target_ptr = &(img.get(k, result_row, 0, 0));  // 現在的目標像素點是哪個

                    for (size_t result_col = 0; result_col < result->shape[2]; result_col++) {  // 一個filter對一張圖片中的一個像素點做捲機計算

                        // 對原始照片的一個像素點做卷積運算後存入result
                        double *filter_ptr = &(filter.get(j, 0, 0, 0));  // 現在要做卷積的filter是哪個
                        double temp = 0;
                        for (size_t i = 0; i < kernel_area_size; i++){

                            double *pixel_a = img_target_ptr + kernel_reflect_img_pos_idx[i];  // 目標像素點對映到捲積計算時要與kernel相乘的像素點位置
                            double *kernel_a =  filter_ptr + i * channel_size;
                            for (size_t z = 0; z < channel_size; z++){
                                temp += pixel_a[z] * kernel_a[z];
                            }

                        }
//                        temp += bias.matrix[j];
                        result->get(k, result_row, result_col, j) = temp;  // 把「一個捲積核」的計算結果存進result matrix
                        img_target_ptr += channel_size;
                    }
                }
            }
        }

        delete[] kernel_reflect_img_pos_idx;
        return *result;
    }


    /***
     * 捲積
     * @param img shape = (img_account, img_row, img_col, img_channel)
     * @param filter shape = (filter_size, kernel_row, kernel_col, kernel_channel)，kernel size 必須要是奇數，filter 必須是正方形，kernel_channel與img_channel相同
     * @param bias shape = (1, 1, 1, filter_size)
     * @return feature_img, shape = (img_account, img_row, img_col, img_channel)
     */
    static Matrix &convolution(Matrix &img, Matrix &filter, Matrix &bias){
        assert(filter.shape[1] < img.shape[1]);
        assert(filter.shape[2] < img.shape[2]);
        assert(filter.shape[3] == img.shape[3]);  // filter 與 img 的 channel 要一樣
//        assert(filter.shape[1] == img.shape[2]);  // filter 必須正方形

        size_t channel_size = img.shape[3];
        size_t filter_size = filter.shape[0];
        const size_t kernel_high = filter.shape[1];
        const size_t kernel_width = filter.shape[2];
//        size_t temp_a = (kernel_width - 1) / 2;  // 因為kernel size關係，img必須從temp_a個像素點開始做捲積計算
        Matrix *result = new Matrix(img.shape[0], img.shape[1] - kernel_high + 1, img.shape[2] - kernel_width + 1, filter.shape[0], 0, true);
        size_t kernel_area_size = kernel_high * kernel_width;
//        size_t kernel_reflect_img_pos_idx[kernel_high][kernel_width];
//        size_t *kernel_reflect_img_pos_idx_1D = kernel_reflect_img_pos_idx[0];  // filter中每一個kernel對應到圖片上需要加減多少個點

        size_t *kernel_reflect_img_pos_idx = new size_t [kernel_high * kernel_width];
//        size_t *kernel_reflect_img_pos_idx_1D = kernel_reflect_img_pos_idx.matrix;

//        size_t *(kernel_reflect_img_pos_idx[kernel_width]);
//        kernel_reflect_img_pos_idx[0] = kernel_reflect_img_pos_idx_1D;


        // 建立kernel_reflect_img_pos_idx
        kernel_reflect_img_pos_idx[0] = 0;
        for (size_t col = 1; col < kernel_width; col++){
            kernel_reflect_img_pos_idx[col] = kernel_reflect_img_pos_idx[0] + col * channel_size;
        }

        size_t source_img_width = img.shape[2];
        for (size_t row = 1; row < kernel_high; row++){  // 最左排完成
            kernel_reflect_img_pos_idx[row * kernel_width + 0] = kernel_reflect_img_pos_idx[(row-1) * kernel_width] + channel_size * source_img_width;
            for (size_t col = 1; col < kernel_width; col++){
                kernel_reflect_img_pos_idx[row * kernel_width + col] = kernel_reflect_img_pos_idx[row * kernel_width] + col * channel_size;
            }
        }


        // 捲積
//        size_t col_jump = kernel_width * channel_size;
        for (size_t k = 0; k < img.shape[0]; k++){  // 對一張圖片

            for (size_t j = 0; j < filter_size; j++){  // 一張圖片對一個filter

                for (size_t result_row = 0; result_row < result->shape[1]; result_row++){
                    double *img_target_ptr = &(img.get(k, result_row, 0, 0));  // 現在的目標像素點是哪個

                    for (size_t result_col = 0; result_col < result->shape[2]; result_col++) {  // 一個filter對一張圖片中的一個像素點做捲機計算

                        // 對原始照片的一個像素點做卷積運算後存入result
                        double *filter_ptr = &(filter.get(j, 0, 0, 0));  // 現在要做卷積的filter是哪個
                        double temp = 0;
                        for (size_t i = 0; i < kernel_area_size; i++){

                            double *pixel_a = img_target_ptr + kernel_reflect_img_pos_idx[i];  // 目標像素點對映到捲積計算時要與kernel相乘的像素點位置
                            double *kernel_a =  filter_ptr + i * channel_size;
                            for (size_t z = 0; z < channel_size; z++){
                                temp += pixel_a[z] * kernel_a[z];
                            }

                        }
                        temp += bias.matrix[j];
                        result->get(k, result_row, result_col, j) = temp;  // 把「一個捲積核」的計算結果存進result matrix
                        img_target_ptr += channel_size;
                    }
                }
            }
        }

        delete[] kernel_reflect_img_pos_idx;

        return *result;
    }

    Matrix &convolution_back_get_per_picture_per_channel_d_w(size_t which_picture, size_t which_channel, Matrix &delta_u){
        Matrix *result = new Matrix(w.shape, 0, true);
        Matrix target_x = Matrix::getPictures(x, which_picture, which_picture+1);
        target_x = Matrix::get_per_channel(target_x, which_channel);

        Matrix target_u = Matrix::getPictures(delta_u, which_picture, which_picture+1);
        Matrix u_change;
        u_change = target_u.per_picture_change_shape_3_to_0_for_conv();

        *result = ConvLayer::convolution(target_x, u_change);

        return *result;
    }

    size_t kernel_size, filter_size;

    ConvLayer(size_t _filter_size, size_t _kernel_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        kernel_size = _kernel_size;
        filter_size = _filter_size;
        active_func = _activeFunc;
        optimizer = _optimizer;
//        w = *(new Matrix(_filter_size, _kernel_size, _kernel_size, chanel_size, 0));
        b = *(new Matrix(1, _filter_size, 0, true));
//        w.random_matrix();
//        b.random_matrix();
//        grad_w = *(new Matrix(w.shape[2], w.shape[3], 0));
        grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
    }


    Matrix &forward(Matrix &_x, bool is_train) override {
        if (w.size_1d == 0){
            w = *(new Matrix(filter_size, kernel_size, kernel_size, _x.shape[3], 0, true));
            w.random_matrix();
            grad_w = *(new Matrix(filter_size, kernel_size, kernel_size, _x.shape[3], 0, true));
        }
//        _x.print_shape();
//        w.print_shape();

        x = _x;
        u = ConvLayer::convolution(x, w, b);
        y = active_func->func_forward(u);

        return y;
    }

    Matrix &backward(Matrix &_delta, bool is_train) override {
//        Matrix *result = new Matrix(true);

        Matrix my_delta = active_func->func_backward(u) * _delta;
        Matrix u_delta = my_delta * u;

        Matrix temp(grad_w.shape[3], grad_w.shape[1], grad_w.shape[2], grad_w.shape[0], 0);

        for(size_t i = 0; i < x.shape[0]; i++){
            Matrix temp2(temp.shape, 0);
            for (size_t _channel = 0; _channel < x.shape[3]; _channel++){
                Matrix w_change_i = convolution_back_get_per_picture_per_channel_d_w(i, _channel, u_delta);
                Matrix w_change_i_3_to_0 = w_change_i.shape_3_to_0();
                temp2.set_per_channel(w_change_i_3_to_0, _channel);
            }
            temp = temp + temp2;
        }

        grad_w = temp.shape_3_to_0() / double(x.shape[0]);

        Matrix w_0_to_3_rotate = w.shape_3_to_0();
        w_0_to_3_rotate = w_0_to_3_rotate.rotate_180();

        Matrix padding_delta = Matrix::padding(u_delta, kernel_size-1);
        delta = ConvLayer::convolution(padding_delta, w_0_to_3_rotate);

        return delta;
    }

    void update() override {
        optimizer->gradient_descent(w, b, grad_w, grad_b);
    }


};

class FlattenLayer: public Layer{
public:
    size_t input_shape[4] = {0, 0, 0, 0};
//    size_t output_shape[4] = {0, 0, 0, 0};
    Matrix &forward(Matrix &_x, bool is_train) override {
        Matrix *result = new Matrix(_x, true);
        input_shape[0] = _x.shape[0];
        input_shape[1] = _x.shape[1];
        input_shape[2] = _x.shape[2];
        input_shape[3] = _x.shape[3];
//        output_shape[0] = 1;
//        output_shape[1] = 1;
//        output_shape[2] = _x.shape[0];
//        output_shape[3] = _x.size_1d * _x.shape[0];
        result->reshape(1, 1, _x.shape[0], _x.shape[1] * _x.shape[2] * _x.shape[3]);
        return *result;
    }

    Matrix &backward(Matrix &_delta, bool is_train) override {
        Matrix *result = new Matrix(_delta, true);
        result->reshape(input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        return *result;
    }

    void update() override {
    }
};

class DenseLayer : public Layer{
public:

//    Matrix x;  // 輸入
//    Matrix y;  // y = xw+b
//    Matrix u;  // u = active_func(y)；此層輸出(下一層的輸入)
//    Matrix w, b;
//    Matrix delta;
//
//    Matrix grad_w;
//    Matrix grad_b;
//
//    ActiveFunc *active_func;
//    Optimizer *optimizer;
    size_t output_size;

    DenseLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(output_size, _activeFunc, _optimizer);
    }

    DenseLayer(size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(output_size, _activeFunc, _optimizer);
    }

    void init(size_t input_size, size_t _output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        w = *(new Matrix(input_size, output_size, 0, true));
        w.random_matrix();
        b = *(new Matrix(1, output_size, 0, true));
        b.random_matrix();
        grad_w = *(new Matrix(w.shape[2], w.shape[3], 0, true));
        grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
        active_func = _activeFunc;
        optimizer = _optimizer;
        output_size = _output_size;
    }

    void init(size_t _output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        active_func = _activeFunc;
        optimizer = _optimizer;
        output_size = _output_size;
    }

    Matrix& forward(Matrix& _x, bool is_train) override{
        if (w.size_1d == 0){
            w = *(new Matrix(_x.shape[3], output_size, 0, true));
            w.random_matrix();
            b = *(new Matrix(1, output_size, 0, true));
            b.random_matrix();
            grad_w = *(new Matrix(w.shape[2], w.shape[3], 0, true));
            grad_b = *(new Matrix(b.shape[2], b.shape[3], 0, true));
        }
        assert(_x.shape[0] == 1 || _x.shape[1] == 1);
        x = _x;
        u = Matrix::dot(x, w);
        u = u + b;
        y = active_func->func_forward(u);
        return y;
    }

    Matrix& backward(Matrix& _delta, bool is_train) override{

        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = _delta * active_func_back;

        Matrix x_t = Matrix::transpose(x);
        grad_w = Matrix::dot(x_t, my_delta);
        grad_w = grad_w / double(x.shape[2]);
        grad_b = _delta.row_sum();
        grad_b = grad_b / double(x.shape[2]);

        Matrix w_t = Matrix::transpose(w);
        delta = Matrix::dot(my_delta, w_t);
        return delta;
    }

    void update() override{
        optimizer->gradient_descent(w, b, grad_w, grad_b);
    }

};

class MyFrame{
    vector<Layer*> layers = vector<Layer*>(0);
    LossFunc *lossFunc;
    int batch;

public:

    MyFrame(LossFunc *_lossFun, int _batch){
        lossFunc = _lossFun;
        batch = _batch;
    }

    ~MyFrame(){
        // 這裡是這樣寫嗎?
        layers.clear();
        layers = vector<Layer*>();
    }

    void add(Layer *layer){
        layers.push_back(layer);
    }

    void train_img_input(size_t epoch, Matrix &x, Matrix &target){
        assert(target.shape[0] == 1);
        assert(target.shape[1] == 1);
        assert(target.shape[2] == x.shape[0]);
        assert(target.shape[3] == layers[layers.size()-1]->y.shape[3]);  // 確認神經網路最終輸出與target的數量是一樣的
        size_t _batch = batch == -1 ? x.shape[0] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[0];  // 存著還有幾筆資料需要訓練

            for (size_t j = 0; data_left_size != 0 ; j++){
                if (data_left_size < _batch){
                    // 如果資料量"不足"填滿一個batch
                    Matrix _x = Matrix::getPictures(x, j * _batch, j * _batch + data_left_size);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + data_left_size);

                    train_one_time(_x, _target);

                    data_left_size = 0;
                }else{
                    // 如果資料量"足夠"填滿一個batch
                    Matrix _x = Matrix::getPictures(x, j * _batch, j * _batch + _batch);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + _batch);

                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }


            }


            cout << "epoch: " << i << endl;
            show(x, target);
        }
    }

    void train(size_t epoch, Matrix &x, Matrix &target, bool img_input = false){  // 這裡擴充batch size
        if (!img_input){
            train_img_input(epoch, x, target);
            return;
        }

        assert(x.shape[0] == 1 && x.shape[1] == 1);  // 不可輸入圖片
        assert(target.shape[0] == 1);
        assert(target.shape[1] == 1);
        assert(target.shape[2] == x.shape[2]);
        assert(target.shape[3] == layers[layers.size()-1]->y.shape[3]);  // 確認神經網路最終輸出與target的數量是一樣的

        size_t _batch = batch == -1 ? x.shape[2] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[2];  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;
            for (size_t j = 0; data_left_size != 0 ; j++){
                if (data_left_size < _batch){
                    // 如果資料量"不足"填滿一個batch
                    Matrix _x = Matrix::getRow(x, j * _batch, j * _batch + data_left_size);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + data_left_size);

                    train_one_time(_x, _target);
                    data_left_size = 0;

                }else{
                    // 如果資料量"足夠"填滿一個batch
                    Matrix _x = Matrix::getRow(x, j * _batch, j * _batch + _batch);
                    Matrix _target = Matrix::getRow(target, j * _batch, j * _batch + _batch);

//                    cout << i << endl;
                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }

            }

            cout << "epoch: " << i;
            show(x, target);
        }
    }

    inline void train_one_time(Matrix &x, Matrix &target){  // 這裡客製化網路輸入
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y, true);
        }

        Matrix delta = lossFunc->backward(y, target);
        for (size_t i = 0; i < layers.size(); i++){  // 這裡會減到零以下，因此不能使用無號數
            delta = layers[layers.size() - 1 - i]->backward(delta, true);
        }

        for (size_t i = 0; i < layers.size(); ++i){
            layers[i]->update();
        }
    }

    void show(Matrix &x, Matrix &target){
        Matrix y = x;
        for (size_t i=0;i<layers.size();++i){
            y = layers[i]->forward(y, false);
        }

//        cout << "\nlabel: " << endl;
//        target.print_matrix();
//
//        cout << "\nresult: " << endl;
//        y.print_matrix();

//        double acc = 0;
//        Matrix temp = x * target;
//        for (size_t i = 0; i < y.shape[0]; i++){
//            double a = 0;
//            double b = 0;
//            for (size_t j = 0; j < y.shape[0]; j++){
//                if (target.get(i, j) == 1){
//
//                }
//            }
//        }

        cout << "\nloss: " << lossFunc->forward(y, target) << endl;

    }

};

Matrix &load_img(string file_name, size_t length){
    char pix[784];
    Matrix *img = new Matrix(length, 28, 28, 1, 0, true);
    ifstream file;
    file.open(file_name, ios::binary);
    assert(file.is_open());

// 先拿出前面16bytes不必要的資訊
    char p[16];
    file.read(p, 16);

// 讀取影像
    for (int b = 0; b < length; b++) {
        file.read(pix, 784);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                img->get(b,r, c, 0) = (double)((unsigned char)(pix[r * 28 + c])) / 255;
            }
        }
    }

// 關閉檔案
    file.close();

    return *img;
}

Matrix &load_label(const string& file_name, size_t length){
    char label[1];// 用來暫存二進制資料
    Matrix *label_matrix = new Matrix(1, 1, length, 10, 0, true);  // 存放label
    std::ifstream file;
    file.open(file_name, ios::binary); // 用二進制方式讀取檔案
    assert(file.is_open());

    // 先拿出前面8bytes不必要的資訊
    char p[8];
    file.read(p, 8);

    // 讀取label
    for (int i = 0; i < length; i++) {
        file.read(label, 1);
        label_matrix->get(i, (unsigned char) label[0]) = 1;
    }

    // 關閉檔案
    file.close();

    return *label_matrix;
}

void mnist(){
    Matrix imgs = load_img("D:\\user\\desktop\\C\\cpp_machine_learning\\train_images.idx3-ubyte", 1000);
    Matrix labels = load_label("D:\\user\\desktop\\C\\cpp_machine_learning\\train_labels.idx1-ubyte", 1000);

    Matrix imgs_test = load_img("D:\\user\\desktop\\C\\cpp_machine_learning\\train_images.idx3-ubyte", 100);
    Matrix labels_test = load_label("D:\\user\\desktop\\C\\cpp_machine_learning\\test_labels.idx1-ubyte", 100);

    MyFrame frame(new CrossEntropy_SoftMax, 256);
    frame.add(new ConvLayer(16, 3, new Relu, new MMT(0.3)));
    frame.add(new ConvLayer(32, 3, new Relu, new MMT(0.3)));
    frame.add(new FlattenLayer());
    frame.add(new DenseLayer(128, new Sigmoid, new MMT(0.3)));
    frame.add(new DropoutLayer(0.4));
    frame.add(new DenseLayer(10, new SoftMax_CrossEntropy, new MMT(0.3)));

    frame.show(imgs_test, labels_test);

    frame.train_img_input(5, imgs, labels);

    frame.show(imgs_test, labels_test);

}

int main() {
    mnist();

    return 0;
}
