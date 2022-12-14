//
// Created by Bacon on 2022/11/29.
//
#ifndef AI_CLASS_MATRIX_H
#define AI_CLASS_MATRIX_H

#define MAX_THREAD 8

#include <iostream>
#include <cmath>
#include <cassert>
#include <thread>

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

//    static Matrix& dot(Matrix &matrix_a, Matrix &matrix_b){
//        size_t row_a = matrix_a.shape[2];
//        size_t col_a = matrix_a.shape[3];
//        size_t row_b = matrix_b.shape[2];
//        size_t col_b = matrix_b.shape[3];
//
//        assert(col_a == row_b);
//
//        const size_t row_result = row_a;
//        const size_t col_result = col_b;
//        Matrix *result = new Matrix(row_result, col_result, 0, true);
//
//        for (int r = 0; r < row_result; r++) {
//            for (int c = 0; c < col_result; c++) {
//                // 指定在result內的哪個位置
//                // 接下來依照指定的result位置取出a、b的值來做計算
//                for (int i = 0; i < col_a; i++) {
//                    result->get(r, c) += matrix_a.get(r, i) * matrix_b.get(i, c);
//                }
//            }
//        }
//        return *result;
//    }


    static Matrix& dot2(Matrix &matrix_a, Matrix &matrix_b){
        size_t row_a = matrix_a.shape[2];
        size_t col_a = matrix_a.shape[3];
        size_t row_b = matrix_b.shape[2];
        size_t col_b = matrix_b.shape[3];

        assert(col_a == row_b);

        Matrix *result = new Matrix(row_a, col_b, 0, true);
        Matrix new_b(col_b, row_b, 0);


        double *result_point = result->matrix;
        double *b_point = matrix_b.matrix;
        double *a_point = matrix_a.matrix;
        double *new_b_point = new_b.matrix;

        double *temp_b(b_point), *temp_new_b;
        for (size_t j=0;j<row_b;j++){
            temp_new_b = new_b_point + j;
            for (size_t i=0;i<col_b;i++){
                *(temp_new_b) = *(temp_b++);
                temp_new_b += row_b;
            }
        }


        a_point = matrix_a.matrix;
        new_b_point = new_b.matrix;

        double *start_a_point = a_point;
        double *end_end_a_point = a_point + matrix_a.size_1d;
        double *end_end_new_b_point = new_b_point + new_b.size_1d;

        double temp = 0;
//        register double temp = 0;

        do{

            do{
                a_point = start_a_point;

                for (size_t i = 0; i < row_b; ++i){
                    temp += *(a_point++) * *(new_b_point++);
                }

                *(result_point++) = temp;
                temp = 0;

            }while(new_b_point != end_end_new_b_point);

        start_a_point = a_point;
        new_b_point = new_b.matrix;

        }while(a_point != end_end_a_point);


        return *result;
    }


    static void thread_mdot_job(double *target, double *a, double *b, size_t a_row, size_t b_size_1d, size_t a_b_col){
        double *start_a_point(a), *end_end_a_point(a + a_row * a_b_col), *start_start_b_point(b), *end_end_b_point(b + b_size_1d);
        double temp = 0;

        do{

            do{
                a = start_a_point;

                for (size_t i = 0; i < a_b_col; ++i){
                    temp += *(a++) * *(b++);
                }

                *(target++) = temp;
                temp = 0;

            }while(b != end_end_b_point);

            start_a_point = a;
            b = start_start_b_point;

        }while(a != end_end_a_point);
    }

    static Matrix& dot(Matrix &matrix_a, Matrix &matrix_b){
        size_t row_a = matrix_a.shape[2];
        size_t col_a = matrix_a.shape[3];
        size_t row_b = matrix_b.shape[2];
        size_t col_b = matrix_b.shape[3];

        assert(col_a == row_b);

        Matrix *result = new Matrix(row_a, col_b, 0, true);
        Matrix new_b(col_b, row_b, 0);


        double *result_point = result->matrix;
        double *b_point = matrix_b.matrix;
        double *a_point = matrix_a.matrix;
        double *new_b_point = new_b.matrix;

        double *temp_b(b_point), *temp_new_b;
        for (size_t j=0;j<row_b;j++){
            temp_new_b = new_b_point + j;
            for (size_t i=0;i<col_b;i++){
                *(temp_new_b) = *(temp_b++);
                temp_new_b += row_b;
            }
        }


        a_point = matrix_a.matrix;
        new_b_point = new_b.matrix;

//        double *start_a_point = a_point;
//        double *end_end_a_point = a_point + matrix_a.size_1d;
//        double *end_end_new_b_point = new_b_point + new_b.size_1d;
//
//        double temp = 0;

        size_t per_thread_a_row_size = size_t(row_a / MAX_THREAD);
        size_t remain_row_size = size_t(row_a % MAX_THREAD);

        thread *thread_array = new thread[MAX_THREAD];

        for (size_t i = 0; i < MAX_THREAD; i++){
//            thread_mdot_job(result_point, a_point, new_b_point, per_thread_a_row_size, new_b.size_1d, col_a);
            if (remain_row_size > 0){
                thread_array[i] = thread(thread_mdot_job, result_point, a_point, new_b_point, per_thread_a_row_size + 1, new_b.size_1d, col_a);
                result_point += (per_thread_a_row_size + 1) * col_b;
                a_point += (per_thread_a_row_size + 1) * col_a;
                remain_row_size -= 1;
            }else{
                thread_array[i] = thread(thread_mdot_job, result_point, a_point, new_b_point, per_thread_a_row_size, new_b.size_1d, col_a);
                result_point += per_thread_a_row_size * col_b;
                a_point += per_thread_a_row_size * col_a;
            }
//            cout << remain_row_size << endl;

        }

        for (size_t i = 0; i < MAX_THREAD; i++){
            thread_array[i].join();
        }

//        while(a_point != end_end_a_point) {
//            start_a_point = a_point;
//            new_b_point = new_b.matrix;
//            do{
//                a_point = start_a_point;
//
//                for (size_t i = 0; i < row_b; ++i){
//                    temp += *(a_point++) * *(new_b_point++);
//                }
//
//                *(result_point++) = temp;
//                temp = 0;
//
//            }while(new_b_point != end_end_new_b_point);
//
//        };

        delete[] thread_array;
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
