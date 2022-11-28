#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>

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

    // 取 start 到 end - 1 的row
    inline static Matrix& getRow(Matrix &_matrix, size_t start_row, size_t end_row){
//        if (
//                start_row > _matrix.shape[2] || end_row > _matrix.shape[2] ||
//                _matrix.shape[0] != 0 || end_row < start_row
//                )
//        {
//            cout << "shape_wrong: Matrix getRow" << endl;
//        }
        assert(!(start_row > _matrix.shape[2] || end_row > _matrix.shape[2] ||
                 _matrix.shape[0] != 0 || end_row < start_row));

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

        if (col_a != row_b){
            std::cout << "shape wrong" << std::endl;
            assert("matrix error - dot");
        }

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
        size_1d = a * b * c * d;
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

#ifdef SHOW_MATRIX_PTR
        cout << "_matrix_point construct " << this << endl;
#endif
    }

    void init(size_t a, size_t b, size_t c, size_t d, double init_val, bool is_calculate){
        is_cal_result = is_calculate;
        size_1d = a * b * c * d;
        shape[3] = d;
        shape[2] = c;
        shape[1] = b;
        shape[0] = a;
        index_reflec_1d_[3] = 1;
        index_reflec_1d_[2] = d;
        index_reflec_1d_[1] = d * c;
        index_reflec_1d_[0] = b * d * c;
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
        for (size_t i = 0; i < size_1d; i++){
            temp[i] += _matrix.matrix[i];
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
        for (size_t i = 0; i < size_1d; i++){
            temp[i] -= _matrix.matrix[i];
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
        for (size_t i = 0; i < size_1d; i++){
            temp[i] *= _matrix.matrix[i];
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
        for (size_t i = 0; i < size_1d; i++){
            temp[i] /= _matrix.matrix[i];
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

        for(size_t r=0; r < y.shape[2];++r)
            for(size_t c = 0; c < y.shape[3];++c)
                result += temp.get(r, c);

        result /= double (y.shape[2] * y.shape[3]);
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(true);
        *result = y - target;
        return *result;
    }
};

class CrossEntropy: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                double _y = y.get(i, j);
                double _target = target.get(i, j);
                result += -_target * log(_y) - (1 - _target) * log(1 - _y);
            }
        }
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(y.shape[2], y.shape[3], 0, true);

        for (size_t i = 0; i< y.shape[2]; i++){
            for (size_t j = 0; j < y.shape[3]; j++){
                double _y = y.get(i, j);
                double _target = target.get(i, j);
                result->get(i, j) = (_y - _target) / (_y * (1 - _y));
            }
        }

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
        return result;
    }

    Matrix& backward(Matrix &y, Matrix &target) override {
        Matrix *result = new Matrix(y.shape[2], y.shape[3], 0, true);
        for (size_t i = 0; i < y.shape[2]; i++)
            for (size_t j = 0; j < y.shape[3]; j++)
                result->get(i, j) = target.get(i, j) != 0 ? y.get(i, j) - 1 : y.get(i, j);

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
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        for (size_t i = 0; i < x.shape[2]; i++){
            for (size_t j = 0; j < x.shape[3]; j++){
                    result->get(i, j) = x.get(i, j) > 0 ? x.get(i, j) : 0;
            }
        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        for (size_t i = 0; i < x.shape[2]; i++){
            for (size_t j = 0; j < x.shape[3]; j++){
                result->get(i, j) = x.get(i, j) > 0 ? 1 : 0;
            }
        }
        return *result;
    }
};

class Sigmoid: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        for (size_t i = 0; i < x.shape[2]; i++){
            for (size_t j = 0; j < x.shape[3]; j++){
                result->get(i, j) = 1 / (1 + exp(-x.get(i, j)));

            }
        }
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
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        Matrix total(x.shape[2], 1, 0);
        Matrix max(x.shape[2], 1, 0);
        for (size_t i=0;i<x.shape[2];i++)
            for (size_t j=0;j<x.shape[3];j++)
                max.get(i, 0) = x.get(i, j) > max.get(i, 0) ? x.get(i, j) : max.get(i, 0);

        for (size_t i=0;i<x.shape[2];i++)
            for (size_t j=0;j<x.shape[3];j++)
                total.get(i, 0) += exp(x.get(i, j) - max.get(i, 0));

        for (size_t i=0;i<x.shape[2];i++)
            for (size_t j=0;j<x.shape[3];j++)
                result->get(i, j) = exp(x.get(i, j) - max.get(i, 0)) / total.get(i, 0);



        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        return *(new Matrix(x.shape[2], x.shape[3], 1, true));
    }
};

class Tanh:public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        Matrix a = x.exp_();
        Matrix b = x.exp_() * -1;
        *result = (a - b) / (b - a);
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        Matrix a = x.exp_();
        Matrix b = x.exp_() * -1;
        Matrix c = (a - b) / (b - a);
        *result = (c * c - 1) * -1;
        return *result;
    }

};

// active function - end


// Optimizer - start

class Optimizer{
public:
    virtual void gradient_decent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) = 0;
};

class SGD: public Optimizer{
public:
    double eta;

    SGD(double _eta){
        eta = _eta;
    }

    void gradient_decent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
        double _eta = eta / w.shape[2];
        w = w - (grad_w * _eta);
        b = b - (grad_b * _eta);
    }

};

class MMT: public Optimizer{
public:
    double eta;
    double beta = 0.9;
    Matrix last_grad_w;
    Matrix last_grad_b;

    MMT(double _eta){
        init(_eta, 0.9);
    }

    MMT(double _eta, double _beta){
        init(_eta, _beta);
    }

    void init(double _eta, double _beta){
        eta = _eta;
        beta = _beta;
    }

    void gradient_decent(Matrix &w, Matrix &b, Matrix &grad_w, Matrix &grad_b) override {
//        last_grad_w =  alpha * grad_w + beta * last_grad_w;
//        w -= last_grad_w;
        if (last_grad_w.shape[2] == 0){
            last_grad_w = *(new Matrix(grad_w.shape[2], grad_w.shape[3], 0));
            last_grad_b = *(new Matrix(grad_b.shape[2], grad_b.shape[3], 0));
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

    double dropout_probability;
    Matrix dropout_matrix;

    DropoutLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer, double _dropout_probability){
        init(input_size, output_size, _activeFunc, _optimizer);
        dropout_probability = _dropout_probability;
    }

    Matrix& construct_random_bool_list(size_t row, size_t col, double probability){
        Matrix *result = new Matrix(row, col, 0, true);
        double a = 0;
        double b = 0;
        for(size_t i = 0; i < row; i++){
            for(size_t j = 0; j < col; j++){
                result->get(i, j) = (double(rand()) / RAND_MAX < probability) ? 0 : 1;
            }
        }
        return *result;
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
    }

    Matrix& forward(Matrix &_x, bool is_train) override {
        x = _x;
        if (!is_train){
            return x * (1 - dropout_probability);
        }

        dropout_matrix = construct_random_bool_list(x.shape[2], x.shape[3], dropout_probability);
        x = x * dropout_matrix;
        return x;
    }

    Matrix& backward(Matrix &_delta, bool is_train) override {
        if (!is_train){
            return _delta;
        }
        delta = delta * dropout_matrix;
        return delta;
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

    DenseLayer(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        init(input_size, output_size, _activeFunc, _optimizer);
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
        w = *(new Matrix(input_size, output_size, 0));
        w.random_matrix();
        b = *(new Matrix(1, output_size, 0));
        b.random_matrix();
        grad_w = *(new Matrix(w.shape[2], w.shape[3], 0));
        grad_b = *(new Matrix(b.shape[2], b.shape[3], 0));
        active_func = _activeFunc;
        optimizer = _optimizer;
    }

    Matrix& forward(Matrix& _x, bool is_train) override{
        x = _x;
        u = Matrix::dot(x, w);
        u = u + b;
        y = active_func->func_forward(u);
        return y;
    }

    Matrix& backward(Matrix& _delta, bool is_train) override{

        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = delta * active_func_back;
//        Matrix my_delta = Matrix::times(&_delta, &active_func_back);

        Matrix x_t = Matrix::transpose(x);
        grad_w = Matrix::dot(x_t, my_delta);

        Matrix w_t = Matrix::transpose(w);
        delta = Matrix::dot(my_delta, w_t);
        return delta;
    }

    void update() override{
        optimizer->gradient_decent(w, b, grad_w, grad_b);
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
        size_t _batch = batch == -1 ? x.shape[0] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[0];  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;

            for (size_t j = 0; data_left_size > 0 ; j++){
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

                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }

            }
        }
    }

    void train(size_t epoch, Matrix &x, Matrix &target){  // 這裡擴充batch size
        size_t _batch = batch == -1 ? x.shape[2] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[2];  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;

            for (size_t j = 0; data_left_size > 0 ; j++){
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

                    train_one_time(_x, _target);
                    data_left_size -= _batch;
                }

            }
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

        cout << "\nlabel: " << endl;
        target.print_matrix();

        cout << "\nresult: " << endl;
        y.print_matrix();

        cout << "\nloss: " << lossFunc->forward(y, target) << endl;

    }

};

int main() {
//    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};

    double temp_x[][25] =
            {
                    {0, 1, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 0, 1, 0, 0,
                            0, 1, 1, 1, 0},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 0,
                            1, 1, 1, 1, 1},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            0, 0, 1, 0, 1,
                            1, 1, 1, 1, 0},
                    {0, 0, 0, 1, 0,
                            0, 0, 1, 1, 0,
                            0, 1, 0, 1, 0,
                            1, 1, 1, 1, 1,
                            0, 0, 0, 1, 0},
                    {1, 1, 1, 1, 1,
                            1, 0, 0, 0, 0,
                            1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            1, 1, 1, 1, 0}
            };
    double temp_target[][5] =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 0, 1}};

    double temp_validation[][25] =
            {
                    {0, 0, 1, 1, 0,
                            0, 0, 1, 1, 0,
                            0, 1, 0, 1, 0,
                            0, 0, 0, 1, 0,
                            0, 1, 1, 1, 0},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 1, 1},
                    {1, 1, 1, 1, 0,
                            0, 0, 0, 0, 1,
                            0, 1, 1, 1, 0,
                            1, 0, 0, 0, 1,
                            1, 1, 1, 1, 0},
                    {0, 1, 1, 1, 0,
                            0, 1, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 0, 1, 1,
                            0, 1, 1, 1, 0},
                    {0, 1, 1, 1, 1,
                            0, 1, 0, 0, 0,
                            0, 1, 1, 1, 0,
                            0, 0, 0, 1, 0,
                            1, 1, 1, 1, 0}
            };

    double temp_validation_target[][5] =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 1, 0}};



//    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
//    vector<vector<double>> temp_target = {{0}, {1}, {1}, {0}};
    // data init
    Matrix x = *(new Matrix(temp_x[0], 5, 5, 5, 1));
    Matrix x_target = *(new Matrix(temp_target[0], 1, 1, 5, 5));
    Matrix validation = *(new Matrix(temp_validation[0], 5, 5, 5, 1));
    Matrix validation_target = *(new Matrix(temp_validation_target[0], 1, 1, 5, 5));

    // active func
    Sigmoid sigmoid = Sigmoid();
    SoftMax_CrossEntropy softmax = SoftMax_CrossEntropy();
    Tanh tanh = Tanh();
    Relu relu = Relu();

    /**
     * loss function: cross entropy with softmax
     * */
    MyFrame frame = MyFrame(new CrossEntropy_SoftMax, -1);


    frame.add(new DenseLayer(25, 32, &relu, new SGD(0.1)));
//    frame.add(new DenseLayer(64, 32, &softmax, new SGD(0.1)));
    frame.add(new DropoutLayer(32, 32, &softmax, new SGD(0.001), 0.5));
    frame.add(new DenseLayer(32, 5, &softmax, new SGD(0.1)));

    frame.train(100, x, x_target);


    frame.show(validation, validation_target);
    frame.show(x, x_target);
    return 0;
}
