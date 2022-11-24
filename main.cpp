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


// loss function - start

class LossFunc{
public:
    virtual double forward(Matrix &y, Matrix &target) = 0;
    virtual Matrix backward(Matrix &y, Matrix &target) = 0;
};

class MSE: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override{
        double result = 0;
        Matrix temp = y - target;
        temp = Matrix::times(&temp, &temp);

        for(size_t r=0;r<y.row();++r)
            for(size_t c=0;c<y.col();++c)
                result += temp.matrix[r][c];

        result /= y.row() * y.col();
        return result;
    }

    Matrix backward(Matrix &y, Matrix &target) override {
        return Matrix::reduce(&y, &target);
    }
};

class CrossEntropy: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.row(); i++){
            for (size_t j = 0; j < y.col(); j++){
                double _y = y.matrix[i][j];
                double _target = target.matrix[i][j];
                result += -_target * log(_y) - (1 - _target) * log(1 - _y);
            }
        }
        return result;
    }

    Matrix backward(Matrix &y, Matrix &target) override {
        Matrix result(y.row(), y.col(), 0);

        for (size_t i = 0; i< y.row(); i++){
            for (size_t j = 0; j < y.col(); j++){
                double _y = y.matrix[i][j];
                double _target = target.matrix[i][j];
                result.matrix[i][j] = (_y - _target) / (_y * (1 - _y));
            }
        }

        return result;
    }
};

class CrossEntropy_SoftMax: public LossFunc{
public:
    double forward(Matrix &y, Matrix &target) override {
        double result = 0;
        for (size_t i = 0; i< y.row(); i++){
            for (size_t j = 0; j < y.col(); j++){
                if (target.matrix[i][j] == 1){
                    result -= log(y.matrix[i][j]);
                    break;
                }else{
                    continue;
                }
            }
        }
        return result;
    }

    Matrix backward(Matrix &y, Matrix &target) override {
        Matrix result = Matrix(y.row(), y.col(), 0);
        for (size_t i = 0; i < y.row(); i++)
            for (size_t j = 0; j < y.row(); j++)
                result.matrix[i][j] = target.matrix[i][j] ? y.matrix[i][j] - 1 : y.matrix[i][j];

        return result;
    }
};

// loss function - end


// active function - start

class ActiveFunc{
public:
    virtual Matrix func_forward(Matrix x) = 0;
    virtual Matrix func_backward(Matrix x) = 0;

};

class Relu: public ActiveFunc{
public:
    Matrix func_forward(Matrix x) override {
        Matrix result = Matrix(x.row(), x.col(), 0);
        for (size_t i = 0; i<x.row(); i++){
            for (size_t j = 0; j<x.col(); j++){
                    result.matrix[i][j] = x.matrix[i][j] > 0 ? x.matrix[i][j] : 0;
            }
        }
        return result;
    }

    Matrix func_backward(Matrix x) override {
        Matrix result = Matrix(x.row(), x.col(), 0);
        for (size_t i = 0; i<x.row(); i++){
            for (size_t j = 0; j<x.col(); j++){
                result.matrix[i][j] = x.matrix[i][j] > 0 ? 1 : 0;
            }
        }
        return result;
    }
};

class Sigmoid: public ActiveFunc{
public:
    Matrix func_forward(Matrix x) override {
        Matrix result(x.row(), x.col(), 0);

        for (int i = 0; i< x.row(); i++){
            for (int j = 0; j< x.col(); j++) {
                result.matrix[i][j] = 1 / (1 + exp(-x.matrix[i][j]));

            }
        }
        return result;
    }

    Matrix func_backward(Matrix x) override {
        Matrix a = func_forward(x);
        Matrix b = Matrix::reduce(&a, 1);
        b = Matrix::times(&b, -1);
        return Matrix::times(&a, &b);
    }
};

class SoftMax_CrossEntropy: public ActiveFunc{
public:
    Matrix func_forward(Matrix x) override {
        Matrix result = Matrix(x.row(), x.col(), 0);
        Matrix total = Matrix(x.row(), 1, 0);
        Matrix max = Matrix(x.row(), 1, 0);
        for (size_t i=0;i<x.row();i++)
            for (size_t j=0;j<x.col();j++)
                max.matrix[i][0] = x.matrix[i][j] > max.matrix[i][0] ? x.matrix[i][j] : max.matrix[i][0];

        for (size_t i=0;i<x.row();i++)
            for (size_t j=0;j<x.col();j++)
                total.matrix[i][0] += exp(x.matrix[i][j] - max.matrix[i][0]);

        for (size_t i=0;i<x.row();i++)
            for (size_t j=0;j<x.col();j++)
                result.matrix[i][j] = exp(x.matrix[i][j] - max.matrix[i][0]) / total.matrix[i][0];



        return result;
    }

    Matrix func_backward(Matrix x) override {
        return Matrix(x.row(), x.col(), 1);
    }
};

class Tanh:public ActiveFunc{
public:
    Matrix func_forward(Matrix x) override {
        Matrix result(x.row(), x.col(), 0);

        for (int i = 0; i< x.row(); i++){
            for (int j = 0; j< x.col(); j++) {
                double a = exp(x.matrix[i][j]);
                double b = exp(-x.matrix[i][j]);
//                double c = (a - b) / (b - a);
                result.matrix[i][j] = (a - b) / (b - a);
            }
        }
        return result;
    }

    Matrix func_backward(Matrix x) override {
        Matrix result(x.row(), x.col(), 0);

        for (int i = 0; i< x.row(); i++){
            for (int j = 0; j< x.col(); j++) {
                double a = exp(x.matrix[i][j]);
                double b = exp(-x.matrix[i][j]);
                double c = (a - b) / (b - a);
                result.matrix[i][j] = 1 - c * c;
            }
        }


        return result;
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
        double _temp = eta / w.row();

        Matrix temp_w = Matrix::times(&grad_w, _temp);
        w = Matrix::reduce(&w, &temp_w);

        Matrix temp_b = Matrix::times(&grad_b, _temp);
        b = Matrix::reduce(&b, &temp_b);
    }

};

class MMT: public Optimizer{
public:
    double eta;
    double beta = 0.9;
    Matrix last_grad_w = Matrix();
    Matrix last_grad_b = Matrix();

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
        if (last_grad_w.row() == 0){
            last_grad_w = Matrix(grad_w.row(), grad_w.col(), 0);
            last_grad_b = Matrix(grad_b.row(), grad_b.col(), 0);
        }

        Matrix temp_a = Matrix::times(&grad_w, eta);
        Matrix temp_b = Matrix::times(&last_grad_w, beta);
        last_grad_w = Matrix::add(&temp_a, &temp_b);
        w = Matrix::reduce(&w, &last_grad_w);

        temp_a = Matrix::times(&grad_b, eta);
        temp_b = Matrix::times(&last_grad_b, beta);
        last_grad_b = Matrix::add(&temp_a, &temp_b);
        b = Matrix::reduce(&b, &last_grad_b);

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

    virtual Matrix forward(Matrix _x, bool is_train) = 0;
    virtual Matrix backward(Matrix _delta, bool is_train) = 0;
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

    Matrix construct_random_bool_list(size_t row, size_t col, double probability){
        Matrix result = Matrix(row, col, 0);
        double a = 0;
        double b = 0;
        for(size_t i = 0; i < row; i++){
            for(size_t j = 0; j < col; j++){
                result.matrix[i][j] = (double(rand()) / RAND_MAX < probability) ? 0 : 1;
            }
        }
        return result;
    }

    void init(size_t input_size, size_t output_size, ActiveFunc *_activeFunc, Optimizer *_optimizer){
    }

    Matrix forward(Matrix _x, bool is_train) override {
        x = _x;
        if (!is_train){
            return Matrix::times(&x, (1-dropout_probability));
        }

        dropout_matrix = construct_random_bool_list(x.row(), x.col(), dropout_probability);
        x = Matrix::times(&x, &dropout_matrix);
        return x;
    }

    Matrix backward(Matrix _delta, bool is_train) override {
        if (!is_train){
            return _delta;
        }
        delta = Matrix::times(&_delta, &dropout_matrix);
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
        w = Matrix(input_size, output_size, 0);
        w.random_matrix();
        b = Matrix(1, output_size, 0);
        b.random_matrix();
        grad_w = Matrix(w.row(), w.col(), 0);
        grad_b = Matrix(b.row(), b.col(), 0);
        active_func = _activeFunc;
        optimizer = _optimizer;
    }

    Matrix forward(Matrix _x, bool is_train) override{
        x = _x;
        u = Matrix::dot(&x, &w);
        u = Matrix::add(&u, &b);
        y = active_func->func_forward(u);
        return y;
    }

    Matrix backward(Matrix _delta, bool is_train) override{

        Matrix active_func_back = active_func->func_backward(u);
        Matrix my_delta = Matrix::times(&_delta, &active_func_back);

        Matrix x_t = x.transpose();
        grad_w = Matrix::dot(&x_t, &my_delta);

        Matrix w_t = w.transpose();
        delta = Matrix::dot(&my_delta, &w_t);
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

    void train(size_t epoch, Matrix &x, Matrix &target){  // 這裡擴充batch size
        size_t _batch = batch == -1 ? x.row() : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.row();  // 存著還有幾筆資料需要訓練
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

    vector<vector<double>> temp_x =
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
    vector<vector<double>> temp_target =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 0, 1}};

    vector<vector<double>> temp_validation =
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

    vector<vector<double>> temp_validation_target =
            {{1, 0, 0, 0, 0},
             {0, 1, 0, 0, 0},
             {0, 0, 1, 0, 0},
             {0, 0, 0, 1, 0},
             {0, 0, 0, 1, 0}};



//    vector<vector<double>> temp_x = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
//    vector<vector<double>> temp_target = {{0}, {1}, {1}, {0}};
    // data init
    Matrix x = Matrix(temp_x);
    Matrix x_target = Matrix(temp_target);
    Matrix validation = Matrix(temp_validation);
    Matrix validation_target = Matrix(temp_validation_target);

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

    frame.train(6000, x, x_target);


    frame.show(validation, validation_target);
    frame.show(x, x_target);
    return 0;
}
