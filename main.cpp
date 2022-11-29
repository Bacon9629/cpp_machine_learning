#include <iostream>
#include <time.h>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include "Matrix.h"

using namespace std;

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
        Matrix *result = new Matrix(y.matrix, y.shape[2], y.shape[3],  true);
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
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? temp : 0;
        }

//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                double temp = x.get(i, j);
//                result->get(i, j) = temp > 0 ? temp : 0;
//            }
//        }
        return *result;
    }

    Matrix& func_backward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
        double* result_matrix_pointer = result->matrix;
        double* x_matrix_pointer = x.matrix;

        for (size_t i = 0; i < x.size_1d; i++){
            double temp = x_matrix_pointer[i];
            result_matrix_pointer[i] = temp > 0 ? 1 : 0;
        }

//        for (size_t i = 0; i < x.shape[2]; i++){
//            for (size_t j = 0; j < x.shape[3]; j++){
//                result->get(i, j) = x.get(i, j) > 0 ? 1 : 0;
//            }
//        }
        return *result;
    }
};

class Sigmoid: public ActiveFunc{
public:
    Matrix& func_forward(Matrix &x) override {
        Matrix *result = new Matrix(x.shape[2], x.shape[3], 0, true);
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
        delta = _delta * dropout_matrix;

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
        grad_b = _delta.row_sum();

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
        size_t _batch = batch == -1 ? x.shape[0] : batch;

        for (size_t i = 0; i < epoch; i++){
            size_t data_left_size = x.shape[0];  // 存著還有幾筆資料需要訓練
//            train_one_time(x, target);
//            continue;

            for (size_t j = 0; data_left_size > 0 ; j++){
                if (data_left_size < _batch){
                    // 如果資料量"不足"填滿一個batch
                    Matrix _x = Matrix::getPicture_row(x, j * _batch, j * _batch + data_left_size);
                    Matrix _target = Matrix::getPicture_row(target, j * _batch, j * _batch + data_left_size);

                    train_one_time(_x, _target);
                    data_left_size = 0;
                }else{
                    // 如果資料量"足夠"填滿一個batch
                    Matrix _x = Matrix::getPicture_row(x, j * _batch, j * _batch + _batch);
                    Matrix _target = Matrix::getPicture_row(target, j * _batch, j * _batch + _batch);

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

//                    cout << i << endl;
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

void img_train(){
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

    x.reshape(1, 1, 5, 25);
    x_target.reshape(1, 1, 5, 5);
    validation.reshape(1, 1, 5, 25);
    validation_target.reshape(1, 1, 5, 5);

    // active func
    Sigmoid sigmoid = Sigmoid();
    SoftMax_CrossEntropy softmax = SoftMax_CrossEntropy();
    Tanh tanh = Tanh();
    Relu relu = Relu();

    /**
     * loss function: cross entropy with softmax
     * */
    MyFrame frame = MyFrame(new CrossEntropy_SoftMax, -1);


    frame.add(new DenseLayer(25, 32, &relu, new MMT(0.1)));
//    frame.add(new DenseLayer(64, 32, &softmax, new SGD(0.1)));
    frame.add(new DropoutLayer(32, 32, &softmax, new MMT(0.001), 0.65));
    frame.add(new DenseLayer(32, 5, &softmax, new MMT(0.1)));

    frame.train(6000, x, x_target);


    frame.show(validation, validation_target);
    frame.show(x, x_target);
}

void type_a_train(){

    double temp_x[][3] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    double temp_target[][2] = {{0, 1}, {0, 1}, {1, 0}, {1, 0}};

    Matrix x(temp_x[0], 4, 3);
    Matrix target(temp_target[0], 4, 2);

    MyFrame myFrame(new CrossEntropy_SoftMax(), -1);

    myFrame.add(new DenseLayer(3, 32, new Relu(), new SGD(0.3)));
//    myFrame.add(new DenseLayer(32, 32, new Relu(), new SGD(0.3)));
    myFrame.add(new DenseLayer(32, 2, new SoftMax_CrossEntropy(), new SGD(0.3)));

    myFrame.train(3000, x, target);

    myFrame.show(x, target);


}

int main() {
//    type_a_train();
    img_train();
    return 0;
}
