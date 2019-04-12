#include <iostream>
#include <functional>

#include "optim.hpp"
#include "armadillo"

using namespace std::placeholders;

class Box {
    public:

        double length;
        double width;
        double height;

        double getVolume(void);
        //double optimizeVolume(const arma::vec&, arma::vec*, void*);
        double optimizeVolume(const arma::vec&, arma::vec*, void*);
        void setLength(double);
        void setWidth(double);
        void setHeight(double);
};

double Box::getVolume(void){
    return length*width*height;
}

//double Box::optimizeVolume(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data){
double Box::optimizeVolume(const arma::vec& vals_inp, arma::vec* grad_out, void* opt_data){    
    return vals_inp[0]*vals_inp[0];//*vals_inp[1]*vals_inp[2];
}

void Box::setLength(double len){
    length = len;
}

void Box::setWidth(double wid){
    width = wid;
}

void Box::setHeight(double hei){
    height = hei;
}

int main(){
    Box box1;
    Box box2;
    double volume = 0.0;

    // Info for box1
    box1.setLength(6.0);
    box1.setWidth(7.0);
    box1.setHeight(5.0);

    // Info for box2
    box2.setLength(12.0);
    box2.setWidth(13.0);
    box2.setHeight(10.0);

    // Volume of box1
    volume = box1.getVolume(); //210
    std::cout << "Box1 volume: " << volume << std::endl;

    // Volume of box2
    volume = box2.getVolume(); //1560
    std::cout << "Box2 volume: " << volume << std::endl;

    //optim stuff
    arma::vec x_1 = arma::zeros(3,1)+ 5.0;

    optim::algo_settings_t settings_1;
    //settings_1.vals_bound = true;
    settings_1.lower_bounds = arma::zeros(3,1) + 0.5;
    settings_1.upper_bounds = arma::zeros(3,1) + 10.0;
    settings_1.pso_initial_lb = arma::zeros(3,1) - 10.0;
    settings_1.pso_initial_ub = arma::zeros(3,1) + 10.0;


    arma::vec* grad_out;
    void* opt_data;

    // Info for box3
    Box box3;
    box3.setLength(1.0);
    box3.setWidth(1.0);
    box3.setHeight(1.0);

    //Setting up a pointer to a member function of class Box
    //std::function<double(void)> BoxPtr = std::bind(&Box::optimizeVolume, &box3);
    std::function<double(const arma::vec&, arma::vec*, void*)> BoxPtr = std::bind(&Box::optimizeVolume, &box3, _1, _2, _3);

    std::cout << "BoxPtr: " << BoxPtr(x_1, grad_out, opt_data) << std::endl;


    //bool success = true;
    bool success = optim::pso(x_1, BoxPtr, nullptr, settings_1);

    if (success){
        std::cout << x_1 << std::endl;
        std::cout << "BoxPtr: " << BoxPtr(x_1, grad_out, opt_data) << std::endl;
    }

    return 0;
}