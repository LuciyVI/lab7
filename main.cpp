#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <set>
#include <fstream>
#include <list>

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

double generateUniqueRandomDouble(double min, double max) {
   
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    static std::set<double> generated;

    double num;
    
        do {
            num = dis(gen);
        } while (generated.count(num) > 0);

        generated.insert(num);
    
    return num;
}


class Matrix_Trust_Agent {

    public:

        Eigen::MatrixXd M;

        Matrix_Trust_Agent(int N) : M(N,N) {

        
            for (int i = 0; i < M.rows(); ++i) {
        
            double min= 0.0;
            double max= 1.0;

                for (int j = 0; j < M.cols(); ++j) {
                    M(i, j) = generateUniqueRandomDouble(min,max);
                    max=max-M(i,j);

                }
            }
 
        }
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix_Trust_Agent& obj) {
        os << obj.M;
        
        return os;
    }
};
bool matrix_verification(Eigen::MatrixXd mat, double eps) {
   
    bool status = false;
   
    for (int i = 0; i < mat.rows(); i++) {
        
        Eigen::VectorXd rows= mat.row(i);
      
        double min_value = rows.minCoeff(); 
        double max_value = rows.maxCoeff();
        
        
        
        std::cout<<std::fixed<<max_value<<"  "<<min_value<<std::endl;


        std::cout<<"\n "<<max_value<<"-"<<min_value<<"="<<max_value-min_value<<std::endl;
    
        if ((max_value - min_value) <= eps) {
        
            status = true;
            break;
        
        }

    }
    
    return status;
}

bool isConvergent(const Eigen::MatrixXd& A, double eps) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    return svd.singularValues()(0) - svd.singularValues()(A.rows() - 1) < eps;
}


int main() {
     
    Eigen::VectorXd agent_opinion(10);

    for(int i=0; i<agent_opinion.size();i++){

        double randomValue=generateUniqueRandomDouble(1,20);
        double roundedValue = std::round(randomValue); // Округление до целого числа
        agent_opinion[i] = roundedValue;

    }

    Matrix_Trust_Agent Agent1(10);
    Eigen::MatrixXd verif_mat(10,10);
    
    verif_mat=Agent1.M;
       

std::list<VectorXd>listOfVectors; // Создание списка векторов

while(!isConvergent(verif_mat,0.001))
        {   
            listOfVectors.push_back((verif_mat * agent_opinion));
            verif_mat = Agent1.M * verif_mat;
            std::cout<<verif_mat<<std::endl;
        }

std::ofstream file("output.txt");

    for(int i = 1; i<agent_opinion.size();)
        {   
        if (i % 10 == 0 && i != 0) {
 
            file << std::endl;
        }
            file << agent_opinion(i) << " ";
            i++;

        }

    return 0;  

}