#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;//优化变量数据
    std::vector<int> drop_set;//待边缘化的优化变量id

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;//残差 IMU:15X1 视觉2X1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors;//所有观测项
    int m, n;//m为要边缘化的变量个数，n为要保留下来的变量个数
    std::unordered_map<long, int> parameter_block_size; //<优化变量内存地址,localSize>
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //<待边缘化的优化变量内存地址,在舒尔补A矩阵中行列式的起始下标> 可以快速寻找A矩阵中的child块位置
    std::unordered_map<long, double *> parameter_block_data;//<优化变量内存地址,数据> 这里的数据是new出来的，跟优化变量内存地址不一样，里面保存的是上一轮优化后的值
                                                            //"不同残差对同一个状态求雅克比时，线性化点必须一致", 这里的线性化点就是当前的最优解，单独保存到parameter_block_data中

    std::vector<int> keep_block_size; //marg后保留的优化变量块代表的变量个数，从parameter_block_size中复制
    std::vector<int> keep_block_idx;  //marg后保留的优化变量在舒尔补A矩阵中行列式的起始下标，也是残差变量b中的下标，从parameter_block_idx中复制
    std::vector<double *> keep_block_data;//marg后保留的优化变量地址，从parameter_block_data中复制，里面的值是优化后的值，跟下一轮ceres优化时迭代计算残差用。不是用来存放ceres优化求解后的结果。

    //https://www.sohu.com/a/301868161_715754 https://blog.csdn.net/heyijia0327/article/details/52822104 两个文章最后都对这两个量有详细描述：”线性化点必须一致"
    //https://blog.csdn.net/weixin_41394379/article/details/89975386 证明过程最清楚。其中Jl​为linearized_jacobians，e0为linearized_residuals
    //这两者会作为先验残差带入到下一轮的先验残差的雅克比和残差的计算当中去，就是上一轮优化后的结果值处展开对应的雅克比和残差值（上一轮结果值为parameter_block_data)
    //进入下一轮的ceres求最优解时作为约束，迭代的解相对keep_block_data的变化量引起的残差量应该刚好把linearized_residuals残差抵消掉
    //"不同残差对同一个状态求雅克比时，线性化点必须一致" 就是通过这两个变量实现的
    //下一轮的ceres求最优解见 MarginalizationFactor::Evaluate()
    Eigen::MatrixXd linearized_jacobians; //舒尔补边缘化后的信息矩阵H中恢复出来线性化点的原始雅克比矩阵J   H=J^T x J    舒尔补是对 Hδx=b 进行的
    Eigen::VectorXd linearized_residuals; //舒尔补边缘化后的残差矩阵b中恢复出来线性化点的原始残差e        b=J^T x e
    const double eps = 1e-8;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
