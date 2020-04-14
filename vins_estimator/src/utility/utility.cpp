#include "utility.h"

// 得到把当前重力g向量旋转到Z轴上的旋转矩阵，该矩阵不需要考虑绕Z轴旋转的yaw角(pitch,roll两个维度就可以到达旋转到Z的效果了)
// 也可以理解为把惯导坐标系中的重力g向量转换到Z轴向上的世界坐标系的旋转：R_w_g
// 也可以理解为得到一个旋转矩阵，应用该矩阵可以把惯导坐标系下的向量中的重力g分量跟世界坐标系的Z轴同方向
// 因为惯导硬件安装的原因,即使在水平面上重力方向也不会正好在Z轴上：例如安装X轴向上
Eigen::Matrix3d Utility::g2R(const Eigen::Vector3d &g)
{
    Eigen::Matrix3d R0;
    Eigen::Vector3d ng1 = g.normalized();
    Eigen::Vector3d ng2{0, 0, 1.0};
    R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    // R0 = Utility::ypr2R(Eigen::Vector3d{-90, 0, 0}) * R0;
    return R0;
}
