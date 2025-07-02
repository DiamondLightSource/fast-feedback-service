#pragma once
#include <Eigen/Dense>
#include <cmath>

using Eigen::Matrix3d;
using Eigen::Vector3d;

Matrix3d dR_from_axis_and_angle(Vector3d axis_, double angle) {
    axis_.normalize();
    double ca = cos(angle);
    double sa = sin(angle);
    return Matrix3d{{sa * axis_[0] * axis_[0] - sa,
                        sa * axis_[0] * axis_[1] - ca * axis_[2],
                        sa * axis_[0] * axis_[2] + ca * axis_[1]},
                        {sa * axis_[1] * axis_[0] + ca * axis_[2],
                        sa * axis_[1] * axis_[1] - sa,
                        sa * axis_[1] * axis_[2] - ca * axis_[0]},
                        {sa * axis_[2] * axis_[0] - ca * axis_[1],
                        sa * axis_[2] * axis_[1] + ca * axis_[0],
                        sa * axis_[2] * axis_[2] - sa}};
                                             }

// axis and angle as rot mat
Matrix3d axis_and_angle_as_rot(Vector3d axis, double angle){
    double q0=0.0;
    double q1=0.0;
    double q2=0.0;
    double q3=0.0;
    if (!(std::fmod(angle, 2.0*M_PI))){
        q0=1.0;
    }
    else {
        double h = 0.5 * angle;
        q0 = cos(h);
        double s = sin(h);
        axis.normalize();
        q1 = axis[0]*s;
        q2 = axis[1]*s;
        q3 = axis[2]*s;
    }
    Matrix3d m{
        {2*(q0*q0+q1*q1)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)},
        {2*(q1*q2+q0*q3),   2*(q0*q0+q2*q2)-1, 2*(q2*q3-q0*q1)},
        {2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   2*(q0*q0+q3*q3)-1}};
    return m;
}
