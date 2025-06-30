#ifndef REFINE_BPARAM
#define REFINE_BPARAM

#include <gemmi/math.hpp> // for symmetric 3x3 matrix SMat33
#include <dx2/crystal.hpp>
#include <Eigen/Dense>
#include <math.h>
using Eigen::Matrix3d;
using Eigen::Vector3d;


double acos_deg(double x) { return 180.0 * std::acos(x) / M_PI; }

gemmi::UnitCell uc_params_from_metrical_matrix(gemmi::SMat33<double> G){
  double p0 = std::sqrt(G.u11);
  double p1 = std::sqrt(G.u22);
  double p2 = std::sqrt(G.u33);
  double p3 = acos_deg(G.u23 / p1 / p2);
  double p4 = acos_deg(G.u13 / p2 / p0);
  double p5 = acos_deg(G.u12 / p0 / p1);
  return gemmi::UnitCell(p0,p1,p2,p3,p4,p5);
}

// Define the BG converter
struct BG {
  // convert orientation matrix B (called A internally here) to metrical
  // matrix g & reverse
  /*The general orientation matrix A is re-expressed in terms of the
    upper-triangular fractionalization matrix F by means of the following
    transformation:
                           F = (D * C * B * A).transpose()
    where D,C,B are three rotation matrices.
  */
  Matrix3d orientation;
  double phi,psi,theta; //in radians
  Matrix3d B,C,D,F;
  gemmi::SMat33<double> G;
  void forward(Matrix3d const& ori){
    orientation = ori;
    Matrix3d A(ori); // i.e. B matrix (unhelpfully called A here)
    phi = std::atan2(A(0,2),-A(2,2));
    B = Matrix3d({
      {std::cos(phi),0.,std::sin(phi)},
      {0.,1.,0.},
      {-std::sin(phi),0.,std::cos(phi)}
    });
    Matrix3d BA(B * A);
    psi = std::atan2(-BA(1,2),BA(2,2));
    C = Matrix3d({
      {1.,0.,0.},
      {0., std::cos(psi),std::sin(psi)},
      {0.,-std::sin(psi),std::cos(psi)}
    });
    Matrix3d CBA (C * BA);

    theta = std::atan2(-CBA(0,1),CBA(1,1));
    D = Matrix3d({
      {std::cos(theta),std::sin(theta),0.},
      {-std::sin(theta),std::cos(theta),0.},
      {0.,0.,1.}
    });
    F = (D * CBA).transpose();
    Matrix3d G9 (A.transpose()*A); //3x3 form of metrical matrix
    G = {G9(0,0),G9(1,1),G9(2,2),G9(0,1),G9(0,2),G9(1,2)};
  }
  void validate_and_setG(gemmi::SMat33<double> const& g){
    // skip validation
    G = {g.u11, g.u22, g.u33, g.u12, g.u13, g.u23};
  }
  Matrix3d back() const {
    gemmi::UnitCell cell = uc_params_from_metrical_matrix(G);
    cell = cell.reciprocal();
    gemmi::Mat33 F = cell.frac.mat;
    Matrix3d Fback;
    Fback << F.a[0][0], 0.0, 0.0,
      F.a[0][1], F.a[1][1], 0.0,
      F.a[0][2], F.a[1][2], F.a[2][2];
    Matrix3d prefact = B.inverse() * C.inverse() * D.inverse();
    return (prefact * Fback);
  }
  Matrix3d back_as_orientation() const {
    return back();
  }
};


std::vector<Matrix3d> calc_dB_dg(BG Bconverter){
    /* note we don't need the first three elements from the equivalent in g_gradients.py

    # G=(g0,g1,g2,g3,g4,g5) 6 elements of the symmetrized metrical matrix,
    #   with current increments already applied ==(a*.a*,b*.b*,c*.c*,a*.b*,a*.c*,b*.c*)*/
    gemmi::SMat33<double> g = Bconverter.G;
    double g0 = g.u11;
    double g1 = g.u22;
    double g2 = g.u33;
    double g3 = g.u12;
    double g4 = g.u13;
    double g5 = g.u23;
    std::vector<Matrix3d> gradients {};

    //# the angles f = phi, p = psi, t = theta, along with convenient trig
    //# expressions
    double f = Bconverter.phi;
    double p = Bconverter.psi;
    double t = Bconverter.theta;

    double cosf = cos(f);
    double sinf = sin(f);
    double cosp = cos(p);
    double sinp = sin(p);
    double cost = cos(t);
    double sint = sin(t);

    double trig1 =  (cosf*cost*sinp)-(sinf*sint);
    double trig2 =  (cost*sinf)+(cosf*sinp*sint);
    double trig3 = (-1.0*cost*sinf*sinp)-(cosf*sint);
    double trig4 =  (cosf*cost)-(sinf*sinp*sint);

    //# Partial derivatives of the 9 A components with respect to metrical matrix elements
    double rad3 = g0-(((g2*g3*g3)+(g1*g4*g4)-(2*g3*g4*g5))/((g1*g2)-(g5*g5)));
    double sqrt_rad3 = sqrt(rad3);

    gradients.push_back(Matrix3d({{
      0.5*trig4/sqrt_rad3,
      0,
      0},
      {0.5*cosp*sint/sqrt_rad3,
      0,
      0},
      {0.5*trig2/sqrt_rad3,
      0,
      0}
    }));

    double fac4 = (g2*g3)-(g4*g5);
    double rad1 = g1-(g5*g5/g2);
    double rad1_three_half = sqrt(rad1*rad1*rad1);
    double fac3 = (g5*g5)-(g1*g2);
    double rad2 = -((g2*g3*g3) +g4*((g1*g4)-(2*g3*g5))+(g0*fac3))/((g1*g2)-(g5*g5));
    double factor_dg1 = (fac4*fac4)/(fac3*fac3*sqrt(rad2));
    double fac5 = g3-(g4*g5/g2);

    gradients.push_back(Matrix3d({{
     -0.5*(fac5*trig3/rad1_three_half) + 0.5*factor_dg1*trig4,
      0.5*trig3/sqrt(rad1),
      0},
     {-0.5*(fac5*cosp*cost/rad1_three_half) + 0.5*factor_dg1*cosp*sint,
      0.5*cosp*cost/sqrt(rad1),
      0},
     {-0.5*(fac5*trig1/rad1_three_half) + 0.5*factor_dg1*trig2,
      0.5*trig1/sqrt(rad1),
      0}
    }));

    double rat5_22 = g5/(g2*g2);
    double fac1 = g5*(g3-(g4*g5/g2));
    double fac2 = ((g1*g4)-(g3*g5));
    double fac2sq = fac2*fac2;

    gradients.push_back(Matrix3d({{
        -0.5*rat5_22*fac1*trig3/rad1_three_half + g4*rat5_22*trig3/sqrt(rad1) +
            0.5*fac2sq*trig4/(fac3*fac3*sqrt(rad2)) + 0.5*g4*cosp*sinf/pow(g2,1.5),
        0.5*rat5_22*(g5*trig3/sqrt(rad1)+sqrt(g2)*cosp*sinf),
        -0.5*cosp*sinf/sqrt(g2)},
        {-0.5*rat5_22*fac1*cosp*cost/rad1_three_half + g4*rat5_22*cosp*cost/sqrt(rad1) +
            0.5*g4*sinp/pow(g2,1.5) + 0.5*(fac2sq/fac3)*cosp*sint/(fac3*sqrt(rad2)),
        0.5*rat5_22*(g5*cosp*cost/sqrt(rad1)+sqrt(g2)*sinp),
        -0.5*sinp/sqrt(g2)},
        {-0.5*rat5_22*fac1*trig1/rad1_three_half + g4*rat5_22*trig1/sqrt(rad1) +
            0.5*fac2sq*trig2/(fac3*fac3*sqrt(rad2)) - 0.5*g4*cosf*cosp/pow(g2,1.5),
        0.5*rat5_22*(g5*trig1/sqrt(rad1)-sqrt(g2)*cosf*cosp),
        0.5*cosf*cosp/sqrt(g2)}}));

    gradients.push_back(Matrix3d({{
        trig3/sqrt(rad1) + fac4*trig4/(fac3*sqrt(rad2)),
        0,
        0},
        {cosp*cost/sqrt(rad1) + fac4*cosp*sint/(fac3*sqrt(rad2)),
        0,
        0},
        {trig1/sqrt(rad1) + fac4*trig2/(fac3*sqrt(rad2)),
        0,
        0}}));

    gradients.push_back(Matrix3d({{
        -g5*trig3/(g2*sqrt(rad1)) + fac2*trig4/(fac3*sqrt(rad2)) - cosp*sinf/sqrt(g2),
        0,
        0},
        {-g5*cosp*cost/(g2*sqrt(rad1)) - sinp/sqrt(g2) + fac2*cosp*sint/(fac3*sqrt(rad2)),
        0,
        0},
        {-g5*trig1/(g2*sqrt(rad1)) + fac2*trig2/(fac3*sqrt(rad2)) + cosf*cosp/sqrt(g2),
        0,
        0}}));

    double better_ratio = (fac2/fac3)*(fac4/fac3);

    gradients.push_back(Matrix3d({{
        fac1*trig3/(g2*rad1_three_half) -g4*trig3/(g2*sqrt(rad1)) +
        better_ratio*trig4/sqrt(rad2),
        -g5*trig3/(g2*sqrt(rad1)) - cosp*sinf/sqrt(g2),
        0},
        {fac1*cosp*cost/(g2*rad1_three_half) - g4*cosp*cost/(g2*sqrt(rad1)) +
        better_ratio*cosp*sint/sqrt(rad2),
        -g5*cosp*cost/(g2*sqrt(rad1)) -sinp/sqrt(g2),
        0},
        {fac1*trig1/(g2*rad1_three_half) - g4*trig1/(g2*sqrt(rad1)) +
        better_ratio*trig2/sqrt(rad2),
        -g5*trig1/(g2*sqrt(rad1))+cosf*cosp/sqrt(g2),
        0}}));
    return gradients;
}

std::vector<Matrix3d> dB_dp(BG Bconverter){
    // If we had symmetry constraints, thes would be applied out here.
    return calc_dB_dg(Bconverter);
}

// A class to manage the translation from B to G and back,
// plus any symmetry constraints (we are sticking to P1 here though.)
class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge();
  void set_orientation(Matrix3d B);
  std::vector<double> forward_independent_parameters();
  Matrix3d backward_orientation(std::vector<double> independent);
  std::vector<Matrix3d> forward_gradients();

private:
  Matrix3d orientation_{};
  BG Bconverter{};
};

SymmetrizeReduceEnlarge::SymmetrizeReduceEnlarge(){}

void SymmetrizeReduceEnlarge::set_orientation(Matrix3d B) {
  orientation_ = B;
}

std::vector<double> SymmetrizeReduceEnlarge::forward_independent_parameters() {
  Bconverter.forward(orientation_);
  std::vector<double> p = {Bconverter.G.u11, Bconverter.G.u22, Bconverter.G.u33, Bconverter.G.u12, Bconverter.G.u13, Bconverter.G.u23};
  return p;
}

Matrix3d SymmetrizeReduceEnlarge::backward_orientation(
  std::vector<double> independent){
  gemmi::SMat33<double> ustar = {independent[0], independent[1], independent[2], independent[3], independent[4], independent[5]};
  Bconverter.validate_and_setG(ustar);
  orientation_ = Bconverter.back_as_orientation();
  return orientation_;
}

std::vector<Matrix3d> SymmetrizeReduceEnlarge::forward_gradients() {
  return dB_dp(Bconverter);
}


class CellParameterisation {
public:
  CellParameterisation(const Crystal& crystal);
  std::vector<double> get_params() const;
  void set_params(std::vector<double>);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0,0.0,0.0,0.0};
  void compose();
  Matrix3d B_{};
  std::vector<Matrix3d> dS_dp{};
  SymmetrizeReduceEnlarge SRE;
};

void CellParameterisation::compose() {
  std::vector<double> vals(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    vals[i] = 1E-5 * params_[i];
  }
  SRE.set_orientation(B_);
  SRE.forward_independent_parameters();
  B_ = SRE.backward_orientation(vals);
  dS_dp = SRE.forward_gradients();
  for (int i = 0; i < dS_dp.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        dS_dp[i](j,k) *= 1E-5;
      }
    }
  }
}

CellParameterisation::CellParameterisation(const Crystal& crystal)
    : B_(crystal.get_B_matrix()), SRE() {
  SRE.set_orientation(B_);
  std::vector<double> X = SRE.forward_independent_parameters();
  params_ = std::vector<double>(X.size());
  for (int i = 0; i < X.size(); ++i) {
    params_[i] = 1E5 * X[i];
  }
  compose();
}

std::vector<double> CellParameterisation::get_params() const {
  return params_;
}
Matrix3d CellParameterisation::get_state() const {
  return B_;
}
void CellParameterisation::set_params(std::vector<double> p) {
  params_ = p;
  compose();
}
std::vector<Matrix3d> CellParameterisation::get_dS_dp() const {
  return dS_dp;
}

#endif  // REFINE_BPARAM