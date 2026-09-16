#pragma once

#include <Eigen/Core>
#include <vector>
#include "ellipsoid_parameterisation.hpp"
#include "target.hpp"

using ParameterVector = Eigen::Matrix<double, 6, 1>;
using FisherMatrix = Eigen::Matrix<double, 6, 6>;

struct RefinementStep {
    ParameterVector parameters;
    double log_likelihood;
    double mse;
};

class FisherScoringMaximumLikelihood {
public:

    FisherScoringMaximumLikelihood(
        Simple6MosaicityParameterisation& model,
        MaximumLikelihoodTarget& target,
        int max_iter = 1000,
        double tolerance = 1e-7,
        double ll_tolerance = 1e-6)
        :
        model_(model),
        target_(target),
        parameters_(model.parameters()),
        max_iter_(max_iter),
        tolerance_(tolerance),
        ll_tolerance_(ll_tolerance)
    {}

    void solve();

    const ParameterVector& parameters() const {
        return parameters_;
    }

    const std::vector<RefinementStep>& history() const {
        return history_;
    }

private:

    double log_likelihood(
        const ParameterVector& x);

    ParameterVector score(
        const ParameterVector& x);

    std::pair<ParameterVector,FisherMatrix>
    score_and_fisher_information(
        const ParameterVector& x);

    ParameterVector solve_update_equation(
        const ParameterVector& S,
        const FisherMatrix& I) const;

    double line_search(
        const ParameterVector& x,
        const ParameterVector& p,
        double tau = 0.5,
        double delta = 1.0);

    ParameterVector gradient_search(
        const ParameterVector& x);

    bool test_LL_convergence() const;

    void callback(
        const ParameterVector& x);

private:

    Simple6MosaicityParameterisation& model_;
    MaximumLikelihoodTarget& target_;

    ParameterVector parameters_;

    int max_iter_;
    double tolerance_;
    double ll_tolerance_;

    std::vector<RefinementStep> history_;
};
