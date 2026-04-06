#ifndef FISK_GLM_HPP
#define FISK_GLM_HPP

#include "FiskData.hpp"
#include "FiskModel.hpp"
#include "FiskParser.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

enum Family { GAUSSIAN, BINOMIAL, POISSON, NEGBINOM, GAMMA, INVGAUSSIAN, WEIBULL };

class FiskGLM {
public:
    static FiskModel execute_mixed(const FiskData& fd, const std::string& ds_name,
                                   const std::string& formula_str, Family fam, const std::string& model_name="") {
        FiskModel m;
        m.dataset_name = ds_name;
        m.formula = formula_str;
        m.model_name = model_name.empty() ? ds_name : model_name;
        
        std::string fam_str = (fam == BINOMIAL) ? "binomial" : 
                              (fam == POISSON) ? "poisson" :
                              (fam == NEGBINOM) ? "negbinom" :
                              (fam == GAMMA) ? "gamma" :
                              (fam == INVGAUSSIAN) ? "invgaussian" :
                              (fam == WEIBULL) ? "weibull" : "gaussian";
        m.family = fam_str;
        
        std::string link = (fam == BINOMIAL) ? "logit" : 
                          (fam == GAMMA) ? "log" :
                          (fam == INVGAUSSIAN) ? "1/mu^2" :
                          (fam == WEIBULL) ? "log" :
                          (fam == POISSON || fam == NEGBINOM) ? "log" : "identity";
        m.link = link;

        FiskParser::Design d = FiskParser::parse_formula(fd, formula_str);
        if (!d.success) return m;

        m.term_names = d.names;
        int n = fd.rows;
        int p = d.X.cols();
        int q = d.Z.cols();
        m.df_null = n - 1;
        m.df_residual = n - p;
        m.design_matrix = d.X;
        m.response_vector = d.y;

        Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
        Eigen::VectorXd u = Eigen::VectorXd::Zero(q);
        Eigen::VectorXd eta = d.X * beta;
        Eigen::VectorXd mu = Eigen::VectorXd::Zero(n);
        const double eps = 1e-9;
        const double pi = 3.14159265358979323846;

        try {
            Eigen::VectorXd weights(n);
            for (int iter = 0; iter < 25; ++iter) {
                Eigen::VectorXd beta_old = beta;

                for (int i = 0; i < n; ++i) {
                    if (fam == BINOMIAL) {
                        mu(i) = 1.0 / (1.0 + std::exp(-eta(i)));
                        weights(i) = std::max(mu(i) * (1.0 - mu(i)), eps);
                    } else if (fam == POISSON) {
                        mu(i) = std::exp(eta(i));
                        weights(i) = std::max(mu(i), eps);
                    } else if (fam == NEGBINOM) {
                        mu(i) = std::exp(eta(i));
                        weights(i) = std::max(mu(i), eps);
                    } else if (fam == GAMMA) {
                        mu(i) = std::exp(eta(i));
                        weights(i) = std::max(mu(i) * mu(i), eps);
                    } else if (fam == INVGAUSSIAN) {
                        mu(i) = std::exp(eta(i));
                        weights(i) = std::max(mu(i) * mu(i) * mu(i), eps);
                    } else if (fam == WEIBULL) {
                        mu(i) = std::exp(eta(i));
                        weights(i) = std::max(mu(i), eps);
                    } else {
                        mu(i) = eta(i);
                        weights(i) = 1.0;
                    }
                }

                Eigen::VectorXd z = eta + (d.y - mu).cwiseQuotient(weights);
                Eigen::MatrixXd W = weights.asDiagonal();

                beta = (d.X.transpose() * W * d.X).ldlt().solve(d.X.transpose() * W * z);

                if (q > 0) {
                    Eigen::MatrixXd ZWZ = d.Z.transpose() * W * d.Z + Eigen::MatrixXd::Identity(q, q);
                    u = ZWZ.ldlt().solve(d.Z.transpose() * W * (z - d.X * beta));
                }

                eta = d.X * beta;
                if (q > 0) eta += d.Z * u;

                if ((beta - beta_old).norm() < 1e-6) {
                    m.iterations = iter + 1;
                    break;
                }
                m.iterations = iter + 1;
            }

            m.coefficients = beta;
            m.random_effects = u;
            m.fitted = mu;
            m.residuals = d.y - mu;
            m.success = true;

            Eigen::MatrixXd Wfinal = weights.asDiagonal();
            m.vcov_matrix = (d.X.transpose() * Wfinal * d.X).inverse();
            m.std_errors = m.vcov_matrix.diagonal().cwiseSqrt();
            Eigen::VectorXd se = m.std_errors;
            for (int i = 0; i < se.size(); ++i) if (se[i] == 0.0) se[i] = eps;
            m.z_values = m.coefficients.cwiseQuotient(se);
            m.p_values = m.z_values.unaryExpr([](double z) { return std::erfc(std::abs(z) / std::sqrt(2.0)); });

            if (fam == GAUSSIAN) {
                m.deviance = m.residuals.squaredNorm();
                Eigen::VectorXd mu0 = Eigen::VectorXd::Constant(n, d.y.mean());
                m.null_deviance = (d.y - mu0).squaredNorm();
                m.dispersion = m.deviance / std::max(1, m.df_residual);
                m.log_likelihood = -0.5 * (n * std::log(2.0 * pi * m.dispersion) + m.deviance / m.dispersion);
            } else if (fam == BINOMIAL) {
                double ll = 0.0;
                double null_ll = 0.0;
                double mu0 = std::max(eps, std::min(1.0 - eps, d.y.mean()));
                for (int i = 0; i < n; ++i) {
                    double y = d.y(i);
                    ll += y * std::log(mu(i) + eps) + (1.0 - y) * std::log(1.0 - mu(i) + eps);
                    null_ll += y * std::log(mu0) + (1.0 - y) * std::log(1.0 - mu0);
                }
                m.log_likelihood = ll;
                m.deviance = -2.0 * ll;
                m.null_deviance = -2.0 * null_ll;
                m.dispersion = 1.0;
            } else if (fam == POISSON || fam == NEGBINOM || fam == GAMMA || fam == INVGAUSSIAN || fam == WEIBULL) {
                double ll = 0.0;
                double null_ll = 0.0;
                double mu0 = std::max(eps, d.y.mean());
                for (int i = 0; i < n; ++i) {
                    double y = d.y(i);
                    ll += y * std::log(mu(i) + eps) - mu(i);
                    null_ll += y * std::log(mu0) - mu0;
                }
                m.log_likelihood = ll;
                m.deviance = -2.0 * ll;
                m.null_deviance = -2.0 * null_ll;
                m.dispersion = 1.0;
            } else {
                m.deviance = m.residuals.squaredNorm();
                m.log_likelihood = -0.5 * m.deviance;
                m.dispersion = 1.0;
            }

            m.aic = 2.0 * p - 2.0 * m.log_likelihood;
        } catch (...) {
            std::cerr << "[!] Math Error: Model failed to converge.\n";
            m.success = false;
        }

        return m;
    }
};

#endif