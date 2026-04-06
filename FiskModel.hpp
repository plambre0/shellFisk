#ifndef FISK_MODEL_HPP
#define FISK_MODEL_HPP

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

struct FiskModel {
    std::string model_name;
    std::string dataset_name;
    std::string formula;
    std::string family;
    std::string link = "identity";
    bool success = false;
    int df_residual = 0;
    int df_null = 0;
    int iterations = 0;
    double aic = 0.0;
    double deviance = 0.0;
    double null_deviance = 0.0;
    double log_likelihood = 0.0;
    double dispersion = 1.0;

    std::vector<std::string> term_names;
    Eigen::VectorXd coefficients;
    Eigen::VectorXd random_effects;
    Eigen::VectorXd fitted;
    Eigen::VectorXd residuals;
    Eigen::VectorXd std_errors;
    Eigen::VectorXd z_values;
    Eigen::VectorXd p_values;
    Eigen::MatrixXd vcov_matrix;
    Eigen::MatrixXd design_matrix; // Store design matrix for predictions
    Eigen::VectorXd response_vector; // Store response for diagnostics

    static std::string abbrev(const std::string& s, size_t len) {
        if (s.size() <= len) return s;
        if (len <= 3) return s.substr(0, len);
        return s.substr(0, len - 3) + "...";
    }

    void plot_residuals() const {
        if (fitted.size() != residuals.size() || fitted.size() == 0) return;

        const int width = 80;
        const int height = 20;
        std::vector<std::string> grid(height, std::string(width, ' '));

        double x_min = *std::min_element(fitted.data(), fitted.data() + fitted.size());
        double x_max = *std::max_element(fitted.data(), fitted.data() + fitted.size());
        double y_min = *std::min_element(residuals.data(), residuals.data() + residuals.size());
        double y_max = *std::max_element(residuals.data(), residuals.data() + residuals.size());

        double x_range = x_max - x_min;
        double y_range = y_max - y_min;
        if (x_range == 0) x_range = 1;
        if (y_range == 0) y_range = 1;

        for (int i = 0; i < fitted.size(); ++i) {
            int x = static_cast<int>((fitted[i] - x_min) / x_range * (width - 1));
            int y = static_cast<int>((y_max - residuals[i]) / y_range * (height - 1)); // invert y
            x = std::max(0, std::min(width - 1, x));
            y = std::max(0, std::min(height - 1, y));
            grid[y][x] = '*';
        }

        std::cout << "\nResiduals vs Fitted Plot:\n";
        std::cout << std::string(width + 2, '-') << "\n";
        for (const auto& row : grid) {
            std::cout << "|" << row << "|\n";
        }
        std::cout << std::string(width + 2, '-') << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Fitted: [" << x_min << ", " << x_max << "]  Residuals: [" << y_min << ", " << y_max << "]\n\n";
    }

    void print_summary() const {
        std::cout << "Call:\n";
        std::cout << "  glm(formula = " << abbrev(formula, 60) << ", family = " << family
                  << ", link = " << link << ")\n\n";

        if (residuals.size() > 0) {
            std::vector<double> sorted(residuals.data(), residuals.data() + residuals.size());
            std::sort(sorted.begin(), sorted.end());
            auto quantile = [&](double q) {
                if (sorted.empty()) return 0.0;
                size_t idx = std::min(sorted.size() - 1, static_cast<size_t>(q * (sorted.size() - 1)));
                return sorted[idx];
            };
            std::cout << "Deviance Residuals:\n";
            std::cout << std::setw(12) << "Min" << std::setw(12) << "1Q" << std::setw(12) << "Median"
                      << std::setw(12) << "3Q" << std::setw(12) << "Max" << "\n";
            std::cout << std::setw(12) << sorted.front()
                      << std::setw(12) << quantile(0.25)
                      << std::setw(12) << quantile(0.50)
                      << std::setw(12) << quantile(0.75)
                      << std::setw(12) << sorted.back() << "\n\n";
        }

        if (coefficients.size() > 0) {
            const std::string stat_label = (family == "gaussian") ? "t value" : "z value";
            std::cout << std::setw(20) << "Term" << std::setw(15) << "Estimate" << std::setw(15) << "Std. Error"
                      << std::setw(12) << stat_label << std::setw(15) << "Pr(>|z|)" << "\n";
            std::cout << std::string(77, '-') << "\n";
            for (int i = 0; i < coefficients.size(); ++i) {
                std::string term = (i < term_names.size()) ? abbrev(term_names[i], 20) : ("beta_" + std::to_string(i));
                double se = (i < std_errors.size()) ? std_errors[i] : 0.0;
                double z = (i < z_values.size()) ? z_values[i] : 0.0;
                double p = (i < p_values.size()) ? p_values[i] : 0.0;
                std::cout << std::setw(20) << term
                          << std::setw(15) << coefficients[i]
                          << std::setw(15) << se
                          << std::setw(12) << z
                          << std::setw(15) << p << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "Dispersion parameter for " << family << " family taken to be " << dispersion << "\n";
        std::cout << "Null deviance: " << null_deviance << "  on " << df_null << " degrees of freedom\n";
        std::cout << "Residual deviance: " << deviance << "  on " << df_residual << " degrees of freedom\n";
        std::cout << "AIC: " << aic << "\n";
        std::cout << "Log-likelihood: " << log_likelihood << "\n";
        std::cout << "Number of Fisher Scoring iterations: " << iterations << "\n";
        plot_residuals();
        print_performance();
    }

    static double p_from_z(double z) {
        return 2.0 * std::erfc(std::abs(z) / std::sqrt(2.0));
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& new_X) const {
        Eigen::VectorXd eta = new_X * coefficients;
        
        if (family == "binomial") {
            return eta.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
        } else if (family == "poisson" || family == "negbinom" || family == "gamma" || family == "invgaussian" || family == "weibull") {
            return eta.unaryExpr([](double x) { return std::exp(x); });
        } else {
            return eta; // identity link (gaussian)
        }
    }

    double rmse() const {
        if (residuals.size() == 0) return 0.0;
        return std::sqrt(residuals.squaredNorm() / residuals.size());
    }

    double mae() const {
        if (residuals.size() == 0) return 0.0;
        return residuals.cwiseAbs().mean();
    }

    double bic() const {
        int p = coefficients.size();
        int n = response_vector.size();
        if (n == 0) return 0.0;
        return 2.0 * p * std::log(n) - 2.0 * log_likelihood;
    }

    void print_performance() const {
        std::cout << "\n=== Model Performance Metrics ===\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "AIC:  " << aic << "\n";
        std::cout << "BIC:  " << bic() << "\n";
        std::cout << "RMSE: " << rmse() << "\n";
        std::cout << "MAE:  " << mae() << "\n";
        std::cout << "Deviance: " << deviance << "\n";
        std::cout << "Null Deviance: " << null_deviance << "\n";
        if (null_deviance > 0) {
            double pseudo_r2 = 1.0 - (deviance / null_deviance);
            std::cout << "McFadden's Pseudo R²: " << pseudo_r2 << "\n";
        }
        std::cout << "\n";
    }
};

#endif