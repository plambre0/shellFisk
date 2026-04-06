#ifndef FISK_TRANSFORM_HPP
#define FISK_TRANSFORM_HPP

#include "FiskData.hpp"
#include <map>
#include <set>
#include <random>
#include <cmath>
#include <stdexcept>

class FiskTransform {
public:
    static void normalize(FiskData& fd, const std::string& col_name) {
        int idx = fd.get_col_idx(col_name);
        if (idx == -1) throw std::runtime_error("Column not found: " + col_name);
        double mean = fd.matrix.col(idx).mean();
        double stddev = std::sqrt((fd.matrix.col(idx).array() - mean).square().sum() / (fd.rows - 1));
        fd.matrix.col(idx) = (fd.matrix.col(idx).array() - mean) / stddev;
    }

    static void factorize(FiskData& fd, const std::string& col_name) {
        int idx = fd.get_col_idx(col_name);
        if (idx == -1) throw std::runtime_error("Column not found: " + col_name);
        std::set<std::string> levels;
        for (int i = 0; i < fd.rows; ++i) levels.insert(fd.raw_strings[i][idx]);
        std::map<std::string,int> mapping;
        int val = 0;
        for (const auto &l : levels) mapping[l] = val++;
        fd.factor_maps[idx] = mapping;
        fd.is_factor[idx] = true;
        for (int i = 0; i < fd.rows; ++i)
            fd.matrix(i,idx) = mapping[fd.raw_strings[i][idx]];
    }

    static void numericize(FiskData& fd, const std::string& col_name) {
        int idx = fd.get_col_idx(col_name);
        if (idx == -1) throw std::runtime_error("Column not found: " + col_name);
        if (!fd.is_factor[idx]) throw std::runtime_error("Column is not a factor: " + col_name);
        fd.is_factor[idx] = false;
        fd.factor_maps.erase(idx);
    }

    static void scale_z(FiskData& fd, const std::string& col_name) {
        int idx = fd.get_col_idx(col_name);
        if (idx == -1) throw std::runtime_error("Column not found: " + col_name);
        Eigen::VectorXd col = fd.matrix.col(idx);
        double mean = col.mean();
        double stddev = std::sqrt((col.array() - mean).square().sum() / (fd.rows - 1));
        if (stddev == 0) throw std::runtime_error("Column has zero variance: " + col_name);
        fd.matrix.col(idx) = (col.array() - mean) / stddev;
    }

    static void impute_mean(FiskData& fd, const std::string& col_name) {
        int idx = fd.get_col_idx(col_name);
        if (idx == -1) throw std::runtime_error("Column not found: " + col_name);
        double sum = 0; int count = 0;
        for (int i = 0; i < fd.rows; ++i)
            if (!std::isnan(fd.matrix(i,idx))) { sum += fd.matrix(i,idx); count++; }
        double mean = sum / count;
        for (int i = 0; i < fd.rows; ++i)
            if (std::isnan(fd.matrix(i,idx))) fd.matrix(i,idx) = mean;
    }

    static void impute_mice(FiskData& fd, int iterations = 5) {
        std::vector<int> missing_cols;
        for (int j = 0; j < fd.cols; ++j) {
            bool has_missing = false;
            for (int i = 0; i < fd.rows; ++i) {
                if (std::isnan(fd.matrix(i, j))) {
                    has_missing = true;
                    break;
                }
            }
            if (has_missing) missing_cols.push_back(j);
        }
        if (missing_cols.empty()) return;

        for (int iter = 0; iter < iterations; ++iter) {
            for (int j : missing_cols) {
                // Collect complete cases for predictors
                std::vector<int> complete_rows;
                for (int i = 0; i < fd.rows; ++i) {
                    bool complete = true;
                    for (int k = 0; k < fd.cols; ++k) {
                        if (k != j && std::isnan(fd.matrix(i, k))) {
                            complete = false;
                            break;
                        }
                    }
                    if (complete) complete_rows.push_back(i);
                }
                if (complete_rows.size() < 2) continue; // not enough data

                int n = complete_rows.size();
                int p = fd.cols - 1; // predictors
                Eigen::MatrixXd X(n, p + 1);
                Eigen::VectorXd y(n);

                int col_idx = 0;
                for (int k = 0; k < fd.cols; ++k) {
                    if (k == j) continue;
                    for (int r = 0; r < n; ++r) {
                        X(r, col_idx) = fd.matrix(complete_rows[r], k);
                    }
                    col_idx++;
                }
                // Intercept
                X.col(p) = Eigen::VectorXd::Ones(n);

                for (int r = 0; r < n; ++r) {
                    y(r) = fd.matrix(complete_rows[r], j);
                }

                // Fit linear regression
                Eigen::VectorXd beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);

                // Impute missing values
                for (int i = 0; i < fd.rows; ++i) {
                    if (std::isnan(fd.matrix(i, j))) {
                        Eigen::VectorXd x_pred(p + 1);
                        int pred_idx = 0;
                        for (int k = 0; k < fd.cols; ++k) {
                            if (k == j) continue;
                            x_pred(pred_idx) = fd.matrix(i, k);
                            pred_idx++;
                        }
                        x_pred(p) = 1.0; // intercept
                        fd.matrix(i, j) = x_pred.dot(beta);
                    }
                }
            }
        }
    }
};

#endif