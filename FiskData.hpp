#ifndef FISK_DATA_HPP
#define FISK_DATA_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
#include <set>
#include <limits>
#include <Eigen/Dense>

struct FiskData {
    int rows = 0, cols = 0;
    std::vector<std::string> headers;
    Eigen::MatrixXd matrix;
    std::vector<std::vector<std::string>> raw_strings;

    // For categorical encoding
    std::map<int, std::map<std::string,int>> factor_maps;
    std::vector<bool> is_factor;

    static std::string abbrev(std::string s, size_t len) {
        if (s.length() <= len) return s;
        return s.substr(0, len - 2) + "..";
    }

    static FiskData from_csv(const std::string& path) {
        FiskData fd;
        std::ifstream file(path);
        if (!file.is_open()) throw std::runtime_error("Cannot open file: " + path);

        std::string line, word;

        // Read headers
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            while (std::getline(ss, word, ',')) {
                word.erase(std::remove(word.begin(), word.end(), '\r'), word.end());
                fd.headers.push_back(word);
            }
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<std::string> row_str;
            while (std::getline(ss, word, ',')) {
                word.erase(std::remove(word.begin(), word.end(), '\r'), word.end());
                row_str.push_back(word);
            }
            if (row_str.empty()) continue;
            fd.raw_strings.push_back(row_str);
            fd.rows++;
        }

        fd.cols = fd.headers.size();
        fd.matrix.resize(fd.rows, fd.cols);
        fd.is_factor.resize(fd.cols, false);

        for (int j = 0; j < fd.cols; ++j) {
            std::set<std::string> levels;
            bool all_numeric = true;

            for (int i = 0; i < fd.rows; ++i) {
                std::string s = fd.raw_strings[i][j];
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
                if (s.empty()) {
                    fd.matrix(i,j) = std::numeric_limits<double>::quiet_NaN();
                    levels.insert("NA");
                    continue;
                }
                try {
                    fd.matrix(i,j) = std::stod(s);
                } catch (...) {
                    all_numeric = false;
                    fd.matrix(i,j) = std::numeric_limits<double>::quiet_NaN();
                }
                levels.insert(s);
            }

            if (!all_numeric) {
                fd.is_factor[j] = true;
                int val = 0;
                for (const auto &l : levels) fd.factor_maps[j][l] = val++;
                for (int i = 0; i < fd.rows; ++i) {
                    std::string s = fd.raw_strings[i][j];
                    s.erase(0, s.find_first_not_of(" \t"));
                    s.erase(s.find_last_not_of(" \t") + 1);
                    if (s.empty()) s = "NA";
                    fd.matrix(i,j) = fd.factor_maps[j][s];
                }
            }
        }

        return fd;
    }

    int get_col_idx(const std::string& name) const {
        for (int i = 0; i < cols; ++i) if (headers[i] == name) return i;
        return -1;
    }

    void append_column(const FiskData& other, const std::string& new_name) {
        if (other.rows != this->rows) {
            throw std::runtime_error("Row count mismatch");
        }

        // 1. Expand the Matrix
        int old_cols = this->cols;
        this->matrix.conservativeResize(rows, old_cols + 1);
        this->matrix.col(old_cols) = other.matrix.col(0);
        this->cols++;

        // 2. Update Headers
        this->headers.push_back(new_name);

        // 3. CRITICAL: Update metadata vectors to prevent crash
        this->is_factor.push_back(false); 
        
        // 4. Sync raw_strings so head/summary works
        for (int i = 0; i < rows; ++i) {
            this->raw_strings[i].push_back(other.raw_strings[i][0]);
        }
    }

    void addColumn(const Eigen::VectorXd& colData, const std::string& name) {
        int oldCols = this->cols;
        this->matrix.conservativeResize(this->rows, oldCols + 1);
        this->matrix.col(oldCols) = colData;
        this->cols++;
        
        // THIS PREVENTS THE CRASH:
        this->headers.push_back(name);
        this->is_factor.push_back(false); 
        
        // Keep raw_strings synced for the 'head' command
        for (int i = 0; i < rows; ++i) {
            this->raw_strings[i].push_back(std::to_string(colData(i)));
        }
    }

    void print_summary() const {
        std::cout << "Dataset Summary (" << rows << " rows, " << cols << " columns)\n";
        std::cout << std::string(80, '=') << "\n\n";

        for (int j = 0; j < cols; ++j) {
            std::cout << "Column: " << headers[j] << "\n";
            if (is_factor[j]) {
                // Factor summary
                std::map<std::string, int> counts;
                for (int i = 0; i < rows; ++i) {
                    std::string level = raw_strings[i][j];
                    counts[level]++;
                }
                std::cout << "Type: Factor (" << counts.size() << " levels)\n";
                std::cout << "Most common levels:\n";
                // Sort by count descending
                std::vector<std::pair<std::string, int>> sorted_counts(counts.begin(), counts.end());
                std::sort(sorted_counts.begin(), sorted_counts.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
                int max_show = 5; // Show top 5
                for (int i = 0; i < std::min(max_show, (int)sorted_counts.size()); ++i) {
                    std::cout << "  " << sorted_counts[i].first << ": " << sorted_counts[i].second << "\n";
                }
                if (sorted_counts.size() > max_show) {
                    std::cout << "  ... and " << (sorted_counts.size() - max_show) << " more\n";
                }
            } else {
                // Numeric summary
                Eigen::VectorXd col = matrix.col(j);
                std::vector<double> vals;
                int n_missing = 0;
                for (int i = 0; i < rows; ++i) {
                    if (std::isnan(col[i])) {
                        n_missing++;
                    } else {
                        vals.push_back(col[i]);
                    }
                }
                int n = vals.size();
                if (n == 0) {
                    std::cout << "Type: Numeric (all missing)\n";
                    continue;
                }

                std::sort(vals.begin(), vals.end());
                double mean = 0.0;
                for (double v : vals) mean += v;
                mean /= n;

                double variance = 0.0;
                for (double v : vals) variance += (v - mean) * (v - mean);
                variance /= (n - 1);
                double std_dev = std::sqrt(variance);

                auto quantile = [&](double q) -> double {
                    if (vals.empty()) return 0.0;
                    size_t idx = std::min(vals.size() - 1, static_cast<size_t>(q * (vals.size() - 1)));
                    return vals[idx];
                };

                std::cout << "Type: Numeric\n";
                std::cout << std::fixed << std::setprecision(4);
                std::cout << "Count: " << n << " (missing: " << n_missing << ")\n";
                std::cout << "Mean: " << mean << "\n";
                std::cout << "Std Dev: " << std_dev << "\n";
                std::cout << "Min: " << vals.front() << "\n";
                std::cout << "1st Qu.: " << quantile(0.25) << "\n";
                std::cout << "Median: " << quantile(0.50) << "\n";
                std::cout << "3rd Qu.: " << quantile(0.75) << "\n";
                std::cout << "Max: " << vals.back() << "\n";
            }
            std::cout << "\n";
        }
    }
};

#endif