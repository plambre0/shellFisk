#ifndef FISK_PARSER_HPP
#define FISK_PARSER_HPP

#include <vector>
#include <string>
#include <regex>
#include <stdexcept>
#include "FiskData.hpp"

class FiskParser {
public:
    struct Design {
        Eigen::MatrixXd X;
        Eigen::VectorXd y;
        Eigen::MatrixXd Z; // Random effects
        std::vector<std::string> names;
        bool success = false;
    };

    static Design parse_formula(const FiskData& fd, const std::string& formula) {
        Design d;
        try {
            size_t eq_pos = formula.find('=');
            if (eq_pos == std::string::npos) throw std::runtime_error("Formula must have '='");

            std::string y_raw = formula.substr(0, eq_pos);
            std::string rhs = formula.substr(eq_pos+1);
            d.y = extract_col(fd, y_raw);

            std::regex term_regex(R"(\(([^)]+)\|([^)]+)\)|([a-zA-Z0-9_.]+))");
            auto words_begin = std::sregex_iterator(rhs.begin(), rhs.end(), term_regex);
            auto words_end = std::sregex_iterator();

            std::vector<Eigen::VectorXd> fixed_cols;
            std::vector<Eigen::VectorXd> random_cols;
            d.names.push_back("(Intercept)");
            fixed_cols.push_back(Eigen::VectorXd::Ones(fd.rows));

            for (auto i = words_begin; i != words_end; ++i) {
                std::smatch m = *i;
                if (m[1].matched && m[2].matched) {
                    // Random effect (1|group)
                    Eigen::VectorXd group = extract_col(fd, m[2].str());
                    random_cols.push_back(group);
                    d.names.push_back("(1|" + m[2].str() + ")");
                } else {
                    std::string var = m[3].matched ? m[3].str() : "";
                    fixed_cols.push_back(extract_col(fd, var));
                    d.names.push_back(var);
                }
            }

            d.X.resize(fd.rows, fixed_cols.size());
            for (size_t i = 0; i < fixed_cols.size(); ++i) d.X.col(i) = fixed_cols[i];

            if (!random_cols.empty()) {
                d.Z.resize(fd.rows, random_cols.size());
                for (size_t i = 0; i < random_cols.size(); ++i) d.Z.col(i) = random_cols[i];
            }

            d.success = true;
        } catch (...) {
            d.success = false;
            throw std::runtime_error("Failed to parse formula");
        }
        return d;
    }

private:
    static Eigen::VectorXd extract_col(const FiskData& fd, const std::string& name) {
        std::string s = name;
        s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
        int idx = fd.get_col_idx(s);
        if (idx == -1) throw std::runtime_error("Column not found: " + s);
        return fd.matrix.col(idx);
    }
};

#endif