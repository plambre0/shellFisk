#ifndef FISK_ANALYZER_HPP
#define FISK_ANALYZER_HPP

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <map>
#include <cmath>
#include "FiskNativePlotter.hpp"

class FiskPlotter {
public:
    struct Series {
        std::string label;
        std::vector<double> x;
        std::vector<double> y;
        std::string style; // "steps", "points", "lines"
    };

    static void send_to_gnuplot(const std::string& title, const std::string& xlabel, const std::string& ylabel, const std::vector<Series>& datasets) {
        // Use _popen for Windows/MSYS2
        FILE* gp = _popen("gnuplot -persistent", "w");
        if (!gp) {
            std::cerr << "[!] Gnuplot not found. Run: pacman -S mingw-w64-x86_64-gnuplot\n";
            return;
        }

        fprintf(gp, "set title '%s'\n", title.c_str());
        fprintf(gp, "set xlabel '%s'\n", xlabel.c_str());
        fprintf(gp, "set ylabel '%s'\n", ylabel.c_str());
        fprintf(gp, "set grid\n");
        fprintf(gp, "set yrange [0:1.05]\n");

        fprintf(gp, "plot ");
        for (size_t i = 0; i < datasets.size(); ++i) {
            fprintf(gp, "'-' with %s title '%s' lw 2%s", 
                    datasets[i].style.c_str(), datasets[i].label.c_str(), 
                    (i == datasets.size() - 1 ? "" : ", "));
        }
        fprintf(gp, "\n");

        for (const auto& ds : datasets) {
            for (size_t i = 0; i < ds.x.size(); ++i) {
                fprintf(gp, "%f %f\n", ds.x[i], ds.y[i]);
            }
            fprintf(gp, "e\n");
        }
        _pclose(gp);
    }
};

class FiskAnalyzer {
public:
    static void trim_str(std::string& s) {
        s.erase(0, s.find_first_not_of(" \t"));
        s.erase(s.find_last_not_of(" \t") + 1);
    }

    static void summary_stats(const FiskData& fd){
        std::cout << std::setw(15) << "Column" << std::setw(12) << "Mean" << std::setw(12) << "Std" << std::setw(12) << "Min" << std::setw(12) << "Max\n";
        for (int j = 0; j < fd.cols; j++) {
            double sum = 0, minv = 1e9, maxv = -1e9;
            int count = 0;
            for (int i = 0; i < fd.rows; i++) {
                double val = fd.matrix(i, j);
                if (!std::isnan(val)) { sum += val; minv = std::min(minv, val); maxv = std::max(maxv, val); count++; }
            }
            double mean = (count > 0) ? sum / count : 0;
            double sqsum = 0;
            for (int i = 0; i < fd.rows; i++) { double val = fd.matrix(i, j); if (!std::isnan(val)) sqsum += (val - mean) * (val - mean); }
            double stdv = (count > 1) ? std::sqrt(sqsum / (count - 1)) : 0;
            std::cout << std::setw(15) << FiskData::abbrev(fd.headers[j], 15) << std::setw(12) << mean << std::setw(12) << stdv << std::setw(12) << minv << std::setw(12) << maxv << "\n";
        }
    }

    static void run_pca(const FiskData& fd) {
        Eigen::MatrixXd X = fd.matrix;
        Eigen::VectorXd mean = X.colwise().mean();
        X.rowwise() -= mean.transpose();
        Eigen::MatrixXd cov = (X.transpose() * X) / (X.rows() - 1);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

        Eigen::VectorXd evals = eig.eigenvalues().reverse();
        Eigen::MatrixXd loadings = eig.eigenvectors().rowwise().reverse();
        int k = evals.size();
        Eigen::VectorXd prop = evals / evals.sum();
        Eigen::VectorXd cum = Eigen::VectorXd::Zero(k);
        for (int i = 0; i < k; ++i) cum(i) = (i == 0) ? prop(i) : cum(i - 1) + prop(i);

        std::cout << "PCA Summary:\n";
        std::cout << std::setw(10) << "PC" << std::setw(14) << "Eigenvalue" << std::setw(14) << "Prop.Var" << std::setw(14) << "Cum.Var" << "\n";
        for (int i = 0; i < k; ++i) {
            std::cout << std::setw(10) << ("PC" + std::to_string(i + 1))
                      << std::setw(14) << evals(i)
                      << std::setw(14) << prop(i)
                      << std::setw(14) << cum(i) << "\n";
        }

        std::cout << "\nPCA Loadings:\n";
        int show = std::min(k, 6);
        std::cout << std::setw(15) << "Variable";
        for (int i = 0; i < show; ++i) std::cout << std::setw(12) << ("PC" + std::to_string(i + 1));
        std::cout << "\n";
        std::cout << std::string(15 + show * 12, '-') << "\n";
        for (int j = 0; j < fd.cols; ++j) {
            std::cout << std::setw(15) << FiskData::abbrev(fd.headers[j], 15);
            for (int i = 0; i < show; ++i) {
                std::cout << std::setw(12) << loadings(j, i);
            }
            std::cout << "\n";
        }
    }

    static std::string value_label(const FiskData& fd, int col, int row) {
        if (col < 0 || col >= fd.cols || row < 0 || row >= fd.rows) return "";
        if (fd.is_factor[col] && row < fd.raw_strings.size() && col < fd.raw_strings[row].size()) {
            return fd.raw_strings[row][col];
        }
        return std::to_string(fd.matrix(row, col));
    }

    static void run_survival(const FiskData& fd, const std::string& time_col, const std::string& event_col, const std::string& group_col = "") {
        int t_idx = fd.get_col_idx(time_col);
        int e_idx = fd.get_col_idx(event_col);
        
        if (t_idx < 0 || e_idx < 0) {
            std::cerr << "[!] Columns not found: " << time_col << " or " << event_col << "\n";
            return;
        }

        // 1. Group the data
        std::map<std::string, std::vector<std::pair<double, int>>> groups;
        int g_idx = group_col.empty() ? -1 : fd.get_col_idx(group_col);

        for (int i = 0; i < fd.rows; ++i) {
            double t = fd.matrix(i, t_idx);
            if (std::isnan(t)) continue;
            int e = (fd.matrix(i, e_idx) > 0.5) ? 1 : 0;
            std::string label = (g_idx >= 0) ? value_label(fd, g_idx, i) : "All";
            groups[label].push_back({t, e});
        }

        // 2. Prepare the Plot Request
        FiskNativePlotter::PlotRequest* req = new FiskNativePlotter::PlotRequest();
        req->title = "Kaplan-Meier Survival Analysis";
        req->xlabel = "Time (" + time_col + ")";
        req->ylabel = "Survival Probability";

        COLORREF colors[] = { RGB(0,0,0), RGB(200,0,0), RGB(0,0,200), RGB(0,150,0) };
        int c_idx = 0;

        for (auto& kv : groups) {
            auto& obs = kv.second;
            std::sort(obs.begin(), obs.end(), [](auto& a, auto& b){ return a.first < b.first; });

            FiskNativePlotter::Series s;
            s.label = kv.first;
            s.style = FiskNativePlotter::STEPS;
            s.color = colors[c_idx++ % 4];

            // Kaplan-Meier Math
            double survival = 1.0;
            int n_at_risk = obs.size();
            
            // Initial point at (0, 1.0)
            s.x.push_back(0.0);
            s.y.push_back(1.0);

            size_t i = 0;
            while (i < obs.size()) {
                double cur_t = obs[i].first;
                int events = 0, censored = 0;
                while (i < obs.size() && std::abs(obs[i].first - cur_t) < 1e-9) {
                    if (obs[i].second == 1) events++;
                    else censored++;
                    i++;
                }
                if (events > 0) {
                    survival *= (1.0 - (double)events / n_at_risk);
                }
                s.x.push_back(cur_t);
                s.y.push_back(survival);
                n_at_risk -= (events + censored);
            }
            req->series.push_back(s);
        }

        FiskNativePlotter::Show(req);
        std::cout << "[+] Survival Plot window opened.\n";
    }

    static void run_subgroup(const FiskData& fd, const std::string& group_col, const std::string& outcome_col) {
        int g_idx = fd.get_col_idx(group_col);
        int o_idx = fd.get_col_idx(outcome_col);
        if (g_idx < 0 || o_idx < 0) {
            std::cerr << "[!] Subgroup analysis error: group or outcome column not found\n";
            return;
        }
        std::map<std::string, std::vector<double>> groups;
        for (int i = 0; i < fd.rows; ++i) {
            std::string label = value_label(fd, g_idx, i);
            double value = fd.matrix(i, o_idx);
            if (!std::isnan(value)) groups[label].push_back(value);
        }
        std::cout << "\nSubgroup summary for " << outcome_col << " by " << group_col << ":\n";
        std::cout << std::setw(20) << "Group" << std::setw(10) << "N" << std::setw(12) << "Mean" << std::setw(12) << "SD" << std::setw(12) << "Min" << std::setw(12) << "Max" << "\n";
        std::cout << std::string(78, '-') << "\n";
        for (auto& kv : groups) {
            auto& values = kv.second;
            int n = values.size();
            if (n == 0) continue;
            double sum = 0;
            for (double v : values) sum += v;
            double mean = sum / n;
            double ss = 0;
            for (double v : values) ss += (v - mean) * (v - mean);
            double sd = (n > 1) ? std::sqrt(ss / (n - 1)) : 0.0;
            std::sort(values.begin(), values.end());
            std::cout << std::setw(20) << FiskData::abbrev(kv.first, 20)
                      << std::setw(10) << n
                      << std::setw(12) << mean
                      << std::setw(12) << sd
                      << std::setw(12) << values.front()
                      << std::setw(12) << values.back() << "\n";
        }
    }

    static void run_anova(const FiskData& fd, const std::string& response_col, const std::string& factor_col) {
        int r_idx = fd.get_col_idx(response_col);
        int f_idx = fd.get_col_idx(factor_col);
        if (r_idx < 0 || f_idx < 0) {
            std::cerr << "[!] ANOVA error: response or factor column not found\n";
            return;
        }
        std::map<std::string, std::vector<double>> groups;
        for (int i = 0; i < fd.rows; ++i) {
            double y = fd.matrix(i, r_idx);
            if (std::isnan(y)) continue;
            std::string label = value_label(fd, f_idx, i);
            groups[label].push_back(y);
        }
        int n = 0;
        double total_sum = 0;
        for (auto& kv : groups) {
            for (double y : kv.second) {
                n += 1;
                total_sum += y;
            }
        }
        if (n == 0 || groups.size() < 2) {
            std::cerr << "[!] ANOVA error: not enough groups or values\n";
            return;
        }
        double overall_mean = total_sum / n;
        double ss_between = 0;
        double ss_within = 0;
        for (auto& kv : groups) {
            int ni = kv.second.size();
            double group_sum = 0;
            for (double y : kv.second) group_sum += y;
            double group_mean = group_sum / ni;
            ss_between += ni * (group_mean - overall_mean) * (group_mean - overall_mean);
            for (double y : kv.second) ss_within += (y - group_mean) * (y - group_mean);
        }
        int df_between = groups.size() - 1;
        int df_within = n - groups.size();
        double ms_between = ss_between / df_between;
        double ms_within = ss_within / df_within;
        double f_stat = (ms_within > 0) ? ms_between / ms_within : 0.0;
        std::cout << "ANOVA result for " << response_col << " by " << factor_col << ":\n";
        std::cout << "  F(" << df_between << ", " << df_within << ") = " << f_stat << "\n";
        std::cout << "  SS(between) = " << ss_between << ", SS(within) = " << ss_within << "\n";
    }

    static std::vector<double> rank_values(const std::vector<double>& values) {
        int n = values.size();
        std::vector<std::pair<double, int>> sorted;
        sorted.reserve(n);
        for (int i = 0; i < n; ++i) sorted.emplace_back(values[i], i);
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::vector<double> ranks(n);
        int i = 0;
        while (i < n) {
            int j = i + 1;
            while (j < n && sorted[j].first == sorted[i].first) j++;
            double rank = 0.5 * (i + 1 + j);
            for (int k = i; k < j; ++k) ranks[sorted[k].second] = rank;
            i = j;
        }
        return ranks;
    }

    static void run_wilcoxon(const FiskData& fd, const std::string& group_col, const std::string& value_col) {
        int g_idx = fd.get_col_idx(group_col);
        int v_idx = fd.get_col_idx(value_col);
        if (g_idx < 0 || v_idx < 0) {
            std::cerr << "[!] Wilcoxon error: group or value column not found\n";
            return;
        }
        std::map<std::string, std::vector<double>> groups;
        for (int i = 0; i < fd.rows; ++i) {
            double value = fd.matrix(i, v_idx);
            if (std::isnan(value)) continue;
            groups[FiskAnalyzer::value_label(fd, g_idx, i)].push_back(value);
        }
        if (groups.size() != 2) {
            std::cerr << "[!] Wilcoxon error: exactly two groups are required\n";
            return;
        }
        std::vector<double> combined;
        std::vector<int> labels;
        std::vector<std::string> names;
        for (auto& kv : groups) {
            names.push_back(kv.first);
            for (double v : kv.second) {
                combined.push_back(v);
                labels.push_back(names.size() - 1);
            }
        }
        auto ranks = rank_values(combined);
        std::vector<double> rank_sum(2, 0.0);
        std::vector<int> n(2, 0);
        for (int i = 0; i < labels.size(); ++i) {
            rank_sum[labels[i]] += ranks[i];
            n[labels[i]] += 1;
        }
        double u1 = rank_sum[0] - n[0] * (n[0] + 1) / 2.0;
        double u2 = n[0] * n[1] - u1;
        double u = std::min(u1, u2);
        double mean_u = n[0] * n[1] / 2.0;
        double sigma_u = std::sqrt(n[0] * n[1] * (n[0] + n[1] + 1) / 12.0);
        double z = sigma_u > 0 ? (u - mean_u) / sigma_u : 0.0;
        double p = 0.5 * std::erfc(std::abs(z) / std::sqrt(2.0));
        std::cout << "Wilcoxon rank-sum test (Mann-Whitney U):\n";
        std::cout << "  Groups: " << names[0] << " (n=" << n[0] << "), " << names[1] << " (n=" << n[1] << ")\n";
        std::cout << "  U = " << u << ", z = " << z << ", p approx = " << p << "\n";
    }

    static void run_kruskal(const FiskData& fd, const std::string& group_col, const std::string& value_col) {
        int g_idx = fd.get_col_idx(group_col);
        int v_idx = fd.get_col_idx(value_col);
        if (g_idx < 0 || v_idx < 0) {
            std::cerr << "[!] Kruskal-Wallis error: group or value column not found\n";
            return;
        }
        std::map<std::string, std::vector<double>> groups;
        for (int i = 0; i < fd.rows; ++i) {
            double value = fd.matrix(i, v_idx);
            if (std::isnan(value)) continue;
            groups[FiskAnalyzer::value_label(fd, g_idx, i)].push_back(value);
        }
        if (groups.size() < 2) {
            std::cerr << "[!] Kruskal-Wallis error: at least two groups are required\n";
            return;
        }
        std::vector<double> combined;
        std::vector<int> sizes;
        for (auto& kv : groups) {
            sizes.push_back(kv.second.size());
            for (double v : kv.second) combined.push_back(v);
        }
        auto ranks = rank_values(combined);
        int index = 0;
        double total_n = combined.size();
        double h = 0.0;
        int k = groups.size();
        int g = 0;
        for (auto& kv : groups) {
            double sum_ranks = 0.0;
            for (int j = 0; j < sizes[g]; ++j) {
                sum_ranks += ranks[index++];
            }
            h += sum_ranks * sum_ranks / sizes[g];
            g++;
        }
        h = (12.0 / (total_n * (total_n + 1))) * h - 3.0 * (total_n + 1);
        std::cout << "Kruskal-Wallis H test:\n";
        std::cout << "  H = " << h << ", df = " << (k - 1) << "\n";
    }

    static void run_spearman(const FiskData& fd, const std::string& x_col, const std::string& y_col) {
        int x_idx = fd.get_col_idx(x_col);
        int y_idx = fd.get_col_idx(y_col);
        if (x_idx < 0 || y_idx < 0) {
            std::cerr << "[!] Spearman error: x or y column not found\n";
            return;
        }
        std::vector<double> x_vals, y_vals;
        for (int i = 0; i < fd.rows; ++i) {
            double x = fd.matrix(i, x_idx);
            double y = fd.matrix(i, y_idx);
            if (std::isnan(x) || std::isnan(y)) continue;
            x_vals.push_back(x);
            y_vals.push_back(y);
        }
        if (x_vals.size() < 2) {
            std::cerr << "[!] Spearman error: not enough paired observations\n";
            return;
        }
        auto rx = rank_values(x_vals);
        auto ry = rank_values(y_vals);
        double sx = 0.0, sy = 0.0, sxy = 0.0;
        double mx = std::accumulate(rx.begin(), rx.end(), 0.0) / rx.size();
        double my = std::accumulate(ry.begin(), ry.end(), 0.0) / ry.size();
        for (int i = 0; i < rx.size(); ++i) {
            sx += (rx[i] - mx) * (rx[i] - mx);
            sy += (ry[i] - my) * (ry[i] - my);
            sxy += (rx[i] - mx) * (ry[i] - my);
        }
        double rho = (sx > 0 && sy > 0) ? sxy / std::sqrt(sx * sy) : 0.0;
        std::cout << "Spearman rank correlation for " << x_col << " and " << y_col << ":\n";
        std::cout << "  rho = " << rho << "\n";
    }

    static void run_chi2(const FiskData& fd, const std::string& row_col, const std::string& col_col) {
        int r_idx = fd.get_col_idx(row_col);
        int c_idx = fd.get_col_idx(col_col);
        if (r_idx < 0 || c_idx < 0) {
            std::cerr << "[!] Chi-squared error: row or column factor not found\n";
            return;
        }
        std::map<std::string, std::map<std::string, int>> table;
        std::map<std::string, int> row_totals;
        std::map<std::string, int> col_totals;
        int total = 0;
        for (int i = 0; i < fd.rows; ++i) {
            std::string r = value_label(fd, r_idx, i);
            std::string c = value_label(fd, c_idx, i);
            if (r.empty() || c.empty()) continue;
            table[r][c]++;
            row_totals[r]++;
            col_totals[c]++;
            total++;
        }
        if (row_totals.size() < 2 || col_totals.size() < 2) {
            std::cerr << "[!] Chi-squared error: at least 2 rows and 2 columns required\n";
            return;
        }
        double chi2 = 0.0;
        for (auto& rkv : table) {
            for (auto& ckv : rkv.second) {
                double observed = ckv.second;
                double expected = double(row_totals[rkv.first]) * double(col_totals[ckv.first]) / total;
                if (expected > 0) chi2 += (observed - expected) * (observed - expected) / expected;
            }
        }
        int df = (row_totals.size() - 1) * (col_totals.size() - 1);
        std::cout << "Chi-squared test of independence:\n";
        std::cout << "  chi2 = " << chi2 << ", df = " << df << "\n";
    }

};

#endif