#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <random>
#include <set>
#include <deque>
#include <filesystem>
#include <memory>
#include <Eigen/Dense>
#include "FiskData.hpp"
#include "FiskModel.hpp"
#include "FiskGLM.hpp"
#include "FiskTransform.hpp"
#include "FiskAnalyzer.hpp"
#include "FiskNativePlotter.hpp"

#include <nlohmann/json.hpp>

struct ShellEnv {
    std::map<std::string, FiskData> data_vars;
    std::map<std::string, FiskModel> models;
    std::string active_ds = "";
    std::vector<std::map<std::string, FiskData>> undo_stack;
    std::vector<std::map<std::string, FiskData>> redo_stack;
    std::vector<std::string> history;
};

static void push_data_state(ShellEnv& env) {
    env.undo_stack.push_back(env.data_vars);
    env.redo_stack.clear();
    if (env.undo_stack.size() > 50) env.undo_stack.erase(env.undo_stack.begin());
}

static void trim_str(std::string& s) {
    s.erase(0, s.find_first_not_of(" \t"));
    s.erase(s.find_last_not_of(" \t") + 1);
}

static bool parse_double(const std::string& text, double& value) {
    try {
        size_t pos;
        value = std::stod(text, &pos);
        return pos == text.size();
    } catch (...) {
        return false;
    }
}

static std::vector<int> parse_index_expr(const std::string& expr, int max_index) {
    std::vector<int> indices;
    std::string cleaned = expr;
    trim_str(cleaned);
    if (cleaned.empty()) return indices;
    if (cleaned.find(':') != std::string::npos) {
        size_t pos = cleaned.find(':');
        int start = std::stoi(cleaned.substr(0, pos));
        int end = std::stoi(cleaned.substr(pos + 1));
        start = std::max(1, start);
        end = std::min(max_index, end);
        for (int i = start; i <= end; ++i) indices.push_back(i - 1);
    } else {
        int idx = std::stoi(cleaned);
        if (idx >= 1 && idx <= max_index) indices.push_back(idx - 1);
    }
    return indices;
}

static std::vector<std::string> split_names(const std::string& names) {
    std::vector<std::string> result;
    std::string item;
    for (char ch : names) {
        if (ch == ',') {
            trim_str(item);
            if (!item.empty()) result.push_back(item);
            item.clear();
        } else {
            item.push_back(ch);
        }
    }
    trim_str(item);
    if (!item.empty()) result.push_back(item);
    return result;
}

static bool parse_subset_expr(const std::string& expr, const ShellEnv& env, FiskData& out, std::string& err) {
    std::string text = expr;
    trim_str(text);
    if (text.rfind("subset", 0) != 0) {
        err = "Unsupported append source expression";
        return false;
    }
    std::stringstream ss(text);
    std::string token, ds_name;
    ss >> token >> ds_name;
    if (ds_name.empty()) {
        err = "Subset source must include a dataset name";
        return false;
    }
    std::string rest;
    std::getline(ss, rest);
    trim_str(rest);
    size_t open = rest.find('(');
    size_t close = rest.rfind(')');
    if (open == std::string::npos || close == std::string::npos || close <= open) {
        err = "Subset source must include parentheses with row and column expressions";
        return false;
    }
    std::string inner = rest.substr(open + 1, close - open - 1);
    size_t comma = inner.find(',');
    if (comma == std::string::npos) {
        err = "Subset source must contain row and column expressions separated by comma";
        return false;
    }
    std::string row_expr = inner.substr(0, comma);
    std::string col_expr = inner.substr(comma + 1);
    trim_str(row_expr);
    trim_str(col_expr);
    if (!env.data_vars.count(ds_name)) {
        err = "Subset source dataset not found: " + ds_name;
        return false;
    }
    const FiskData& src = env.data_vars.at(ds_name);
    std::vector<int> row_indices = row_expr.empty() ? std::vector<int>() : parse_index_expr(row_expr, src.rows);
    std::vector<int> col_indices = col_expr.empty() ? std::vector<int>() : parse_index_expr(col_expr, src.cols);
    if (!row_expr.empty() && row_indices.empty()) {
        err = "Invalid row expression in subset source";
        return false;
    }
    if (!col_expr.empty() && col_indices.empty()) {
        err = "Invalid column expression in subset source";
        return false;
    }
    if (row_indices.empty()) {
        row_indices.resize(src.rows);
        std::iota(row_indices.begin(), row_indices.end(), 0);
    }
    if (col_indices.empty()) {
        col_indices.resize(src.cols);
        std::iota(col_indices.begin(), col_indices.end(), 0);
    }
    if (col_indices.size() != 1) {
        err = "Subset source must resolve to exactly one column for append";
        return false;
    }
    out = src;
    FiskData dst;
    dst.rows = row_indices.size();
    dst.cols = 1;
    dst.headers = { src.headers[col_indices[0]] };
    dst.is_factor = { src.is_factor[col_indices[0]] };
    if (src.factor_maps.count(col_indices[0])) {
        dst.factor_maps[0] = src.factor_maps.at(col_indices[0]);
    }
    dst.matrix = Eigen::MatrixXd(dst.rows, 1);
    for (size_t i = 0; i < row_indices.size(); ++i) {
        dst.matrix(i, 0) = src.matrix(row_indices[i], col_indices[0]);
    }
    out = dst;
    return true;
}

static bool parse_nrow_ncol(const std::string& text, int& nrow, int& ncol) {
    std::string tmp = text;
    trim_str(tmp);
    if (tmp.size() < 5 || tmp.front() != '(' || tmp.back() != ')') return false;
    tmp = tmp.substr(1, tmp.size() - 2);
    size_t comma = tmp.find(',');
    if (comma == std::string::npos) return false;
    std::string rows = tmp.substr(0, comma);
    std::string cols = tmp.substr(comma + 1);
    trim_str(rows);
    trim_str(cols);
    try {
        nrow = std::stoi(rows);
        ncol = std::stoi(cols);
        return nrow > 0 && ncol > 0;
    } catch (...) {
        return false;
    }
}

static bool parse_number_list(const std::string& text, std::vector<double>& values) {
    std::string tmp = text;
    trim_str(tmp);
    if (tmp.size() >= 2 && tmp.front() == '(' && tmp.back() == ')') {
        tmp = tmp.substr(1, tmp.size() - 2);
    }
    values.clear();
    std::string token;
    std::stringstream ss(tmp);
    while (std::getline(ss, token, ',')) {
        trim_str(token);
        if (token.empty()) continue;
        try {
            values.push_back(std::stod(token));
        } catch (...) {
            return false;
        }
    }
    return !values.empty();
}

static void quickplot_scatter(const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    int width = 80;
    int height = 20;
    std::vector<std::string> grid(height, std::string(width, ' '));
    if (x.empty() || y.empty() || x.size() != y.size()) {
        std::cout << "[!] Plot error: invalid data\n";
        return;
    }
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    double y_min = *std::min_element(y.begin(), y.end());
    double y_max = *std::max_element(y.begin(), y.end());
    double x_range = x_max - x_min;
    double y_range = y_max - y_min;
    if (x_range == 0) x_range = 1;
    if (y_range == 0) y_range = 1;
    for (size_t i = 0; i < x.size(); ++i) {
        int xi = std::clamp((int)((x[i] - x_min) / x_range * (width - 1)), 0, width - 1);
        int yi = std::clamp((int)((y_max - y[i]) / y_range * (height - 1)), 0, height - 1);
        grid[yi][xi] = '*';
    }
    std::cout << "\n" << title << "\n";
    std::cout << std::string(width + 2, '-') << "\n";
    for (const auto& row : grid) {
        std::cout << "|" << row << "|\n";
    }
    std::cout << std::string(width + 2, '-') << "\n";
    std::cout << "X range: [" << x_min << ", " << x_max << "]  Y range: [" << y_min << ", " << y_max << "]\n";
}

static void quickplot_roc_curve(const std::vector<double>& scores, const std::vector<int>& labels, const std::string& title) {
    if (scores.size() != labels.size() || scores.empty()) {
        std::cout << "[!] Plot error: invalid ROC data\n";
        return;
    }
    std::vector<std::pair<double,int>> points;
    for (size_t i = 0; i < scores.size(); ++i) {
        points.emplace_back(scores[i], labels[i]);
    }
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b){ return a.first > b.first; });

    int positives = 0;
    int negatives = 0;
    for (auto& p : points) {
        if (p.second == 1) positives++; else negatives++;
    }
    if (positives == 0 || negatives == 0) {
        std::cout << "[!] ROC error: dataset must contain both positive and negative labels\n";
        return;
    }

    std::vector<double> tpr;
    std::vector<double> fpr;
    int tp = 0;
    int fp = 0;
    double prev_score = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].first != prev_score) {
            tpr.push_back(tp / double(positives));
            fpr.push_back(fp / double(negatives));
            prev_score = points[i].first;
        }
        if (points[i].second == 1) tp++; else fp++;
    }
    tpr.push_back(tp / double(positives));
    fpr.push_back(fp / double(negatives));

    int width = 80;
    int height = 20;
    std::vector<std::string> grid(height, std::string(width, ' '));
    for (size_t i = 0; i < fpr.size(); ++i) {
        int xi = std::clamp((int)(fpr[i] * (width - 1)), 0, width - 1);
        int yi = std::clamp((int)((1.0 - tpr[i]) * (height - 1)), 0, height - 1);
        grid[yi][xi] = '*';
    }
    for (int d = 0; d < width; ++d) {
        int y = height - 1 - d * (height - 1) / (width - 1);
        if (y >= 0 && y < height && grid[y][d] == ' ') grid[y][d] = '.';
    }

    double auc = 0.0;
    for (size_t i = 1; i < fpr.size(); ++i) {
        auc += 0.5 * (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]);
    }

    std::cout << "\n" << title << " (AUC=" << std::fixed << std::setprecision(3) << auc << ")\n";
    std::cout << std::string(width + 2, '-') << "\n";
    for (const auto& row : grid) {
        std::cout << "|" << row << "|\n";
    }
    std::cout << std::string(width + 2, '-') << "\n";
    std::cout << "FPR from 0 to 1 horizontally, TPR from 1 to 0 vertically\n";
}

static std::string sanitize_name(const std::string& text) {
    std::string result;
    for (char ch : text) {
        if (std::isalnum((unsigned char)ch) || ch == '_' || ch == '-') result.push_back(ch);
        else result.push_back('_');
    }
    return result;
}

static bool write_dataset_csv(const FiskData& fd, const std::filesystem::path& path, std::string& err) {
    std::ofstream file(path);
    if (!file.is_open()) {
        err = "Unable to open file for writing: " + path.string();
        return false;
    }
    for (size_t i = 0; i < fd.headers.size(); ++i) {
        if (i > 0) file << ",";
        file << fd.headers[i];
    }
    file << "\n";
    for (int i = 0; i < fd.rows; ++i) {
        for (int j = 0; j < fd.cols; ++j) {
            if (j > 0) file << ",";
            if (fd.is_factor[j] && fd.factor_maps.count(j)) {
                std::string value = "";
                for (auto& kv : fd.factor_maps.at(j)) {
                    if (kv.second == (int)fd.matrix(i, j)) {
                        value = kv.first;
                        break;
                    }
                }
                file << value;
            } else {
                file << fd.matrix(i, j);
            }
        }
        file << "\n";
    }
    return true;
}

static bool export_environment(const ShellEnv& env, const std::filesystem::path& script_path, std::string& err) {
    std::filesystem::path folder = script_path.parent_path();
    if (folder.empty()) folder = std::filesystem::current_path();
    std::ofstream script_file(script_path);
    if (!script_file.is_open()) {
        err = "Unable to open export file: " + script_path.string();
        return false;
    }
    for (auto& kv : env.data_vars) {
        const std::string& name = kv.first;
        const FiskData& fd = kv.second;
        std::string safe_name = sanitize_name(name);
        std::filesystem::path data_file = folder / (script_path.stem().string() + "_" + safe_name + ".csv");
        std::string write_err;
        if (!write_dataset_csv(fd, data_file, write_err)) {
            err = write_err;
            return false;
        }
        script_file << "load " << data_file.string() << " " << name << "\n";
    }
    if (!env.models.empty()) {
        script_file << "# NOTE: models are not exported automatically; recreate with glm commands if needed\n";
    }
    return true;
}

static bool write_momento(const ShellEnv& env, const std::filesystem::path& path, std::string& err) {
    std::ofstream out(path);
    if (!out.is_open()) {
        err = "Unable to open momento file: " + path.string();
        return false;
    }
    for (const std::string& line : env.history) {
        out << line << "\n";
    }
    return true;
}

static std::mt19937 global_rng(0);

static std::mt19937& get_global_rng() {
    return global_rng;
}

static void set_global_seed(unsigned int seed) {
    global_rng.seed(seed);
}

static bool generate_random_matrix(const std::string& family, int nrow, int ncol, const std::vector<double>& params, Eigen::MatrixXd& out, std::string& err) {
    std::mt19937& rng = get_global_rng();
    out = Eigen::MatrixXd(nrow, ncol);
    if (family == "gaussian" || family == "normal") {
        double mean = params.size() > 0 ? params[0] : 0.0;
        double sd = params.size() > 1 ? params[1] : 1.0;
        if (sd <= 0) { err = "Gaussian sd must be positive"; return false; }
        std::normal_distribution<double> dist(mean, sd);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
    }
    if (family == "poisson") {
        double lambda = params.size() > 0 ? params[0] : 1.0;
        if (lambda <= 0) { err = "Poisson lambda must be positive"; return false; }
        std::poisson_distribution<int> dist(lambda);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
    }
    if (family == "binomial") {
        if (params.size() < 2) { err = "Binomial requires n and p"; return false; }
        int trials = int(params[0]);
        double p = params[1];
        if (trials <= 0 || p < 0 || p > 1) { err = "Binomial n must be positive and p between 0 and 1"; return false; }
        std::binomial_distribution<int> dist(trials, p);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
    }
    if (family == "gamma") {
        if (params.size() < 2) { err = "Gamma requires shape and scale"; return false; }
        double shape = params[0];
        double scale = params[1];
        if (shape <= 0 || scale <= 0) { err = "Gamma shape and scale must be positive"; return false; }
        std::gamma_distribution<double> dist(shape, scale);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
    }
    if (family == "weibull") {
        if (params.size() < 2) { err = "Weibull requires shape and scale"; return false; }
        double a = params[0];
        double b = params[1];
        if (a <= 0 || b <= 0) { err = "Weibull shape and scale must be positive"; return false; }
        std::weibull_distribution<double> dist(a, b);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
    }
    if (family == "invgaussian" || family == "inverse_gaussian") {
#ifdef __cpp_lib_random
        if (params.size() < 2) { err = "Inverse Gaussian requires mean and shape"; return false; }
        double mean = params[0];
        double shape = params[1];
        if (mean <= 0 || shape <= 0) { err = "Inverse Gaussian mean and shape must be positive"; return false; }
        std::inverse_gaussian_distribution<double> dist(mean, shape);
        for (int i = 0; i < nrow; ++i)
            for (int j = 0; j < ncol; ++j)
                out(i, j) = dist(rng);
        return true;
#else
        err = "Inverse Gaussian distribution not supported by this compiler";
        return false;
#endif
    }
    err = "Unsupported random matrix family: " + family;
    return false;
}

static void plot_scatter(const std::vector<double>& x, const std::vector<double>& y, const std::string& title) {
    int width = 80;
    int height = 20;
    std::vector<std::string> grid(height, std::string(width, ' '));
    if (x.empty() || y.empty() || x.size() != y.size()) {
        std::cout << "[!] Plot error: invalid data\n";
        return;
    }
    double x_min = *std::min_element(x.begin(), x.end());
    double x_max = *std::max_element(x.begin(), x.end());
    double y_min = *std::min_element(y.begin(), y.end());
    double y_max = *std::max_element(y.begin(), y.end());
    double x_range = x_max - x_min;
    double y_range = y_max - y_min;
    if (x_range == 0) x_range = 1;
    if (y_range == 0) y_range = 1;
    for (size_t i = 0; i < x.size(); ++i) {
        int xi = std::clamp((int)((x[i] - x_min) / x_range * (width - 1)), 0, width - 1);
        int yi = std::clamp((int)((y_max - y[i]) / y_range * (height - 1)), 0, height - 1);
        grid[yi][xi] = '*';
    }
    std::cout << "\n" << title << "\n";
    std::cout << std::string(width + 2, '-') << "\n";
    for (const auto& row : grid) {
        std::cout << "|" << row << "|\n";
    }
    std::cout << std::string(width + 2, '-') << "\n";
    std::cout << "X range: [" << x_min << ", " << x_max << "]  Y range: [" << y_min << ", " << y_max << "]\n";
}

static void plot_roc_curve(const std::vector<double>& scores, const std::vector<int>& labels, const std::string& title) {
    if (scores.size() != labels.size() || scores.empty()) {
        std::cout << "[!] Plot error: invalid ROC data\n";
        return;
    }
    std::vector<std::pair<double,int>> points;
    for (size_t i = 0; i < scores.size(); ++i) {
        points.emplace_back(scores[i], labels[i]);
    }
    std::sort(points.begin(), points.end(), [](const auto& a, const auto& b){ return a.first > b.first; });

    int positives = 0;
    int negatives = 0;
    for (auto& p : points) {
        if (p.second == 1) positives++; else negatives++;
    }
    if (positives == 0 || negatives == 0) {
        std::cout << "[!] ROC error: dataset must contain both positive and negative labels\n";
        return;
    }

    std::vector<double> tpr;
    std::vector<double> fpr;
    int tp = 0;
    int fp = 0;
    double prev_score = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].first != prev_score) {
            tpr.push_back(tp / double(positives));
            fpr.push_back(fp / double(negatives));
            prev_score = points[i].first;
        }
        if (points[i].second == 1) tp++; else fp++;
    }
    tpr.push_back(tp / double(positives));
    fpr.push_back(fp / double(negatives));

    double auc = 0.0;
    for (size_t i = 1; i < fpr.size(); ++i) {
        auc += 0.5 * (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1]);
    }

    int width = 80;
    int height = 20;
    std::vector<std::string> grid(height, std::string(width, ' '));
    for (size_t i = 0; i < fpr.size(); ++i) {
        int x = std::clamp((int)(fpr[i] * (width - 1)), 0, width - 1);
        int y = std::clamp((int)((1.0 - tpr[i]) * (height - 1)), 0, height - 1);
        grid[y][x] = '*';
    }
    for (int d = 0; d < width; ++d) {
        int y = height - 1 - d * (height - 1) / (width - 1);
        if (y >= 0 && y < height && grid[y][d] == ' ') grid[y][d] = '.';
    }

    std::cout << "\n" << title << " (AUC=" << std::fixed << std::setprecision(3) << auc << ")\n";
    std::cout << std::string(width + 2, '-') << "\n";
    for (const auto& row : grid) {
        std::cout << "|" << row << "|\n";
    }
    std::cout << std::string(width + 2, '-') << "\n";
    std::cout << "FPR from 0 to 1 horizontally, TPR from 1 to 0 vertically\n";
}

int main() {
    ShellEnv env;
    std::string line;
    std::deque<std::string> command_queue;
    std::set<std::string> active_scripts;

    std::cout <<   " _      _        _  _   _   \n"
                << "(_ |_| |_ |  |  |_  |  (_ |_/ \n"
                << "__)| | |_ |_ |_ |  _|_ __)| \\ \n"
                << "\n";

    while (true) {
        if (command_queue.empty()) {
            if (!(std::cout << "fisk> " && std::getline(std::cin, line))) break;
        } else {
            line = command_queue.front();
            command_queue.pop_front();
        }

        if (line.rfind("#__SCRIPT_END__:", 0) == 0) {
            std::string script_marker = line.substr(sizeof("#__SCRIPT_END__:") - 1);
            active_scripts.erase(script_marker);
            continue;
        }

        trim_str(line);
        if (line.empty() || line.rfind("#", 0) == 0) continue;
        env.history.push_back(line);

        std::stringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "load") {
            std::string file, name;
            ss >> file >> name;
            if (file.empty() || name.empty()) {
                std::cerr << "[!] Usage: load <file.csv> <name>\n";
                continue;
            }
            FiskData fd = FiskData::from_csv(file);
            if (fd.rows > 0) {
                push_data_state(env);
                env.data_vars[name] = fd;
                env.active_ds = name;
                std::cout << "[+] Loaded " << name << " (" << fd.rows << "x" << fd.cols << ")\n";
            } else {
                std::cerr << "[!] Failed to load " << file << "\n";
            }
        }
        else if (cmd == "ls") {
            for (auto const& [k, v] : env.data_vars)
                std::cout << " [D] " << k << " (" << v.rows << "x" << v.cols << ")\n";
            for (auto const& [k, v] : env.models)
                std::cout << " [M] " << k << "\n";
        }
        else if (cmd == "head") {
            std::string name; 
            if (!(ss >> name)) name = env.active_ds;
            if (name.empty() || !env.data_vars.count(name)) {
                std::cerr << "[!] No dataset specified or loaded.\n";
                continue;
            }
            FiskData& fd = env.data_vars[name];
            for (auto& h : fd.headers)
                std::cout << std::setw(12) << FiskData::abbrev(h, 10);
            std::cout << "\n" << std::string(fd.cols * 12, '-') << "\n";

            for (int i = 0; i < std::min(fd.rows, 5); ++i) {
                for (int j = 0; j < fd.cols; ++j) {
                    if (fd.is_factor[j] && fd.factor_maps.count(j)) {
                        auto& fmap = fd.factor_maps[j];
                        std::string val = "";
                        for (auto& kv : fmap)
                            if (kv.second == (int)fd.matrix(i,j)) { 
                                val = kv.first; 
                                break; 
                            }
                        std::cout << std::setw(12) << FiskData::abbrev(val, 10);
                    } else {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(4) << fd.matrix(i,j);
                        std::cout << std::setw(12) << FiskData::abbrev(ss.str(), 10);
                    }
                }
                std::cout << "\n";
            }
        }
        else if (cmd == "pca") {
            std::string ds_name;
            ss >> ds_name;
            if (ds_name.empty()) {
                std::cerr << "[!] Usage: pca <dataset>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_pca(env.data_vars[ds_name]);
        }
        else if (cmd == "impute") {
            std::string method;
            ss >> method;
            if (method == "mice") {
                int iterations = 5;
                std::string ds_name;
                ss >> iterations >> ds_name;
                if (ds_name.empty()) {
                    std::cerr << "[!] Usage: impute mice <iterations> <dataset>\n";
                    continue;
                }
                if (!env.data_vars.count(ds_name)) {
                    std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                    continue;
                }
                try {
                    push_data_state(env);
                    FiskTransform::impute_mice(env.data_vars[ds_name], iterations);
                    std::cout << "[+] MICE imputation completed with " << iterations << " iterations on dataset '" << ds_name << "'.\n";
                } catch (std::exception& e) {
                    std::cerr << "[!] Impute error: " << e.what() << "\n";
                }
            } else if (method == "mean") {
                std::string ds_name, col;
                ss >> ds_name >> col;
                if (col.empty()) {
                    std::cerr << "[!] Usage: impute mean <dataset> <column>\n";
                    continue;
                }
                if (!env.data_vars.count(ds_name)) {
                    std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                    continue;
                }
                try {
                    push_data_state(env);
                    FiskTransform::impute_mean(env.data_vars[ds_name], col);
                    std::cout << "[+] Mean imputation completed for column '" << col << "' in dataset '" << ds_name << "'.\n";
                } catch (std::exception& e) {
                    std::cerr << "[!] Impute error: " << e.what() << "\n";
                }
            } else {
                std::cerr << "[!] Unknown imputation method: " << method << "\n";
            }
        }
        else if (cmd == "scale") {
            std::string ds_name, col;
            ss >> ds_name >> col;
            if (col.empty()) {
                std::cerr << "[!] Usage: scale <dataset> <column>\n";
                continue;
            }
            if (ds_name.empty() || !env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            try {
                push_data_state(env);
                FiskTransform::scale_z(env.data_vars[ds_name], col);
                std::cout << "[+] Column '" << col << "' z-scaled in dataset '" << ds_name << "'.\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Scale error: " << e.what() << "\n";
            }
        }
        else if (cmd == "numeric") {
            std::string ds_name, col;
            ss >> ds_name >> col;
            if (col.empty()) {
                std::cerr << "[!] Usage: numeric <dataset> <column>\n";
                continue;
            }
            if (ds_name.empty() || !env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            try {
                push_data_state(env);
                FiskTransform::numericize(env.data_vars[ds_name], col);
                std::cout << "[+] Column '" << col << "' converted back to numeric in dataset '" << ds_name << "'.\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Numeric error: " << e.what() << "\n";
            }
        }
        else if (cmd == "factor") {
            std::string ds_name, col;
            ss >> ds_name >> col;
            if (ds_name.empty() || !env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& fd = env.data_vars[ds_name];
            int col_idx = fd.get_col_idx(col);
            
            // FIX: Added bounds check and redundancy check
            if (col_idx < 0 || col_idx >= fd.cols) {
                std::cerr << "[!] Column not found: " << col << "\n";
                continue;
            }
            if (fd.is_factor[col_idx]) {
                std::cout << "[*] Column '" << col << "' is already a factor.\n";
                continue;
            }

            push_data_state(env);
            FiskTransform::factorize(fd, col); // Assuming this handles the internal mapping
            std::cout << "[+] Column '" << col << "' factorized.\n";
        }
        else if (cmd == "glm") {
            std::string fam_type, ds_name, model_name;
            ss >> fam_type >> ds_name >> model_name;
            std::string formula;
            std::getline(ss, formula);
            formula.erase(0, formula.find_first_not_of(" \t"));
            if (fam_type.empty() || ds_name.empty() || formula.empty() || model_name.empty()) {
                std::cerr << "[!] Usage: glm <family> <dataset> <model_name> <formula>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }

            Family f = (fam_type == "poisson") ? POISSON :
                       (fam_type == "binomial") ? BINOMIAL :
                       (fam_type == "negbinom") ? NEGBINOM :
                       (fam_type == "gamma") ? GAMMA :
                       (fam_type == "invgaussian") ? INVGAUSSIAN :
                       (fam_type == "weibull") ? WEIBULL : GAUSSIAN;

            try {
                FiskModel m = FiskGLM::execute_mixed(
                    env.data_vars[ds_name], ds_name, formula, f, model_name
                );

                env.models[model_name] = m;
                std::cout << "[+] Model " << model_name << " saved and fit.\n";
            } catch(std::exception& e) {
                std::cerr << "[!] GLM error: " << e.what() << "\n";
            }
        }
        else if (cmd == "summary") {
            std::string name;
            if (!(ss >> name)) {
                if (env.models.empty()) { 
                    std::cerr << "[!] No models available.\n"; 
                    continue; 
                }
                name = env.models.rbegin()->first; // last inserted model
            }
            if (env.models.count(name)) {
                env.models[name].print_summary();
            } else if (env.data_vars.count(name)) {
                env.data_vars[name].print_summary();
            } else {
                std::cerr << "[!] Model or dataset not found: " << name << "\n";
            }
        }
        else if (cmd == "predict") {
            std::string model_name, ds_name, output_col;
            ss >> model_name >> ds_name >> output_col;
            if (model_name.empty() || ds_name.empty()) {
                std::cerr << "[!] Usage: predict <model> <dataset> [output_column]\n";
                continue;
            }
            if (!env.models.count(model_name)) {
                std::cerr << "[!] Model not found: " << model_name << "\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            if (output_col.empty()) output_col = "pred_" + model_name;
            
            try {
                FiskModel& m = env.models[model_name];
                FiskData& fd = env.data_vars[ds_name];
                
                FiskParser::Design d = FiskParser::parse_formula(fd, m.formula);
                if (!d.success) {
                    std::cerr << "[!] Cannot parse formula for predictions\n";
                    continue;
                }
                
                Eigen::VectorXd preds = m.predict(d.X);
                
                int col_idx = -1;
                for (int i = 0; i < fd.headers.size(); ++i) {
                    if (fd.headers[i] == output_col) {
                        col_idx = i;
                        break;
                    }
                }
                
                if (col_idx == -1) {
                    fd.headers.push_back(output_col);
                    fd.cols++;
                    Eigen::MatrixXd new_matrix(fd.rows, fd.cols);
                    new_matrix.leftCols(fd.cols - 1) = fd.matrix;
                    new_matrix.col(fd.cols - 1) = preds;
                    fd.matrix = new_matrix;
                    fd.is_factor.push_back(false);
                } else {
                    fd.matrix.col(col_idx) = preds;
                }
                
                std::cout << "[+] Predictions saved to column '" << output_col << "'.\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Predict error: " << e.what() << "\n";
            }
        }
        else if (cmd == "save") {
            std::string ds_name, filename;
            ss >> ds_name >> filename;
            if (ds_name.empty() || filename.empty()) {
                std::cerr << "[!] Usage: save <dataset> <filename.csv>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            
            try {
                FiskData& fd = env.data_vars[ds_name];
                std::ofstream file(filename);
                
                for (size_t i = 0; i < fd.headers.size(); ++i) {
                    if (i > 0) file << ",";
                    file << fd.headers[i];
                }
                file << "\n";
                
                for (int i = 0; i < fd.rows; ++i) {
                    for (int j = 0; j < fd.cols; ++j) {
                        if (j > 0) file << ",";
                        if (fd.is_factor[j] && fd.factor_maps.count(j)) {
                            auto& fmap = fd.factor_maps[j];
                            std::string val = "";
                            for (auto& kv : fmap) {
                                if (kv.second == (int)fd.matrix(i,j)) {
                                    val = kv.first;
                                    break;
                                }
                            }
                            file << val;
                        } else {
                            file << fd.matrix(i,j);
                        }
                    }
                    file << "\n";
                }
                
                std::cout << "[+] Dataset saved to " << filename << "\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Save error: " << e.what() << "\n";
            }
        }
        else if (cmd == "subset") {
            std::string ds_name;
            ss >> ds_name;
            std::string rest;
            std::getline(ss, rest);
            size_t open = rest.find('(');
            size_t close = rest.find(')');
            if (ds_name.empty() || open == std::string::npos || close == std::string::npos || close <= open) {
                std::cerr << "[!] Usage: subset <dataset> (<row_expr>,<col_expr>) <output_name>\n";
                std::cerr << "   blank row_expr or col_expr means all rows/cols\n";
                continue;
            }
            std::string inner = rest.substr(open + 1, close - open - 1);
            std::string output_name;
            std::stringstream after(rest.substr(close + 1));
            after >> output_name;
            if (output_name.empty()) {
                std::cerr << "[!] Subset requires output dataset name\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            std::string row_expr, col_expr;
            size_t comma = inner.find(',');
            if (comma == std::string::npos) {
                std::cerr << "[!] Subset requires both row and column expressions separated by comma\n";
                continue;
            }
            row_expr = inner.substr(0, comma);
            col_expr = inner.substr(comma + 1);
            auto trim = [&](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
                s.erase(s.find_last_not_of(" \t") + 1);
            };
            trim(row_expr);
            trim(col_expr);
            FiskData src = env.data_vars[ds_name];
            std::vector<int> row_indices = row_expr.empty() ? std::vector<int>() : parse_index_expr(row_expr, src.rows);
            std::vector<int> col_indices = col_expr.empty() ? std::vector<int>() : parse_index_expr(col_expr, src.cols);
            if (!row_expr.empty() && row_indices.empty()) {
                std::cerr << "[!] Invalid row expression\n";
                continue;
            }
            if (!col_expr.empty() && col_indices.empty()) {
                std::cerr << "[!] Invalid column expression\n";
                continue;
            }
            if (row_indices.empty()) {
                row_indices.resize(src.rows);
                std::iota(row_indices.begin(), row_indices.end(), 0);
            }
            if (col_indices.empty()) {
                col_indices.resize(src.cols);
                std::iota(col_indices.begin(), col_indices.end(), 0);
            }
            push_data_state(env);
            try {
                FiskData dst = src;
                dst.cols = col_indices.size();
                dst.rows = row_indices.size();
                dst.headers.clear();
                dst.is_factor.clear();
                dst.factor_maps.clear();
                Eigen::MatrixXd new_matrix(dst.rows, dst.cols);
                for (size_t i = 0; i < row_indices.size(); ++i) {
                    for (size_t j = 0; j < col_indices.size(); ++j) {
                        new_matrix(i, j) = src.matrix(row_indices[i], col_indices[j]);
                    }
                }
                for (size_t j = 0; j < col_indices.size(); ++j) {
                    dst.headers.push_back(src.headers[col_indices[j]]);
                    dst.is_factor.push_back(src.is_factor[col_indices[j]]);
                    if (src.factor_maps.count(col_indices[j])) {
                        dst.factor_maps[j] = src.factor_maps.at(col_indices[j]);
                    }
                }
                dst.matrix = new_matrix;
                env.data_vars[output_name] = dst;
                std::cout << "[+] Subset created: " << output_name << " (" << dst.rows << " rows, " << dst.cols << " cols)\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Subset error: " << e.what() << "\n";
            }
        }
        else if (cmd == "select") {
            std::string ds_name, output_name;
            ss >> ds_name >> output_name;
            if (ds_name.empty() || output_name.empty()) {
                std::cerr << "[!] Usage: select <dataset> <output> <col1> [col2] ...\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            
            FiskData src = env.data_vars[ds_name];
            std::vector<int> col_indices;
            std::string col;
            
            while (ss >> col) {
                int idx = src.get_col_idx(col);
                if (idx >= 0) {
                    col_indices.push_back(idx);
                } else {
                    std::cerr << "[!] Column not found: " << col << "\n";
                }
            }
            
            if (col_indices.empty()) {
                std::cerr << "[!] No valid columns selected\n";
                continue;
            }
            push_data_state(env);
            
            FiskData dst = src;
            dst.cols = col_indices.size();
            dst.headers.clear();
            dst.is_factor.clear();
            dst.factor_maps.clear();
            
            Eigen::MatrixXd new_matrix(src.rows, col_indices.size());
            for (size_t j = 0; j < col_indices.size(); ++j) {
                dst.headers.push_back(src.headers[col_indices[j]]);
                new_matrix.col(j) = src.matrix.col(col_indices[j]);
                dst.is_factor.push_back(src.is_factor[col_indices[j]]);
                if (src.factor_maps.count(col_indices[j])) {
                    dst.factor_maps[j] = src.factor_maps.at(col_indices[j]);
                }
            }
            
            dst.matrix = new_matrix;
            env.data_vars[output_name] = dst;
            std::cout << "[+] Selected dataset created: " << output_name << "\n";
        }
        else if (cmd == "append") {
            std::string ds_name, rest;
            ss >> ds_name;
            std::getline(ss, rest);
            trim_str(rest);

            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Target dataset not found: " << ds_name << "\n";
                continue;
            }

            FiskData& target = env.data_vars[ds_name];
            FiskData source_data;
            std::string new_col_name = "";
            
            std::stringstream tmp(rest);
            std::string token;
            tmp >> token;
            
            std::string maybe_name;
            std::getline(tmp, maybe_name);
            trim_str(maybe_name);
            if (!maybe_name.empty()) new_col_name = maybe_name;

            double constant_value;
            if (parse_double(token, constant_value)) {
                // Case 1: Constant numeric value
                source_data.rows = target.rows;
                source_data.cols = 1;
                source_data.matrix = Eigen::MatrixXd::Constant(target.rows, 1, constant_value);
                source_data.headers = { new_col_name.empty() ? ("const_" + token) : new_col_name };
                source_data.is_factor = { false };
                // Populate raw strings for metadata consistency
                source_data.raw_strings.resize(target.rows, { std::to_string(constant_value) });
            } 
            else if (env.data_vars.count(token)) {
                // Case 2: Source is another dataset (must be 1 column)
                const FiskData& candidate = env.data_vars.at(token);
                if (candidate.cols != 1) {
                    std::cerr << "[!] Append source dataset must have exactly one column\n";
                    continue;
                }
                if (candidate.rows != target.rows) {
                    std::cerr << "[!] Row count mismatch: " << candidate.rows << " vs " << target.rows << "\n";
                    continue;
                }
                source_data = candidate;
                if (new_col_name.empty()) new_col_name = candidate.headers[0];
            } else {
                // Case 3: Source is an existing column within the same dataset
                int idx = target.get_col_idx(token);
                if (idx < 0) {
                    std::cerr << "[!] Append source not found: " << token << "\n";
                    continue;
                }

                source_data.rows = target.rows;
                source_data.cols = 1;
                source_data.headers = { new_col_name.empty() ? token : new_col_name };
                source_data.is_factor = { target.is_factor[idx] };

                if (target.factor_maps.count(idx)) {
                    source_data.factor_maps[0] = target.factor_maps.at(idx);
                }

                source_data.matrix = target.matrix.col(idx);

                // FIX STARTS HERE: Use a safer copy for raw_strings
                source_data.raw_strings.clear(); 
                source_data.raw_strings.resize(target.rows);
                for(int i = 0; i < target.rows; ++i) {
                    // Check if the specific row actually contains the index to avoid Segfault
                    if (idx < (int)target.raw_strings[i].size()) {
                        source_data.raw_strings[i].push_back(target.raw_strings[i][idx]);
                    } else {
                        // Fallback if metadata is out of sync
                        source_data.raw_strings[i].push_back(std::to_string(target.matrix(i, idx)));
                    }
                }
            }

            if (new_col_name.empty()) new_col_name = "appended_col";

            // PERFORM THE ACTUAL APPEND
            try {
                push_data_state(env);
                int old_cols = target.cols;
                
                // 1. Resize the Eigen Matrix
                target.matrix.conservativeResize(Eigen::NoChange, old_cols + 1);
                target.matrix.col(old_cols) = source_data.matrix.col(0);
                
                // 2. FIX: Sync the metadata vectors immediately
                target.headers.push_back(new_col_name);
                target.is_factor.push_back(source_data.is_factor[0]);
                target.cols++;

                // 3. FIX: Populate the raw_strings buffer for EVERY row
                // Failure to do this causes a segfault during display
                for (int i = 0; i < target.rows; ++i) {
                    if (source_data.is_factor[0]) {
                        target.raw_strings[i].push_back(source_data.raw_strings[i][0]);
                    } else {
                        target.raw_strings[i].push_back(std::to_string(target.matrix(i, old_cols)));
                    }
                }

                // 4. FIX: Copy factor maps if they exist
                if (source_data.factor_maps.count(0)) {
                    target.factor_maps[old_cols] = source_data.factor_maps[0];
                }

                std::cout << "[+] Appended column '" << new_col_name << "' to " << ds_name << "\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Append crash prevented: " << e.what() << "\n";
            }
        }
        else if (cmd == "drop" || cmd == "delete") {
            std::string ds_name;
            ss >> ds_name;
            if (ds_name.empty()) {
                std::cerr << "[!] Usage: delete <dataset> [col1,col2,...] or drop <dataset> <col1,col2,...>\n";
                continue;
            }
            std::string rest;
            std::getline(ss, rest);
            trim_str(rest);
            if (rest.empty()) {
                if (cmd == "delete") {
                    if (!env.data_vars.count(ds_name)) {
                        std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                        continue;
                    }
                    push_data_state(env);
                    env.data_vars.erase(ds_name);
                    std::cout << "[+] Deleted dataset '" << ds_name << "'\n";
                } else {
                    std::cerr << "[!] Usage: drop <dataset> <col1,col2,...>\n";
                }
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            std::vector<std::string> vars = split_names(rest);
            if (vars.empty()) {
                std::cerr << "[!] No columns specified to drop/delete\n";
                continue;
            }
            FiskData& target = env.data_vars[ds_name];
            std::vector<bool> keep(target.cols, true);
            int remove_count = 0;
            for (auto& name : vars) {
                int idx = target.get_col_idx(name);
                if (idx < 0) {
                    std::cerr << "[!] Column not found: " << name << "\n";
                } else if (keep[idx]) {
                    keep[idx] = false;
                    remove_count += 1;
                }
            }
            if (remove_count == 0) {
                std::cerr << "[!] No valid columns removed\n";
                continue;
            }
            if (remove_count >= target.cols) {
                std::cerr << "[!] Cannot remove all columns from dataset\n";
                continue;
            }
            push_data_state(env);
            FiskData dst;
            dst.rows = target.rows;
            dst.cols = target.cols - remove_count;
            dst.matrix = Eigen::MatrixXd(target.rows, dst.cols);
            for (int j = 0, nj = 0; j < target.cols; ++j) {
                if (!keep[j]) continue;
                dst.headers.push_back(target.headers[j]);
                dst.is_factor.push_back(target.is_factor[j]);
                if (target.factor_maps.count(j)) {
                    dst.factor_maps[nj] = target.factor_maps.at(j);
                }
                dst.matrix.col(nj) = target.matrix.col(j);
                nj += 1;
            }
            env.data_vars[ds_name] = dst;
            std::cout << "[+] " << (cmd == "delete" ? "Deleted" : "Dropped") << " columns from dataset '" << ds_name << "'\n";
        }
        else if (cmd == "replace") {
            std::string ds_name, col_name, source;
            ss >> ds_name >> col_name >> source;
            if (ds_name.empty() || col_name.empty() || source.empty()) {
                std::cerr << "[!] Usage: replace <dataset> <column> <source>\n";
                std::cerr << "   source can be a numeric constant, another column, or a one-column env dataset\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& target = env.data_vars[ds_name];
            int target_idx = target.get_col_idx(col_name);
            if (target_idx < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            FiskData source_data;
            if (source.rfind("subset", 0) == 0) {
                std::string rest = source + std::string(std::istreambuf_iterator<char>(ss.rdbuf()), {});
                trim_str(rest);
                std::string err;
                if (!parse_subset_expr(rest, env, source_data, err)) {
                    std::cerr << "[!] Replace error: " << err << "\n";
                    continue;
                }
            } else {
                double constant_value;
                if (parse_double(source, constant_value)) {
                    source_data.rows = target.rows;
                    source_data.cols = 1;
                    source_data.headers = { col_name };
                    source_data.is_factor = { false };
                    source_data.matrix = Eigen::MatrixXd::Constant(target.rows, 1, constant_value);
                } else if (env.data_vars.count(source)) {
                    const FiskData& candidate = env.data_vars.at(source);
                    if (candidate.cols != 1) {
                        std::cerr << "[!] Replace source dataset must have exactly one column\n";
                        continue;
                    }
                    if (candidate.rows != target.rows) {
                        std::cerr << "[!] Replace source dataset row count must match target dataset\n";
                        continue;
                    }
                    source_data = candidate;
                } else {
                    int src_idx = target.get_col_idx(source);
                    if (src_idx < 0) {
                        std::cerr << "[!] Replace source not found: " << source << "\n";
                        continue;
                    }
                    source_data.rows = target.rows;
                    source_data.cols = 1;
                    source_data.headers = { source };
                    source_data.is_factor = { target.is_factor[src_idx] };
                    if (target.factor_maps.count(src_idx)) {
                        source_data.factor_maps[0] = target.factor_maps.at(src_idx);
                    }
                    source_data.matrix = Eigen::MatrixXd(target.rows, 1);
                    source_data.matrix.col(0) = target.matrix.col(src_idx);
                }
            }
            if (source_data.rows != target.rows) {
                std::cerr << "[!] Replace source row count must match target dataset\n";
                continue;
            }
            if (source_data.cols != 1) {
                std::cerr << "[!] Replace source must be a single column\n";
                continue;
            }
            push_data_state(env);
            target.matrix.col(target_idx) = source_data.matrix.col(0);
            target.is_factor[target_idx] = source_data.is_factor[0];
            if (source_data.factor_maps.count(0)) {
                target.factor_maps[target_idx] = source_data.factor_maps.at(0);
            } else {
                target.factor_maps.erase(target_idx);
            }
            std::cout << "[+] Replaced column '" << col_name << "' in dataset '" << ds_name << "'\n";
        }
        else if (cmd == "matrix") {
            std::string mode;
            ss >> mode;
            if (mode == "defined") {
                std::string name, values_text, dims_text;
                ss >> name >> values_text >> dims_text;
                if (name.empty() || values_text.empty() || dims_text.empty()) {
                    std::cerr << "[!] Usage: matrix defined <name> (<val1>,<val2>,...) (<nrow,ncol>)\n";
                    continue;
                }
                std::vector<double> values;
                if (!parse_number_list(values_text, values)) {
                    std::cerr << "[!] Invalid defined matrix values\n";
                    continue;
                }
                int nrow = 0, ncol = 0;
                if (!parse_nrow_ncol(dims_text, nrow, ncol)) {
                    std::cerr << "[!] Invalid matrix dimensions\n";
                    continue;
                }
                if (values.size() != size_t(nrow * ncol)) {
                    std::cerr << "[!] Defined matrix entries count does not match dimensions\n";
                    continue;
                }
                push_data_state(env);
                FiskData dst;
                dst.rows = nrow;
                dst.cols = ncol;
                dst.matrix = Eigen::MatrixXd(nrow, ncol);
                dst.headers.resize(ncol);
                dst.is_factor.assign(ncol, false);
                for (int j = 0; j < ncol; ++j) dst.headers[j] = "V" + std::to_string(j + 1);
                for (int i = 0; i < nrow; ++i)
                    for (int j = 0; j < ncol; ++j)
                        dst.matrix(i, j) = values[i * ncol + j];
                env.data_vars[name] = dst;
                env.active_ds = name;
                std::cout << "[+] Created matrix dataset '" << name << "' (" << nrow << "x" << ncol << ")\n";
            } else if (mode == "random") {
                std::string name, family, dims_text;
                ss >> name >> family >> dims_text;
                if (name.empty() || family.empty() || dims_text.empty()) {
                    std::cerr << "[!] Usage: matrix random <name> <family> (<nrow,ncol>) [params...]\n";
                    continue;
                }
                int nrow = 0, ncol = 0;
                if (!parse_nrow_ncol(dims_text, nrow, ncol)) {
                    std::cerr << "[!] Invalid matrix dimensions\n";
                    continue;
                }
                std::vector<double> params;
                std::string token;
                while (ss >> token) {
                    double v;
                    if (!parse_double(token, v)) {
                        std::cerr << "[!] Invalid distribution parameter: " << token << "\n";
                        continue;
                    }
                    params.push_back(v);
                }
                Eigen::MatrixXd mat;
                std::string err;
                if (!generate_random_matrix(family, nrow, ncol, params, mat, err)) {
                    std::cerr << "[!] Random matrix error: " << err << "\n";
                    continue;
                }
                push_data_state(env);
                FiskData dst;
                dst.rows = nrow;
                dst.cols = ncol;
                dst.matrix = mat;
                dst.headers.resize(ncol);
                dst.is_factor.assign(ncol, false);
                for (int j = 0; j < ncol; ++j) dst.headers[j] = "V" + std::to_string(j + 1);
                env.data_vars[name] = dst;
                env.active_ds = name;
                std::cout << "[+] Created random matrix dataset '" << name << "' (" << nrow << "x" << ncol << ") from " << family << "\n";
            } else if (mode == "op") {
                std::string left, op, right, output;
                ss >> left >> op >> right >> output;
                if (left.empty() || op.empty() || right.empty() || output.empty()) {
                    std::cerr << "[!] Usage: matrix op <left> <add|sub|mul|div|dot> <right|scalar> <output>\n";
                    continue;
                }
                if (!env.data_vars.count(left)) {
                    std::cerr << "[!] Matrix not found: " << left << "\n";
                    continue;
                }
                const FiskData& a = env.data_vars.at(left);
                Eigen::MatrixXd result;
                if (op == "dot") {
                    if (!env.data_vars.count(right)) {
                        std::cerr << "[!] Matrix not found: " << right << "\n";
                        continue;
                    }
                    const FiskData& b = env.data_vars.at(right);
                    if (a.cols != b.rows) {
                        std::cerr << "[!] Incompatible matrix sizes for dot product\n";
                        continue;
                    }
                    result = a.matrix * b.matrix;
                } else {
                    bool right_is_matrix = env.data_vars.count(right);
                    Eigen::MatrixXd bmat;
                    if (right_is_matrix) {
                        const FiskData& b = env.data_vars.at(right);
                        if (a.rows != b.rows || a.cols != b.cols) {
                            std::cerr << "[!] Incompatible matrix dimensions for elementwise operation\n";
                            continue;
                        }
                        bmat = b.matrix;
                    } else {
                        double scalar;
                        if (!parse_double(right, scalar)) {
                            std::cerr << "[!] Right operand must be a matrix name or scalar\n";
                            continue;
                        }
                        bmat = Eigen::MatrixXd::Constant(a.rows, a.cols, scalar);
                    }
                    if (op == "add") result = a.matrix + bmat;
                    else if (op == "sub") result = a.matrix - bmat;
                    else if (op == "mul") result = a.matrix.cwiseProduct(bmat);
                    else if (op == "div") result = a.matrix.cwiseQuotient(bmat);
                    else {
                        std::cerr << "[!] Unsupported matrix op: " << op << "\n";
                        continue;
                    }
                }
                push_data_state(env);
                FiskData dst;
                dst.rows = result.rows();
                dst.cols = result.cols();
                dst.matrix = result;
                dst.headers.resize(dst.cols);
                dst.is_factor.assign(dst.cols, false);
                for (int j = 0; j < dst.cols; ++j) dst.headers[j] = "V" + std::to_string(j + 1);
                env.data_vars[output] = dst;
                env.active_ds = output;
                std::cout << "[+] Matrix operation " << op << " stored in " << output << "\n";
            } else if (mode == "pow") {
                std::string name, exp_str, output;
                ss >> name >> exp_str >> output;
                if (name.empty() || exp_str.empty() || output.empty()) {
                    std::cerr << "[!] Usage: matrix pow <name> <exponent> <output>\n";
                    continue;
                }
                if (!env.data_vars.count(name)) {
                    std::cerr << "[!] Matrix not found: " << name << "\n";
                    continue;
                }
                double exponent;
                try { exponent = std::stod(exp_str); } catch (...) {
                    std::cerr << "[!] Invalid exponent: " << exp_str << "\n";
                    continue;
                }
                const FiskData& a = env.data_vars.at(name);
                Eigen::MatrixXd result = a.matrix.array().pow(exponent).matrix();
                push_data_state(env);
                FiskData dst;
                dst.rows = result.rows();
                dst.cols = result.cols();
                dst.matrix = result;
                dst.headers.resize(dst.cols);
                dst.is_factor.assign(dst.cols, false);
                for (int j = 0; j < dst.cols; ++j) dst.headers[j] = "V" + std::to_string(j + 1);
                env.data_vars[output] = dst;
                env.active_ds = output;
                std::cout << "[+] Created powered matrix " << output << "\n";
            } else if (mode == "transpose") {
                std::string name, output;
                ss >> name >> output;
                if (name.empty() || output.empty()) {
                    std::cerr << "[!] Usage: matrix transpose <name> <output>\n";
                    continue;
                }
                if (!env.data_vars.count(name)) {
                    std::cerr << "[!] Matrix not found: " << name << "\n";
                    continue;
                }
                const FiskData& a = env.data_vars.at(name);
                Eigen::MatrixXd result = a.matrix.transpose();
                push_data_state(env);
                FiskData dst;
                dst.rows = result.rows();
                dst.cols = result.cols();
                dst.matrix = result;
                dst.headers.resize(dst.cols);
                dst.is_factor.assign(dst.cols, false);
                for (int j = 0; j < dst.cols; ++j) dst.headers[j] = "V" + std::to_string(j + 1);
                env.data_vars[output] = dst;
                env.active_ds = output;
                std::cout << "[+] Created transposed matrix " << output << "\n";
            } else {
                std::cerr << "[!] Unsupported matrix mode: " << mode << "\n";
            }
        }
        else if (cmd == "rename") {
            std::string ds_name, old_name, new_name;
            ss >> ds_name >> old_name >> new_name;
            
            if (ds_name.empty() || old_name.empty() || new_name.empty()) {
                std::cerr << "[!] Usage: rename <dataset> <old_name> <new_name>\n";
                continue;
            }

            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }

            FiskData& target = env.data_vars[ds_name];
            int idx = target.get_col_idx(old_name);
            
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << old_name << "\n";
                continue;
            }

            if (target.get_col_idx(new_name) != -1) {
                std::cerr << "[!] Error: Column name '" << new_name << "' already exists.\n";
                continue;
            }

            // CRITICAL FIX: Initialize raw_strings if they are missing (common after 'subset' or 'matrix')
            if (target.raw_strings.empty() && target.rows > 0) {
                target.raw_strings.resize(target.rows);
            }

            for (int i = 0; i < target.rows; ++i) {
                // If this specific row is empty or too short, fill it up to the current column count
                if ((int)target.raw_strings[i].size() < target.cols) {
                    int missing = target.cols - target.raw_strings[i].size();
                    for (int m = 0; m < missing; ++m) {
                        int col_to_fill = target.raw_strings[i].size();
                        target.raw_strings[i].push_back(std::to_string(target.matrix(i, col_to_fill)));
                    }
                }
            }

            target.headers[idx] = new_name;
            std::cout << "[+] Renamed '" << old_name << "' to '" << new_name << "'\n";
        }
        else if (cmd == "fillna") {
            std::string ds_name, col_name, value;
            ss >> ds_name >> col_name >> value;
            if (ds_name.empty() || col_name.empty() || value.empty()) {
                std::cerr << "[!] Usage: fillna <dataset> <column> <value>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& target = env.data_vars[ds_name];
            int idx = target.get_col_idx(col_name);
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            double numeric_value;
            bool is_const = parse_double(value, numeric_value);
            push_data_state(env);
            for (int i = 0; i < target.rows; ++i) {
                if (std::isnan(target.matrix(i, idx))) {
                    if (is_const) {
                        target.matrix(i, idx) = numeric_value;
                    } else if (target.is_factor[idx]) {
                        if (!target.factor_maps.count(idx)) target.factor_maps[idx];
                        if (target.factor_maps[idx].count(value) == 0) {
                            int code = target.factor_maps[idx].size();
                            target.factor_maps[idx][value] = code;
                        }
                        target.matrix(i, idx) = target.factor_maps[idx][value];
                    } else {
                        std::cerr << "[!] Cannot fill non-numeric missing value with non-numeric token\n";
                        break;
                    }
                }
            }
            std::cout << "[+] Filled missing values in " << ds_name << "." << col_name << "\n";
        }
        else if (cmd == "dropna") {
            std::string ds_name;
            ss >> ds_name;
            if (ds_name.empty()) {
                std::cerr << "[!] Usage: dropna <dataset> [col1,col2,...]\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            std::string rest;
            std::getline(ss, rest);
            trim_str(rest);
            std::vector<std::string> cols = rest.empty() ? std::vector<std::string>() : split_names(rest);
            FiskData& target = env.data_vars[ds_name];
            std::vector<int> col_indices;
            if (cols.empty()) {
                col_indices.resize(target.cols);
                std::iota(col_indices.begin(), col_indices.end(), 0);
            } else {
                for (auto& col : cols) {
                    int idx = target.get_col_idx(col);
                    if (idx < 0) {
                        std::cerr << "[!] Column not found: " << col << "\n";
                    } else {
                        col_indices.push_back(idx);
                    }
                }
                if (col_indices.empty()) {
                    std::cerr << "[!] No valid columns specified for dropna\n";
                    continue;
                }
            }
            std::vector<int> keep_rows;
            for (int i = 0; i < target.rows; ++i) {
                bool missing = false;
                for (int j : col_indices) {
                    if (std::isnan(target.matrix(i, j))) { missing = true; break; }
                }
                if (!missing) keep_rows.push_back(i);
            }
            if (keep_rows.size() == target.rows) {
                std::cout << "[+] No rows dropped from " << ds_name << "\n";
                continue;
            }
            push_data_state(env);
            FiskData dst = target;
            dst.rows = keep_rows.size();
            dst.matrix = Eigen::MatrixXd(dst.rows, dst.cols);
            for (int i = 0; i < dst.rows; ++i) dst.matrix.row(i) = target.matrix.row(keep_rows[i]);
            env.data_vars[ds_name] = dst;
            std::cout << "[+] Dropped " << (target.rows - dst.rows) << " rows from " << ds_name << "\n";
        }
        else if (cmd == "distinct") {
            std::string ds_name, col_name, output_name;
            ss >> ds_name >> col_name >> output_name;
            if (ds_name.empty() || col_name.empty()) {
                std::cerr << "[!] Usage: distinct <dataset> <column> [output] \n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            const FiskData& target = env.data_vars[ds_name];
            int idx = target.get_col_idx(col_name);
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            std::set<double> seen;
            std::vector<double> values;
            for (int i = 0; i < target.rows; ++i) {
                double v = target.matrix(i, idx);
                if (std::isnan(v)) continue;
                if (!seen.count(v)) {
                    seen.insert(v);
                    values.push_back(v);
                }
            }
            if (output_name.empty()) output_name = ds_name + "_distinct_" + col_name;
            FiskData dst;
            dst.rows = values.size();
            dst.cols = 1;
            dst.headers = { col_name };
            dst.is_factor = { false };
            dst.matrix = Eigen::MatrixXd(dst.rows, 1);
            for (int i = 0; i < dst.rows; ++i) dst.matrix(i, 0) = values[i];
            env.data_vars[output_name] = dst;
            std::cout << "[+] Created distinct values dataset: " << output_name << "\n";
        }
        else if (cmd == "count") {
            std::string ds_name, col_name;
            ss >> ds_name >> col_name;
            if (ds_name.empty() || col_name.empty()) {
                std::cerr << "[!] Usage: count <dataset> <column>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            const FiskData& target = env.data_vars[ds_name];
            int idx = target.get_col_idx(col_name);
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            std::map<std::string, int> counts;
            for (int i = 0; i < target.rows; ++i) {
                std::string value = target.is_factor[idx] ? target.raw_strings[i][idx] : std::to_string(target.matrix(i, idx));
                counts[value]++;
            }
            std::cout << "Counts for " << col_name << " in " << ds_name << ":\n";
            for (auto& kv : counts) {
                std::cout << "  " << kv.first << ": " << kv.second << "\n";
            }
        }
        else if (cmd == "mutate") {
            std::string ds_name, var_name;
            ss >> ds_name >> var_name;
            if (ds_name.empty() || var_name.empty()) {
                std::cerr << "[!] Usage: mutate <dataset> <newvar> <expression>\n";
                std::cerr << "   example: mutate data newcol col1 + col2\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            
            FiskData& fd = env.data_vars[ds_name];
            std::string expr;
            std::getline(ss, expr);
            expr.erase(0, expr.find_first_not_of(" \t"));
            
            try {
                std::vector<double> result(fd.rows, 0.0);
                bool is_expr = (expr.find('+') != std::string::npos || 
                               expr.find('-') != std::string::npos ||
                               expr.find('*') != std::string::npos ||
                               expr.find('/') != std::string::npos);
                
                if (is_expr && expr.find(',') == std::string::npos) {
                    std::stringstream expr_ss(expr);
                    std::string col1, op, col2;
                    expr_ss >> col1 >> op >> col2;
                    
                    int idx1 = fd.get_col_idx(col1);
                    int idx2 = fd.get_col_idx(col2);
                    
                    if (idx1 >= 0 && idx2 >= 0) {
                        for (int i = 0; i < fd.rows; ++i) {
                            double v1 = fd.matrix(i, idx1);
                            double v2 = fd.matrix(i, idx2);
                            if (op == "+") result[i] = v1 + v2;
                            else if (op == "-") result[i] = v1 - v2;
                            else if (op == "*") result[i] = v1 * v2;
                            else if (op == "/") result[i] = v2 != 0 ? v1 / v2 : std::numeric_limits<double>::quiet_NaN();
                        }
                    }
                }
                
                push_data_state(env);
                fd.headers.push_back(var_name);
                fd.cols++;
                fd.is_factor.push_back(false);
                
                Eigen::MatrixXd new_matrix(fd.rows, fd.cols);
                new_matrix.leftCols(fd.cols - 1) = fd.matrix;
                new_matrix.col(fd.cols - 1) = Eigen::VectorXd::Map(result.data(), result.size());
                fd.matrix = new_matrix;
                
                std::cout << "[+] Variable '" << var_name << "' created in dataset '" << ds_name << "'.\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Mutate error: " << e.what() << "\n";
            }
        }
        else if (cmd == "power") {
            std::string ds_name, newvar, source, exp_str;
            ss >> ds_name >> newvar >> source >> exp_str;
            if (ds_name.empty() || newvar.empty() || source.empty() || exp_str.empty()) {
                std::cerr << "[!] Usage: power <dataset> <newvar> <source> <exponent>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& fd = env.data_vars[ds_name];
            int idx = fd.get_col_idx(source);
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << source << "\n";
                continue;
            }
            double exponent = std::stod(exp_str);
            push_data_state(env);
            std::vector<double> result(fd.rows);
            for (int i = 0; i < fd.rows; ++i) {
                result[i] = std::pow(fd.matrix(i, idx), exponent);
            }
            fd.headers.push_back(newvar);
            fd.cols++;
            fd.is_factor.push_back(false);
            Eigen::MatrixXd new_matrix(fd.rows, fd.cols);
            new_matrix.leftCols(fd.cols - 1) = fd.matrix;
            new_matrix.col(fd.cols - 1) = Eigen::VectorXd::Map(result.data(), result.size());
            fd.matrix = new_matrix;
            std::cout << "[+] Created power variable '" << newvar << "' in " << ds_name << "\n";
        }
        else if (cmd == "log") {
            std::string ds_name, newvar, source;
            ss >> ds_name >> newvar >> source;
            if (ds_name.empty() || newvar.empty() || source.empty()) {
                std::cerr << "[!] Usage: log <dataset> <newvar> <source>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& fd = env.data_vars[ds_name];
            int idx = fd.get_col_idx(source);
            if (idx < 0) {
                std::cerr << "[!] Column not found: " << source << "\n";
                continue;
            }
            push_data_state(env);
            std::vector<double> result(fd.rows);
            for (int i = 0; i < fd.rows; ++i) {
                result[i] = std::log(fd.matrix(i, idx));
            }
            fd.headers.push_back(newvar);
            fd.cols++;
            fd.is_factor.push_back(false);
            Eigen::MatrixXd new_matrix(fd.rows, fd.cols);
            new_matrix.leftCols(fd.cols - 1) = fd.matrix;
            new_matrix.col(fd.cols - 1) = Eigen::VectorXd::Map(result.data(), result.size());
            fd.matrix = new_matrix;
            std::cout << "[+] Created log variable '" << newvar << "' in " << ds_name << "\n";
        }
        else if (cmd == "quickplot") {
            std::string kind;
            ss >> kind;
            if (kind.empty()) {
                std::cerr << "[!] Usage: quickplot <dataset|pca|roc> ...\n";
                continue;
            }
            if (kind == "pca") {
                std::string ds_name;
                int pc1, pc2;
                ss >> ds_name >> pc1 >> pc2;
                if (ds_name.empty() || pc1 <= 0 || pc2 <= 0) {
                    std::cerr << "[!] Usage: quickplot pca <dataset> <pc1> <pc2>\n";
                    continue;
                }
                if (!env.data_vars.count(ds_name)) {
                    std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                    continue;
                }
                FiskData& fd = env.data_vars[ds_name];
                Eigen::MatrixXd X = fd.matrix;
                Eigen::VectorXd mean = X.colwise().mean();
                X.rowwise() -= mean.transpose();
                Eigen::MatrixXd cov = (X.transpose() * X) / (X.rows() - 1);
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
                Eigen::MatrixXd scores = X * eig.eigenvectors().rowwise().reverse();
                int n = scores.rows();
                int i1 = pc1 - 1;
                int i2 = pc2 - 1;
                if (i1 < 0 || i1 >= scores.cols() || i2 < 0 || i2 >= scores.cols()) {
                    std::cerr << "[!] PCA component index out of range\n";
                    continue;
                }
                std::vector<double> xs(n), ys(n);
                for (int i = 0; i < n; ++i) {
                    xs[i] = scores(i, i1);
                    ys[i] = scores(i, i2);
                }
                quickplot_scatter(xs, ys, "PCA scatter: PC" + std::to_string(pc1) + " vs PC" + std::to_string(pc2));
            } else if (kind == "roc") {
                std::string ds_name, score_col, label_col;
                ss >> ds_name >> score_col >> label_col;
                if (ds_name.empty() || score_col.empty() || label_col.empty()) {
                    std::cerr << "[!] Usage: quickplot roc <dataset> <score_col> <label_col>\n";
                    continue;
                }
                if (!env.data_vars.count(ds_name)) {
                    std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                    continue;
                }
                FiskData& fd = env.data_vars[ds_name];
                int score_idx = fd.get_col_idx(score_col);
                int label_idx = fd.get_col_idx(label_col);
                if (score_idx < 0 || label_idx < 0) {
                    std::cerr << "[!] Score or label column not found\n";
                    continue;
                }
                std::vector<double> scores(fd.rows);
                std::vector<int> labels(fd.rows);
                for (int i = 0; i < fd.rows; ++i) {
                    scores[i] = fd.matrix(i, score_idx);
                    std::string lab = FiskAnalyzer::value_label(fd, label_idx, i);
                    trim_str(lab);
                    if (lab == "1" || lab == "true" || lab == "True" || lab == "TRUE" || lab == "yes" || lab == "Y") labels[i] = 1;
                    else labels[i] = 0;
                }
                quickplot_roc_curve(scores, labels, "ROC curve: " + ds_name + " (" + score_col + " vs " + label_col + ")");
            } else {
                // If we reach here, 'kind' actually contains the dataset name
                std::string ds_name = kind; 
                int c1, c2;
                
                // Now we only need to pull the two column indices
                if (!(ss >> c1 >> c2)) {
                    std::cerr << "[!] Usage: quickplot <dataset> <col1_idx> <col2_idx>\n";
                    continue;
                }

                if (!env.data_vars.count(ds_name)) {
                    std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                    continue;
                }
                
                FiskData& fd = env.data_vars[ds_name];
                int i1 = c1 - 1;
                int i2 = c2 - 1;

                if (i1 < 0 || i1 >= fd.cols || i2 < 0 || i2 >= fd.cols) {
                    std::cerr << "[!] Column index out of range (Dataset has " << fd.cols << " columns)\n";
                    continue;
                }

                std::vector<double> xs(fd.rows), ys(fd.rows);
                for (int i = 0; i < fd.rows; ++i) {
                    xs[i] = fd.matrix(i, i1);
                    ys[i] = fd.matrix(i, i2);
                }
                
                quickplot_scatter(xs, ys, "Scatter: " + ds_name + " (Col " + std::to_string(c1) + " vs " + std::to_string(c2) + ")");
            }
        }
        else if (cmd == "plot") {
            std::string type, ds_name, x_col, y_col;
            ss >> type >> ds_name >> x_col >> y_col;

            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }

            FiskData& fd = env.data_vars[ds_name];
            int idxX = fd.get_col_idx(x_col);
            int idxY = fd.get_col_idx(y_col);

            if (idxX < 0 || idxY < 0) {
                std::cerr << "[!] Columns not found.\n";
                continue;
            }

            // Extract the data from the matrix
            std::vector<double> vx, vy;
            for (int i = 0; i < fd.rows; ++i) {
                vx.push_back(fd.matrix(i, idxX));
                vy.push_back(fd.matrix(i, idxY));
            }

            FiskNativePlotter::PlotRequest* req = new FiskNativePlotter::PlotRequest();
            req->title = type + " Plot: " + ds_name;
            req->xlabel = x_col;
            req->ylabel = y_col;
            
            FiskNativePlotter::Style s = FiskNativePlotter::POINTS;
            if (type == "line") s = FiskNativePlotter::LINES;
            if (type == "survival") s = FiskNativePlotter::STEPS;

            // Use the renamed 'series' member and provide a color
            req->series.push_back({ds_name, vx, vy, s, RGB(0, 0, 0)});
            
            FiskNativePlotter::Show(req);
            std::cout << "[+] Plot window opened for " << ds_name << "\n";
        }
        else if (cmd == "survival") {
            std::string ds_name, time_col, event_col, group_col;
            ss >> ds_name >> time_col >> event_col >> group_col;
            if (ds_name.empty() || time_col.empty() || event_col.empty()) {
                std::cerr << "[!] Usage: survival <dataset> <time_col> <event_col> [group_col]\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            if (group_col.empty()) {
                FiskAnalyzer::run_survival(env.data_vars[ds_name], time_col, event_col);
            } else {
                FiskAnalyzer::run_survival(env.data_vars[ds_name], time_col, event_col, group_col);
            }
        }
        else if (cmd == "subgroup") {
            std::string ds_name, group_col, outcome_col;
            ss >> ds_name >> group_col >> outcome_col;
            if (ds_name.empty() || group_col.empty() || outcome_col.empty()) {
                std::cerr << "[!] Usage: subgroup <dataset> <group_col> <outcome_col>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_subgroup(env.data_vars[ds_name], group_col, outcome_col);
        }
        else if (cmd == "anova") {
            std::string ds_name, response_col, factor_col;
            ss >> ds_name >> response_col >> factor_col;
            if (ds_name.empty() || response_col.empty() || factor_col.empty()) {
                std::cerr << "[!] Usage: anova <dataset> <response> <factor>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_anova(env.data_vars[ds_name], response_col, factor_col);
        }
        else if (cmd == "wilcoxon") {
            std::string ds_name, group_col, value_col;
            ss >> ds_name >> group_col >> value_col;
            if (ds_name.empty() || group_col.empty() || value_col.empty()) {
                std::cerr << "[!] Usage: wilcoxon <dataset> <group_col> <value_col>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_wilcoxon(env.data_vars[ds_name], group_col, value_col);
        }
        else if (cmd == "kruskal") {
            std::string ds_name, group_col, value_col;
            ss >> ds_name >> group_col >> value_col;
            if (ds_name.empty() || group_col.empty() || value_col.empty()) {
                std::cerr << "[!] Usage: kruskal <dataset> <group_col> <value_col>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_kruskal(env.data_vars[ds_name], group_col, value_col);
        }
        else if (cmd == "spearman") {
            std::string ds_name, x_col, y_col;
            ss >> ds_name >> x_col >> y_col;
            if (ds_name.empty() || x_col.empty() || y_col.empty()) {
                std::cerr << "[!] Usage: spearman <dataset> <x_col> <y_col>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_spearman(env.data_vars[ds_name], x_col, y_col);
        }
        else if (cmd == "chi2") {
            std::string ds_name, row_col, col_col;
            ss >> ds_name >> row_col >> col_col;
            if (ds_name.empty() || row_col.empty() || col_col.empty()) {
                std::cerr << "[!] Usage: chi2 <dataset> <row_col> <col_col>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskAnalyzer::run_chi2(env.data_vars[ds_name], row_col, col_col);
        }
        else if (cmd == "diagnose") {
            std::string model_name;
            ss >> model_name;
            if (model_name.empty()) {
                std::cerr << "[!] Usage: diagnose <model>\n";
                continue;
            }
            if (!env.models.count(model_name)) {
                std::cerr << "[!] Model not found: " << model_name << "\n";
                continue;
            }
            env.models[model_name].print_performance();
        }
        else if (cmd == "undo") {
            if (env.undo_stack.empty()) {
                std::cerr << "[!] Nothing to undo\n";
                continue;
            }
            env.redo_stack.push_back(env.data_vars);
            env.data_vars = env.undo_stack.back();
            env.undo_stack.pop_back();
            std::cout << "[+] Undo complete\n";
        }
        else if (cmd == "redo") {
            if (env.redo_stack.empty()) {
                std::cerr << "[!] Nothing to redo\n";
                continue;
            }
            env.undo_stack.push_back(env.data_vars);
            env.data_vars = env.redo_stack.back();
            env.redo_stack.pop_back();
            std::cout << "[+] Redo complete\n";
        }
        else if (cmd == "set") {
            std::string ds_name, row_str, col_name, value;
            ss >> ds_name >> row_str >> col_name >> value;
            if (ds_name.empty() || row_str.empty() || col_name.empty() || value.empty()) {
                std::cerr << "[!] Usage: set <dataset> <row> <column> <value>\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            FiskData& fd = env.data_vars[ds_name];
            int row = std::stoi(row_str) - 1;
            int col = fd.get_col_idx(col_name);
            if (row < 0 || row >= fd.rows) {
                std::cerr << "[!] Row out of bounds\n";
                continue;
            }
            if (col < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            try {
                push_data_state(env);
                if (fd.is_factor[col]) {
                    if (!fd.factor_maps.count(col)) fd.factor_maps[col];
                    if (fd.factor_maps[col].count(value) == 0) {
                        int new_code = fd.factor_maps[col].size();
                        fd.factor_maps[col][value] = new_code;
                    }
                    fd.matrix(row, col) = fd.factor_maps[col][value];
                } else {
                    fd.matrix(row, col) = std::stod(value);
                }
                std::cout << "[+] Set [" << row + 1 << ", " << col_name << "] = " << value << "\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Set error: " << e.what() << "\n";
            }
        }
        else if (cmd == "filter") {
            std::string ds_name, col_name, op, value, output_name;
            ss >> ds_name >> col_name >> op >> value >> output_name;
            if (ds_name.empty() || col_name.empty() || op.empty() || value.empty() || output_name.empty()) {
                std::cerr << "[!] Usage: filter <dataset> <column> <op> <value> <output>\n";
                std::cerr << "   ops: == != < > <= >=\n";
                std::cerr << "   example: filter data age > 30 adults\n";
                continue;
            }
            if (!env.data_vars.count(ds_name)) {
                std::cerr << "[!] Dataset not found: " << ds_name << "\n";
                continue;
            }
            
            FiskData src = env.data_vars[ds_name];
            int col_idx = src.get_col_idx(col_name);
            if (col_idx < 0) {
                std::cerr << "[!] Column not found: " << col_name << "\n";
                continue;
            }
            
            try {
                double threshold = std::stod(value);
                std::vector<int> row_indices;
                
                for (int i = 0; i < src.rows; ++i) {
                    double v = src.matrix(i, col_idx);
                    if (std::isnan(v)) continue;
                    
                    bool keep = false;
                    if (op == "==") keep = (v == threshold);
                    else if (op == "!=") keep = (v != threshold);
                    else if (op == "<") keep = (v < threshold);
                    else if (op == ">") keep = (v > threshold);
                    else if (op == "<=") keep = (v <= threshold);
                    else if (op == ">=") keep = (v >= threshold);
                    
                    if (keep) row_indices.push_back(i);
                }
                
                push_data_state(env);
                FiskData dst = src;
                dst.rows = row_indices.size();
                Eigen::MatrixXd new_matrix(row_indices.size(), src.cols);
                for (size_t i = 0; i < row_indices.size(); ++i) {
                    new_matrix.row(i) = src.matrix.row(row_indices[i]);
                }
                
                dst.matrix = new_matrix;
                env.data_vars[output_name] = dst;
                std::cout << "[+] Filtered dataset created: " << output_name << " (" << row_indices.size() << " rows)\n";
            } catch (std::exception& e) {
                std::cerr << "[!] Filter error: " << e.what() << "\n";
            }
        }
        else if (cmd == "seed") {
            std::string seed_text;
            ss >> seed_text;
            if (seed_text.empty()) {
                std::cerr << "[!] Usage: seed <integer>\n";
                continue;
            }
            try {
                unsigned int seed = std::stoul(seed_text);
                set_global_seed(seed);
                std::cout << "[+] Global RNG seeded to " << seed << "\n";
            } catch (...) {
                std::cerr << "[!] Invalid seed value: " << seed_text << "\n";
            }
        }
        else if (cmd == "script") {
            std::string file;
            ss >> file;
            if (file.empty()) {
                std::cerr << "[!] Usage: script <file>\n";
                continue;
            }
            std::filesystem::path script_path(file);
            if (!script_path.is_absolute()) script_path = std::filesystem::current_path() / script_path;
            if (!std::filesystem::exists(script_path)) {
                std::cerr << "[!] Script file not found: " << script_path.string() << "\n";
                continue;
            }
            std::string abs_path = std::filesystem::absolute(script_path).string();
            if (!active_scripts.insert(abs_path).second) {
                std::cerr << "[!] Recursive script include blocked: " << abs_path << "\n";
                continue;
            }
            std::ifstream script_file(abs_path);
            if (!script_file.is_open()) {
                std::cerr << "[!] Unable to open script file: " << abs_path << "\n";
                active_scripts.erase(abs_path);
                continue;
            }
            std::vector<std::string> script_lines;
            std::string script_line;
            while (std::getline(script_file, script_line)) {
                trim_str(script_line);
                if (script_line.empty() || script_line.rfind("#", 0) == 0) continue;
                script_lines.push_back(script_line);
            }
            for (auto it = script_lines.rbegin(); it != script_lines.rend(); ++it) {
                command_queue.push_front(*it);
            }
            command_queue.push_front("#__SCRIPT_END__:" + abs_path);
            std::cout << "[+] Queued script " << abs_path << " (" << script_lines.size() << " commands)\n";
        }
        else if (cmd == "export") {
            std::string filename;
            ss >> filename;
            if (filename.empty()) {
                std::cerr << "[!] Usage: export <file>\n";
                continue;
            }
            std::filesystem::path script_path(filename);
            std::string err;
            if (!export_environment(env, script_path, err)) {
                std::cerr << "[!] Export failed: " << err << "\n";
                continue;
            }
            std::cout << "[+] Environment exported to " << script_path.string() << "\n";
        }
        else if (cmd == "momento") {
            std::string filename;
            ss >> filename;
            if (filename.empty()) {
                std::cerr << "[!] Usage: momento <file>\n";
                continue;
            }
            std::filesystem::path momento_path(filename);
            std::string err;
            if (!write_momento(env, momento_path, err)) {
                std::cerr << "[!] Momento write failed: " << err << "\n";
                continue;
            }
            std::cout << "[+] Momento written to " << momento_path.string() << "\n";
        }
        else if (cmd == "help" || cmd == "?") {
            std::cout << "\nAvailable Commands:\n";
            std::cout << std::string(80, '=') << "\n\n";
            
            std::cout << "Data Loading & Manipulation:\n";
            std::cout << "  load <file.csv> <name>              Load CSV file into dataset\n";
            std::cout << "  ls                                  List all datasets and models\n";
            std::cout << "  head [dataset]                      Show first 5 rows of dataset\n";
            std::cout << "  save <dataset> <filename.csv>       Save dataset to CSV file\n";
            std::cout << "  subset <dataset> (<rows>,<cols>) <output>    Subset rows/columns\n";
            std::cout << "    blank rows or cols means all\n";
            std::cout << "  select <dataset> <output> <col1> [col2]...   Select columns\n";
            std::cout << "  append <dataset> <source> [new_column_name]  Append a new column from constant, column, or subset source\n";
            std::cout << "  replace <dataset> <column> <source>          Replace a dataset column while keeping its name\n";
            std::cout << "  rename <dataset> <old_name> <new_name>      Rename a column\n";
            std::cout << "  fillna <dataset> <column> <value>           Fill missing values in a column\n";
            std::cout << "  dropna <dataset> [col1,col2,...]            Remove rows with missing values\n";
            std::cout << "  distinct <dataset> <column> [output]         Extract distinct values\n";
            std::cout << "  count <dataset> <column>                    Count unique values in a column\n";
            std::cout << "  delete <dataset>                        Remove dataset from environment\n";
            std::cout << "  delete <dataset> <col1,col2,...>       Drop columns from dataset\n";
            std::cout << "  drop <dataset> <col1,col2,...>         Alias for dropping columns\n";
            std::cout << "  mutate <dataset> <newvar> <expression>      Add computed column\n";
            std::cout << "    expr: col1 + col2, col1 * col2, etc.\n";
            std::cout << "  power <dataset> <newvar> <source> <exp>     Power transform\n";
            std::cout << "  log <dataset> <newvar> <source>             Natural log transform\n";
            std::cout << "  matrix defined <name> (<vals>) (<nrow,ncol>)              Create custom matrix dataset\n";
            std::cout << "  matrix random <name> <family> (<nrow,ncol>) [params...]   Create random matrix from distribution\n";
            std::cout << "  matrix op <left> <add|sub|mul|div|dot> <right|scalar> <output>   Matrix arithmetic\n";
            std::cout << "  matrix pow <name> <exponent> <output>            Elementwise power\n";
            std::cout << "  matrix transpose <name> <output>                 Transpose matrix dataset\n";
            std::cout << "  set <dataset> <row> <column> <value>        Modify single entry\n";
            std::cout << "  filter <dataset> <col> <op> <val> <output>  Filter rows by condition\n";
            std::cout << "    ops: == != < > <= >=\n";
            std::cout << "  undo                                Undo last dataset change\n";
            std::cout << "  redo                                Redo last undone change\n";
            
            std::cout << "Data Transformation:\n";
            std::cout << "  factor <dataset> <column>      Convert column to factor (categorical)\n";
            std::cout << "  numeric <dataset> <column>     Convert factor back to numeric\n";
            std::cout << "  scale <dataset> <column>       Z-score normalization (standardize)\n";
            std::cout << "  impute mice <iter> <dataset>   MICE imputation (default 5 iterations)\n";
            std::cout << "  impute mean <dataset> <col>    Mean imputation for numeric column\n\n";            
            std::cout << "Statistical Analysis:\n";
            std::cout << "  pca <dataset>                  Principal Component Analysis\n";
            std::cout << "  quickplot roc <dataset> <score_col> <label_col>  Quick inline ROC curve\n";
            std::cout << "  quickplot <dataset> <col1> <col2>              Quick inline scatter plot\n";
            std::cout << "  glm <fam> <ds> <name> <form>  Fit GLM model\n";
            std::cout << "    families: binomial | poisson | gaussian | negbinom | gamma | invgaussian | weibull\n";
            std::cout << "    formula: response = fixed + (1|random)\n";
            std::cout << "    example: glm binomial titanic model1 Survived = Sex + Age\n";
            std::cout << "  anova <dataset> <response> <factor>             One-way ANOVA by factor\n";
            std::cout << "  wilcoxon <dataset> <group_col> <value_col>      Wilcoxon rank-sum test (two groups)\n";
            std::cout << "  kruskal <dataset> <group_col> <value_col>       Kruskal-Wallis rank test\n";
            std::cout << "  spearman <dataset> <x_col> <y_col>              Spearman rank correlation\n";
            std::cout << "  chi2 <dataset> <row_col> <col_col>              Chi-squared test of independence\n";
            std::cout << "  survival <dataset> <time> <event> [group]       Kaplan-Meier survival estimate\n";
            std::cout << "  subgroup <dataset> <group> <outcome>            Subgroup summary analysis\n";
            std::cout << "  diagnose <model>                                  Print model diagnostics and performance\n\n";
            
            std::cout << "Model & Data Inspection:\n";
            std::cout << "  summary [name]                 Summary of model or dataset\n";
            std::cout << "  predict <model> <dataset> [col] Get predictions from model\n";
            std::cout << "    (col auto-named if omitted)\n\n";
            
            std::cout << "System:\n";
            std::cout << "  help, ?                        Show this help message\n";
            std::cout << "  seed <integer>                 Seed the random generator for reproducible draws\n";
            std::cout << "  script <file>                  Execute commands from a script file\n";
            std::cout << "  export <file>                  Save current datasets and a load script for this env\n";
            std::cout << "  momento <file>                 Write command history to a replay script\n";
            std::cout << "  exit                           Exit the shell\n\n";
        }
        else if (cmd == "exit") {
            break;
        }
        else {
            std::cerr << "[!] Unknown command: " << cmd << "\n";
        }
    }

    return 0;
}