### Introduction

shellFish is as a self-contained environment to perform reproducible data-engineering and statistical analysis at high performance for common biostatistics tasks. The language was deisgned to resemble UNIX commands for ease of access and simplicity. The environment supports undo/redo functionality, the ability to write all commands made during a session to a file for traceability, and the ability to export an environment state, similar to R. The project is currently under development and note all features work as intended.

<img width="357" height="357" alt="shellFisk" src="https://github.com/user-attachments/assets/af7fa8f7-b8ac-4ee7-8092-1a63526ce876" />

### Commands

```text
Data Loading & Manipulation:
load <file.csv> <name>                 Load CSV file into dataset
ls                                     List all datasets and models
head [dataset]                         Show first 5 rows of dataset
save <dataset> <filename.csv>          Save dataset to CSV file
subset <dataset> (<rows>,<cols>) <out> Subset rows/columns (blank for all)
select <dataset> <output> <col1>...    Select columns
append <dataset> <src> [new_name]      Append column from constant/subset
replace <dataset> <column> <source>    Replace column while keeping name
rename <dataset> <old_name> <new_name> Rename a column
fillna <dataset> <column> <value>      Fill missing values
dropna <dataset> [col1,col2,...]       Remove rows with missing values
distinct <dataset> <column> [output]   Extract distinct values
count <dataset> <column>               Count unique values
delete <dataset>                       Remove dataset from environment
delete <dataset> <col1,col2,...>       Drop columns from dataset
mutate <dataset> <newvar> <expr>       Add computed column (e.g. col1 + col2)
power <dataset> <newvar> <src> <exp>   Power transform
log <dataset> <newvar> <source>        Natural log transform

Matrix Operations:
matrix defined <name> (<vals>) (<r,c>) Create custom matrix dataset
matrix random <name> <fam> (<r,c>)     Create random matrix from distribution
matrix op <l> <op> <r> <out>           Matrix arithmetic (add|sub|mul|div|dot)
matrix pow <name> <exp> <out>          Elementwise power
matrix transpose <name> <out>          Transpose matrix dataset
set <dataset> <row> <col> <val>        Modify single entry
filter <ds> <col> <op> <val> <out>     Filter rows (== != < > <= >=)
undo / redo                            Revert or repeat last dataset change

Data Transformation:
factor <dataset> <column>              Convert column to categorical
numeric <dataset> <column>             Convert factor back to numeric
scale <dataset> <column>               Z-score normalization
impute mice <iter> <dataset>           MICE imputation (default 5)
impute mean <dataset> <col>            Mean imputation for numeric column

Statistical Analysis:
pca <dataset>                          Principal Component Analysis
quickplot <ds> <col1> <col2>           Quick inline scatter/ROC plot
glm <fam> <ds> <name> <form>           Fit GLM model (binomial|poisson|etc)
anova <ds> <response> <factor>         One-way ANOVA by factor
wilcoxon <ds> <grp_col> <val_col>      Wilcoxon rank-sum test
kruskal <ds> <grp_col> <val_col>       Kruskal-Wallis rank test
spearman <ds> <x_col> <y_col>          Spearman rank correlation
chi2 <ds> <row_col> <col_col>          Chi-squared test
survival <ds> <time> <event> [grp]     Kaplan-Meier survival estimate
diagnose <model>                       Print model diagnostics

System:
help, ?                                Show this help message
seed <integer>                         Seed the random generator
script <file>                          Execute commands from a script file
export <file>                          Save datasets and load script
momento <file>                         Write command history to replay script
exit                                   Exit the shell
