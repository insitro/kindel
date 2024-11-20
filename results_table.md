# Full results table


Table 1: mapk14 target results 
| Target | Split strategy   | Metric                                   | Data split           | Value (mean ± standard deviation) |
| ------ | ---------------- | ---------------------------------------- | -------------------- | --------------------------------- |
| mapk14 | random           | RMSE                                     | test set             | 0.128 ± 0.144                     |
| mapk14 | random           | Spearman Correlation coefficient         |  on DNA:In-library   | 0.410 ± 0.153                     |
| mapk14 | random           | Spearman Correlation coefficient         | off DNA:In-library   | 0.406 ± 0.124                     |
| mapk14 | random           | Kendall's tau                            | off DNA:In-library   | 0.272 ± 0.084                     |
| mapk14 | random           | Spearman Correlation coefficient         |  on DNA:all          | 0.089 ± 0.186                     |
| mapk14 | random           | Spearman Correlation coefficient         | off DNA:all          | 0.418 ± 0.113                     |
| mapk14 | random           | Kendall's tau                            | off DNA:all          | 0.280 ± 0.081                     |
| mapk14 | disynthon        | RMSE                                     | test set             | 0.168 ± 0.059                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         |  on DNA:In-library   | 0.046 ± 0.238                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         | off DNA:In-library   | 0.246 ± 0.055                     |
| mapk14 | disynthon        | Kendall's tau                            | off DNA:In-library   | 0.167 ± 0.043                     |
| mapk14 | disynthon        | Spearman Correlation coefficient         |  on DNA:all          | -0.059 ± 0.171                    |
| mapk14 | disynthon        | Spearman Correlation coefficient         | off DNA:all          | 0.227 ± 0.096                     |
| mapk14 | disynthon        | Kendall's tau                            | off DNA:all          | 0.154 ± 0.073                     |
| ddr1   | random           | RMSE                                     | test set             | 0.530 ± 0.117                     |
| ddr1   | random           | Spearman Correlation coefficient         | on DNA:In-library    | 0.458 ± 0.065                     |
| ddr1   | random           | Spearman Correlation coefficient         | off DNA:In-library   | 0.181 ± 0.092                     |
| ddr1   | random           | Kendall's tau                            | off:DNA:In-library   | 0.126 ± 0.060                     |
| ddr1   | random           | Spearman Correlation coefficient         | on DNA:all           | 0.512 ± 0.049                     |
| ddr1   | random           | Spearman Correlation coefficient         | off:DNA:all          | 0.170 ± 0.082                     |
| ddr1   | random           | Kendall's tau                            | off:DNA:all          | 0.121 ± 0.055                     |
| ddr1   | disynthon        | RMSE                                     | test set             | 1.722 ± 1.061                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | on DNA:In-library    | 0.367 ± 0.214                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | off:DNA:In-library   | 0.090 ± 0.086                     |
| ddr1   | disynthon        | Kendall's tau                            | off:DNA:In-library   | 0.060 ± 0.055                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | on DNA:all           | 0.486 ± 0.137                     |
| ddr1   | disynthon        | Spearman Correlation coefficient         | off:DNA:all          | 0.087 ± 0.083                     |
| ddr1   | disynthon        | Kendall's tau                            | off:DNA:all          | 0.059 ± 0.054                     |