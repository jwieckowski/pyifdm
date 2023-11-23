# pyifdm

Python 3 package to perform Multi-Criteria Decision Analysis in the Intuitionistic Fuzzy environment

---

# Installation

The package can be download using pip:

```Bash
pip install pyifdm
```

# Testing

The modules performance can be verified with pytest library

```Bash
pip install pytest
pytest tests
```

---

# Modules and functionalities

- MCDA methods based on Intuitionistic Fuzzy Sets (IFS):

| Abbreviation | Full name                                                                    | Reference      |
| ------------ | ---------------------------------------------------------------------------- | -------------- |
| ARAS         | Additive Ratio ASsessment                                                    | [[1]](#ref1)   |
| CODAS        | COmbinative Distance-based ASsessment                                        | [[2]](#ref2)   |
| COPRAS       | COmplex PRoportional ASsessment                                              | [[3]](#ref3)   |
| EDAS         | Evaluation based on Distance from Average Solution                           | [[4]](#ref4)   |
| MABAC        | Multi-Attributive Border Approximation area Comparison                       | [[5]](#ref5)   |
| MAIRCA       | MultiAttributive Ideal-Real Comparative Analysis                             | [[6]](#ref6)   |
| MARCOS       | Measurement of Alternatives and Ranking according to the Compromise Solution | [[31]](#ref31) |
| MOORA        | Multi-Objective Optimization Method by Ratio Analysis                        | [[7]](#ref7)   |
| OCRA         | Operational Competitiveness Rating                                           | [[30]](#ref30) |
| TOPSIS       | Technique for the Order of Prioritisation by Similarity to Ideal Solution    | [[8]](#ref8)   |
| VIKOR        | VIseKriterijumska Optimizacija I Kompromisno Resenje                         | [[9]](#ref9)   |
| WASPAS       | Weighted Aggregated Sum Product Assessment                                   | [[38]](#ref38) |
| WSM          | Weighted Sum Method                                                          | [[37]](#ref37) |
| WPM          | Weighted Product Method                                                      | [[37]](#ref37) |

- Weighting methods:

| Name                    |   Reference    |
| ----------------------- | :------------: |
| Burillo entropy weights | [[25]](#ref25) |
| Equal weights           | [[10]](#ref10) |
| Entropy weights         |  [[9]](#ref9)  |
| Liu entropy weights     | [[27]](#ref27) |
| Szmidt entropy weights  | [[26]](#ref26) |
| Thakur entropy weights  |  [[3]](#ref3)  |
| Ye entropy weights      | [[24]](#ref24) |

- Normalization methods:

| Name                  |   Reference    |
| --------------------- | :------------: |
| Ecer normalization    | [[10]](#ref10) |
| Max normalization     | [[37]](#ref37) |
| Min-Max normalization |  [[6]](#ref6)  |
| Supriya normalization | [[11]](#ref11) |
| Swap normalization    |  [[2]](#ref2)  |

- Score functions:

| Name             |   Reference    |
| ---------------- | :------------: |
| Chen score 1     | [[29]](#ref29) |
| Chen score 2     | [[29]](#ref29) |
| Kharal score 1   | [[15]](#ref15) |
| Kharal score 2   | [[15]](#ref15) |
| Liu Wang score   | [[28]](#ref28) |
| Supriya score    | [[11]](#ref11) |
| Thakur score     |  [[3]](#ref3)  |
| Wan Dong score 1 | [[13]](#ref13) |
| Wan Dong score 2 | [[13]](#ref13) |
| Wei score        | [[12]](#ref12) |
| Zhang Xu score 1 | [[14]](#ref14) |
| Zhang Xu score 2 | [[14]](#ref14) |

- Distance measures:

| Name                          |   Reference    |
| ----------------------------- | :------------: |
| Euclidean distance            | [[16]](#ref16) |
| Grzegorzewski distance        | [[17]](#ref17) |
| Hamming distance              | [[16]](#ref16) |
| Hausdorf Euclidean distance   | [[32]](#ref32) |
| Luo Distance                  |  [[9]](#ref9)  |
| Normalized Euclidean distance | [[16]](#ref16) |
| Normalized Hamming distance   | [[16]](#ref16) |
| Wang Xin distance 1           | [[18]](#ref18) |
| Wang Xin distance 2           | [[18]](#ref18) |
| Yang Chiclana distance        | [[19]](#ref19) |

- IFS-similarity measures:

| Name                 |   Reference    |
| -------------------- | :------------: |
| Chen similarity      | [[38]](#ref38) |
| Fan-Zhang similarity | [[38]](#ref38) |
| Hong-Kim similarity  | [[38]](#ref38) |
| Li similarity        | [[38]](#ref38) |
| Li-Xu similarity     | [[38]](#ref38) |
| Ye similarity        | [[38]](#ref38) |

- Correlation coefficients:

| Name                                      |   Reference    |
| ----------------------------------------- | :------------: |
| Pearson correlation coefficient           | [[21]](#ref21) |
| Spearman correlation coefficient          | [[20]](#ref20) |
| Weighted Spearman correlation coefficient | [[22]](#ref22) |
| WS Rank Similarity coefficient            | [[23]](#ref23) |

- Intuitionistic Fuzzy Set [[33]](#ref33), [[34]](#ref34), [[35]](#ref35), [[36]](#ref36) :

| Functionality name |
| ------------------ |
| Addition           |
| Subtraction        |
| Multiplication     |
| Division           |
| Power              |
| Equality           |
| AND (Intersection) |
| OR (Union)         |
| Invert             |
| OWA aggregation    |
| Dominance          |
| Jaccard similarity |
| Fuzzy relation     |

- Graphs:

| Functionality name    |
| --------------------- |
| IFS criteria bar plot |
| IFS heatmap plot      |
| IFS radar plot        |
| Single IFS bar plot   |
| Single IFS pie plot   |

- Helpers methods
  - rank
  - generate ifs matrix

# Usage example

Below the sample example of the Intuitionistic Fuzzy EDAS method application is presented.
More examples of package functionalities can be found in [Jupyter examples](https://github.com/jwieckowski/pyifdm/blob/main/examples/examples.ipynb).

```python
from pyifdm.methods import ifEDAS
from pyifdm.helpers import rank
import numpy as np

if __name__ == '__main__':
    matrix = np.array([
        [[0.4745, 0.5255], [0.4752, 0.5248], [0.2981, 0.7019], [0.4374, 0.5627]],
        [[0.5346, 0.4654], [0.5532, 0.4468], [0.6300, 0.3700], [0.5901, 0.4099]],
        [[0.4324, 0.5676], [0.4030, 0.5970], [0.4298, 0.5702], [0.4361, 0.5639]],
        [[0.5235, 0.4765], [0.4808, 0.5192], [0.5667, 0.4333], [0.2913, 0.7087]],
        [[0.4168, 0.5832], [0.4923, 0.5077], [0.4732, 0.5268], [0.4477, 0.5523]]
    ])

    weights = np.array([0.2, 0.3, 0.15, 0.35])

    types = np.array([1, -1, 1, 1])

    if_edas = ifEDAS()
    pref = if_edas(matrix, weights, types)

    print(f'IF-EDAS preferences: {pref}')
    print(f'IF-EDAS ranking: {rank(pref)}')
```

Output:

```bash
IF-EDAS preferences: 0.276 0.259 0.523 0.995 0.322
IF-EDAS ranking: 4 5 2 1 3
```

# References

<a name="ref1">**[1]**</a> Raj Mishra, A., Sisodia, G., Raj Pardasani, K., & Sharma, K. (2020). Multi-criteria IT personnel selection on intuitionistic fuzzy information measures and ARAS methodology. Iranian Journal of Fuzzy Systems, 17(4), 55-68.

<a name="ref2">**[2]**</a> Buyukozkan, G., & Göçer, F. (2019, August). Prioritizing the strategies to enhance smart city logistics by intuitionistic fuzzy CODAS. In 11th Conference of the European Society for Fuzzy Logic and Technology (EUSFLAT 2019) (pp. 805-811). Atlantis Press.

<a name="ref3">**[3]**</a> Thakur, P., Kizielewicz, B., Gandotra, N., Shekhovtsov, A., Saini, N., Saeid, A. B., & Sałabun, W. (2021). A New Entropy Measurement for the Analysis of Uncertain Data in MCDA Problems Using Intuitionistic Fuzzy Sets and COPRAS Method. Axioms, 10(4), 335.

<a name="ref4">**[4]**</a> Liang, Y. (2020). An EDAS method for multiple attribute group decision-making under intuitionistic fuzzy environment and its application for evaluating green building energy-saving design projects. Symmetry, 12(3), 484.

<a name="ref5">**[5]**</a> Li, Y. (2021). IF-MABAC Method for Evaluating the Intelligent Transportation System with Intuitionistic Fuzzy Information. Journal of Mathematics, 2021.

<a name="ref6">**[7]**</a> Ecer, F. (2022). An extended MAIRCA method using intuitionistic fuzzy sets for coronavirus vaccine selection in the age of COVID-19. Neural Computing and Applications, 34(7), 5603-5623.

<a name="ref8">**[8]**</a> Pérez-Domínguez, L., Alvarado-Iniesta, A., Rodríguez-Borbón, I., & Vergara-Villegas, O. (2015). Intuitionistic fuzzy MOORA for supplier selection. Dyna, 82(191), 34-41.

<a name="ref9">**[9]**</a> Boran, F. E., Boran, K. U. R. T. U. L. U. Ş., & Menlik, T. (2012). The evaluation of renewable energy technologies for electricity generation in Turkey using intuitionistic fuzzy TOPSIS. Energy Sources, Part B: Economics, Planning, and Policy, 7(1), 81-90.

<a name="ref10">**[10]**</a> Ying-Yu, W., & De-Jian, Y. (2011, September). Extended VIKOR for multi-criteria decision making problems under intuitionistic environment. In 2011 International Conference on Management Science & Engineering 18th Annual Conference Proceedings (pp. 118-122). IEEE.

<a name="ref11">**[11]**</a> Ecer, F., & Pamucar, D. (2021). MARCOS technique under intuitionistic fuzzy environment for determining the COVID-19 pandemic performance of insurance companies in terms of healthcare services. Applied Soft Computing, 104, 107199.

<a name="ref12">**[12]**</a> De, S. K., Biswas, R., & Roy, A. R. (2000). Some operations on intuitionistic fuzzy sets. Fuzzy sets and Systems, 114(3), 477-484.

<a name="ref13">**[13]**</a> Wei P, Gao ZH, Guo TT (2012) An intuitionistic fuzzy entropy measure based on the trigonometric function. Control Decis 27:571–574

<a name="ref14">**[14]**</a> Wan, S., & Dong, J. (2020). A selection method based on MAGDM with interval-valued intuitionistic fuzzy sets. In Decision Making Theories and Methods Based on Interval-Valued Intuitionistic Fuzzy Sets (pp. 115-137). Springer, Singapore.

<a name="ref15">**[15]**</a> Zhang, X., & Xu, Z. (2012). A new method for ranking intuitionistic fuzzy values and its application in multi-attribute decision making. Fuzzy Optimization and Decision Making, 11(2), 135-146.

<a name="ref16">**[16]**</a> Kharal, A. (2009). Homeopathic drug selection using intuitionistic fuzzy sets. Homeopathy, 98(1), 35-39.

<a name="ref17">**[17]**</a> Çalı, S., & Balaman, Ş. Y. (2019). A novel outranking based multi criteria group decision making methodology integrating ELECTRE and VIKOR under intuitionistic fuzzy environment. Expert Systems with Applications, 119, 36-50.

<a name="ref18">**[18]**</a> Grzegorzewski, P. (2004). Distances between intuitionistic fuzzy sets and/or interval-valued fuzzy sets based on the Hausdorff metric. Fuzzy sets and systems, 148(2), 319-328.

<a name="ref19">**[19]**</a> Wang, W., & Xin, X. (2005). Distance measure between intuitionistic fuzzy sets. Pattern recognition letters, 26(13), 2063-2069.

<a name="ref19">**[19]**</a> Yang, Y., & Chiclana, F. (2012). Consistency of 2D and 3D distances of intuitionistic fuzzy sets. Expert Systems with Applications, 39(10), 8665-8670.

<a name="ref20">**[20]**</a> Spearman, C. (1910). Correlation calculated from faulty data. British Journal of Psychology, 1904‐1920, 3(3), 271-295.

<a name="ref21">**[21]**</a> Pearson, K. (1895). VII. Note on regression and inheritance in the case of two parents. proceedings of the royal society of London, 58(347-352), 240-242.

<a name="ref22">**[22]**</a> Dancelli, L., Manisera, M., & Vezzoli, M. (2013). On two classes of Weighted Rank Correlation measures deriving from the Spearman’s ρ. In Statistical Models for Data Analysis (pp. 107-114). Springer, Heidelberg.

<a name="ref23">**[23]**</a> Sałabun, W., & Urbaniak, K. (2020, June). A new coefficient of rankings similarity in decision-making problems. In International Conference on Computational Science (pp. 632-645). Springer, Cham.

<a name="ref24">**[24]**</a> Ye, J. Two effective measures of intuitionistic fuzzy entropy. Computing 2010, 87, 55–62.

<a name="ref25">**[25]**</a> Burillo, P.; Bustince, H. Entropy on intuitionistic fuzzy sets and on interval-valued fuzzy sets. Fuzzy Sets Syst. 1996, 78, 305–316.

<a name="ref26">**[26]**</a> Szmidt, E.; Kacprzyk, J. Entropy for intuitionistic fuzzy sets. Fuzzy Sets Syst. 2001, 118, 467–477.

<a name="ref27">**[27]**</a> Liu, M.; Ren, H. A new intuitionistic fuzzy entropy and application in multi-attribute decision making. Information 2014, 5, 587–601.

<a name="ref28">**[28]**</a> Liu, H. W., & Wang, G. J. (2007). Multi-criteria decision-making methods based on intuitionistic fuzzy sets. European Journal of Operational Research, 179(1), 220-233.

<a name="ref29">**[29]**</a> Chen, T. Y. (2011). A comparative analysis of score functions for multiple criteria decision making in intuitionistic fuzzy settings. Information Sciences, 181(17), 3652-3676.

<a name="ref30">**[30]**</a> Mishra, A. R., Rani, P., Cavallaro, F., Hezam, I. M., & Lakshmi, J. (2023). An integrated intuitionistic fuzzy closeness coefficient-based OCRA method for sustainable urban transportation options selection. Axioms, 12(2), 144.

<a name="ref31">**[31]**</a> Ecer, F., & Pamucar, D. (2021). MARCOS technique under intuitionistic fuzzy environment for determining the COVID-19 pandemic performance of insurance companies in terms of healthcare services. Applied Soft Computing, 104, 107199.

<a name="ref32">**[32]**</a> Li, M., Wu, C., Zhang, L., & You, L. N. (2015). An intuitionistic fuzzy-TODIM method to solve distributor evaluation and selection problem. International Journal of Simulation Modelling, 14(3), 511-524.

<a name="ref33">**[33]**</a> Ejegwa, P. A., Akowe, S. O., Otene, P. M., & Ikyule, J. M. (2014). An overview on intuitionistic fuzzy sets. Int. J. Sci. Technol. Res, 3(3), 142-145.

<a name="ref34">**[34]**</a> Du, W. S. (2021). Subtraction and division operations on intuitionistic fuzzy sets derived from the Hamming distance. Information Sciences, 571, 206-224.

<a name="ref35">**[35]**</a> Mitchell, H. B. (2004). An intuitionistic OWA operator. International journal of uncertainty, fuzziness and knowledge-based systems, 12(06), 843-860.

<a name="ref36">**[36]**</a> Ngan, S. C. (2016). An activation detection based similarity measure for intuitionistic fuzzy sets. Expert Systems with Applications, 60, 62-80.

<a name="ref37">**[37]**</a> Xiong, L., Zhong, S., Liu, S., Zhang, X., & Li, Y. (2020). An approach for resilient-green supplier selection based on WASPAS, BWM, and TOPSIS under intuitionistic fuzzy sets. Mathematical Problems in Engineering, 2020.

<a name="ref38">**[38]**</a> Mishra, A. R., Singh, R. K., & Motwani, D. (2019). Multi-criteria assessment of cellular mobile telephone service providers using intuitionistic fuzzy WASPAS method with similarity measures. Granular Computing, 4, 511-529.
