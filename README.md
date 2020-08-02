## Recommender Systems

### 1. About This Repository

This repository contains the source code for the Master Thesis of Robin Opdam, for his completion of the MSc. Business Analytics at the Vrije Universiteit Amsterdam. The subject of this thesis is a ["Comparison of Deep Learning Recommendation Engines in Different Settings"](https://science.vu.nl/en/education/internship-office-for-mathematics-and-computer-science/master-project-ba/internship-papers-online/index.aspx) and compares the following models:

* Bayesian Personalised Ranking
    - Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). BPR: [Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf). arXiv preprint arXiv:1205.2618.

* Collaborative Filtering with Recurrent Neural Networks
    - Devooght, R., & Bersini, H. (2016). [Collaborative filtering with recurrent neural networks](https://arxiv.org/pdf/1608.07400.pdf) arXiv preprint arXiv:1608.07400.

* Neural Matrix Factorisation (Neural Collaborative Filtering) 
    - Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](https://dl.acm.org/doi/10.1145/3038912.3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.
  
The data used for this research:

* [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

* [Amazon Shoes Clothing and Jewellery 5-core subset (Amazon 20K Users)](https://grouplens.org/datasets/movielens/1m/)
    - Ni, J., Li, J., & McAuley, J. (2019, November). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 188-197).

* [Amazon Shoes Clothing and Jewellery 5-core subset structured more like MovieLens 1M (Am-like-ML)](https://grouplens.org/datasets/movielens/1m/)
    - Ni, J., Li, J., & McAuley, J. (2019, November). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 188-197).
    
The data needs to be in the following format

| user_id | item_id | datetime            |
|---------|---------|---------------------|
| 0       | 392     | 2000-12-31 22:00:19 |
| 0       | 3189    | 2000-01-15 10:00:00 |
| 0       | 1093    | 2000-01-21 16:15:29 |
|...      |...      |...                  |
| n       | 457     | 2000-01-13 09:34:00 |

### 2. Repository Map
```
recommender_systems
├── README.md
│
├── Notebooks
│   ├── Example
│   │     └── all_models.ipynb
│   └── Thesis
│        ├── BPR.ipynb
│        ├── CFRNN.ipynb    
│        ├── NCF.ipynb
│        ├── data_used.ipynb
│        ├── final_results.ipynb
│        └── all_models_GS.ipynb
│
├── Data_prep.py
├── BPR.py 
├── CFRNN.py
├── NCF.py
├── Evaluation.py
├── visualize_results.py
├── Helpers.py
│
└── Results
    ├── Plots
    └── Thesis  
```

### 3. Usage
Each model is written within its own class, an example of how to call the algorithms can be found in Example/all_models.ipynb. For the Thesis I used the individual notebooks BPR.ipynb, CFRNN.ipynb, NCF.ipynb and prepared the data using Data_prep.py
    
Required Python Libraries for the repository: 
    
* pandas
* numpy
* os
* tensorflow >= 2.1
* time
* progressbar
* math

****
### 5. Contact

This is the work of Robin Opdam, if you feel the need to contact me:

* robinopdam@hotmail.com


<!-- ##### This repository is confidential and shall in no case be shared with entities outside of the organizaiton  -->

