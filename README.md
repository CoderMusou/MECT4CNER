# MECT4CNER
The source code of the MECT for ACL 2021 paper:
> Shuang Wu, Xiaoning Song, and Zhenhua Feng. 2021. MECT: Multi-metadata embedding based cross- transformer for Chinese named entity recognition. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Lan- guage Processing (Volume 1: Long Papers), pages 1529-1539, Online. Association for Computational Linguistics.

Models and results can be found at our paper in [ACL 2021](https://aclanthology.org/2021.acl-long.121/) or [arXiv](https://arxiv.org/abs/2107.05418).  

## Introduction
MECT has the lattice and radical streams, which not only possesses FLAT’s word boundary and semantic learning ability but also increases the structure information of Chinese character radicals. With the structural characteristics of Chinese characters, MECT can better capture the semantic information of Chinese characters for Chinese NER.

## Citation
If you want to use our codes in your research, please cite:
```
@inproceedings{wu-etal-2021-mect,
    title = "{MECT}: {M}ulti-Metadata Embedding based Cross-Transformer for {C}hinese Named Entity Recognition",
    author = "Wu, Shuang  and
      Song, Xiaoning  and
      Feng, Zhenhua",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.121",
    doi = "10.18653/v1/2021.acl-long.121",
    pages = "1529--1539",
}
```

## Environment Requirement
The code has been tested under Python 3.7. The required packages are as follows:
```
torch==1.5.1
numpy==1.18.5
FastNLP==0.5.0
fitlog==0.3.2
```
you can click [here](https://fastnlp.readthedocs.io/zh/latest/) to know more about FastNLP. And you can click [here](https://fitlog.readthedocs.io/zh/latest/) to know more about Fitlog.

## Example to Run the Codes
1. Download the pretrained character embeddings and word embeddings and put them in the data folder.
    * Character embeddings (gigaword_chn.all.a2b.uni.ite50.vec): [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Bi-gram embeddings (gigaword_chn.all.a2b.bi.ite50.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
    * Word(Lattice) embeddings (ctb.50d.vec): [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
  
2. Get Chinese character structure components(radicals). The radicals used in the paper are from the [online Xinhua dictionary](http://tool.httpcn.com/Zi/). Due to copyright reasons, these data cannot be published. There is a method that can be replaced by [漢語拆字字典](https://github.com/kfcd/chaizi), but inconsistent character decomposition methods cannot guarantee repeatability.

3. Modify the `Utils/paths.py` to add the pretrained embedding and the dataset

4. Run following commands
    * Weibo dataset
    ```shell
    python Utils/preprocess.py
    python main.py --dataset weibo
    ```
    * Resume dataset
    ```shell
    python Utils/preprocess.py
    python main.py --dataset resume
    ```
    * Ontonotes dataset
    ```shell
    python Utils/preprocess.py
    python main.py --dataset ontonotes
    ```
    * MSRA dataset
    ```shell
    python Utils/preprocess.py --clip_msra
    python main.py --dataset msra
    ```

## Acknowledgements
* Thanks to Dr. Li and his team for contributing the [FLAT source code](https://github.com/LeeSureman/Flat-Lattice-Transformer).
* Thanks to the author team and contributors of [FastNLP](https://github.com/fastnlp/fastNLP).

