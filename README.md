# Deep Learning for Multi-Label Text Classification

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Multi-Label-Text-Classification.svg?branch=master)](https://travis-ci.org/RandolphVI/Multi-Label-Text-Classification) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c45aac301b244316830b00b9b0985e3e)](https://www.codacy.com/app/chinawolfman/Multi-Label-Text-Classification?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Multi-Label-Text-Classification&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Multi-Label-Text-Classification.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Multi-Label-Text-Classification.svg)](https://github.com/RandolphVI/Multi-Label-Text-Classification/issues)

I'm trying Multi Label Classification for CPC of Patent.

This code is made by "黄威，Randolph" and I am now understanding the code and managing to adapt for my patent data.

## Requirements

- Python 3.6
- Tensorflow 1.1 +
- Numpy
- Gensim


## Data

The sample data is in `data` folder. I successed to make my patent abstract data to this sample format.

Original Data : 
{"id":"US001899016A",
 "pd":"19330228",
 "cpc":"C25D1/18|B29B15/1",
 "title":"Dehydrating rubber deposited from aqueous dispersions",
 "p":"Electric endosmose is employed in the removal ~~. The invention may be applied to the dehydration of coated 
      fabrics."}

Target Data :  
{"testid": "3930328",
 "features_content": ["anchor", "slidably", "adjustably", "carried", "tie-line", "floatable", "decoy", "duck",
                      "storage", "tie-line", "tension", "decoy", "ready", "re-use"],
 "labels_index": [9, 17, 288],
 "labels_num": 3}
 

## Network Structure

### FastText

![](https://farm2.staticflickr.com/1917/45609842012_30f370a0ee_o.png)

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![](https://farm2.staticflickr.com/1965/44745949305_50f831a579_o.png)

References:

- **Personal ideas 🙃**

---

### TextCNN

![](https://farm2.staticflickr.com/1927/44935475604_1d6b8f71a3_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1925/30719666177_6665038ea2_o.png)

#### TODO
1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1915/43842346360_e4660c5921_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

![](https://farm2.staticflickr.com/1950/31788031648_b5cba7bbf0_o.png)

References:

- **Personal ideas 🙃**

---

### TextHAN

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Find the format of "/data/content.txt"
2. Making training data
3. Using CNN, HAN, RCNN...

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

## I am

Donghee Han

Korea Univ

Email : ehdgml76@gmail.com

## Original Code is made by

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)

github: [randolph's github](https://github.com/RandolphVI/Multi-Label-Text-Classification)