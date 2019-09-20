### Deep Learning for Multi-Label Text Classification

- Goal : Making Multi Label Classification Model for CPC of Patent.

- Baseline code is made by "黄威，Randolph". I am now understanding the code and manipulating to adapt for my patent data.

### Requirements

- Python 3.6, Tensorflow 1.1 +, Numpy, Gensim


### Data

- The sample data is in `data` folder. I'm using the patent abstract.

- Original Data : {"id":"US001899016A", "pd":"19330228", "cpc":"C25D1/18|B29B15/1",
 "title":"Dehydrating rubber deposited from aqueous dispersions",
 "p":"Electric endosmose is employed in the removal ~~. The invention may be applied to the dehydration of coated 
      fabrics."}

- Target Data : {"testid": "3930328", "features_content": ["anchor", "slidably", "adjustably", "carried", "tie-line",
 "floatable", "decoy", "duck", "storage", "tie-line", "tension", "decoy", "ready", "re-use"], 
 "labels_index": [9, 17, 288], "labels_num": 3}
 
---

## My Work

1. Understanding the whole code
2. Making training data
3. dh.load_data_and_labels 수정(load_data_and_labels_v2)


Name : Donghee Han

Institute : Korea Univ

Email : ehdgml76@gmail.com

## Baseline Code is made by

黄威，Randolph

SCU SE Bachelor; USTC CS Master

github: https://github.com/RandolphVI/Multi-Label-Text-Classification
Email: chinawolfman@hotmail.com