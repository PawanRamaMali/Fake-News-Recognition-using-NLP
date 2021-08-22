# Fake News Recognition using NLP

The degree of authenticity of the news posted online cannot be definitively measured, since the manual classification of news is tedious and time-consuming and is also subject to bias.

## Problem Statement 

To tackle the growing problem, detection, classification of fake news 

## Methodology 

The categories, bs (i.e. bullshit), junksci(i.e. junk science), hate, fake, conspiracy, bias, satire and state declare the category under which untrustworthy or false news fall under.

The first step, which is text preprocessing was performed using the following:

* Taking care of null/missing values
* Transforming categorical data with the help of label encoders
* Uppercase to lowercase
* Number removal
* Tokenization
* Stop Word Removal, Stemming and Lemmatization (with POS tagging) using the Natural Language Toolkit Library (NLTK) 

For feature engineering, the TF-IDF technique is used. This processed and embedded text is provided as an input to Machine learning models, where the data is made to fit the model, to get a prediction as an output.

```Python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

```

### Imports

```
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import plotly
import plotly.express as px
import seaborn as sns

from IPython.core.display import HTML
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
from collections import OrderedDict 
from colorama import Fore, Back, Style
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL
```
