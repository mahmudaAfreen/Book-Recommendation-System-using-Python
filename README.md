# Book-Recommendation-System-using-Python
It's a Machine Learning project that will cover the basic idea of a collaborative recommendation system from several aspects. 

![buy-1024x576](https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/8baca444-687e-4b43-9aec-26154291073c)


# Abstract 
This work will introduce a Collaborative Book Recommendation System using two popular techniques: K-Nearest Neighbors (KNN) and Matrix Factorization (MF) methods such as correlation and Singular Value Decomposition (SVD). The proposed system aims to provide personalized book recommendations based on user preferences and historical data. The KNN algorithm identifies similar users and recommends books based on their reading habits. Additionally, the Matrix Factorization approach, including correlation and SVD, is used to extract latent features from the user-item matrix, enabling accurate predictions of user preferences. The combination of these techniques results in an effective collaborative book recommendation system that enhances user satisfaction and engagement.

## Libraries 
``` python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')
```

## Introduction of the Data
This goodbooks-10k dataset(https://github.com/zygmuntz/goodbooks-10k) contains almost 6 million ratings of 10 thousand popular books.As of the record every book have more eor less 100 reviews. The book IDs range from 1 to 10,000, while the user IDs range from 1 to 53,424. All users in the system have provided a minimum of two ratings, ensuring a comprehensive dataset. The median number of ratings per user is eight, indicating a sufficient amount of user feedback for accurate recommendation generation. 

```python
# Load the books data
books_df = pd.read_csv('books.csv')

# Load the ratings data
ratings_df = pd.read_csv('ratings.csv')
```
#### Explore the dataset

```python
books_df.head(2)
ratings_df.head(2)
books_df.info()
```
