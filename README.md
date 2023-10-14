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
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
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
books_df.describe()
```
## Data Visualization
```python
import matplotlib.pyplot as plt

# Count top 10 books
top_rated_books = ratings_df['book_id'].value_counts().head(10)
top_rated_books = books_df.merge(top_rated_books, left_on='book_id', right_index=True)
plt.figure(figsize=(15, 4))
plt.barh(top_rated_books['title'], top_rated_books['book_id'], color='skyblue')
plt.xlabel('Number of Ratings')
plt.ylabel('Book Title')
plt.title('Top 10 Most Rated Books')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```
<img width="930" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/c2e37477-2837-4169-a318-b36ba48fb58a">


```python
# Distribution of Book Ratings
import matplotlib.pyplot as plt

# Rename the column labels
rating_counts = books_df.rename(columns={'ratings_1': 'Rating 1', 'ratings_2': 'Rating 2',
                                         'ratings_3': 'Rating 3', 'ratings_4': 'Rating 4',
                                         'ratings_5': 'Rating 5'})

# Count the number of books for each rating
rating_sum = rating_counts[['Rating 1', 'Rating 2', 'Rating 3', 'Rating 4', 'Rating 5']].sum()

# Define custom colors for each rating category
colors = ['#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#3498DB']

# Create pie chart with smaller size
plt.figure(figsize=(15, 10))

# Plot the pie chart
plt.pie(rating_sum, labels=rating_sum.index, autopct='%1.1f%%', startangle=90, colors=colors,
        textprops={'color': 'black', 'fontsize': 12})

# Set equal aspect ratio to ensure circular shape
plt.axis('equal')

# Add a title to the chart
plt.title('Distribution of Book Ratings', fontsize=16, fontweight='bold')

# Move the percentage labels inside the pie slices
plt.gca().set_position([0, 0, 0.5, 1])  # Adjust the position of the pie chart
plt.gca().text(0.1, 0.1, 'Total\n' + str(rating_sum.sum()), ha='center', va='center', fontsize=14, fontweight='bold')

plt.show()
```
<img width="527" alt="Screenshot 2023-10-14 015524" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/3ae70098-3b14-4772-bbea-166b332f2295">

## Collaborative Recommendation System using KNN

The collaborative recommendation system is a type of filtering system that suggests items or content to users based on their past behaviors and preferences, as well as the behaviors and preferences of other similar users.This system relies on the assumption that users with similar tastes and preferences in the past will have similar tastes and preferences in the future.
As of k-nearest neighbor (KNN) based recommender system is a type of collaborative filtering system that uses the ratings given by users to other items to make recommendations. The system works by calculating the similarity between each pair of items, and then using the similarities to predict how a user will rate a given item.
[Reference-1](https://www.aurigait.com/blog/recommendation-system-using-knn/) [Reference-2](https://www.itm-conferences.org/articles/itmconf/abs/2017/04/itmconf_ita2017_04008/itmconf_ita2017_04008.html)

## What I have done here
In the above part i have used KNN algorithm to perform a collaborative recommended system and i have used RMSE and MAE to measure the performance of the algorithm. The core functionality is encapsulated within a function called "get_recomm". This function accepts a book title, the number of neighbors to consider, and a display flag as parameters.It retrieves the index of the queried book in the ratings_df pivot table, using a hypothetical get_book_id function. The KNN model is then queried to obtain the distances and indices of the nearest neighbors for the queried book.By iterating through the distances and indices of the nearest neighbors, the function retrieves the recommended book IDs. If the display flag is set to True, it prints the recommendations, including the book ID and the corresponding distance.

#### Create a copy of the origina dataset 

```python
books = books_df.copy()
ratings = ratings_df.copy()
```

#### To clean and preprocess the data by removing missing values and duplicate entries, ensuring the data frames are ready for further analysis or modeling tasks.

```python
books = books.dropna()
ratings = ratings.sort_values("user_id")
ratings.drop_duplicates(subset=["user_id","book_id"], keep='first', inplace=True) 
books.drop_duplicates(subset='original_title', keep='first', inplace=True)
```

#### To prepares a new data frame "df" that contains relevant information for further analysis.

```python
merged_df = pd.merge(books, ratings, how='left', left_on=['id'], right_on=['book_id'])
df = merged_df[['id','original_title', 'user_id', 'rating']]

df = df.rename(columns = {'id':'book_id'})
df.head(2)
```

####  To create the pivot table and filling missing values with zeros, the code organizes the data in a matrix-like format, where the rows represent books, the columns represent users, and the values represent ratings.

```python
ratings_df = df.pivot_table(index='book_id',columns='user_id',values='rating').fillna(0)

pd.set_option('display.max_columns', 100)
ratings_df.head()
```
```python
ratings_df.shape
```
```python
ratings_matrix = csr_matrix(ratings_df.values)
```
```python
model_knn = NearestNeighbors(metric='cosine', algorithm = 'brute')
model_knn.fit(ratings_matrix)
```

#### Helper Function
```python
def get_book_id(book_title):
    target_df = df.loc[df['original_title'] == book_title]
    return target_df['book_id'].iloc[0]

id_TheHungerGames = get_book_id('The Hunger Games')
print(id_TheHungerGames)
```

```python
def get_title(book_id):
    target_df = df.loc[df['book_id'] == book_id]
    return target_df['original_title'].iloc[0]

print(get_title(1))
```

```python
def get_recomm(book_title, num_neighbors=10, display=False): 
    book_ids = []
    
    query_index = get_book_id(book_title) - 1
    
    if num_neighbors > 0:
        distances, indices = model_knn.kneighbors(ratings_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = num_neighbors + 1)
    else:
        distances, indices = model_knn.kneighbors(ratings_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 10 + 1)
    
    for i in range(0, len(distances.flatten())):
        if display is True:
            if i == 0:
                print('Recommendations for ', book_title, '\n')
            else:    
                print('{0}\t Book ID: {1}\t  Distance: {2}:\n'.format(i, ratings_df.index[indices.flatten()[i]], distances.flatten()[i]))
        
        book_ids.append(ratings_df.index[indices.flatten()[i]])
    
    return book_ids
```

#### Test the Result
```python
recommendations_for_TheHungerGames = get_recomm('The Hunger Games', num_neighbors=10, display=True)
```
```python

```
