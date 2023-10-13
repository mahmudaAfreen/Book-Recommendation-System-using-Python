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
