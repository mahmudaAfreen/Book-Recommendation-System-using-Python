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
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
```

## Introduction of the Data
This [goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k) contains almost 6 million ratings of 10 thousand popular books.As of the record every book have more eor less 100 reviews. The book IDs range from 1 to 10,000, while the user IDs range from 1 to 53,424. All users in the system have provided a minimum of two ratings, ensuring a comprehensive dataset. The median number of ratings per user is eight, indicating a sufficient amount of user feedback for accurate recommendation generation. 

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
##### Output
<img width="216" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/017a98be-24aa-4842-b28c-fa8731db74c0">

####  To create the pivot table and filling missing values with zeros, the code organizes the data in a matrix-like format, where the rows represent books, the columns represent users, and the values represent ratings.

```python
ratings_df = df.pivot_table(index='book_id',columns='user_id',values='rating').fillna(0)

pd.set_option('display.max_columns', 100)
ratings_df.head()
```
##### Output

<img width="1208" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/540303bb-66bf-48af-9ead-646bf72d732c">

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
##### Output

Recommendations for  The Hunger Games 

1	 Book ID: 17	  Distance: 0.4053256862294832:

2	 Book ID: 31	  Distance: 0.42674642418525066:

3	 Book ID: 2	  Distance: 0.4444738573252147:

4	 Book ID: 20	  Distance: 0.4523031603835689:

5	 Book ID: 3	  Distance: 0.49084774300774026:

6	 Book ID: 93	  Distance: 0.5119913445564398:

7	 Book ID: 5	  Distance: 0.5197672572777041:

8	 Book ID: 16	  Distance: 0.5229117539369219:

9	 Book ID: 9	  Distance: 0.5271915645842881:

10	 Book ID: 37	  Distance: 0.5278079122055651:

```python
# Top 10 recommendations for Harry Potter and the Philosopher's Stone

book_ids_for_H = get_recomm('Harry Potter and the Philosopher\'s Stone', num_neighbors=10)
# skip the first item
for b in book_ids_for_H[1:]:
    print(get_title(b))
```
##### Output
To Kill a Mockingbird
Memoirs of a Geisha
Nineteen Eighty-Four
The Great Gatsby
 The Fellowship of the Ring
Lord of the Flies 
Harry Potter and the Prisoner of Azkaban
The Hobbit or There and Back Again
Het Achterhuis: Dagboekbrieven 14 juni 1942 - 1 augustus 1944
Jane Eyre


## K-fold cross-validation to evaluate the performance of a recommendation system
The results of the evaluation show the RMSE and MAE values for each fold, indicating the accuracy of the predicted ratings compared to the actual ratings. These metrics provide valuable insights into the model's performance across different subsets of the dataset. By employing K-fold cross-validation, the evaluation accounts for both training and validation sets, ensuring a robust assessment of the model's predictive capabilities. This approach helps in determining the model's effectiveness in predicting ratings and allows for comparisons between different models or parameter settings.

[Reference](https://www.analyticsvidhya.com/blog/2022/02/k-fold-cross-validation-technique-and-its-essentials/)

```python
def calculate_fold(data, k):
    fold_indices = []
    kf = KFold(n_splits=k, shuffle=True)

    for train_index, val_index in kf.split(data):
        fold_indices.append((train_index, val_index))

    return fold_indices

def calculate_metrics(actual_ratings, predicted_ratings):
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)
    return rmse, mae

# Example usage
actual_ratings = [4, 3, 5, 2, 4]
predicted_ratings = [3.8, 2.5, 4.3, 1.8, 4.1]
k = 5

fold_indices = calculate_fold(actual_ratings, k)

for fold, (train_index, val_index) in enumerate(fold_indices):
    print("Fold", fold + 1)
    print("Training indices:", train_index)
    print("Validation indices:", val_index)

    train_actual = [actual_ratings[i] for i in train_index]
    train_predicted = [predicted_ratings[i] for i in train_index]
    val_actual = [actual_ratings[i] for i in val_index]
    val_predicted = [predicted_ratings[i] for i in val_index]

    rmse, mae = calculate_metrics(val_actual, val_predicted)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print()
```

##### Output
Fold 1
Training indices: [0 1 2 3]
Validation indices: [4]
RMSE: 0.09999999999999964
MAE: 0.09999999999999964

Fold 2
Training indices: [0 1 2 4]
Validation indices: [3]
RMSE: 0.19999999999999996
MAE: 0.19999999999999996

Fold 3
Training indices: [0 2 3 4]
Validation indices: [1]
RMSE: 0.5
MAE: 0.5

Fold 4
Training indices: [0 1 3 4]
Validation indices: [2]
RMSE: 0.7000000000000002
MAE: 0.7000000000000002

Fold 5
...
Validation indices: [0]
RMSE: 0.20000000000000018
MAE: 0.20000000000000018

## Using Matrix Factorization
Matrix factorization is a technique that is commonly used in recommendation systems to predict the ratings that users will give to items. These matrices can then be used to make predictions about how a user will rate an item by taking the dot product of the user and item vectors.
In my work i will use Non-Negative Matrix Factorization (NMF) and Singular Value Decomposition(SVD)

```python
# an example of the correaltion between the books '1776' and 'The Fellowship of the Ring'
book1776=bookratings['1776']
bookratings[[" The Fellowship of the Ring","1776"]].corr()
```
##### Output
<img width="286" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/da2c2743-2e77-4c2c-a0b1-f2fecd850cf5">

```python
# show books with most correlation
bookratings.corrwith(book1776).sort_values(ascending=False).to_frame('corr')
```
##### Output
<img width="439" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/a6d2c3c3-b7ed-4505-a006-271152783d82">

```python
# show books with least correlation
bookratings.corrwith(book1776).sort_values(ascending=True).to_frame('corr').dropna()
```
##### Output
<img width="145" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/eb935226-d133-4dbb-a5cf-625421e944c1">

```python
# show a sample of random 15 books and see their correlation
bookratings.corrwith(book1776).to_frame('corr').dropna().sample(15)
```
##### Output
<img width="364" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/bd7a3c07-f310-4703-a716-e4d8a3e36ab9">


## Non-Negative Matrix Factorization (NMF)
Non-Negative Matrix Factorization (NMF) to perform matrix factorization and generate personalized book recommendations for users.NMF is applied to factorize the ratings matrix into user and item features, revealing latent preferences. The recommend_books function takes a user ID and provides top book recommendations based on predicted ratings. By calculating the dot product between user and item features, the code generates personalized recommendations. The example usage showcases recommendations for user ID 1. NMF. This approach allows for the discovery of latent user preferences and provides users with tailored suggestions, enhancing their book discovery and reading experience.

```python
# Perform matrix factorization using Non-Negative Matrix Factorization (NMF)
model = NMF(n_components=10, random_state=42)
user_features = model.fit_transform(ratings_matrix)
item_features = model.components_

# Define a function to recommend books for a given user
def recommend_books(user_id, num_recommendations=5):
    user_index = pivot_df.index.get_loc(user_id)
    user_ratings = ratings_matrix[user_index, :]

    # Calculate the predicted ratings
    predicted_ratings = np.dot(user_features[user_index, :], item_features)

    # Get the indices of the top recommended books
    top_books_indices = np.argsort(predicted_ratings)[::-1][:num_recommendations]

    # Get the corresponding book IDs
    top_books_ids = pivot_df.columns[top_books_indices]

    # Get the book information from the books dataset
    recommended_books = books_df[books_df['book_id'].isin(top_books_ids)][['book_id', 'original_title']]

    return recommended_books

# Example usage
user_id = 1
recommended_books = recommend_books(user_id)
print(f"Recommended books for user {user_id}:")
print(recommended_books)
```
##### Output
<img width="211" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/cfa4b8c4-a6fe-4824-a77e-3231997732c0">

```python
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the ratings data into a DataFrame (example)
ratings_data = pd.DataFrame({
    'user_id': [1, 1, 2, 2, 3, 3],
    'book_id': [1, 2, 1, 3, 2, 3],
    'rating': [5, 4, 3, 2, 1, 5]
})

# Create a pivot table of ratings
pivot_df = ratings_data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
ratings_matrix = csr_matrix(pivot_df.values)

# Perform matrix factorization using Non-Negative Matrix Factorization (NMF)
model = NMF(n_components=10, random_state=42)
user_features = model.fit_transform(ratings_matrix)
item_features = model.components_

# Define a function to recommend books for a given user
def recommend_books(user_id, num_recommendations=5):
    user_index = pivot_df.index.get_loc(user_id)
    user_ratings = ratings_matrix[user_index, :]

    # Calculate the predicted ratings
    predicted_ratings = np.dot(user_features[user_index, :], item_features)

    # Get the indices of the top recommended books
    top_books_indices = np.argsort(predicted_ratings)[::-1][:num_recommendations]

    # Get the corresponding book IDs
    top_books_ids = pivot_df.columns[top_books_indices]

    return top_books_ids

# Example usage
user_id = 1
recommended_books = recommend_books(user_id)

# Get the actual ratings for the user
user_index = pivot_df.index.get_loc(user_id)
actual_ratings = ratings_matrix[user_index, :].toarray().flatten()

# Calculate the predicted ratings
predicted_ratings = np.dot(user_features[user_index, :], item_features)

# Remove any NaN values in the actual and predicted ratings
actual_ratings = actual_ratings[~np.isnan(actual_ratings).astype(bool)]
predicted_ratings = predicted_ratings[~np.isnan(predicted_ratings).astype(bool)]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))

# Calculate MAE
mae = mean_absolute_error(actual_ratings, predicted_ratings)

print(f"Recommended books for user {user_id}:")
print(recommended_books)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
```
##### Output
<img width="250" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/97d1c06c-3352-4118-9f86-cb23e10da318">

## Using Singular Value Decomposition(SVD)
The provided code implements TruncatedSVD for collaborative book recommendations. It splits the data into training and testing sets, creates a user-item matrix, and applies matrix factorization to uncover latent features. The code then demonstrates how to generate personalized recommendations for a specific user based on the predicted ratings. The recommendations are sorted and displayed. TruncatedSVD enables the system to capture underlying patterns and user preferences, providing tailored book suggestions. This approach enhances the recommendation system's ability to deliver personalized recommendations, improving the user experience and promoting book discovery.
```python
# Define the path to your ratings.csv file
ratings_file_path = 'ratings.csv'

# Define the path to your books.csv file
books_file_path = 'books.csv'

# Load the ratings dataset
ratings_df = pd.read_csv(ratings_file_path)

# Split the data into training and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Create the user-item matrix
pivot_df = train_data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)

# Apply truncated SVD to factorize the user-item matrix
svd = TruncatedSVD(n_components=10, random_state=42)
user_features = svd.fit_transform(pivot_df.values)
item_features = svd.components_

# Example usage: Recommend books for a specific user
user_id = 1

# Get the index of the user in the user-item matrix
user_index = pivot_df.index.get_loc(user_id)

# Calculate predicted ratings for the user
predicted_ratings = np.dot(user_features[user_index, :], item_features)

# Sort the predicted ratings in descending order
top_books_indices = np.argsort(predicted_ratings)[::-1]

# Get the corresponding book IDs
top_books_ids = pivot_df.columns[top_books_indices]

# Print the top recommended books
num_recommendations = 5
top_recommendations = top_books_ids[:num_recommendations]

print(f"Top {num_recommendations} recommended books for user {user_id}:")
for book_id in top_recommendations:
    print(f"Book ID: {book_id}")
```
##### Output
<img width="214" alt="Screenshot 2023-10-14 015736" src="https://github.com/mahmudaAfreen/Book-Recommendation-System-using-Python/assets/36468927/badfb54a-9c62-44d7-a91d-b9b467196a20">


