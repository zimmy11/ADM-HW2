import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


def explore(df):
    print("List of Columns of the DataFrame:", df.columns, "\n")
    print("Shape of the DataFrame:", df.shape, "\n")
    print("Data Types of the columns:\n", df.dtypes, "\n")
    print("Number of missing values:",df.isnull().sum())


def ouliers(df):
    # IQR method for outliers
    Q1 = df['comment_count'].quantile(0.25)
    Q3 = df['comment_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['comment_count'] < lower_bound) | (df['comment_count'] > upper_bound)]
    print("Number of outliers in number of comments per app:", outliers.shape[0])


def stats(df):
    total_reviews = df['review_id'].nunique()
    print(f"Total reviews: {total_reviews}\n")

    unique_apps = df['app_id'].nunique()
    print(f"Unique apps: {unique_apps}\n")

    average_votes_helpful = df['votes_helpful'].mean()
    print(f"Average helpful votes: {average_votes_helpful}\n")

    reviews_per_app = df['app_name'].value_counts().head()
    print(f"Number of reviews for app (first rows): {reviews_per_app}\n")

    top_5_apps = df['app_name'].value_counts().head(5)
    print(f"Top 5 apps with the most reviews:  {top_5_apps}\n")

    top_5_authors = df['author.steamid'].value_counts().head(5)
    print(f"Top 5 authors with the most reviews: {top_5_authors}\n")

    average_playtime_forever = df['author.playtime_forever'].mean()
    print(f"Average playtime forever: {average_playtime_forever}")

    top_5_languages = df['language'].value_counts().head()
    print(f"Top 5 Languages most used for reviews: {top_5_languages}\n")









def visualize(df):

    # Bar Plot Histogram
    top_5_apps = df['app_name'].value_counts().head(5)

    plt.figure(figsize=(12, 8))
    sns.barplot(data = top_5_apps)
    plt.title('Distribution of Reviews of the Top 5 Most Reviewed Apps')
    plt.xlabel('Apps')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

    top_5_reviewers = df['author.steamid'].value_counts().head(5)

    plt.figure(figsize=(10, 6))
    sns.barplot(data = top_5_reviewers)
    plt.title('Distribution of Reviews of the Top 5 Reviewers')
    plt.xlabel('ReviewersId')
    plt.ylabel('Value')
    plt.show()

    # Bar Plot
    language_by_review = df['language'].value_counts()

    plt.figure(figsize=(12, 8))
    sns.barplot(x=language_by_review.index, y=language_by_review)
    plt.title('Number of Reviews by Language', fontsize=16, pad=20)
    plt.xlabel('Number of Reviews')
    plt.ylabel('Languages')
    plt.yticks(ticks=range(0, language_by_review.max() + 1, 1000000)) 
    plt.xticks(rotation=45)
    plt.show()


    # Scatter Plot
    unique_authors_df = df.drop_duplicates(subset='author.steamid')
    plt.figure(figsize=(10, 6))
    plt.scatter(x=unique_authors_df['author.num_reviews'], y = unique_authors_df['author.playtime_forever'], color = 'blue' , alpha = 0.6)
    plt.title('Correlation between number of reviews and amount of time played')
    plt.yticks(ticks=range(0, unique_authors_df['author.num_reviews'].max() + 1, 1000000)) 
    plt.xticks(ticks=range(0, unique_authors_df['author.playtime_forever'].max() + 1, 1000000)) 
    plt.xlabel('Value')
    plt.show()


    #Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data = df[['comment_count', 'app_id']], x = 'app_id', y = 'comment_count')
    plt.xticks([])
    plt.title('Distribution of Number of Comments amongst Reviews')
    plt.show()

def highest_lowest_reviews_applications(df):

    # Compute gathering all the data with the same app_id, adding a new columns for that
    top_review_counts = df.groupby(['app_id','app_name']).size().reset_index(name='count_reviews')

    # Find the max
    max_count = top_review_counts['count_reviews'].max()
    
    # Select all the apps with the max number of reviews
    top_apps = top_review_counts[top_review_counts['count_reviews'] == max_count][['app_name','count_reviews']]

    print(f"Top apps with the most reviews:  {top_apps}\n")


    # Compute gathering all the data with the same app_id, adding a new columns for that
    bottom_review_counts = df.groupby(['app_id','app_name']).size().reset_index(name='count_reviews')

    # Find the minimum
    min_count = bottom_review_counts['count_reviews'].min()
    
    # Select all the apps with the minimum number of reviews
    bottom_apps = bottom_review_counts[bottom_review_counts['count_reviews'] == min_count][['app_name','count_reviews']]

    print(f"Top apps with the minimum reviews:  {bottom_apps}\n")

def reviews_count_plot(df):
    
    review_counts = df.groupby('app_name').size().reset_index(name='count_reviews')
    review_counts = review_counts.sort_values(by='count_reviews', ascending=False)



    #Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=review_counts, x='app_name', y='count_reviews')
    plt.title('Number of Reviews for Each Application', fontsize=12)
    plt.xlabel('Application Name', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    current_ticks = plt.xticks()[0]

    plt.xticks(ticks=[current_ticks[0], current_ticks[-1]], 
           labels=[review_counts['app_name'].iloc[0], review_counts['app_name'].iloc[-1]], 
           rotation=90)
    plt.show()


def puchased_gratis_reviewers(df):

    top_5_apps = df['app_name'].value_counts().head(5)
    top_5_apps_names = top_5_apps.index
    total_number_of_reviewers =   pd.DataFrame(top_5_apps)

    reviewers_purchased = df[(df['app_name'].isin(top_5_apps_names)) & (df['steam_purchase'] == True)].groupby('app_name').count()
    reviewers_gratis = df[(df['app_name'].isin(top_5_apps_names)) & (df['received_for_free'] == True)].groupby('app_name').count()
    reviewers_purchased = pd.DataFrame({
    'percentage_of_purchase_reviewers': round(reviewers_purchased['steam_purchase'] * 100 / total_number_of_reviewers['count'], 2)
    })
    reviewers_gratis = pd.DataFrame({
    'percentage_of_free_reviewers': round(reviewers_gratis['received_for_free'] * 100 / total_number_of_reviewers['count'], 2)
    })
    print(f"Percentage of Reviewers that purchased the app for the Top 5 Most Reviewed Apps:\n {reviewers_purchased}")
    print(f"Percentage of Reviewers that received for free the app for the Top 5 Most Reviewed Apps:\n {reviewers_gratis}")


def most_least_recommended_reviews_applications(df):

    # Compute gathering all the data with the same app_id, adding a new columns for that
    top_review_counts = df.groupby(['app_id','app_name']).size().reset_index(name='count_reviews')

    # Find the max
    max_count = top_review_counts['count_reviews'].max()
    
    # Select all the apps with the max number of reviews
    top_apps = top_review_counts[top_review_counts['count_reviews'] == max_count][['app_name','count_reviews']]

    print(f"Top Most recommended apps with the most reviews:  {top_apps}\n")


    # Compute gathering all the data with the same app_id, adding a new columns for that
    bottom_review_counts = df.groupby(['app_id','app_name']).size().reset_index(name='count_reviews')

    # Find the minimum
    min_count = bottom_review_counts['count_reviews'].min()
    
    # Select all the apps with the minimum number of reviews
    bottom_apps = bottom_review_counts[bottom_review_counts['count_reviews'] == min_count][['app_name','count_reviews']]

    print(f"Top apps with the minimum reviews:  {bottom_apps}\n")
