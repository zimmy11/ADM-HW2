import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
from textblob import TextBlob
from langdetect import detect
from googletrans import Translator
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def explore(df):
    print("List of Columns of the DataFrame:", df.columns, "\n")
    print("Shape of the DataFrame:", df.shape, "\n")
    print("Data Types of the columns:\n", df.dtypes, "\n")
    print("Number of missing values:\n",df.isnull().sum())


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
    top_apps.set_index('app_name', inplace=True)

    print(f"Top apps with the most reviews:\n  {top_apps}\n")


    # Compute gathering all the data with the same app_id, adding a new columns for that
    bottom_review_counts = df.groupby(['app_id','app_name']).size().reset_index(name='count_reviews')

    # Find the minimum
    min_count = bottom_review_counts['count_reviews'].min()
    
    # Select all the apps with the minimum number of reviews
    bottom_apps = bottom_review_counts[bottom_review_counts['count_reviews'] == min_count][['app_name','count_reviews']]
    bottom_apps.set_index('app_name', inplace=True)

    print(f"Top apps with the minimum reviews:\n  {bottom_apps}\n")

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

    reviewers_purchased = df[(df['app_name'].isin(top_5_apps_names))].groupby('app_name')['steam_purchase'].sum()
    reviewers_gratis = df[(df['app_name'].isin(top_5_apps_names))].groupby('app_name')['received_for_free'].sum()
    reviewers_purchased = pd.DataFrame({
    'percentage_of_purchase_reviewers': round(reviewers_purchased * 100 / total_number_of_reviewers['count'], 2)
    })
    reviewers_gratis = pd.DataFrame({
    'percentage_of_free_reviewers': round(reviewers_gratis * 100 / total_number_of_reviewers['count'], 2)
    })
    print(f"Percentage of Reviewers that purchased the app for the Top 5 Most Reviewed Apps:\n {reviewers_purchased}")
    print(f"Percentage of Reviewers that received for free the app for the Top 5 Most Reviewed Apps:\n {reviewers_gratis}")


def most_least_recommended_reviews_applications(df):

    # Compute gathering all the data with the same app_name, and we use the function sum to sum the True values (will be treated as 1) in the 'recommended' column
    recommended_counts = df.groupby('app_name')['recommended'].sum()

    # Find the max
    max_count = recommended_counts.max()
    
    # Select all the apps with the max number of reviews
    top_apps = recommended_counts[recommended_counts == max_count]

    print(top_apps)
    print(f"Most recommended apps:\n  {pd.DataFrame(top_apps).rename(columns={'recommended': 'count_of_recommended_reviews'})}\n")

   # Find the minimum
    min_count = recommended_counts.min()
    

    # Select all the apps with the minimum number of reviews
    bottom_apps = recommended_counts[recommended_counts == min_count]

    
    print(f"Least Recommended apps:\n  {pd.DataFrame(bottom_apps).rename(columns={'recommended': 'count_of_recommended_reviews'})}\n")


def statistical_correlation(df):
    
    # We compute the number of reviews that recommend the app per application
    recommended_counts = df.groupby('app_name')['recommended'].sum()

    # We compute the number of total reviews per app (i use the count because i know that there arent Nan values in the 'recommended' column, so i dont skip any review)
    total_counts = df.groupby('app_name')['recommended'].count()

    # We transform into Dataframes the previous Series in order to add a new 'reviews_score' column
    total_counts = pd.DataFrame(total_counts)
    reviews_score = pd.DataFrame(recommended_counts)
    
    # We do the calculus of score_review dividing the number of recommended review for the number of total reviews for each app and we normalise in (0,10)

    reviews_score['reviews_score'] = round((reviews_score['recommended'] / total_counts['recommended']) * 10, 2)

    # Visualize the relationship

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=reviews_score, x='recommended', y='reviews_score')
    plt.title('Relationship Between Recommendations and Review Scores')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Review Scores')
    plt.grid(True)
    plt.show()
    
    # Calculate correlation coefficients
    pearson_corr, pearson_pvalue = pearsonr(reviews_score['recommended'], reviews_score['reviews_score'])
    spearman_corr, spearman_pvalue = spearmanr(reviews_score['recommended'], reviews_score['reviews_score'])
    kendall_corr, kendall_pvalue = kendalltau(reviews_score['recommended'], reviews_score['reviews_score'])

    print(f"Pearson correlation coefficient: {pearson_corr:.2f}, p-value: {pearson_pvalue:.4f}")
    print(f"Spearman correlation coefficient: {spearman_corr:.2f}, p-value: {spearman_pvalue:.4f}")
    print(f"Kendall correlation coefficient: {kendall_corr:.2f}, p-value: {kendall_pvalue:.4f}")

    # Interpretation
    alpha = 0.05  # significance level
    if pearson_pvalue < alpha:
        print("There is a statistically significant linear correlation (Pearson).")
    else:
        print("There is no statistically significant linear correlation (Pearson).")

    if spearman_pvalue < alpha:
        print("There is a statistically significant monotonic correlation (Spearman).")
    else:
        print("There is no statistically significant monotonic correlation (Spearman).")

    if kendall_pvalue < alpha:
        print("There is a statistically significant monotonic correlation (Kendall).")
    else:
        print("There is no statistically significant monotonic correlation (Kendall).")


    
translator = Translator()

language_codes = {
    'english': 'en',
    'schinese': 'zh-cn',  
    'russian': 'ru',
    'brazilian': 'pt',    
    'spanish': 'es',
    'german': 'de',
    'turkish': 'tr',
    'korean': 'ko',
    'french': 'fr',
    'polish': 'pl',
    'tchinese': 'zh-tw',  
    'czech': 'cs',
    'italian': 'it',
    'thai': 'th',
    'japanese': 'ja',
    'portuguese': 'pt',   
    'swedish': 'sv',
    'dutch': 'nl',
    'hungarian': 'hu',
    'latam': 'es-419',   
    'danish': 'da',
    'finnish': 'fi',
    'norwegian': 'no',
    'romanian': 'ro',
    'ukrainian': 'uk',
    'greek': 'el',
    'bulgarian': 'bg',
    'vietnamese': 'vi'
}
# Function to translate non-English text to English
def translate_to_english(text, lang_name):
    try:
        lang = language_codes.get(lang_name)
        if lang != 'en':
            translated = translator.translate(text, src=lang, dest='en').text
            return translated
        else:
            return text
    except Exception as e:
        return text


nltk.download('vader_lexicon')

# Initialize the sentiment analyzer of Vader
sia = SentimentIntensityAnalyzer()

# Function to classify sentiment
def classify_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    polarity = sentiment_scores['compound']
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'
    
def sentiment_classification(df):

     # Count the occurrences of each language
    top_languages = df['language'].value_counts().nlargest(1).index.tolist()

    # Filter the DataFrame to only include the top 3 languages
    df_top_languages = df[df['language'].isin(top_languages)]

    df_top_languages = df_top_languages[df_top_languages['review'].notna()]


    
    # Translate non-English reviews to English
    # df_top_languages['translated_review'] = df_top_languages.apply(lambda row: translate_to_english(row['review'], row['language']), axis=1)

    #  Perform sentiment analysis
    df_top_languages['sentiment'] = df_top_languages['review'].apply(classify_sentiment)

    return df_top_languages

def compute_distribution(df):
    sentiment_distribution = df['sentiment'].value_counts(normalize=True) * 100
    # Display sentiment distribution
    print("Sentiment Distribution (in %):")
    print(sentiment_distribution)

def comparison_sentiment_recommendations(df):
    # Analyze sentiment distribution for recommended and non-recommended reviews
    sentiment_summary = df.groupby(['recommended', 'sentiment']).size().unstack(fill_value=0)

    # Calculate the percentage of each sentiment category for recommended and non-recommended reviews
    sentiment_percentage = sentiment_summary.div(sentiment_summary.sum(axis=1), axis=0) * 100

    print("Sentiment Distribution by Recommendation:")
    print(sentiment_summary)

    print("\nSentiment Percentage by Recommendation:")
    print(sentiment_percentage)


def correlation_sentiment_helpful_votes(df):
    df['sentiment'] = df['sentiment'].astype('category').cat.codes


    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sentiment', y='votes_helpful')

    plt.title('Scatter Plot: Sentiment vs Number of Helpful Votes')
    plt.xlabel('Sentiment (Positive, Neutral, Negative)')
    plt.ylabel('Number of Helpful Votes')


    # Show
    plt.grid(True)
    plt.show()



def algorithmic_question(n, k):
    # if the list can't be made it will not print
    no_fail = True
    # creating a list of k elements that contains the floor division
    lista = [n//k] * k
    # computing how much we miss from the original number
    reminder = (n-sum(lista))

    # if we miss a even number we can just add it to the first element
    # and the list will keep it's polarity
    # if it is odd we need to swap the polarity of the list, this is possible only if the 
    # list is odd as the first element is changed by the reminder and the k-1 elements 
    # are changed by loopign +1 and -1.
    # if k-1 is odd we can't match the -1 and +1 and therfore cant cahnge the polarity

    if reminder % 2 != 0: # if reminder is odd
        if k % 2 == 0 or lista[0] == 1: # if cant +1 -1 without change or get 0
            no_fail = False # will not print the list
            print("NO")
        else:
            for i in range(1, k):# alternate +1 -1 from second to last element
                if i % 2 == 0:
                    lista[i] += 1
                else:
                    lista[i] -= 1
    if no_fail: # if the list is good
        lista[0] += reminder # sum reminder to the first element
        print("YES\n", *lista) # print yes and the items

