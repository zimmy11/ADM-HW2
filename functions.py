import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import chi2_contingency


def explore(df):
    missing_values_mask = df.isnull().sum()
    print("List of Columns of the DataFrame:", df.columns, "\n")
    print("Shape of the DataFrame:", df.shape, "\n")
    print("Data Types of the columns:\n", df.dtypes, "\n")
    print("Number of missing values:\n",missing_values_mask[missing_values_mask > 0])


def ouliers(df):
    # IQR method for outliers
    Q1 = df['comment_count'].quantile(0.25)
    Q3 = df['comment_count'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # We Filter for all the elements the have comment_count smaller than lower_bound
    #  and greater than upper_bound
    outliers = df[(df['comment_count'] < lower_bound) | (df['comment_count'] > upper_bound)]
    print("Number of outliers in number of comments:", outliers.shape[0])


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
    
    # Group by app_name and compute the number of reviews per app 
    review_counts = df.groupby('app_name').size().reset_index(name='count_reviews')
    # Sort the Dataframe in descending order
    review_counts = review_counts.sort_values(by='count_reviews', ascending=False)

    #Create the bar plot with x = 'app_name' and y = 'count_reviews'
    plt.figure(figsize=(10, 6))
    sns.barplot(data=review_counts, x='app_name', y='count_reviews', palette='viridis')
    plt.title('Number of Reviews for Each Application', fontsize=12)
    plt.xlabel('Application Name', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    current_ticks = plt.xticks()[0]

    plt.xticks(ticks=[current_ticks[0], current_ticks[-1]], 
           labels=[review_counts['app_name'].iloc[0], review_counts['app_name'].iloc[-1]], 
           rotation=90)
    
        # Draw a line connecting the tops of the bars
    bin_centers = np.arange(len(review_counts))  # X-coordinates for the line
    plt.plot(bin_centers, review_counts['count_reviews'].values, color='#5B2E91', marker='o', linestyle='-', linewidth=2)

    # Show grid and adjust layout
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def puchased_gratis_reviewers(df):

    # Obtain the 5 apps with the most reviews
    top_5_apps = df['app_name'].value_counts().head(5)
    top_5_apps_names = top_5_apps.index
    # Create a Datframe from a Series
    total_number_of_reviewers =   pd.DataFrame(top_5_apps)

    # Filter based on the app_name is in the top most reviewed apps and sum the numbers
    # of reviewers that have purchased the app or received for free with the sum()
    reviewers_purchased = df[(df['app_name'].isin(top_5_apps_names))].groupby('app_name')['steam_purchase'].sum()
    reviewers_gratis = df[(df['app_name'].isin(top_5_apps_names))].groupby('app_name')['received_for_free'].sum()
    
    # Create 2 Dataframes with the distributions of the reviewers
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
    sns.scatterplot(data=reviews_score, x='recommended', y='reviews_score', palette='coolwarm',
                    sizes=(20, 200),alpha=0.7, edgecolor='w',linewidth=0.5)
    plt.title('Relationship Between Recommendations and Review Scores')
    plt.xlabel('Number of Recommendations')
    plt.ylabel('Review Scores')
    plt.legend(title='Review Score', loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Calculate correlation coefficients
    pearson_corr, pearson_pvalue = pearsonr(reviews_score['recommended'], reviews_score['reviews_score'])

    print(f"Pearson correlation coefficient: {pearson_corr:.2f}, p-value: {pearson_pvalue:.4f}")
    # Interpretation
    alpha = 0.05  # significance level
    if pearson_pvalue < alpha:
        print("There is a statistically significant linear correlation (Pearson).")
    else:
        print("There is no statistically significant linear correlation (Pearson).")



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

    # Define vibrant colors for the pie chart
    colors = ['#FF9999', '#66B3FF', '#99FF99']

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_distribution, 
            labels=sentiment_distribution.index, 
            colors=colors,                        
            autopct='%1.1f%%',                  
            startangle=140,                     
            explode=(0.1,) * len(sentiment_distribution)  
        )
    plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
    # Equal aspect ratio to give a circular aspect to the figure
    plt.axis('equal') 
    plt.show()

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
    # Classify in (0,1,2) the 3 categories (neutral, positive, negative)
    df['sentiment'] = df['sentiment'].astype('category').cat.codes

    # Create a scatter plot between the number of helpful votes and the sentiment category
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


def probability_one_helpful_vote(df):

    # Compute the number of reviews with zero helpful votes
    # Calculate a Series with the number of votes for each unique value present
    vote_counts = df[df['votes_helpful'] == 0]


    # Compute the total number of reviews
    total_helpful_reviews = df.shape[0]
    zero_helpful_reviews = vote_counts.shape[0]


    return zero_helpful_reviews / total_helpful_reviews if total_helpful_reviews > 0 else 0


def conditional_probability_recommended(df):
    # Compute the number of reviews with zero helpful votes
    # Calculate a Series with the number of votes for each unique value present filtering by reviews not reccomended (intersection)
    total_not_recommended_reviews = df[df['recommended'] == False]
    total_not_recommended_and_more_than_one_vote_reviews = total_not_recommended_reviews[total_not_recommended_reviews['votes_helpful'] >= 1].shape[0]


    # Total Number of not recommended reviews 
    total_not_recommended_reviews = total_not_recommended_reviews.shape[0]

    
    return total_not_recommended_and_more_than_one_vote_reviews / total_not_recommended_reviews if total_not_recommended_reviews > 0 else 0


def check_probability_independence(df):
    
    # We Sort the DataFrame based on the column timestamp_created in descending order
    df = df.sort_values(by='timestamp_created', ascending=False)

    # Drop the Duplicates of the column author.steamid because we only wanna extract the last value
    # of the column 'author.num_reviews', the most recent one
    df = df.drop_duplicates(subset='author.steamid', keep='first')
    # Create a contingency table
    contingency_table = pd.crosstab(df['votes_helpful'] >= 1, df['author.num_reviews'] >= 5)

    # Apply the Chi-Square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"Chi-Square Test: Chi2={chi2}, p-value={p_value}")

    # Result interpretation
    if p_value < 0.05:
        print("The two events are NOT independent (significant at 95%)")
    else:
        print("The two events ARE independent (not significant)")


def check_correlation(df):

    # Data Preparation to get only the rows with unique reviewers and their latest infos
    # About num_games_owned and num_reviews_submitted
    df = df.sort_values(by='timestamp_created', ascending=False)

    # Drop the Duplicates of the column author.steamid
    df = df.drop_duplicates(subset='author.steamid', keep='first')

    # Calculate summary statistics for the two columns
    print(df[['author.num_games_owned', 'author.num_reviews']].describe())
    
    # Correlation
    # We Compute the Pearson Index to quantify the correlation between the two features
    # Pearson correlation
    pearson_corr, p_value_pearson = pearsonr(df['author.num_games_owned'], df['author.num_reviews'])
    print("Pearson correlation coefficient:", pearson_corr, "p-value:", p_value_pearson)



def time_vs_relevance(df):

    # trnasform the second played before the review to hours
    df.loc[:,'playtime'] = df['author.playtime_at_review'] / 3600

    # initialize the relevance of the reviews
    relevence = df.review_score

    # compute the robust scaling to normalize the value
    relevence = (relevence - np.median(relevence)) / (np.percentile(relevence, 75) - np.percentile(relevence, 25))

    # fix the values to be in a range between 10-0
    df.loc[:,"review_score"] = 10*(relevence - np.min(relevence)) / (np.max(relevence) - np.min(relevence))

    # return the relevant columns
    return df[["review_score",'playtime']]




def plot_time_vs_revewscore(plot_df, label = str):
    # to differentiate plots i change color based on raccomandation
    col = "green" if label == "Positive" else "blue"
    
    # Create an empty plot
    plt.figure(figsize=(10, 6))
    # scatterplot of review score over hours of playtime
    sns.scatterplot(x="playtime", y="review_score", color=col, data= plot_df)
    # here i get the value of the first and third quartile to display the skewness of the distribution
    q1 = np.percentile(plot_df.playtime, 25)
    q3 = np.percentile(plot_df.playtime, 75)
    # here i plot the lines of the quartiles
    plt.axvline(q1, color='red', linestyle='--', label=f'25th Percentile (Q3): {q1:.2f}')
    plt.axvline(q3, color='red', linestyle='--', label=f'75th Percentile (Q3): {q3:.2f}')
    # here i add a tile to the plot
    plt.title(label + " Review Score. " + str(len(plot_df)) + " reviews analized")
    # here i add axis names to the plot
    plt.xlabel("Hours Played at Review")
    plt.ylabel("Normalized "+ label +" Review Score")
    # i add a legend to the plot
    plt.legend(loc="upper right")
    plt.show()  # Show the plot

def plot_experience_vs_reviewscore(df_plot):
    # create an empty plot
    plt.figure(figsize=(10,6))
    # scatterplot of total time spent on the pc versus the review socre, a person levaing multiple review will have the sae total play time
    sns.scatterplot(x= "play_time", y = "score", data = df_plot)
    # first and thirs quartile to observe skewness
    q1 = np.percentile(df_plot.play_time, 25)
    q3 = np.percentile(df_plot.play_time, 75)
    # plot lines at the quartiles
    plt.axvline(q1, color='red', linestyle='--', label=f'25th Percentile (Q3): {q1:.2f}')
    plt.axvline(q3, color='red', linestyle='--', label=f'75th Percentile (Q3): {q3:.2f}')
    # axis names
    plt.xlabel("Total time spent gameing")
    plt.ylabel("joint normalized review score")
    # a name to the plot
    plt.title("relation between experience and review relevance")
    # add a legend to the plot
    plt.legend(loc="upper right")
    plt.show()  # Show the plot
    
