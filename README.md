# Recommendation-System
• Build a Content-based and Collaborative Filtering Recommendation System based on the Yelp review dataset.
<br>
• For Content-based Recommendation System, construct business profile vectors by using all review texts for each business
and using the top 200 words with the highest TF-IDF score to describe the business then create vectors to represent the user profile by aggregating the profiles of the businesses that the user has reviewed. By calculating the cosine similarity between the user vector and business vector to measure if the user prefers to review the business.<br>
• For item-based Collaborative Filtering Recommendation System, computes the Pearson correlation for the business pairs, and predict the rating for a given pair of user and business by using N business neighbors that are most similar to the target business.<br>
• Apply Min-Hash and LSH algorithms in a user-based CF recommendation system to eliminate dissimilar user pairs. Then compute the Pearson correlation for user pairs by their rating on each business and predict the rating for a given pair of user and business.
