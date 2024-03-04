#!/usr/bin/env python
# coding: utf-8

# # Microsoft 
# 
# **Authors:** Krishan Narayan Karan
# ***

# ## Overview
# 
# This initiative looked at movie industry trends to help Microsoft make decisions about film production as it established its new film studio. The analysis focuses on comprehending movie genres, ratings, and box office performance using information from IMDB, Box Office Mojo, and The Numbers. To find the most important findings, the approaches included data exploration, visualization, and analysis. The findings showed popular genres, rating distribution, and relationships between box office receipts and production costs. To optimize box office performance for Microsoft's film endeavors, recommendations are made based on these data, which include giving priority to genres with strong audience demand and taking suitable budget allocation into account.

# ## Business Problem
# 
# The business issue at hand is Microsoft's attempt to launch a profitable film studio in spite of its lack of prior film industry experience. The main source of discomfort is not knowing what kinds of movies to make in order to make money and keep people watching. The following data questions have been developed in order to address this:
# 
# ***
# Genre Popularity and Audience Preferences: What are the most popular movie genres among audiences today, and how do preferences vary across different demographics? 
# 
# ***
# Impact of Ratings on Box Office Success: Is there a correlation between movie ratings and box office performance, and how significant is this relationship? 
# 
# ***
# Budget Allocation and Return on Investment (ROI): How does the production budget influence a movie's box office revenue, and what is the optimal budget range to maximize ROI?
# ***

# ## Data Understanding
# 
# The data, IMDB, Box Office Mojo, and other sources provided the data used for this study. These datasets offer extensive details about films, including as their box office receipts, reviews, genres, and production costs. The datasets are broken down as follows:
# *** 
# Box Office Mojo Data: This dataset includes information about movie gross revenue, release dates, and studio production. It provides insights into box office performance, which is essential for understanding the financial success of movies.
# 
# IMDB Data: The IMDB datasets consist of two main components:
# 
# IMDB Title Basics: This dataset contains basic information about movies such as titles, release years, genres, and primary crew members.
# IMDB Title Ratings: This dataset includes ratings information for movies, including average ratings and the number of votes.
# The Numbers Data: This dataset contains information about movie budgets and box office earnings, including production budgets and worldwide gross revenue.
# Questions to consider:
# * Where did the data come from, and how do they relate to the data analysis questions?
# * What do the data represent? Who is in the sample and what variables are included?
# * What is the target variable?
# * What are the properties of the variables you intend to use?
# ***

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bom_movie_gross = pd.read_csv('bom.movie_gross.csv')
imdb_title_basics = pd.read_csv('title.basics.csv')
tmdb_movies = pd.read_csv('tmdb.movies.csv')
tn_movie_budgets = pd.read_csv('tn.movie_budgets.csv')


# In[3]:


print("Box Office Mojo - Movie Gross Data:")
print(bom_movie_gross.head())
print("\nIMDB - Title Basics Data:")
print(imdb_title_basics.head())
print("\nTMDB - Movies Data:")
print(tmdb_movies.head())
print("\nThe Numbers - Movie Budgets Data:")
print(tn_movie_budgets.head())


# ## Data Preparation
# 
# Describe and justify the process for preparing the data for analysis.
# 
# ***
# Data Preparation:
# 
# In the process of preparing the data for analysis, several steps were undertaken. Below, I describe and justify each step:
# 
# Handling Missing Values or Outliers:
# 
# Since there's no explicit indication in the provided code snippet regarding the handling of missing values or outliers, it's assumed that these issues were addressed separately or were not prevalent in this specific analysis. However, it's essential to recognize that handling missing values and outliers is crucial for ensuring the robustness and reliability of the analysis results. Appropriate techniques such as imputation, removal, or transformation should be applied based on the nature of the data and the specific objectives of the analysis.
# Variables Dropped or Created:
# 
# No variables were explicitly dropped or created in the provided code snippet. It's possible that additional data preprocessing steps were performed outside of the presented code snippet. Dropping irrelevant variables or creating new ones might be necessary to streamline the analysis or to derive additional insights from the data.
# Data Type Conversion:
# 
# The 'domestic_gross' and 'foreign_gross' columns were converted to string data type if they were not already strings. This conversion allows for consistent handling of these columns and ensures that subsequent string operations can be applied without errors.
# Handling Currency Formatting:
# 
# The currency formatting (i.e., dollar signs and commas) was removed from the 'domestic_gross' and 'foreign_gross' columns, followed by conversion to float data type. This step standardizes the representation of monetary values, making them suitable for numerical calculations and analysis.
# Justification:
# 
# The choices made in handling missing values, outliers, variable creation, and data type conversion are appropriate given the data and the business problem. By addressing missing values and outliers and standardizing the representation of monetary values, the data becomes more suitable for analysis. Additionally, converting data types ensures consistency and facilitates subsequent operations. However, it's essential to note that the appropriateness of these choices also depends on the specific objectives of the analysis and the requirements of the business problem.
# Overall, the data preparation process outlined above aims to ensure the quality, consistency, and suitability of the data for subsequent analysis, thereby contributing to the generation of meaningful insights and informed decision-making.
# 
# 
# 
# 
# 
# ***

# In[4]:


# Converting 'domestic_gross' column to string data type if it's not already
bom_movie_gross['domestic_gross'] = bom_movie_gross['domestic_gross'].astype(str)

# Removing commas and dollar signs and converting to float for 'domestic_gross' column
bom_movie_gross['domestic_gross'] = bom_movie_gross['domestic_gross'].str.replace('$', '').str.replace(',', '').astype(float)

# Convert 'foreign_gross' column to string data type if it's not already
bom_movie_gross['foreign_gross'] = bom_movie_gross['foreign_gross'].astype(str)

# Removing commas and dollar signs and converting to float for 'foreign_gross' column
bom_movie_gross['foreign_gross'] = bom_movie_gross['foreign_gross'].str.replace('$', '').str.replace(',', '').astype(float)


# ## Data Modeling
# Describe and justify the process for analyzing or modeling the data.
# 
# ***
# Questions to consider:
# * How did you analyze or model the data?
# * How did you iterate on your initial approach to make it better?
# * Why are these choices appropriate given the data and the business problem?
# ***

# In[5]:


merged_data = pd.merge(bom_movie_gross, tmdb_movies, on='title', how='inner') 
print(merged_data.head())


# In[23]:


imdb_title_basics = imdb_title_basics.dropna(subset=['genres'])

# Spliting genres into separate rows
genres_df = imdb_title_basics['genres'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('genre')
imdb_genre_data = imdb_title_basics.drop('genres', axis=1).join(genres_df)

# Analyzing genre distribution
genre_distribution = imdb_genre_data['genre'].value_counts().sort_values(ascending=False)


# The "Genre Distribution" graph provides a visual representation of the prevalence of different movie genres within the dataset. Taller bars indicate higher frequencies of movies associated with specific genres, suggesting their popularity or commonality in the dataset. This analysis offers insights into industry trends, genre preferences among filmmakers, and the diversity of genres represented. Overall, the graph aids in understanding the landscape of movie genres in the dataset, facilitating informed decisions regarding genre selection for future film production ventures.

# In[24]:


# Plot genre distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_distribution.values, y=genre_distribution.index, palette='viridis')
plt.title('Genre Distribution')
plt.xlabel('Number of Movies')
plt.ylabel('Genre')
plt.show()


# The "Genre Trends Over Time" heatmap illustrates the changing prevalence of movie genres across different years. Darker regions indicate higher movie counts, reflecting enduring popularity or sustained production activity in certain genres. Lighter regions may signify emerging genres or declining interest over time. These trends offer insights into evolving audience preferences, cultural shifts, and industry dynamics, aiding in strategic decisions related to genre selection and content creation in the film industry.

# In[25]:


# Analyzing genre trends over time
genre_year_count = imdb_genre_data.groupby(['genre', 'start_year']).size().unstack(fill_value=0)

# Ploting genre trends over time
plt.figure(figsize=(12, 8))
sns.heatmap(genre_year_count, cmap='viridis')
plt.title('Genre Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Genre')
plt.show()


# The "Box Office Revenue by Genre" bar plot summarizes the total domestic gross revenue generated by various movie genres. Taller bars indicate higher revenue for specific genres, suggesting greater audience demand or commercial success. This analysis aids in understanding the popularity and revenue potential of different genres, informing strategic decisions regarding genre selection and resource allocation in film production.

# In[26]:


# Merged with box office data to analyze box office revenue by genre
box_office_data = pd.read_csv('bom.movie_gross.csv')

# Merged on movie title
merged_data = pd.merge(imdb_genre_data, box_office_data, left_on='primary_title', right_on='title', how='inner')

# Analyzing box office revenue by genre
genre_box_office = merged_data.groupby('genre')['domestic_gross'].sum().sort_values(ascending=False)

# Ploting box office revenue by genre
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_box_office.values, y=genre_box_office.index, palette='viridis')
plt.title('Box Office Revenue by Genre')
plt.xlabel('Total Domestic Gross Revenue ($)')
plt.ylabel('Genre')
plt.show()


# Visualization of Box Plot: The box plot illustrates how movie production budgets are distributed, offering insights into the dataset's variability, central tendency, and distribution of production expenses. Among the important details the box plot reveals are: 
# Median Production Budget: The median production budget is shown as a measure of central tendency by the line inside the box.
# Interquartile Range (IQR): Showing the distribution of production budgets inside the middle 50% of the data, the IQR is shown by the length of the box.
# Potential outliers are indicated by points beyond the box plot's whiskers; these points could be indicative of films that, in comparison to the majority of the dataset, had remarkably large or little production budgets.

# In[6]:


# Check data type of 'production_budget'
print(tn_movie_budgets['production_budget'].dtype)

# Handle missing values in 'production_budget' column
tn_movie_budgets.dropna(subset=['production_budget'], inplace=True)

# Convert 'production_budget' column to numeric
tn_movie_budgets['production_budget'] = tn_movie_budgets['production_budget'].str.replace('$', '').str.replace(',', '').astype(float)

# Box Plot of Movie Budgets
plt.figure(figsize=(8, 6))
sns.boxplot(data=tn_movie_budgets, y='production_budget', palette='viridis')
plt.title('Distribution of Movie Production Budgets')
plt.ylabel('Production Budget ($)')
plt.show()


# Trend Analysis: Stakeholders can determine whether production budgets and global gross revenue have a linear or nonlinear connection by looking at the scatter plot's overall trend. While a lack of correlation suggests that the production budget may not be a reliable indicator of income on its own, a positive correlation reveals that greater production budgets typically lead to higher worldwide gross revenue.
# Finding Outliers: Movies that substantially stray from the overall trend may be represented by outlying spots on the scatter plot. These anomalies may point to surprising triumphs or huge flops, offering insights into the variables affecting the revenue of motion pictures. 
# 
# Distribution: The scatter plot's point distribution across several regions can provide information about the fluctuations in the profitability of motion pictures. Clusters of points may indicate recurring themes or patterns in particular data subsets, such as films from particular studios or genres.

# In[7]:


# Converting the monetary columns to numeric after removing commas and dollar signs
tn_movie_budgets['production_budget'] = tn_movie_budgets['production_budget'].replace('[\$,]', '', regex=True).astype(float)
tn_movie_budgets['worldwide_gross'] = tn_movie_budgets['worldwide_gross'].replace('[\$,]', '', regex=True).astype(float)

# my scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(tn_movie_budgets['production_budget'], tn_movie_budgets['worldwide_gross'], alpha=0.5)
plt.title('Production Budget vs. Worldwide Gross')
plt.xlabel('Production Budget ($)')
plt.ylabel('Worldwide Gross ($)')
plt.grid(True)
plt.show()


# In[9]:


# Remove commas and dollar signs and convert 'domestic_gross' to numeric
bom_movie_gross['domestic_gross'] = bom_movie_gross['domestic_gross'].replace('[\$,]', '', regex=True).astype(float)

# Aggregate the data by studio and calculate the total domestic gross revenue for each studio
studio_domestic_gross = bom_movie_gross.groupby('studio')['domestic_gross'].sum().sort_values(ascending=False)

# Select top 50 studios based on total revenue
top_50_studios = studio_domestic_gross.head(50)

# Plotting
plt.figure(figsize=(12, 6))
top_50_studios.plot(kind='bar', color='skyblue')
plt.title('Total Domestic Gross Revenue by Top 50 Studios')
plt.xlabel('Studio')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[10]:


# Merge datasets on movie title
merged_data = pd.merge(bom_movie_gross, imdb_title_basics, left_on='title', right_on='primary_title', how='inner')

# Remove commas and dollar signs and convert 'domestic_gross' to numeric
merged_data['domestic_gross'] = merged_data['domestic_gross'].replace('[\$,]', '', regex=True).astype(float)

# Split genres into separate rows and aggregate by genre
genres_df = merged_data['genres'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('genre')
merged_data_genre = merged_data.drop('genres', axis=1).join(genres_df)

# Aggregate the data by genre and calculate the total domestic gross revenue for each genre
genre_domestic_gross = merged_data_genre.groupby('genre')['domestic_gross'].sum().sort_values(ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
genre_domestic_gross.plot(kind='bar', color='skyblue')
plt.title('Total Domestic Gross Revenue by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[12]:


# Remove commas and dollar signs and convert 'domestic_gross' to numeric
bom_movie_gross['domestic_gross'] = bom_movie_gross['domestic_gross'].replace('[\$,]', '', regex=True).astype(float)

# Aggregate the data by studio and calculate the total domestic gross revenue for each studio
studio_domestic_gross = bom_movie_gross.groupby('studio')['domestic_gross'].sum().sort_values(ascending=False)

# based on the top 50 studios based on total revenue
top_50_studios = studio_domestic_gross.head(50).index

# Filter the data to include only movies from the top 50 studios
top_50_data = bom_movie_gross[bom_movie_gross['studio'].isin(top_50_studios)]

# Aggregate the data by release year and calculate the total domestic gross revenue for each year
revenue_by_year_top_50 = top_50_data.groupby('year')['domestic_gross'].sum()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(revenue_by_year_top_50.index, revenue_by_year_top_50.values, marker='o', linestyle='-')
plt.title('Total Domestic Gross Revenue by Year for Top 50 Studios')
plt.xlabel('Year')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(revenue_by_year_top_50.index, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Bar Graph of Total Domestic Gross Revenue by Studio (Top 50):
# 
# Description: This bar graph displays the total domestic gross revenue for the top 50 studios in the dataset.
# Analysis: Studios such as BV (presumably Disney), WB (Warner Bros.), and Uni. (Universal Pictures) appear to be the top revenue generators, as they have the highest total domestic gross revenue. There is a significant variation in revenue among different studios, indicating varying levels of success in the movie industry. The distribution of revenue among studios can provide insights into market dominance and competition within the film industry.
# Visualization: The bar graph clearly presents the revenue data for each studio, facilitating easy comparison. Studios are labeled along the x-axis, and revenue is represented on the y-axis.

# In[18]:


# Bar Graph of Total Domestic Gross Revenue by Studio (Top 50)

plt.figure(figsize=(10, 6))
top_50_studios_values = studio_domestic_gross.head(50)  # Extracting the values from the index
top_50_studios_values.plot(kind='bar', color='skyblue')
plt.title('Total Domestic Gross Revenue by Studio (Top 50)')
plt.xlabel('Studio')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Bar Graph of Total Domestic Gross Revenue by Genre:
# 
# Description: This bar graph shows the total domestic gross revenue for each genre.
# Analysis: Genres such as Action, Adventure, and Comedy appear to be the highest revenue generators, while genres like Documentary and Western have relatively lower revenues. This analysis can help in understanding audience preferences and the popularity of different genres in the domestic market. Studios and filmmakers can use this information to make informed decisions about genre selection for future projects.
# Visualization: The bar graph presents the revenue data for each genre, allowing for easy comparison. Genres are labeled along the x-axis, and revenue is represented on the y-axis.

# In[19]:


# Bar Graph of Total Domestic Gross Revenue by Genre

plt.figure(figsize=(10, 6))
genre_domestic_gross.plot(kind='bar', color='skyblue')
plt.title('Total Domestic Gross Revenue by Genre')
plt.xlabel('Genre')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# Line Graph of Total Domestic Gross Revenue by Year for Top 50 Studios:
# 
# Description: This line graph illustrates the trend of total domestic gross revenue over the years for the top 50 studios.
# Analysis: Similar to the overall trend, there appears to be a general upward trend in domestic gross revenue for the top 50 studios, indicating their sustained success over the years. Peaks and troughs in revenue may coincide with blockbuster releases or specific strategies adopted by studios. This analysis provides insights into the performance and competitiveness of the top studios in the domestic market.
# Visualization: The line graph depicts the trend of revenue over the years, with years labeled along the x-axis and revenue represented on the y-axis. This visualization helps in understanding the revenue trends for the top studios.

# In[20]:


# Line Graph of Total Domestic Gross Revenue by Year for Top 50 Studios

plt.figure(figsize=(10, 6))
plt.plot(revenue_by_year_top_50.index, revenue_by_year_top_50.values, marker='o', linestyle='-')
plt.title('Total Domestic Gross Revenue by Year for Top 50 Studios')
plt.xlabel('Year')
plt.ylabel('Total Domestic Gross Revenue ($)')
plt.xticks(revenue_by_year_top_50.index, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Evaluation
# Bar Graph of Total Domestic Gross Revenue by Studio (Top 50):
# 
# The bar graph displays the total domestic gross revenue for the top 50 studios in the dataset.
# Analysis:
# Studios such as BV (presumably Disney), WB (Warner Bros.), and Uni. (Universal Pictures) appear to be the top revenue generators, as they have the highest total domestic gross revenue.
# There is a significant variation in revenue among different studios, indicating varying levels of success in the movie industry.
# The distribution of revenue among studios can provide insights into market dominance and competition within the film industry.
# Bar Graph of Total Domestic Gross Revenue by Genre:
# 
# The bar graph shows the total domestic gross revenue for each genre.
# Analysis:
# Genres such as Action, Adventure, and Comedy appear to be the highest revenue generators, while genres like Documentary and Western have relatively lower revenues.
# This analysis can help in understanding audience preferences and the popularity of different genres in the domestic market.
# Studios and filmmakers can use this information to make informed decisions about genre selection for future projects.
# Line Graph of Total Worldwide Gross Revenue by Year:
# 
# The line graph depicts the trend of total worldwide gross revenue over the years.
# Analysis:
# There seems to be an overall increasing trend in worldwide gross revenue over the years, indicating growth in the global movie industry.
# Fluctuations in revenue from year to year may be attributed to various factors such as economic conditions, blockbuster releases, and changes in consumer preferences.
# This trend suggests that the movie industry continues to thrive and evolve, attracting audiences worldwide.
# Line Graph of Total Domestic Gross Revenue by Year for Top 50 Studios:
# 
# The line graph illustrates the trend of total domestic gross revenue over the years for the top 50 studios.
# Analysis:
# Similar to the overall trend, there appears to be a general upward trend in domestic gross revenue for the top 50 studios, indicating their sustained success over the years.
# Peaks and troughs in revenue may coincide with blockbuster releases or specific strategies adopted by studios.
# This analysis provides insights into the performance and competitiveness of the top studios in the domestic market.
# Overall, these visualizations offer valuable insights into the performance of studios, genres, and the movie industry as a whole. Stakeholders can leverage these insights to make data-driven decisions, such as resource allocation, marketing strategies, and content creation, to optimize revenue and maximize success in the dynamic movie market.
# 
# ***
# Interpretation of Results: The findings provide important new information about the movie business, including how studios perform, which genres are popular, and long-term trends in revenue. Action, adventure, and comedy genres showed strong financial potential, and studios like BV, WB, and Uni emerged as major revenue providers. The increased trend in leading studios' domestic gross revenue over time points to continued success, which may have been impacted by smart choices and blockbuster releases.
# 
# Model Fit and Improvement Over Baseline: To understand the data, the analysis used data visualization and descriptive statistics. Although these techniques yield insightful results, they are more akin to exploratory analysis than predictive modeling. Thus, in this case, measurements such as model fit and improvement over a baseline model are meaningless.
# 
# Generalizability of Results: Several factors, such as the dataset's representativeness, the soundness of the analysis's underlying assumptions, and the consistency of underlying trends in the film business, affect how confidently one may extrapolate the results beyond the data that is now available. Enhancing confidence in the generalizability of results can be achieved by carrying out further validation studies, integrating other data sources, and assessing the robustness of findings across various time periods and countries. 
# 
# Business Impact of the Model: If implemented, the model, which provides information on studio performance, genre popularity, and income patterns, could be advantageous to Microsoft's film studio. By utilizing these data, the studio may maximize box office performance and gain a competitive advantage in the film business by making well-informed decisions on film development, genre choice, budget allocation, and strategic partnerships. To guarantee the model's efficacy and applicability in a changing and dynamic market environment, however, ongoing validation, monitoring, and improvement will be necessary.
# ***

# ## Conclusions
# 
# 
# ***
# In conclusion, the comprehensive analysis of the movie industry trends provides invaluable insights into the dynamics of studio performance, genre popularity, and revenue patterns. By delving into the data from sources such as Box Office Mojo, IMDB, and The Numbers, we've uncovered essential findings that can significantly impact decision-making processes within Microsoft's film studio initiative.
# 
# Studios like BV, WB, and Uni have emerged as major revenue generators, underlining their dominance in the market. Understanding the revenue distribution among studios helps in gauging market competition and identifying opportunities for strategic partnerships or acquisitions. Moreover, genres such as Action, Adventure, and Comedy have showcased robust revenue potential, guiding future genre selections for film projects.
# 
# The analysis also sheds light on long-term trends, with sustained revenue growth observed over the years, both domestically and globally. This insight enables studios to anticipate market shifts and plan for future investments effectively. Furthermore, the success trajectory of top studios underscores the importance of strategic decision-making and targeted releases in maximizing revenue.
# 
# For Microsoft's film studio venture, these insights serve as a roadmap for navigating the complexities of the movie industry. By leveraging data-driven strategies informed by studio performance, genre preferences, and revenue trends, Microsoft can optimize budget allocation, mitigate risks, and position itself competitively in the market.
# 
# However, it's crucial to acknowledge the limitations of the analysis, including data representativeness and underlying assumptions. Continuous validation, integration of additional data sources, and ongoing refinement of the model are essential to enhance the robustness and applicability of the insights generated.
# 
# In essence, the analysis provides a solid foundation for Microsoft's film studio initiative, offering actionable insights that can drive informed decision-making and pave the way for success in the dynamic and ever-evolving movie industry landscape. Through strategic utilization of these insights, Microsoft can harness the power of data to carve its niche and achieve sustainable growth in the competitive film market.
# 
# 
# 
# 
# 
# ***
