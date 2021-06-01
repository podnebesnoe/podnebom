import pandas as pd

# http://bharatreddy.github.io/2015/01/15/sql-pandas1/

# Movie table: mID | title | year | director
# Rating table: rID | mID | stars | ratingDate
# Reviewer table: rID | name


df_movie = pd.DataFrame()
df_rating = pd.DataFrame()
df_reviewer = pd.DataFrame()

### part 1

# Find the titles of all movies directed by Steven Spielberg

'''
SELECT title FROM movie WHERE director == "STEVEN"
'''

df_movie[df_movie['director'] == 'STEVEN']['title']

# Find all years that have a movie that received a rating of 4 or 5, and sort them in increasing order

'''
SELECT DISTINCT movie.year FROM movie INNER JOIN rating ON movie.mID = rating.mID WHERE rating.stars >= 4 and rating.stars <= 5 ORDER BY rating ASC
'''

df_rating_2 = df_rating[df_rating['stars'] >= 4]
res = pd.merge(df_rating_2, df_movie, on='mID', how='inner')
res = res.sort_values(['year'], ascending=True)['year'].unique()

# Find the titles of all movies that have no ratings

'''
SELECT movie.title FROM movie LEFT JOIN rating ON movie.mID = raging.mID WHERE rating.mID IS NULL 
'''

res = pd.merge(df_rating, df_movie, on='mID', how='left')
res = res[res['stars'].isnull()]['title']

# Some reviewers didn't provide a date with their rating. Find the names of all reviewers who have ratings with a NULL value for the date

'''
SELECT DISTINCT reviewer.name FROM reviwer INNER JOIN rating ON reviewer.rID = rating.rID WHERE reviewer.ratingDate IS NULL
'''

res = pd.merge(df_rating, df_reviewer, on='rID', how='inner')
res[res['rattingDate'].isnull()]['name']

# Write a query to return the ratings data in a more readable format: reviewer name, movie title, stars, and ratingDate.
# Also, sort the data, first by reviewer name, then by movie title, and lastly by number of stars

'''
SELECT reviwer.name, movie,title, rating,ratingDate from movie INNER JOIN rating ON movie.mID = rating.mID INNER JOIN rating.rID = reviwer.rID
ORDER BY reviewer.name, movie.TITLE, rating.stars
'''

res = pd.merge(df_movie, df_rating, on='mID', how='inner')
res = pd.merge(res, df_reviewer, on='rID', how='inner')
res = res.sort_values(['name','title','stars'])['name', 'title', 'stars', 'ratingDate']

# For all cases where the same reviewer rated the same movie twice and gave it a higher rating the second time, return the reviewer's name and the title of the movie

'''
SELECT reviewer.name, movie.title FROM reviewer INNER JOIN rating ra1 ON ra1,ID = reviewer.ID INNER JOIN movie ON movie.ID = ra1.mID
INNER JOIN rating ra2 ON ra1.rID = ra2.rID AND ra1.mID = ra2.mID
WHERE ra2.stars > ra1.stars AND ra2.ratingDate > ra1.ratingDate
'''

res = pd.merge(df_rating, df_rating, on=['rID','mID'], how='inner')
res = res[(res['stars_y'] > res['stars_x']) & (res['ratingDate_y'] > res['ratingDate_x'])]
res = pd.merge(res, df_movie, on=['mID'], how='inner')
res = pd.merge(res, df_reviewer, on=['rID'], how='inner')
res = res[['name', 'title']]

# For each movie that has at least one rating, find the highest number of stars that movie received. Return the movie title and number of stars. Sort by movie title

'''
SELECT movie.title, tab.max_stars FROM movie INNER JOIN
(SELECT mID, MAX(stars) max_stars FROM rating GROUP BY mID HAVING COUNT(mID) > 1) tab
ON movie.mID = tab.mID ORDER BY movie.title
'''

# pandas - too long

# For each movie, return the title and the 'rating spread', that is, the difference between highest and lowest ratings given to that movie.
# Sort by rating spread from highest to lowest, then by movie title

'''
SELECT movie.title, tab.spread FROM movie INNER JOIN
(SELECT mID, MIN(stars) - MAX(stars) AS spread FROM rating GROUP BY mID) tab
ON movie.mID = tab.mID
ORDER BY tab.spread DESC, movie.title
'''

# pandas - too long

# Find the difference between the average rating of movies released before 1980 and the average rating of movies released after 1980

# SQL 0 too long

res = pd.merge(df_movie, df_rating, on='mID', how='inner')
before = res[res['ratingDate'] < 1980]
after = res[res['ratingDate'] >= 1980]
before = before.groupby(['mID'])['stars'].mean()
after = after.groupby(['mID'])['stars'].mean()
after.mean() - before.mean()

### part 2

# Find the names of all reviewers who rated Gone with the Wind

'''
SELECT DISTINCT reviewer.name from review INNER JOIN rating ON reviewer.rID = rating.rID INNER JOIN movie ON rating.mID = movie.mID
WHERE movie.title = 'Wind'
'''

res = pd.merge(df_reviewer, df_rating, on='rID', how = 'inner')
res = pd.merge(res, df_movie, on='mID', how = 'inner')
res[res['title'] == 'Wind']['name'].unique()


# For any rating where the reviewer is the same as the director of the movie, return the reviewer name, movie title, and number of stars

'''
SELECT reviewer.name, movie.title, rating.stars FROM movie INNER JOIN rating ON movie.mid = rating.mID 
INNER JOIN reviwer ON rating.rID = reviewer.rID WHERE movie.director = reviewer.name
'''

res = pd.merge(df_reviewer, df_rating, on='rID', how = 'inner')
res = pd.merge(res, df_movie, on='mID', how = 'inner')
res[res['director'] == res['name']][['name', 'title', 'stars']]

# Return all reviewer names and movie names together in a single list, alphabetized

'''
SELECT name AS list FROM reviewer
UNION
SELECT title AS list from movie ORDER BY list
'''

pd.concat(df_movie['title'], df_reviewer['name']).sort_values()

# you can use:
# df.rename(columns={"A": "a"})

# Find the titles of all movies not reviewed by Chris Jackson

'''
SELECT title FROM movie WHERE title NOT IN
(SELECT title FROM movie  INNER JOIN rating ON movie.mID = rating.mID INNER JOIN reviewer ON reviewer.rID = movie.rID
WHERE reviewer.name = 'Chris')
'''

res = pd.merge(df_rating, df_reviewer, on='rID', how='inner')
res = res[res['name'] == 'Chris']
res = pd.merge(df_movie, res, on='mID', how='left')
res[res['rID'].isnull()]['title']

