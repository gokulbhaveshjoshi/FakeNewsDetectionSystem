# Fake_News_Detection_System
 Fake news detection system is based on Machine learning.
<img src="GIF/gif.gif">

#Introduction
Fake news detection system is an web application which tells you about news, news is fake or not.

#Abstract 
Fake news Detction system, It will be trained by a bidirectional neural network and LSTM based deep learning model to detected fake news from given corpus. This project could be pratically used by predict whether the circulating news is fake or not.

#File structure
<img src="images/file-structure.png">

# Datasets
Dowloads datasets [link](https://ravi-finds-fake-news.herokuapp.com) 6336 * 4
* fake_or_real_news.csv: A datasets with the following attributes
  1.Id: unique id for a news article.
  2.Title: The title of a news article.
  3. Text: Text of the news article.
  4. Label: REAL or FAKE.
  size of datasets id 6336*4


# Program Structure
* fake_news_detection.py - This contains code fot our Machine Learning model to classify the model 
* app.py - This contains Flask APIs that receives news url through GUI or API calls, extracts the article from the url, feeds it to the model and returns the prediction.
* templates - This folder contains the HTML template to allow user to enter url and displays whether the news is fake or real.
* static - This folder contains the CSS file.
* libraries.txt - It contains the list of libraries required to run the heroku app




# Graphs and table
* Confussion Matrix
<img src="images/confussion_matrix.png">

* Heat Map
<img src="images/Heat_map.png">

* Reports
<img src="images/Report.png">


# Run project
* first go to project home directory
* Create machine learning model by running command -
python fake_news_detection.py
* Automatically create a serialized version of our model into one file model.pickle
* Run app.py on command prompt-
python app.py
By default, flask will run on port 5000.

Navigate to URL http://127.0.0.1:5000 
