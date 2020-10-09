# NLP projects
This repository contains projects on __Parsing Analysis__, __Bag-of-Words__ language model, and __term frequency-inverse document frequency (tf-idf)__ using Python's `nltk` 
and `scikit-learn` libraries.

## Classic Text Project
Novels and text contain insights into ideologies and places that are often originally unknown to the reader. By reading a written piece, you uncover the opinions of the author 
on their chosen topic and come to understand both the topic and how the author thinks.

In this project we will perform a _Natural Language Parsing Analysis_ to gain deeper insight into one of the often most discussed novels in the public domain: Oscar Wilde’s 
The Picture of Dorian Gray. One of the beauties of natural language parsing with _regular expressions_ is the ability to gain insight into lengthy pieces of text without 
a formal read!

By the end of this project, we will find out the main topics of discussion in the novel of choosing and can begin to discern some of the author’s thoughts and beliefs.

## Mystery Friend
We’ve received an anonymous postcard from a friend who we haven’t seen in years. This friend did not leave a name, and so far, we’ve narrowed our search down to three friends, 
based on handwriting:

* Emma Goldman
* Matthew Henson
* TingFang Wu

But which one sent the card?

Just like we can classify a message as spam or not spam with a spam filter, we can classify writing as related to one friend or another by building a kind of friend writing 
classifier. In this project, we'll use `scikit-learn`’s _Bag-of-Words_ and _Naive Bayes classifier_ to determine who the mystery friend is!


## News Analysis
Newspapers and their online formats supply the public with the information we need to understand the events occurring in the world around us. Given the vast amount of news 
articles in circulation, identifying and organizing articles by topic is a useful activity. This can help us sift through the enormous amount of information out there so we 
can find the news relevant to our interests, or even allow us to build a news recommendation engine!

[The News International](https://www.thenews.com.pk/) is the largest English language newspaper in Pakistan, covering local and international news across a variety of sectors. 
A selection of articles from a [Kaggle Dataset of The News International articles](https://www.kaggle.com/asad1m9a9h6mood/news-articles) is used in this projcet.

In this project we use _term frequency-inverse document frequency (tf-idf)_ to analyze each article’s content and uncover the terms that best describe each article, providing 
quick insight into each article’s topic.

## U.S.A. Presidential Vocabulary
Whenever a United States of America president is elected or re-elected, an inauguration ceremony takes place to mark the beginning of the president’s term. During the ceremony, 
the president gives an inaugural address to the nation, dictating the tone and focus of the next four years of leadership.

In this project we analyze the inaugural addresses of the presidents of the United States of America, as collected by 
[the Natural Language Toolkit](https://www.nltk.org/book/ch02.html), using __word embeddings__.

By training sets of word embeddings on subsets of inaugural address versus the collection of presidents as a whole, we can learn about the different ways in which the presidents 
use language to convey their agenda.
