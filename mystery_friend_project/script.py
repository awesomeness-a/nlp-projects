from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# setting up the combined list of friends' writing samples
friends_docs = goldman_docs + henson_docs + wu_docs
# setting up labels for your three friends
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

# print out a document from each friend:
print(goldman_docs[56])
print(henson_docs[34])
print(wu_docs[99])

#mystery_postcard = """
#My friend,
#From the 10th of July to the 13th, a #fierce storm raged, clouds of
#freeing spray broke over the ship, #incasing her in a coat of icy mail,
#and the tempest forced all of the ice out of the lower end of the
#channel and beyond as far as the eye could see, but the _Roosevelt_
#still remained surrounded by ice.
#Hope to see you soon.
#"""

mystery_postcard = """
Free love? As if love is anything but free! Man has bought brains, but all 
the millions in the world have failed to buy love. Man has subdued bodies, 
but all the power on earth has been unable to subdue love. Man has conquered 
whole nations, but all his armies could not conquer love. Man has chained and 
fettered the spirit, but he has been utterly helpless before love. High on a 
throne, with all the splendor and pomp his gold can command, man is yet poor 
and desolate, if love passes him by. And if it stays, the poorest hovel is 
radiant with warmth, with life and color. Thus love has the magic power to make 
of a beggar a king. Yes, love is free; it can dwell in no other atmosphere. In 
freedom it gives itself unreservedly, abundantly, completely. All the laws on 
the statutes, all the courts in the universe, cannot tear it from the soil, once 
love has taken root. If, however, the soil is sterile, how can marriage make it 
bear fruit? It is like the last desperate struggle of fleeting life against death.
"""

# create bow_vectorizer:
bow_vectorizer = CountVectorizer()

# define friends_vectors:
friends_vectors = bow_vectorizer.fit_transform(friends_docs)

# define mystery_vector: 
mystery_vector = bow_vectorizer.transform([mystery_postcard])

# define friends_classifier:
friends_classifier = MultinomialNB()

# train the classifier:
friends_classifier.fit(friends_vectors, friends_labels)

# change predictions:
#predictions = ["None Yet"]
#predictions = friends_classifier.predict(mystery_vector)
predictions = friends_classifier.predict_proba(mystery_vector)

#mystery_friend = predictions[0] if predictions[0] else "someone else"

#print("The postcard was from {}!".format(mystery_friend))
print(predictions)