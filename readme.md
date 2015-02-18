## Spam Filter API

### Concept
This is a simple MVP which exposes only a few HTTP endpoints to allow real time  classification of emails as either `spam` or `ham` (not spam). As an added feature, the ability to help the algorithm learn further is provided.

### Usage

There are three endpoints to the application:
```
POST: '/classify/'
POST: '/trainham/'
POST: '/trainspam/'
```
Requests to any of the above endpoints should contain file attachments representing emails. It is intended that each of the attachments to `classify` requests will be named with a useful identifier, so that the response

```json
{"file1": "ham"|"spam", "file2": "ham"| "spam", ... }
```
will be of greatest utility.

Files sent to `/trainham/` or `/trainspam/` endpoints should be *known* to consist solely of either ham or spam mail. This design choice is discussed more below. Correct use of these endpoints will help the algorithm continue to develop and become more accurate.
### Design

This spam filter is built on Django. Message classification is performed by a [naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier).  It is backed by a Distribution model treated as a singleton which holds the algorithm's learned probabilities. The choice was made to hold the probabilities in a model and store them in the app's SQLite3 database both to increase responsiveness after the server is downed (Heroku sleeps apps and shuts down dynos daily) and to allow continued learning.

####Decisions

I mainly chose to write this application in Python for its low ramp-up time and my familiarity in performing math operations with it.

With my experience mainly in Java and Python, if I were not time-constrained by the challenge and believed in the possibility of extensions to the application, Java likely would have been my language of choice. This is due to my greater confidence in building a robust system in Java and suspicion that the relatively time-consuming algorithm used here might benefit from Java's agility.

The decision to make the app a real-time classifier was quick for me, as I figured a lot of the utility of a spam classifier was to efficiently filter good emails *to* you and spam away from you as quickly as possible. Hence a real-time classifier modeled most closely something I feel I would wish to make on my own time.

The concept of a batch classifier is interesting. The potential gain I see from that design would be an increase in efficiency due to focussed processing. By the nature of my naive Bayes algorithm, however, I saw limited potential gain from batch processing.

####Scaling

By the time-intensive nature of the algorithm used, scaling would warrant many more processors/multiple threads. Random routing of users would probably be a sufficient way to distribute, although it would be worth researching if certain types of spam are more likely in certain locales. If so, location-based routing, with different Distribution models for each server location, could provide more than a latency decrease.


The biggest challenge in scaling this app would be to decide what should be done with the Distribution model when it comes to continued learning. Although hits to `/trainham/` or `/trainspam/` endpoints are processed quickly, concurrency issues could arise. This could be solved by carefully synchronising access to the app's databse. A more interesting solution to me, however, would be to have each instance of the app hold its own Distribution object, and sync up all the distributions perhaps nightly. This would avoid concurrency issues, and provide the added benefit of decentralisation. Nightly syncs would ensure that no one server fatality would result in significant information loss - the other app instances could continue processing with essentially similar up-to-date distributions. The location-specific idea above would merely introduce another abstraction layer to this model.

### Algorithm

The naive Bayes classifier implemented here works by learning the probabilities of certain words occuring given that the email is either spam or ham. Then, when asked to classify a new message, the algorithm will calculate the probability of that email's composition given either that it's spam or ham. The classification corresponding to the most probable scenario is the classification given. This is known as the [MAP decision rule](http://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation).


The algorithm is "naive" because it treats all word occurences as independent of one another. Although this is an imperfection, complexity greatly increases once one attempts to incorporate word relationships.

#### Improvements

On its first implementation, my classifier performed rather poorly on a testing set, classifying perfectly ham messages, but actually missing more spam than random. Inspection revealed that the classifier was consistently over-aggressive in classifying ham, and that spam messages were only *just* classified as ham. Thus a linear offset (~100 between log-probabilities works well) did a solid job of increasing spam classification with relating little impact on hamming rates.

I suspected the issue was due to an insufficient spam training set/high variability of spam mail.

The emails used to train and test this algorithm contained headers and often the majority of the text in a file would come from them. I'm skeptical of my algorithm's ability to find anything helpful in those headers, especially as the training and test data used does not correspond to a specific user (and thus spam is likely to come from a wide variety of sources). It might therefore be beneficial to ignore those headers during training and classification to save processing time and unnecessary noise.

As another suggestion, it is entirely possible for motivated spammers to minimally mutate words that correspond commonly with spam, slipping past this classifier as a neutral word. Fuzzy matching could help eliminate this issue, at the cost of some more processing to pick the strongest from a set of matches.

#### Performance

In a run on one half of testing, the app correctly classified 90/100 spam and 100/100 ham. In the remaining half of testing, the app classified 76/100 spam and 85/100 ham. So overall we saw false positive rates of 7.5%, and failed to classify about 17% of spam mail.

### Improvements

Improvements on the algorithm, improvements to design.
