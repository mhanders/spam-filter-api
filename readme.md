## Spam Filter API

### Concept
This is a simple application which exposes only a few HTTP endpoints to allow classification of emails as either `spam` or `ham` (not spam), and the ability to help the algorithm learn further.

### Usage

There are three endpoints to application:
```
POST: '/classify/'
POST: '/trainham/'
POST: '/trainspam/'
```
Requests to any of the above endpoints should contain file attachments. It is intended that each attachments to `classify` requests will be named with a useful identifier, so that the response

```json
{"file1": "ham"|"spam", "file2": "ham"| "spam, ... "}
```
will be of greatest utility.

Files sent to `/trainham/` or `/trainspam/` endpoints should be *known* to consist solely of either ham or spam mail. This design choice is discussed more below. Correct use of these endpoints will help the algorithm continue to develop and become more accurate.
### Design

This spam filter is built on Django. Message classification is performed by a [naive Bayes classifier](http://en.wikipedia.org/wiki/Naive_Bayes_classifier).  It is backed by a Distribution model treated as a singleton which holds the algorithm's learned probabilities. The choice was made to hold the probabilities in a model and store them in the app's SQLite3 database both to increase responsiveness after the server is downed (Heroku sleeps apps and shuts down dynos daily) and to allow continued learning. The continued learning is facilitated through exposure of two more endpoints, in addition to the

####Decisions

I mainly chose to write this application in Python for its low ramp-up time and my familiarity in performing math operations with it.

With my experience mainly in Java and Python, if I were not time-constrained by the challenge and believed in the possibility of extensions to the application, Java would have been my language of choice. This is due to my greater confidence in building a robust system in Java and suspicion that the relatively time-consuming algorithm used here might benefit from Java's agility.

#### Algorithm

### Improvements
