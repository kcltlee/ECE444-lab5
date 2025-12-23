# ECE444 lab 5 - Deploying an ML APP to the Cloud
The objective of this assignment is for students to get familiar with deploying machine learning (ML)
models to the cloud using a cloud provider. We will focus on deploying a model using AWS Elastic
Beanstalk, making the model accessible as a service using a REST API call.

Fake news has recently become a topic of concern as more of the information we receive about the world
is delivered through the web. You, as a developer on the Fake News detection team at Not Fake News Co.
have been tasked with building a barebones REST API that takes in a snippet of text (from a news article
or source) and determines if it is considered fake news or not (by returning a 1 if it is Fake News, 0
otherwise).

<img width="705" height="381" alt="image" src="https://github.com/user-attachments/assets/a0297e4a-db48-405d-a203-75b436a34c6d" />

## Boxplot to visualize the performance results for each test case:

<img width="449" height="335" alt="image" src="https://github.com/user-attachments/assets/4d71cc06-4951-4b5a-8e3a-8abac56b86cf" />

Average Latency per Test Case (seconds):
| Test Case | Average Latency (s) |
|------------|---------------------|
| Fake_1     | 0.056498            |
| Fake_2     | 0.051598            |
| Real_1     | 0.052962            |
| Real_2     | 0.052982            |

