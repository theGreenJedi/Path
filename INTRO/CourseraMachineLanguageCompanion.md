
Coursera: Machine Learning 
Read View source View history 
Created by: Colorado Reed 
Intended for: Coursera Machine Learning Students 
Metacademy Primer
Metacademy is an open source platform designed to help you efficiently learn about any topic that you're interested in---it currently specializes in machine learning and artificial intelligence topics. The idea is that you click on a concept that interests you, and Metacademy produces a "learning plan" that will help you learn the concept and all of its prerequisite concepts that you don't already know.
Metacademy's learning experience revolves around two central components:
a "learning plan", e.g here's one for logistic regression 
and a "graph view" to help you explore relationships among concepts, e.g. here's the graph for logistic regression 
You can tell Metacademy that you understand a [prerequisite] concept by clicking the checkmark next to the concept's title in the graph or list view. Furthermore, you can then click the "hide" button in the upper right to hide the concepts you understand (Metacademy remembers the concepts you've learned, so it'll automatically apply these in the future).
Coursera Roadmap
This roadmap is a supplement to Andrew Ng's Coursera machine learning course. You should use this roadmap to review the essential concepts presented during each section, find detailed resources for each of the discussed concepts, and also to brush up on necessary prerequisite concepts for the covered material (especially if you'd like to learn the concepts in greater detail). This roadmap is a work in progress, but you may still find it useful in its current incarnation. If you have any comments or suggestions, you can contact me at _olorado@meta_ademy.org (replace _ with a c).
PS) After completing this course, I highly recommend furthering your machine learning knowledge via Roger Grosse's excellent Bayesian Machine Learning Roadmap.
Section I (Introduction)
No specific concepts for this section, which provides a general motivation for machine learning and quickly covers higher-level concepts such as supervised learning and unsupervised learning. 
If you're new to machine learning, take a moment to read Pedro Domingos's practical machine learning overview paper, and make sure to reflect on this paper as you progress in your machine learning endeavors. 
Section II (Linear Regression with One Variable )
linear regression, pay attention to the single variate case 
loss function also known as a "cost function" 
gradient descent 
additional concepts mentioned but not discussed in detail include:
convex functions 
convex optimization 
Section III (Linear Algebra Review)
matrix inverse and its prerequisites will bring you up to speed on the linear algebra concepts necessary for most of this course 
Section IV (Linear Regression with Multiple Variables)
linear regression (covered in section II), pay attention to the multivariate case 
basis function expansions, which ties in with notion of "feature selection" as discussed in the lecture (a particular choice of basis functions for linear regression yields "polynomial regression," as discussed in the lectures) 
linear regression closed form solution: is typically used in lieu of gradient descent optimization for smaller datasets (N < 10000) -- this solution yields the so-called "normal equations" 
Section VI (Logistic Regression)
logistic regression note the content on regularized logistic regression (this topic is covered in the next section) 
Section VII (Regularization)
generalization also known as avoiding over/under-fitting. 
ridge regression also known as regularized linear regression 
Section VIII (Neural Networks: Representation)
This section provides a higher level overview/justification of neural-networks and multiclass classification. 
If you'd like to dive deeper into neural networks after completing Section VIII and IX, consider enrolling in Geoffry Hinton's Neural Networks for Machine Learning Coursera course. 
Section IX (Neural Networks: Learning)
feed forward neural networks also simply referred to as a neural network. 
backpropagation 
The following concepts may also be useful (not explicitly discussed in the class lectures):
weight decay in neural networks a technique for preventing extremely unbalanced weights in the neural network 
early stopping is a technique for improving the performance of neural networks by stopping the training once the performance on a held out portion of the dataset has stopped (before the weights converge) 
learning invariances in neural networks can help improve neural network performance 
tikhonov regularization is a way of learning invariances in neural networks 
Section X (Advice for Applying Machine Learning)
bias variance tradeoff 
model selection 
cross validation 
Section XI (Machine Learning System Design)
Common metrics used to evaluate machine learning systems include:
precision and recall 
F measure 
Section XII (Support Vector Machines)
support vector machine 
svm vs logistic_regression 
kernel trick (also see constructing kernels if you're curious how you can construct kernels -- not in the course) 
kernel svm 
SVMs have a number of extensions (not covered in this course):
soft margin svm 
multiclass svm 
support vector regression 
Section XII [Clustering]
k-means 
Also see k means++ for a simple k-means initialization routine that yields an optimality guarantee on the final result 
Section XIV (Dimensionality Reduction)
principal component analysis 
pca preprocessing 
PCA has a number of extensions/generalizations not mentioned in this course
kernel pca 
probabilistic pca 
bayesian pca 
Section XV (Anomaly Detection)
gaussian distribution 
multivariate gaussian distribution 
Section XVI (Recommender Systems)
matrix factorization 
optional: also take a look at factor analysis for a probabilistic perspective on matrix factorization 
Section XVII (Large Scale Machine Learning)
stochastic gradient descent 
Section XVIII (Application Example: Photo OCR)
sorry, nothing for this section 
What next?
Consider taking one of the following Coursera courses to further your machine learning knowledge:
Daphne Koller's Probabilistic Graphical Models Course -- it's quite a bit harder than Andrew's machine learning course, but if you're serious about doing machine learning (e.g. research or professionally) then you will need to learn this content at some point (you'll save yourself a lot of pain if you start on it early). 
Geoffry Hinton's Neural Networks for Machine Learning Coursera course 
Pedro Domingos's Machine Learning Coursera course (this is more oriented toward applied machine learning & data mining). 
EdX offer's a fantastic artificial intelligence course that will give you a broader view of AI. 
Udacity has a course on robotics that will show you how to use some of your new machine learning knowledge to program an autonomous vehicle. 
Consider working through the material in Roger Grosse's excellent Bayesian Machine Learning Roadmap. Do you understand Markov Chain Monte Carlo inference, mixture of Gaussians model, hidden Markov models, or the junction tree algorithm? These concepts, and many more in his roadmap, should be in the toolbelt of any machine learner worth his/her salt. 

