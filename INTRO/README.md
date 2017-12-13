A Few Useful Things to Know about Machine Learning
Pedro Domingos
Department of Computer Science and Engineering
University of Washington
Seattle, WA 98195-2350, U.S.A.
pedrod@cs.washington.edu
ABSTRACT

Machine learning algorithms can figure out how to perform
important tasks by generalizing from examples. This is of-
ten feasible and cost-effective where manual programming
is not. As more data becomes available, more ambitious
problems can be tackled. As a result, machine learning is
widely used in computer science and other fields. However,
developing successful machine learning applications requires
a substantial amount of “black art” that is hard to find in
textbooks. This article summarizes twelve key lessons that
machine learning researchers and practitioners have learned.
These include pitfalls to avoid, important issues to focus on,
and answers to common questions.
1. INTRODUCTION

Machine learning systems automatically learn programs from
data. This is often a very attractive alternative to manually
constructing them, and in the last decade the use of machine
learning has spread rapidly throughout computer science
and beyond. Machine learning is used in Web search, spam
filters, recommender systems, ad placement, credit scoring,
fraud detection, stock trading, drug design, and many other
applications. A recent report from the McKinsey Global In-
stitute asserts that machine learning (a.k.a. data mining or
predictive analytics) will be the driver of the next big wave of
innovation [16]. Several fine textbooks are available to inter-
ested practitioners and researchers (e.g, [17, 25]). However,
much of the “folk knowledge” that is needed to successfully
develop machine learning applications is not readily avail-
able in them. As a result, many machine learning projects
take much longer than necessary or wind up producing less-
than-ideal results. Yet much of this folk knowledge is fairly
easy to communicate. This is the purpose of this article.

Many different types of machine learning exist, but for il-
lustration purposes I will focus on the most mature and
widely used one: classification. Nevertheless, the issues I
will discuss apply across all of machine learning. Aclassi-
fieris a system that inputs (typically) a vector of discrete
and/or continuousfeature valuesand outputs a single dis-
crete value, theclass. For example, a spam filter classifies
email messages into“spam”or“not spam,”and its input may
be a Boolean vectorx= (x 1 ,... , xj,... , xd), wherexj= 1
if thejth word in the dictionary appears in the email and
xj= 0 otherwise. Alearnerinputs atraining setofexam-
ples(xi, yi), wherexi= (xi, 1 ,... , xi,d) is an observed input
andyiis the corresponding output, and outputs a classifier.
The test of the learner is whether this classifier produces the

correct outputytfor future examplesxt(e.g., whether the
spam filter correctly classifies previously unseen emails as
spam or not spam).

2. LEARNING = REPRESENTATION +
EVALUATION + OPTIMIZATION

Suppose you have an application that you think machine
learning might be good for. The first problem facing you
is the bewildering variety of learning algorithms available.
Which one to use? There are literally thousands available,
and hundreds more are published each year. The key to not
getting lost in this huge space is to realize that it consists
of combinations of just three components. The components
are:

Representation.A classifier must be represented in some
formal language that the computer can handle. Con-
versely, choosing a representation for a learner is tan-
tamount to choosing the set of classifiers that it can
possibly learn. This set is called thehypothesis space
of the learner. If a classifier is not in the hypothesis
space, it cannot be learned. A related question, which
we will address in a later section, is how to represent
the input, i.e., what features to use.
Evaluation.An evaluation function (also calledobjective
functionorscoring function)is needed to distinguish
good classifiers from bad ones. The evaluation function
used internally by the algorithm may differ from the
external one that we want the classifier to optimize, for
ease of optimization (see below) and due to the issues
discussed in the next section.
Optimization.Finally, we need a method to search among
the classifiers in the language for the highest-scoring
one. The choice of optimization technique is key to the
efficiency of the learner, and also helps determine the
classifier produced if the evaluation function has more
than one optimum. It is common for new learners to
start out using off-the-shelf optimizers, which are later
replaced by custom-designed ones.

Table 1 shows common examples of each of these three com-
ponents. For example,k-nearest neighbor classifies a test
example by finding thekmost similar training examples
and predicting the majority class among them. Hyperplane-
based methods form a linear combination of the features per
class and predict the class with the highest-valued combina-
tion. Decision trees test one feature at each internal node,

Table 1: The three components of learning algorithms.

Representation Evaluation Optimization
Instances Accuracy/Error rate Combinatorial optimization
K-nearest neighbor Precision and recall Greedy search
Support vector machines Squared error Beam search
Hyperplanes Likelihood Branch-and-bound
Naive Bayes Posterior probability Continuous optimization
Logistic regression Information gain Unconstrained
Decision trees K-L divergence Gradient descent
Sets of rules Cost/Utility Conjugate gradient
Propositional rules Margin Quasi-Newton methods
Logic programs Constrained
Neural networks Linear programming
Graphical models Quadratic programming
Bayesian networks
Conditional random fields

with one branch for each feature value, and have class predic-
tions at the leaves. Algorithm 1 shows a bare-bones decision
tree learner for Boolean domains, using information gain and
greedy search [21]. InfoGain(xj,y) is the mutual information
between featurexjand the classy. MakeNode(x,c 0 ,c 1 ) re-
turns a node that tests featurexand hasc 0 as the child for
x= 0 andc 1 as the child forx= 1.

Of course, not all combinations of one component from each
column of Table 1 make equal sense. For example, dis-
crete representations naturally go with combinatorial op-
timization, and continuous ones with continuous optimiza-
tion. Nevertheless, many learners have both discrete and
continuous components, and in fact the day may not be
far when every single possible combination has appeared in
some learner!

Most textbooks are organized by representation, and it’s
easy to overlook the fact that the other components are
equally important. There is no simple recipe for choosing
each component, but the next sections touch on some of the
key issues. And, as we will see below, some choices in a
machine learning project may be even more important than
the choice of learner.
3. IT’S GENERALIZATION THAT COUNTS

The fundamental goal of machine learning is togeneralize
beyond the examples in the training set. This is because,
no matter how much data we have, it is very unlikely that
we will see those exact examples again at test time. (No-
tice that, if there are 100,000 words in the dictionary, the
spam filter described above has 2^100 ,^000 possible different in-
puts.) Doing well on the training set is easy (just memorize
the examples). The most common mistake among machine
learning beginners is to test on the training data and have
the illusion of success. If the chosen classifier is then tested
on new data, it is often no better than random guessing. So,
if you hire someone to build a classifier, be sure to keep some
of the data to yourself and test the classifier they give you
on it. Conversely, if you’ve been hired to build a classifier,
set some of the data aside from the beginning, and only use
it to test your chosen classifier at the very end, followed by
learning your final classifier on the whole data.

Algorithm 1 LearnDT(TrainSet)
ifall examples inTrainSethave the same classy∗then
returnMakeLeaf(y∗)
ifno featurexjhas InfoGain(xj,y)> 0 then
y∗←Most frequent class inTrainSet
returnMakeLeaf(y∗)
x∗←argmaxxjInfoGain(xj,y)
TS 0 ←Examples inTrainSetwithx∗= 0
TS 1 ←Examples inTrainSetwithx∗= 1
returnMakeNode(x∗, LearnDT(TS 0 ), LearnDT(TS 1 ))

Contamination of your classifier by test data can occur in
insidious ways, e.g., if you use test data to tune parameters
and do a lot of tuning. (Machine learning algorithms have
lots of knobs, and success often comes from twiddling them
a lot, so this is a real concern.) Of course, holding out
data reduces the amount available for training. This can
be mitigated by doing cross-validation: randomly dividing
your training data into (say) ten subsets, holding out each
one while training on the rest, testing each learned classifier
on the examples it did not see, and averaging the results to
see how well the particular parameter setting does.

In the early days of machine learning, the need to keep train-
ing and test data separate was not widely appreciated. This
was partly because, if the learner has a very limited repre-
sentation (e.g., hyperplanes), the difference between train-
ing and test error may not be large. But with very flexible
classifiers (e.g., decision trees), or even with linear classifiers
with a lot of features, strict separation is mandatory.

Notice that generalization being the goal has an interesting
consequence for machine learning. Unlike in most other op-
timization problems, we don’t have access to the function
we want to optimize! We have to use training error as a sur-
rogate for test error, and this is fraught with danger. How
to deal with it is addressed in some of the next sections. On
the positive side, since the objective function is only a proxy
for the true goal, we may not need to fully optimize it; in
fact, a local optimum returned by simple greedy search may
be better than the global optimum.

4. DATA ALONE IS NOT ENOUGH

Generalization being the goal has another major consequence:
data alone is not enough, no matter how much of it you have.
Consider learning a Boolean function of (say) 100 variables
from a million examples. There are 2^100 − 106 examples
whose classes you don’t know. How do you figure out what
those classes are? In the absence of further information,
there is just no way to do this that beats flipping a coin. This
observation was first made (in somewhat different form) by
the philosopher David Hume over 200 years ago, but even
today many mistakes in machine learning stem from failing
to appreciate it. Every learner must embody some knowl-
edge or assumptions beyond the data it’s given in order to
generalize beyond it. This was formalized by Wolpert in
his famous “no free lunch” theorems, according to which no
learner can beat random guessing over all possible functions
to be learned [26].

This seems like rather depressing news. How then can we
ever hope to learn anything? Luckily, the functions we want
to learn in the real world arenotdrawn uniformly from
the set of all mathematically possible functions! In fact,
very general assumptions—like smoothness, similar exam-
ples having similar classes, limited dependences, or limited
complexity—are often enough to do very well, and this is a
large part of why machine learning has been so successful.
Like deduction, induction (what learners do) is a knowledge
lever: it turns a small amount of input knowledge into a
large amount of output knowledge. Induction is a vastly
more powerful lever than deduction, requiring much less in-
put knowledge to produce useful results, but it still needs
more than zero input knowledge to work. And, as with any
lever, the more we put in, the more we can get out.

A corollary of this is that one of the key criteria for choos-
ing a representation is which kinds of knowledge are easily
expressed in it. For example, if we have a lot of knowledge
about what makes examples similar in our domain, instance-
based methods may be a good choice. If we have knowl-
edge about probabilistic dependencies, graphical models are
a good fit. And if we have knowledge about what kinds of
preconditions are required by each class, “IF.. .THEN.. .”
rules may be the the best option. The most useful learners
in this regard are those that don’t just have assumptions
hard-wired into them, but allow us to state them explicitly,
vary them widely, and incorporate them automatically into
the learning (e.g., using first-order logic [22] or grammars
[6]).

In retrospect, the need for knowledge in learning should not
be surprising. Machine learning is not magic; it can’t get
something from nothing. What it does is get more from
less. Programming, like all engineering, is a lot of work:
we have to build everything from scratch. Learning is more
like farming, which lets nature do most of the work. Farm-
ers combine seeds with nutrients to grow crops. Learners
combine knowledge with data to grow programs.
5. OVERFITTING HAS MANY FACES

What if the knowledge and data we have are not sufficient
to completely determine the correct classifier? Then we run
the risk of just hallucinating a classifier (or parts of it) that
is not grounded in reality, and is simply encoding random
High
Bias
Low
Bias
Low
Variance
High
Variance

Figure 1: Bias and variance in dart-throwing.

quirks in the data. This problem is calledoverfitting, and is
the bugbear of machine learning. When your learner outputs
a classifier that is 100% accurate on the training data but
only 50% accurate on test data, when in fact it could have
output one that is 75% accurate on both, it has overfit.

Everyone in machine learning knows about overfitting, but
it comes in many forms that are not immediately obvious.
One way to understand overfitting is by decomposing gener-
alization error intobiasandvariance[9]. Bias is a learner’s
tendency to consistently learn the same wrong thing. Vari-
ance is the tendency to learn random things irrespective of
the real signal. Figure 1 illustrates this by an analogy with
throwing darts at a board. A linear learner has high bias,
because when the frontier between two classes is not a hyper-
plane the learner is unable to induce it. Decision trees don’t
have this problem because they can represent any Boolean
function, but on the other hand they can suffer from high
variance: decision trees learned on different training sets
generated by the same phenomenon are often very different,
when in fact they should be the same. Similar reasoning
applies to the choice of optimization method: beam search
has lower bias than greedy search, but higher variance, be-
cause it tries more hypotheses. Thus, contrary to intuition,
a more powerful learner is not necessarily better than a less
powerful one.

Figure 2 illustrates this.^1 Even though the true classifier
is a set of rules, with up to 1000 examples naive Bayes is
more accurate than a rule learner. This happens despite
naive Bayes’s false assumption that the frontier is linear!
Situations like this are common in machine learning: strong
false assumptions can be better than weak true ones, because
a learner with the latter needs more data to avoid overfitting.

(^1) Training examples consist of 64 Boolean features and a
Boolean class computed from them according to a set of “IF

.. .THEN.. .” rules. The curves are the average of 100 runs
with different randomly generated sets of rules. Error bars
are two standard deviations. See Domingos and Pazzani [11]
for details.

Cross-validation can help to combat overfitting, for example
by using it to choose the best size of decision tree to learn.
But it’s no panacea, since if we use it to make too many
parameter choices it can itself start to overfit [18].

Besides cross-validation, there are many methods to combat
overfitting. The most popular one is adding aregulariza-
tion termto the evaluation function. This can, for exam-
ple, penalize classifiers with more structure, thereby favoring
smaller ones with less room to overfit. Another option is to
perform a statistical significance test like chi-square before
adding new structure, to decide whether the distribution of
the class really is different with and without this structure.
These techniques are particularly useful when data is very
scarce. Nevertheless, you should be skeptical of claims that
a particular technique “solves” the overfitting problem. It’s
easy to avoid overfitting (variance) by falling into the op-
posite error of underfitting (bias). Simultaneously avoiding
both requires learning a perfect classifier, and short of know-
ing it in advance there is no single technique that will always
do best (no free lunch).

A common misconception about overfitting is that it is caused
by noise, like training examples labeled with the wrong class.
This can indeed aggravate overfitting, by making the learner
draw a capricious frontier to keep those examples on what
it thinks is the right side. But severe overfitting can occur
even in the absence of noise. For instance, suppose we learn a
Boolean classifier that is just the disjunction of the examples
labeled“true”in the training set. (In other words, the classi-
fier is a Boolean formula in disjunctive normal form, where
each term is the conjunction of the feature values of one
specific training example). This classifier gets all the train-
ing examples right and every positive test example wrong,
regardless of whether the training data is noisy or not.

The problem ofmultiple testing[14] is closely related to over-
fitting. Standard statistical tests assume that only one hy-
pothesis is being tested, but modern learners can easily test
millions before they are done. As a result what looks signif-
icant may in fact not be. For example, a mutual fund that
beats the market ten years in a row looks very impressive,
until you realize that, if there are 1000 funds and each has
a 50% chance of beating the market on any given year, it’s
quite likely that one will succeed all ten times just by luck.
This problem can be combatted by correcting the signifi-
cance tests to take the number of hypotheses into account,
but this can lead to underfitting. A better approach is to
control the fraction of falsely accepted non-null hypotheses,
known as thefalse discovery rate[3].
6. INTUITION FAILS IN HIGH
DIMENSIONS

After overfitting, the biggest problem in machine learning
is thecurse of dimensionality. This expression was coined
by Bellman in 1961 to refer to the fact that many algo-
rithms that work fine in low dimensions become intractable
when the input is high-dimensional. But in machine learn-
ing it refers to much more. Generalizing correctly becomes
exponentially harder as the dimensionality (number of fea-
tures) of the examples grows, because a fixed-size training
set covers a dwindling fraction of the input space. Even with
a moderate dimension of 100 and a huge training set of a
50
55
60
65
70
75
80
10 100 1000 10000
Test-Set Accuracy (%)
Number of Examples
Bayes
C4.

Figure 2: Naive Bayes can outperform a state-of-
the-art rule learner (C4.5rules) even when the true
classifier is a set of rules.

trillion examples, the latter covers only a fraction of about
10 −^18 of the input space. This is what makes machine learn-
ing both necessary and hard.

More seriously, the similarity-based reasoning that machine
learning algorithms depend on (explicitly or implicitly) breaks
down in high dimensions. Consider a nearest neighbor clas-
sifier with Hamming distance as the similarity measure, and
suppose the class is justx 1 ∧x 2. If there are no other fea-
tures, this is an easy problem. But if there are 98 irrele-
vant featuresx 3 ,... , x 100 , the noise from them completely
swamps the signal inx 1 andx 2 , and nearest neighbor effec-
tively makes random predictions.

Even more disturbing is that nearest neighbor still has a
problem even if all 100 features are relevant! This is because
in high dimensions all examples look alike. Suppose, for
instance, that examples are laid out on a regular grid, and
consider a test examplext. If the grid isd-dimensional,xt’s
2 dnearest examples are all at the same distance from it.
So as the dimensionality increases, more and more examples
become nearest neighbors ofxt, until the choice of nearest
neighbor (and therefore of class) is effectively random.

This is only one instance of a more general problem with
high dimensions: our intuitions, which come from a three-
dimensional world, often do not apply in high-dimensional
ones. In high dimensions, most of the mass of a multivari-
ate Gaussian distribution is not near the mean, but in an
increasingly distant “shell” around it; and most of the vol-
ume of a high-dimensional orange is in the skin, not the pulp.
If a constant number of examples is distributed uniformly in
a high-dimensional hypercube, beyond some dimensionality
most examples are closer to a face of the hypercube than
to their nearest neighbor. And if we approximate a hyper-
sphere by inscribing it in a hypercube, in high dimensions
almost all the volume of the hypercube is outside the hyper-
sphere. This is bad news for machine learning, where shapes
of one type are often approximated by shapes of another.

Building a classifier in two or three dimensions is easy; we

can find a reasonable frontier between examples of different
classes just by visual inspection. (It’s even been said that if
people could see in high dimensions machine learning would
not be necessary.) But in high dimensions it’s hard to under-
stand what is happening. This in turn makes it difficult to
design a good classifier. Naively, one might think that gath-
ering more features never hurts, since at worst they provide
no new information about the class. But in fact their bene-
fits may be outweighed by the curse of dimensionality.

Fortunately, there is an effect that partly counteracts the
curse, which might be called the“blessing of non-uniformity.”
In most applications examples are not spread uniformly thr-
oughout the instance space, but are concentrated on or near
a lower-dimensional manifold. For example,k-nearest neigh-
bor works quite well for handwritten digit recognition even
though images of digits have one dimension per pixel, be-
cause the space of digit images is much smaller than the
space of all possible images. Learners can implicitly take
advantage of this lower effective dimension, or algorithms
for explicitly reducing the dimensionality can be used (e.g.,
[23]).
7. THEORETICAL GUARANTEES ARE
NOT WHAT THEY SEEM

Machine learning papers are full of theoretical guarantees.
The most common type is a bound on the number of ex-
amples needed to ensure good generalization. What should
you make of these guarantees? First of all, it’s remarkable
that they are even possible. Induction is traditionally con-
trasted with deduction: in deduction you can guarantee that
the conclusions are correct; in induction all bets are off. Or
such was the conventional wisdom for many centuries. One
of the major developments of recent decades has been the
realization that in fact we can have guarantees on the re-
sults of induction, particularly if we’re willing to settlefor
probabilistic guarantees.

The basic argument is remarkably simple [5]. Let’s say a
classifier is bad if its true error rate is greater thanǫ. Then
the probability that a bad classifier is consistent withnran-
dom, independent training examples is less than (1−ǫ)n.
Letbbe the number of bad classifiers in the learner’s hy-
pothesis spaceH. The probability that at least one of them
is consistent is less thanb(1−ǫ)n, by the union bound.
Assuming the learner always returns a consistent classifier,
the probability that this classifier is bad is then less than
|H|(1−ǫ)n, where we have used the fact thatb≤ |H|. So
if we want this probability to be less thanδ, it suffices to
maken >ln(δ/|H|)/ln(1−ǫ)≥^1 ǫ
(

ln|H|+ ln^1 δ

)
.

Unfortunately, guarantees of this type have to be taken with
a large grain of salt. This is because the bounds obtained in
this way are usually extremely loose. The wonderful feature
of the bound above is that the required number of examples
only grows logarithmically with|H|and 1/δ. Unfortunately,
most interesting hypothesis spaces aredoublyexponential in
the number of featuresd, which still leaves us needing a num-
ber of examples exponential ind. For example, consider the
space of Boolean functions ofdBoolean variables. If there
areepossible different examples, there are 2epossible dif-
ferent functions, so since there are 2dpossible examples, the

total number of functions is 2^2

d

. And even for hypothesis
spaces that are “merely” exponential, the bound is still very
loose, because the union bound is very pessimistic. For ex-
ample, if there are 100 Boolean features and the hypothesis
space is decision trees with up to 10 levels, to guarantee
δ=ǫ= 1% in the bound above we need half a million ex-
amples. But in practice a small fraction of this suffices for
accurate learning.

Further, we have to be careful about what a bound like this
means. For instance, it does not say that, if your learner
returned a hypothesis consistent with a particular training
set, then this hypothesis probably generalizes well. What
it says is that, given a large enough training set, with high
probability your learner will either return a hypothesis that
generalizes well or be unable to find a consistent hypothesis.
The bound also says nothing about how to select a good hy-
pothesis space. It only tells us that, if the hypothesis space
contains the true classifier, then the probability that the
learner outputs a bad classifier decreases with training set
size. If we shrink the hypothesis space, the bound improves,
but the chances that it contains the true classifier shrink
also. (There are bounds for the case where the true classi-
fier is not in the hypothesis space, but similar considerations
apply to them.)

Another common type of theoretical guarantee is asymp-
totic: given infinite data, the learner is guaranteed to output
the correct classifier. This is reassuring, but it would be rash
to choose one learner over another because of its asymptotic
guarantees. In practice, we are seldom in the asymptotic
regime (also known as “asymptopia”). And, because of the
bias-variance tradeoff we discussed above, if learner A is bet-
ter than learner B given infinite data, B is often better than
A given finite data.

The main role of theoretical guarantees in machine learning
is not as a criterion for practical decisions, but as a sourceof
understanding and driving force for algorithm design. In this
capacity, they are quite useful; indeed, the close interplay
of theory and practice is one of the main reasons machine
learning has made so much progress over the years. But
caveat emptor: learning is a complex phenomenon, and just
because a learner has a theoretical justification and works in
practice doesn’t mean the former is the reason for the latter.

8. FEATURE ENGINEERING IS THE KEY

At the end of the day, some machine learning projects suc-
ceed and some fail. What makes the difference? Easily
the most important factor is the features used. If you have
many independent features that each correlate well with the
class, learning is easy. On the other hand, if the class is
a very complex function of the features, you may not be
able to learn it. Often, the raw data is not in a form that is
amenable to learning, but you can construct features from it
that are. This is typically where most of the effort in a ma-
chine learning project goes. It is often also one of the most
interesting parts, where intuition, creativity and “black art”
are as important as the technical stuff.

First-timers are often surprised by how little time in a ma-
chine learning project is spent actually doing machine learn-
ing. But it makes sense if you consider how time-consuming

it is to gather data, integrate it, clean it and pre-process it,
and how much trial and error can go into feature design.
Also, machine learning is not a one-shot process of build-
ing a data set and running a learner, but rather an iterative
process of running the learner, analyzing the results, modi-
fying the data and/or the learner, and repeating. Learning
is often the quickest part of this, but that’s because we’ve
already mastered it pretty well! Feature engineering is more
difficult because it’s domain-specific, while learners can be
largely general-purpose. However, there is no sharp frontier
between the two, and this is another reason the most useful
learners are those that facilitate incorporating knowledge.

Of course, one of the holy grails of machine learning is to
automate more and more of the feature engineering process.
One way this is often done today is by automatically gen-
erating large numbers of candidate features and selecting
the best by (say) their information gain with respect to the
class. But bear in mind that features that look irrelevant
in isolation may be relevant in combination. For example,
if the class is an XOR ofkinput features, each of them by
itself carries no information about the class. (If you want
to annoy machine learners, bring up XOR.) On the other
hand, running a learner with a very large number of fea-
tures to find out which ones are useful in combination may
be too time-consuming, or cause overfitting. So there is ul-
timately no replacement for the smarts you put into feature
engineering.
9. MORE DATA BEATS A CLEVERER
ALGORITHM

Suppose you’ve constructed the best set of features you
can, but the classifiers you’re getting are still not accurate
enough. What can you do now? There are two main choices:
design a better learning algorithm, or gather more data
(more examples, and possibly more raw features, subject to
the curse of dimensionality). Machine learning researchers
are mainly concerned with the former, but pragmatically
the quickest path to success is often to just get more data.
As a rule of thumb, a dumb algorithm with lots and lots of
data beats a clever one with modest amounts of it. (After
all, machine learning is all about letting data do the heavy
lifting.)

This does bring up another problem, however: scalability.
In most of computer science, the two main limited resources
are time and memory. In machine learning, there is a third
one: training data. Which one is the bottleneck has changed
from decade to decade. In the 1980’s it tended to be data.
Today it is often time. Enormous mountains of data are
available, but there is not enough time to process it, so it
goes unused. This leads to a paradox: even though in prin-
ciple more data means that more complex classifiers can be
learned, in practice simpler classifiers wind up being used,
because complex ones take too long to learn. Part of the
answer is to come up with fast ways to learn complex classi-
fiers, and indeed there has been remarkable progress in this
direction (e.g., [12]).

Part of the reason using cleverer algorithms has a smaller
payoff than you might expect is that, to a first approxima-
tion, they all do the same. This is surprising when you
consider representations as different as, say, sets of rules
SVM
N. Bayes
kNN
D. Tree

Figure 3: Very different frontiers can yield similar
class predictions. (+and−are training examples of
two classes.)

and neural networks. But in fact propositional rules are
readily encoded as neural networks, and similar relation-
ships hold between other representations. All learners es-
sentially work by grouping nearby examples into the same
class; the key difference is in the meaning of “nearby.” With
non-uniformly distributed data, learners can produce widely
different frontiers while still making the same predictions in
the regions that matter (those with a substantial number of
training examples, and therefore also where most test ex-
amples are likely to appear). This also helps explain why
powerful learners can be unstable but still accurate. Fig-
ure 3 illustrates this in 2-D; the effect is much stronger in
high dimensions.

As a rule, it pays to try the simplest learners first (e.g., naive
Bayes before logistic regression,k-nearest neighbor before
support vector machines). More sophisticated learners are
seductive, but they are usually harder to use, because they
have more knobs you need to turn to get good results, and
because their internals are more opaque.

Learners can be divided into two major types: those whose
representation has a fixed size, like linear classifiers, and
those whose representation can grow with the data, like deci-
sion trees. (The latter are sometimes called non-parametric
learners, but this is somewhat unfortunate, since they usu-
ally wind up learning many more parameters than paramet-
ric ones.) Fixed-size learners can only take advantage of so
much data. (Notice how the accuracy of naive Bayes asymp-
totes at around 70% in Figure 2.) Variable-size learners can
in principle learn any function given sufficient data, but in
practice they may not, because of limitations of the algo-
rithm (e.g., greedy search falls into local optima) or compu-
tational cost. Also, because of the curse of dimensionality,
no existing amount of data may be enough. For these rea-
sons, clever algorithms—those that make the most of the
data and computing resources available—often pay off in
the end, provided you’re willing to put in the effort. There
is no sharp frontier between designing learners and learn-
ing classifiers; rather, any given piece of knowledge could be
encoded in the learner or learned from data. So machine
learning projects often wind up having a significant compo-
nent of learner design, and practitioners need to have some
expertise in it [13].

In the end, the biggest bottleneck is not data or CPU cycles,
but human cycles. In research papers, learners are typically
compared on measures of accuracy and computational cost.
But human effort saved and insight gained, although harder
to measure, are often more important. This favors learn-
ers that produce human-understandable output (e.g., rule
sets). And the organizations that make the most of ma-
chine learning are those that have in place an infrastructure
that makes experimenting with many different learners, data
sources and learning problems easy and efficient, and where
there is a close collaboration between machine learning ex-
perts and application domain ones.
10. LEARN MANY MODELS, NOT JUST
ONE

In the early days of machine learning, everyone had their fa-
vorite learner, together with somea priorireasons to believe
in its superiority. Most effort went into trying many varia-
tions of it and selecting the best one. Then systematic em-
pirical comparisons showed that the best learner varies from
application to application, and systems containing many dif-
ferent learners started to appear. Effort now went into try-
ing many variations of many learners, and still selecting just
the best one. But then researchers noticed that, if instead
of selecting the best variation found, we combine many vari-
ations, the results are better—often much better—and at
little extra effort for the user.

Creating suchmodel ensemblesis now standard [1]. In the
simplest technique, calledbagging, we simply generate ran-
dom variations of the training set by resampling, learn a
classifier on each, and combine the results by voting. This
works because it greatly reduces variance while only slightly
increasing bias. Inboosting, training examples have weights,
and these are varied so that each new classifier focuses on
the examples the previous ones tended to get wrong. In
stacking, the outputs of individual classifiers become the in-
puts of a “higher-level” learner that figures out how best to
combine them.

Many other techniques exist, and the trend is toward larger
and larger ensembles. In the Netflix prize, teams from all
over the world competed to build the best video recom-
mender system (http://netflixprize.com). As the competi-
tion progressed, teams found that they obtained the best
results by combining their learners with other teams’, and
merged into larger and larger teams. The winner and runner-
up were both stacked ensembles of over 100 learners, and
combining the two ensembles further improved the results.
Doubtless we will see even larger ones in the future.

Model ensembles should not be confused with Bayesian model
averaging (BMA). BMA is the theoretically optimal approach
to learning [4]. In BMA, predictions on new examples are
made by averaging the individual predictions ofallclassifiers
in the hypothesis space, weighted by how well the classifiers
explain the training data and how much we believe in them
a priori. Despite their superficial similarities, ensembles and
BMA are very different. Ensembles change the hypothesis
space (e.g., from single decision trees to linear combinations
of them), and can take a wide variety of forms. BMA assigns
weights to the hypotheses in the original space according to
a fixed formula. BMA weights are extremely different from

those produced by (say) bagging or boosting: the latter are
fairly even, while the former are extremely skewed, to the
point where the single highest-weight classifier usually dom-
inates, making BMA effectively equivalent to just selecting
it [8]. A practical consequence of this is that, while model
ensembles are a key part of the machine learning toolkit,
BMA is seldom worth the trouble.

11. SIMPLICITY DOES NOT IMPLY
ACCURACY

Occam’s razor famously states that entities should not be
multiplied beyond necessity. In machine learning, this is of-
ten taken to mean that, given two classifiers with the same
training error, the simpler of the two will likely have the
lowest test error. Purported proofs of this claim appear reg-
ularly in the literature, but in fact there are many counter-
examples to it, and the “no free lunch” theorems imply it
cannot be true.

We saw one counter-example in the previous section: model
ensembles. The generalization error of a boosted ensem-
ble continues to improve by adding classifiers even after the
training error has reached zero. Another counter-example is
support vector machines, which can effectively have an infi-
nite number of parameters without overfitting. Conversely,
the function sign(sin(ax)) can discriminate an arbitrarily
large, arbitrarily labeled set of points on thexaxis, even
though it has only one parameter [24]. Thus, contrary to in-
tuition, there is no necessary connection between the number
of parameters of a model and its tendency to overfit.

A more sophisticated view instead equates complexity with
the size of the hypothesis space, on the basis that smaller
spaces allow hypotheses to be represented by shorter codes.
Bounds like the one in the section on theoretical guarantees
above might then be viewed as implying that shorter hy-
potheses generalize better. This can be further refined by
assigning shorter codes to the hypothesis in the space that
we have somea prioripreference for. But viewing this as
“proof” of a tradeoff between accuracy and simplicity is cir-
cular reasoning: we made the hypotheses we prefer simpler
by design, and if they are accurate it’s because our prefer-
ences are accurate, not because the hypotheses are “simple”
in the representation we chose.

A further complication arises from the fact that few learners
search their hypothesis space exhaustively. A learner with a
larger hypothesis space that tries fewer hypotheses from it
is less likely to overfit than one that tries more hypotheses
from a smaller space. As Pearl [19] points out, the size of
the hypothesis space is only a rough guide to what really
matters for relating training and test error: the procedure
by which a hypothesis is chosen.

Domingos [7] surveys the main arguments and evidence on
the issue of Occam’s razor in machine learning. The conclu-
sion is that simpler hypotheses should be preferred because
simplicity is a virtue in its own right, not because of a hy-
pothetical connection with accuracy. This is probably what
Occam meant in the first place.

12. REPRESENTABLE DOES NOT IMPLY
LEARNABLE

Essentially all representations used in variable-size learners
have associated theorems of the form “Every function can
be represented, or approximated arbitrarily closely, using
this representation.” Reassured by this, fans of the repre-
sentation often proceed to ignore all others. However, just
because a function can be represented does not mean it can
be learned. For example, standard decision tree learners
cannot learn trees with more leaves than there are training
examples. In continuous spaces, representing even simple
functions using a fixed set of primitives often requires an
infinite number of components. Further, if the hypothesis
space has many local optima of the evaluation function, as
is often the case, the learner may not find the true function
even if it is representable. Given finite data, time and mem-
ory, standard learners can learn only a tiny subset of all pos-
sible functions, and these subsets are different for learners
with different representations. Therefore the key question is
not “Can it be represented?”, to which the answer is often
trivial, but “Can it be learned?” And it pays to try different
learners (and possibly combine them).

Some representations are exponentially more compact than
others for some functions. As a result, they may also re-
quire exponentially less data to learn those functions. Many
learners work by forming linear combinations of simple ba-
sis functions. For example, support vector machines form
combinations of kernels centered at some of the training ex-
amples (the support vectors). Representing parity ofnbits
in this way requires 2nbasis functions. But using a repre-
sentation with more layers (i.e., more steps between input
and output), parity can be encoded in a linear-size classifier.
Finding methods to learn these deeper representations is one
of the major research frontiers in machine learning [2].
13. CORRELATION DOES NOT IMPLY
CAUSATION

The point that correlation does not imply causation is made
so often that it is perhaps not worth belaboring. But, even
though learners of the kind we have been discussing can only
learn correlations, their results are often treated as repre-
senting causal relations. Isn’t this wrong? If so, then why
do people do it?

More often than not, the goal of learning predictive mod-
els is to use them as guides to action. If we find that beer
and diapers are often bought together at the supermarket,
then perhaps putting beer next to the diaper section will
increase sales. (This is a famous example in the world of
data mining.) But short of actually doing the experiment
it’s difficult to tell. Machine learning is usually applied to
observationaldata, where the predictive variables are not
under the control of the learner, as opposed toexperimental
data, where they are. Some learning algorithms can poten-
tially extract causal information from observational data,
but their applicability is rather restricted [20]. On the other
hand, correlation is a sign of a potential causal connection,
and we can use it as a guide to further investigation (for
example, trying to understand what the causal chain might
be).

Many researchers believe that causality is only a convenient
fiction. For example, there is no notion of causality in phys-
ical laws. Whether or not causality really exists is a deep
philosophical question with no definitive answer in sight,
but the practical points for machine learners are two. First,
whether or not we call them “causal,” we would like to pre-
dict the effects of our actions, not just correlations between
observable variables. Second, if you can obtain experimental
data (for example by randomly assigning visitors to different
versions of a Web site), then by all means do so [15].

14. CONCLUSION

Like any discipline, machine learning has a lot of “folk wis-
dom” that can be hard to come by, but is crucial for suc-
cess. This article summarized some of the most salient items.
A good place to learn more is my bookThe Master Algo-
rithm, a non-technical introduction to machine learning [10].
For a complete online machine learning course, check out
http://www.cs.washington.edu/homes/pedrod/class. There
is also a treasure trove of machine learning lectures at http:-
//www.videolectures.net. A widely used open source ma-
chine learning toolkit is Weka [25]. Happy learning!

15. REFERENCES

[1] E. Bauer and R. Kohavi. An empirical comparison of
voting classification algorithms: Bagging, boosting
and variants.Machine Learning, 36:105–142, 1999.
[2] Y. Bengio. Learning deep architectures for AI.
Foundations and Trends in Machine Learning,
2:1–127, 2009.
[3] Y. Benjamini and Y. Hochberg. Controlling the false
discovery rate: A practical and powerful approach to
multiple testing.Journal of the Royal Statistical
Society, Series B, 57:289–300, 1995.
[4] J. M. Bernardo and A. F. M. Smith.Bayesian Theory.
Wiley, New York, NY, 1994.
[5] A. Blumer, A. Ehrenfeucht, D. Haussler, and M. K.
Warmuth. Occam’s razor.Information Processing
Letters, 24:377–380, 1987.
[6] W. W. Cohen. Grammatically biased learning:
Learning logic programs using an explicit antecedent
description language.Artificial Intelligence,
68:303–366, 1994.
[7] P. Domingos. The role of Occam’s razor in knowledge
discovery.Data Mining and Knowledge Discovery,
3:409–425, 1999.
[8] P. Domingos. Bayesian averaging of classifiers and the
overfitting problem. InProceedings of the Seventeenth
International Conference on Machine Learning, pages
223–230, Stanford, CA, 2000. Morgan Kaufmann.
[9] P. Domingos. A unified bias-variance decomposition
and its applications. InProceedings of the Seventeenth
International Conference on Machine Learning, pages
231–238, Stanford, CA, 2000. Morgan Kaufmann.
[10] P. Domingos.The Master Algorithm: How the Quest
for the Ultimate Learning Machine Will Remake Our
World. Basic Books, New York, NY, 2015.
[11] P. Domingos and M. Pazzani. On the optimality of the
simple Bayesian classifier under zero-one loss.Machine
Learning, 29:103–130, 1997.
[12] G. Hulten and P. Domingos. Mining complex models
from arbitrarily large databases in constant time. In

Proceedings of the Eighth ACM SIGKDD International
Conference on Knowledge Discovery and Data Mining,
pages 525–531, Edmonton, Canada, 2002. ACM Press.
[13] D. Kibler and P. Langley. Machine learning as an
experimental science. InProceedings of the Third
European Working Session on Learning, London, UK,

    Pitman.
    [14] A. J. Klockars and G. Sax.Multiple Comparisons.
    Sage, Beverly Hills, CA, 1986.

[15] R. Kohavi, R. Longbotham, D. Sommerfield, and
R. Henne. Controlled experiments on the Web: Survey
and practical guide.Data Mining and Knowledge
Discovery, 18:140–181, 2009.

[16] J. Manyika, M. Chui, B. Brown, J. Bughin, R. Dobbs,
C. Roxburgh, and A. Byers. Big data: The next
frontier for innovation, competition, and productivity.
Technical report, McKinsey Global Institute, 2011.

[17] T. M. Mitchell.Machine Learning. McGraw-Hill, New
York, NY, 1997.
[18] A. Y. Ng. Preventing “overfitting” of cross-validation
data. InProceedings of the Fourteenth International
Conference on Machine Learning, pages 245–253,
Nashville, TN, 1997. Morgan Kaufmann.
[19] J. Pearl. On the connection between the complexity
and credibility of inferred models.International
Journal of General Systems, 4:255–264, 1978.

[20] J. Pearl.Causality: Models, Reasoning, and Inference.
Cambridge University Press, Cambridge, UK, 2000.

[21] J. R. Quinlan.C4.5: Programs for Machine Learning.
Morgan Kaufmann, San Mateo, CA, 1993.

[22] M. Richardson and P. Domingos. Markov logic
networks.Machine Learning, 62:107–136, 2006.
[23] J. Tenenbaum, V. Silva, and J. Langford. A global
geometric framework for nonlinear dimensionality
reduction.Science, 290:2319–2323, 2000.

[24] V. N. Vapnik.The Nature of Statistical Learning
Theory. Springer, New York, NY, 1995.

[25] I. Witten, E. Frank, and M. Hall.Data Mining:
Practical Machine Learning Tools and Techniques.
Morgan Kaufmann, San Mateo, CA, 3rd edition, 2011.
[26] D. Wolpert. The lack ofa prioridistinctions between
learning algorithms.Neural Computation,
8:1341–1390, 1996.

This is a offline tool, your data stays locally and is not send to any server!

Feedback & Bug Reports
