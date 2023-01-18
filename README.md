# The Bigger Fish: A comparison of state-of-the-art QSAR models on low-resourced aquatic toxicity regression tasks

![Graphical Abstract](https://user-images.githubusercontent.com/37438887/213145440-94bfe4a5-48c6-4f50-88b0-50da54bbc0ad.jpg)

Abstract: 
Toxicological information as needed for risk assessments of chemical compounds is often sparse. 
Unfortunately, gathering new toxicological information experimentally often involves animal testing. 
Therefore, simulated alternatives, such as Quantitative Structure-Activity Relationship (QSAR) models, that use known toxicity values to infer the toxicity of a new compound, are preferred. Indeed, the European Union allows chemicals emitted into the environment to be registered with aquatic toxicity information via simulated experiments. These aquatic toxicities are calculated by considering the impact of a given chemical on different aquatic species. 

Aquatic toxicity data collections, thus, consist of many related tasks - each predicting the toxicity of new compounds on a given species.
Since many of these tasks are inherently low-resource, i.e., involve few associated compounds, this is a challenging problem. 
Meta-learning, a subfield of artificial intelligence, enables the utilisation of information captured across tasks, leading to more accurate models. 

In our work, we benchmark various state-of-the-art meta-learning techniques for building QSAR models, focusing on knowledge sharing between species.
Specifically, we employ and compare transformational machine learning, model-agnostic meta-learning, fine-tuning, as well as multitask models. 
Our experiments show that established knowledge-sharing techniques outperform single-task approaches. 

Based on our results, we recommend the use of multitask random forest models for aquatic toxicity QSAR modelling, which matched or exceeded the performance of other approaches and robustly produced good results in low-resource settings. This model functions on a species level predicting toxicity for multiple species across phyla with flexible exposure duration and on a large chemical applicability domain. 
