# PB2PV Submission to Metadata Automation Dream Challenge

Author: Christopher E. Marks


## Summary

Using a graph data model, full-text search, and machine learning to automatically match input data columns to common metadata fields.

## Background

This project is a proposed solution to the [Metadata Automation DREAM Challenge](https://www.synapse.org/#!Synapse:syn18065891/wiki/588180).  The underlying idea I employed is that the data elements documented in the caDSR database, along with the concepts and terms documented in the NCI Thesaurus, were related through an underlying graph structure that could be exploited in a useful way through a graph data model.  I hypothesized that a column of data from a given table could be matched to the correct data element in the caDSR by first scoring text matches between the column data and the nodes in the graph data model, then propagating the scores through the network, and finally by using supervised learning to identify how these propagated scores were predictive of matching data elements.  In implementation, I used [neo4j](https://neo4j.com/) as the database engine and its built-in Lucene full-text search capability to find nodes with text attributes that matched text found in the column data.  I propagated these scores through the network and aggregated them on the data elements, producing a set of features that corresponded to different full-text search score aggregations.  Finally, I used logistic regression to identify probable "NOMATCH" fields, and ridge regression to learn and predict which data elements were the best matches to the column data.

${image?fileName=data%2Dfig%2Epng&align=None&scale=100&responsive=true&altText=Graph Data Model}

## Methods

### Preparation: Implementing the Data Model

I began by implementing the data model depicted in the figure in neo4j.  This data model captures all of the data elements in the caDSR and all of the concepts in the NCI thesaurus, as well as the relationships between the two.  Most of the relationships depicted are intuitive from an inspection of the data, e.g., each "Concept" in the NCI thesaurus has associated "Synonyms".  In a few cases relationships are labeled for clarity.  Relevant information in the data that is not depicted as a node in the figure is included as node attributes in the database.  For example, caDSR data elements (CDEs) each have a `CDE_LONG_NAME`; this field is captured as an attribute for each caDSR data element node.  However, CDEs could have more than one `CDE_SHORT_NAME`. For this reason I modeled this field as a separate node type and used neo4j graph relationships to model their connections to their associated CDEs.  The resulting neo4j database became the backbone of the matching algorithm for this project.

Neo4j includes [full text indexing](https://neo4j.com/developer/kb/fulltext-search-in-neo4j/) and the [Lucene Search](https://lucene.apache.org/) engine. To enable the matching of column data to nodes in the neo4j database, I created two full-text indices on the data:

* An index on permissible value text (`ANSINDEX`).
* An index on synonym text, `CDE_LONG_NAME` text, `CDE_SHORT_NAME` text, DEC name text, and CDE Questions text (`NAMEINDEX`).

### Matching Algorithm

#### Overview

With a graph data model in place, the process for matching a column of data to the best caDSR data element proceeds generally as follows:

1. Preprocess and parse the column header and column data values into a set of neo4j full-text search queries.
1. Propagate the resulting search scores through a set of specific pathways to the CDE nodes (aggregating using `max` or `sum`) to build an initial feature set.  Each feature in this set results from a combination of a specific full-text search and specific network propagation of the score.
1. Calculate statistics capturing the results of the searches.  The statistics I used are the max score returned from each search, the total number of unique CDEs identified from all searches combined, and the percentage of the total CDEs that came from each individual search.  (Note: these percentages could sum to more than 1, as some CDEs could be returned by multiple searches).
1. Using all of this information, determine the probability of the data column having no matching CDE (`NOMATCH`) in the caDSR database, and identify the best matching CDEs identified from the searches.
1. Based on these results, determine the top three matches (including `NOMATCH`, based on probability), in order of consideration.

#### Steps 1-3: Producing the scored CDE candidate set

For example, I preprocessed the column header by replacing some characters (e.g., "\_") with spaces, and inserted spaces between certain combinations (e.g., lowercase-uppercase sequences) in order to convert the column name to a sequence of words.  One of the searches I ran to identify candidate CDEs took this sequence of words and searched the `NAMEINDEX`.  Then I took the synonym nodes that were returned by the query and propagated the search scores along the path: Synonym -- NCI Concept -[DEC property class]- caDSR Data Element Concept -- caDSR Data Element.  If more than one score converged on a node, I aggregated using the `max` operator.  The result was a list of CDEs with associated score propagation values.  The values became one feature in the data set, and the CDEs were added to the list of candidate matches with these associated feature values.

All of the initial features were produced using this general method: a full text search (on `NAMEINDEX` or `ANSINDEX`) using some text derived from the column data, followed by some propagation of scores through specified paths to CDE nodes.  The result was a list of candidate CDEs with multiple scores.  Many candidate CDEs were only identified by one or two search--pathway combinations; in these cases the remaining feature scores were set to 0 by default.  In the event the initial searches were not very productive, I executed an additional preprocessing step and searched again in attempt to build a larger set of candidate CDEs.

#### Step 4: Identifying the best matches

With a list of candidate CDE matches, along with information about how closely they matched the column data in various aspects (based on the various searches and propagation pathways used to produce the scores), my hypothesis was it would be possible to determine the best set of matching CDEs with some accuracy.  I also discovered the importance of identifying columns for which there was not a matching CDE in the caDSR, as many of the columns provided for training were annotated as `NOMATCH`.  To accomplish both tasks, I used two models: a classification model to identify `NOMATCH` probabilities, and a regression model to predict which CDEs might be good matches to the column data.

#### Classification model

I used logistic regression to determine the probability of a `NOMATCH`.  For each column, I used summary statistics of the set of candidate CDEs as features in the logistic regression model, e.g., max values for each feature and total number of candidates. From a subset of these statistics I was able to consistently achieve good out-of-sample predictive power on which columns had no matching CDE using L2-regularized logistic regression.

#### Regression model

I used ridge regression to predict the score a CDE would achieve, based on concept-overlap with the gold standard annotated value (see the Individual result score: "Value Coverage" section of the [Challenge Wiki](https://www.synapse.org/#!Synapse:syn18065891/wiki/600432)).  I found that over-regularizing generally resulted in better performance, measured again by "value coverage" as opposed to using traditional squared error metrics.  I used a model that I found consistently performed well on out-of-sample data.

#### Step 5: Putting it all together

For each column, I used the set of searches and network propagation pathways to produce the set of candidate CDEs with feature scores.  I determined the probability the column was a `NOMATCH` using the logistic regression, and identified the best matches from the caDSR using the ridge regression model.  Using trial and error, I found that cutoff values for `NOMATCH` probabilities that I used to determine whether or where to include `NOMATCH` in the submitted solution set, i.e., if the probability of `NOMATCH` (coming from the logistic regression model) was higher than 0.4, I included `NOMATCH` as the second ranked CDE match for that column, and if the probability was higher than 0.5, I included `NOMATCH` as the top-ranked CDE match for that column.  Using these criteria, I never included `NOMATCH` as the third-ranked CDE match for any column.

I used the highest values from the ridge regression results to determine the other CDE matches and rankings.


##Conclusion/Discussion

The real value in this approach is the use of the inherent graph nature of the underlying data to identify and score candidate matches.  I am certain that someone could develop a better set of searches and network score propagations to obtain better results, and likewise there are probably better ways of identifying and ranking the best solutions.  Furthermore, the subnetwork comprised by the different CDEs represented in the same data table might provide additional insight.  I only considered each column individually, absent the information provided by other columns in the same table.  It might be that the columns present in the same table exhibit a common structure in the CDE-concept data graph, that can be exploited to better match groups of columns to groups of CDEs.

The algorithm and models I have submitted for this contest should generalize well.  I intentionally did not build separate models for each data set, in order to avoid over-fitting.  However, I only used the real datasets in training, as I assumed that the fake data would not necessarily exhibit the same patterns and would therefore result in less-useful model fitting. 

## Technology Used

As noted I used the open source [neo4j](https://neo4j.com/) as a graph database.  I used [Python](https://www.python.org/) and select python modules, notably [scikit-learn](https://scikit-learn.org/stable/) for all data processing, machine learning, and submission work flows.  As required by the project, I used [Docker](https://www.docker.com/) to containerize the submission into a single executable.

I have added my somewhat organized code to a Github repository [https://github.com/cemarks/cancer/](https://github.com/cemarks/cancer/).  If there is interest in reproducing any of these models, I will clean it up and document it to make it easy.

## References

* [Metadata Automation Dream Challenge](https://www.synapse.org/#!Synapse:syn18065891/wiki/588180)
* [Neo4j](https://neo4j.com/)
* [Python](https://www.python.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)

## Authors Statement

This project is exclusively my work, using existing open source capabilities as discussed.  I will clean up the [code](https://github.com/cemarks/cancer/) to make everything easy to reproduce if there is sufficient interest.