### ctmatch


category_data order of features:
cancer,cardiac,endocrine,gastrointestinal,genetic,healthy,infection,neurological,other,pediatric,psychological,pulmonary,renal,reproductive


**repo in development for matching clinical trials to patient text descriptions**

This package is designed generally for the task described in the precision medicine track of TREC since 2021,
that is, an information retrieval task to match patient descriptions (topics) to clinical trials data (xml documents)

ctmatch leverages several tools to build the representations in the dataset against which the topics are matched,
as well as langugage models that have been fine-tuned on the curated ctmatch dataset of relevance-labelled topic, document 
pairs. 

The pipeline currently matches user input topics to the static snapshot of clinical trials data downloaded for the TREC task from december of 2015, 
but can be updated with a current dataset of clinical trials data easily using ctproc to process, and we have plans in the future to 
add support for querying against the database hosted at clinicaltrials.gov.


## pipeline filters

Currently 3 filters are applied to the set of documents:

1. The first filter is based on token similarity, combining the cosine similarity of whole texts (512-dim): (topic_tfidf_representations, doc_elig_crit_representations)
   and the cosine similarity of the extracted category probabilities* (16-dim): (topic_category_vector_representations, doc_category_vector_representation) into a single score. 
   
   *The category vectors are arbitrarily selected 16 classes i.e. pulmonary, cardiac, health, other.... with 
   probabilites as softmax of the output from a zero-shot classification of {large language model} applied to the 
   'condition' field of the ct documents and the raw text of the topic.

   The docs are ranked by this combined score and the top {1000} are selected (out of ~250k)


2. The second filter uses the embedded representations as extracted from the last hidden layer of a LM fine-tuned on the ctmatch data.
   an SVM is used to learn a decision boundary between the topic as one class and the documents as another, then takes the top {100} documents


3. The third filter again uses the fine-tuned LM from filter 2., with prompting, to apply a kind of constrastive reasoning approach (in pseudocode):
   for each remaining document:
      lm is prompted with examples pairs of document, topic, and is asked to create an example of topic for which the document would be relevant
	  lm is prompted with examples pairs of document, topic, and is asked to create an example of topic for which the document would NOT be relevant
	  compute dist to each

	return the documents closest to their respective positive representations created by the lm

## api

IN DEVELOPMENT

