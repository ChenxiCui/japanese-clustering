japanese-clustering
===================

K-means clustering of Japanese sentences from Tanaka Corpus (http://www.edrdg.org/wiki/index.php/Tanaka_Corpus).  
Uses Kuromoji (https://github.com/atilika/kuromoji) via Lucene's JapaneseAnalyzer for tokenizing and removing stopwords. 

Work in Progress:  
While the whole pipeline from reading in the sentences from a text file to clustering the tfidf-vectors works, 
no parameter tuning has been done yet. Therefore, the resulting clusters are not yet semantically grouped.
