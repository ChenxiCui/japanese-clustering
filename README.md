japanese-clustering
===================

K-means clustering of Japanese sentences from Tanaka Corpus (http://www.edrdg.org/wiki/index.php/Tanaka_Corpus).  
Uses Kuromoji (https://github.com/atilika/kuromoji) for tokenizing and filtering for nouns.

Use clusterdumper for additional output:  
/bin/mahout clusterdump -dt sequencefile -d data/output/vectors/dictionary.file-* -i data/output/result/clusters-8-final/ -b 10 -n 10


Work in Progress:  
While the whole pipeline from reading in the sentences from a text file to clustering the tfidf-vectors works, 
no parameter tuning has been done yet. Therefore, the resulting clusters are not yet semantically grouped.
