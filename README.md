# Questions-CS50AI

## Background

This question answering system perform two tasks: document retrieval and passage retrieval. The system has access to a corpus of text documents. When presented with a query (a question in English asked by the user), document retrieval will first identify which document(s) are most relevant to the query. Once the top documents are found, the top document(s) will be subdivided into passages (in this case, sentences) so that the most relevant passage to the question can be determined.

How do we find the most relevant documents and passages? To find the most relevant documents, I used tf-idf to rank documents based both on term frequency for words in the query as well as inverse document frequency for words in the query. Once I've found the most relevant documents, there many possible metrics for scoring passages, but I used a combination of inverse document frequency and a query term density measure.
