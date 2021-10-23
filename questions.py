import nltk
import math
import string
from pathlib import Path
import sys

FILE_MATCHES = 5
SENTENCE_MATCHES = 1


def count_words(documents):
    """
    Given a tokenized dictionary dictionary mapping names of documents
    to a list of words in that document it returns a dictionary
    with the occurrences of every word in the documents.
    """
    occurr = {}
    sets = list()

    #  Create sets from the words of different documents
    for document in documents:
        sets.append(set(documents[document]))

    # Count the number of times a word appears in the different sets
    for single_set in sets:
        for word in single_set:
            if word in occurr.keys():
                occurr[word] += 1
            else:
                occurr[word] = 1
    return occurr

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = {}
    for child in Path(directory).glob('*.txt'):
        if child.is_file():
            files[child.name] = child.read_text()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Remove punctuation
    document = document.translate(str.maketrans('', '', string.punctuation))

    # Tokenize and make all lowercase words
    temp_tokens = nltk.word_tokenize(document.lower())

    # Remove stopwords
    tokens = [w for w in temp_tokens if w not in nltk.corpus.stopwords.words("english")]

    return tokens

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Total number of documents
    n_documents = len(documents.keys())

    # Dictionary with the occurrences of every word in the documents
    occurr = count_words(documents)

    # Compute and replace the occurrences for the IDF
    idfs = {word: (math.log(n_documents / count )) for word, count in occurr.items()}

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    # Create a dictionary with filenames and an score of 0 for all of them
    files_scores = {file: 0 for file in files.keys()}

    # For every file, iterate through all their words 
    # and calculate their tf-idf
    for file in files:
        for word in query:
            if word in files[file]:

                # Only sum to the ranking if the word is in the sentence
                files_scores[file] += files[file].count(word) * idfs[word]

    # Order the files by their score descendent order
    files_scores = sorted(files_scores.items(), key=lambda item: item[1], reverse=True)

    # Create list and only return the first 'n'
    files_ranking = [file[0] for file in files_scores][:n]
    
    return files_ranking

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_s = list()

    # For every sentence in sentences
    for sentence in sentences:
        # Create an score starting with 0 for both MWM and QTD
        sentence_score = [sentence, 0, 0]

        # For every word in the query
        for word in query:
            # If the word is in the sentence update the sentence score 
            if word in sentences[sentence]:
                # Add the idfs value of the to the sentence score 
                sentence_score[1] += idfs[word]
                # Add the number of times the word is repeated in the sentence
                sentence_score[2] += sentences[sentence].count(word)
        
        # Then, divide the number of words that appeared in the query for the length of the sentence
        sentence_score[2] /= len(sentences[sentence])
        # Add the sentence score to the list of sentences
        top_s.append(sentence_score)
    
    # Order the list of sentences by their scores and only store the sentence itself
    top_s = [sentence for sentence, mwm, qtd in sorted(top_s, key=lambda item: (item[1], item[2]), reverse=True)]

    # Return only the first 'n'
    return top_s[:n]



if __name__ == "__main__":
    main()
