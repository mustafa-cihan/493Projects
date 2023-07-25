import os
import string
import re
from html import unescape
from collections import defaultdict

# Returns a dictionary of documents
def preprocess(text):
    # Remove HTML entities
    text = unescape(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Split into tokens by whitespace
    tokens = text.split()
    new_tokens = []
    for token in tokens:
        # Remove punctuation
        token = token.translate(str.maketrans('','',string.punctuation))
        # Make lowercase
        token = token.lower()

        # Check if token is alphanumeric
        if token.isalnum():
            new_tokens.append(token)

    # Return preprocessed tokens
    return new_tokens

# Returns a dictionary of documents
def get_documents():
    documents = {}
    for filename in os.listdir('reuters21578'):
        if filename.endswith('.sgm'):
            with open(os.path.join('reuters21578', filename), 'r', encoding='latin-1') as f:
                text = f.read()
                # Split into individual news stories
                stories = text.split('<REUTERS')
                for story in stories:
                    # Extract NEWID and text of news story
                    start = story.find('NEWID="') + len('NEWID="')
                    end = story.find('"', start)
                    doc_id = story[start:end]
                    # Extract title of news story
                    start = story.find('<TITLE>') + len('<TITLE>')
                    end = story.find('</TITLE>', start)
                    title = story[start:end]
                    # Extract body of news story
                    start = story.find('<BODY>') + len('<BODY>')
                    end = story.find('</BODY>', start)

                    # Check if body exists
                    if not (start == 5 and end == -1):    
                        body = story[start:end]
                    else:
                        body = ''
                    # Check if doc_id is numeric
                    if not doc_id.isdigit():
                            continue
                    # Preprocess title and body
                    tokens = preprocess(title + ' ' + body)
                    # Store tokens for document
                    documents[doc_id] = tokens
    # Return documents
    return documents

# Returns the postings list for a term
def get_postings_list(term, inverted_index):
    if term in inverted_index:
        return sorted(inverted_index[term])
    else:
        return []
    
# Returns the postings list for a phrase or proximity query
def merge_posting_lists(posting_list_1, posting_list_2, k=None):
    
    merged_postings = []
    i = 0
    j = 0

    # Iterate through posting lists
    # If doc_ids don't match, increment the smaller doc_id
    # If positions are within k, add to merged postings list
    # If positions are not within k, increment the greater position
    while i < len(posting_list_1) and j < len(posting_list_2):

        # Get doc_id and position for each posting
        doc_id_1, pos_1 = posting_list_1[i]
        doc_id_2, pos_2 = posting_list_2[j]

        # Check if doc_ids match
        if doc_id_1 == doc_id_2:
            merged_pos = None
            # Check if positions are within k
            if k is None or 0<= (pos_2 - pos_1) <= k:
                merged_pos = pos_2

            # Add to merged postings list
            if merged_pos:
                merged_postings.append((doc_id_1, merged_pos))

            # Increment greater doc_id
            if((pos_2 - pos_1) < k):
                j += 1
            else:
                i += 1

        # Increment smaller doc_id
        elif doc_id_1 < doc_id_2:
            i += 1
        else:
            j += 1

    return merged_postings



documents = {}
index = defaultdict(list)

# Check if inverted index exists
# If it does, load it
if os.path.exists('inverted_index.txt'):
    print('Loading inverted index...')
    with open('inverted_index.txt', 'r') as f:
        for line in f:
            # delete newline character
            line = line[:-1]
            token, postings = line.split(':')
            postings = postings.split(';')
            postings = [posting.split(',') for posting in postings if posting != '']
            postings = [(int(posting[0]), int(posting[1])) for posting in postings]
            index[token] = postings
# If it doesn't, create it
# and save it to a file
else:
    print('Creating inverted index...')
    # Get documents
    documents = get_documents()
    for doc_id, tokens in documents.items():
        # Iterate through each token
        for i, token in enumerate(tokens):
            # Add document to postings list for token
            index[token].append((int(doc_id), int(i)))
    print('Saving inverted index...')
    with open('inverted_index.txt', 'w') as f:
        for token, postings in index.items():
            # Write token and postings to file
            f.write(token + ':')
            for posting in postings:
                f.write(str(posting[0]) + ',' + str(posting[1]) + ';')
            f.write('\n')
    print('Done!')

def handle_phrase_query(query, index):
    print("Handling phrase query: " + query)
    # Preprocess query
    query_terms = preprocess(query)
    ans = []
    # Get postings list for each term
    postings_lists = [get_postings_list(term, index) for term in query_terms]
    
    # Merge postings lists
    # If any postings list is empty, return
    # Otherwise, merge
    for i in range(0,len(postings_lists)-1):
        if not postings_lists[i] or not postings_lists[i+1]:
            print(ans)
            return
        postings_lists[0] = merge_posting_lists(postings_lists[0], postings_lists[i+1],1)
    
    # If postings list is empty, return
    if not postings_lists[0]:
        print(ans)
        return
    
    # Get doc_ids from postings list
    ans.append(postings_lists[0][0][0])
    for i, tup in enumerate(postings_lists[0]):
        if tup[0] == ans[-1]:
            continue
        else:
            ans.append(tup[0])
    print(ans)

# Handles proximity queries
# Returns a list of doc_ids
def handle_proximity_query(query, index):
    print("Handling proximity query: " + query)
    # Split query into terms
    query_terms = re.split(r'\s+', query)
    if len(query_terms) != 3:
        print('Invalid query')
        return
    # Preprocess query
    w1 = preprocess(query_terms[0])[0]
    w2 = preprocess(query_terms[2])[0]
    k = int(query_terms[1])
    postings_lists = []

    # Get postings list for each term
    postings_lists.append(get_postings_list(w1, index))
    postings_lists.append(get_postings_list(w2, index))

    # If any postings list is empty, return []
    if not postings_lists[0] or not postings_lists[1]:
        print([])
        return
    
    # If k == 0, run w1 w2 and w2 w1
    if k == 0:
        merged_lists1 = merge_posting_lists(postings_lists[0], postings_lists[1], 1)
        merged_lists2 = merge_posting_lists(postings_lists[1], postings_lists[0], 1)

        if not merged_lists1 or not merged_lists2:
            print([])
            return
        # Merge lists
        merged_lists = merged_lists1 + merged_lists2
        # Sort merged lists by doc_id
        merged_lists = sorted(merged_lists, key=lambda x: x[0])
        ans = []
        ans.append(merged_lists[0][0])
        for i, tup in enumerate(merged_lists):
            if tup[0] == ans[-1]:
                continue
            else:
                ans.append(tup[0])
        print(ans)

    # If k > 0, run w1 num w2
    else:
        merged_lists = merge_posting_lists(postings_lists[0], postings_lists[1], k+1)
        if not merged_lists:
            print([])
            return
        ans = []
        ans.append(merged_lists[0][0])
        for i, tup in enumerate(merged_lists):
            if tup[0] == ans[-1]:
                continue
            else:
                ans.append(tup[0])
        print(ans)

# Main loop
while True:
    # Get query
    query = input("Enter query (type exit to exit):")
    if query == 'exit':
        break

    if '"' in query:
        handle_phrase_query(query, index)
    else:
        handle_proximity_query(query, index)