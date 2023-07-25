import re
import html
import numpy as np
import random

# tokenize text
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower()) # return list of tokens

# load stopwords from file
def load_stopwords(stopword_file):
    with open(stopword_file, 'r', encoding='latin-1') as file:
        stopwords = file.readlines()
        stopwords = [word.strip() for word in stopwords]
    return set(stopwords)

# load file and unescape html
def load_file(filename):
    with open(filename, 'r', encoding='latin-1') as file:
        data = file.read()
        data = html.unescape(data) # unescape html
    return data

# remove stopwords from file
def remove_stopwords(words, stopwords):
    cleaned_words = [word.lower() for word in words if word.lower() not in stopwords] 
    return cleaned_words

# extract text from file and return list of tuples (text, topics, lewis_split) where text is a list of tokens, topics is a list of topics, and lewis_split is the lewis split
def extract_text(data, stopwords): 
    text_pattern = r'<REUTERS(.*?)</REUTERS>' # match text between <REUTERS> tags
    title_pattern = r'<TITLE>(.*?)</TITLE>' # match text between <TITLE> tags
    body_pattern = r'<BODY>(.*?)</BODY>' # match text between <BODY> tags
    text_matches = re.findall(text_pattern, data, re.DOTALL | re.IGNORECASE) # find all matches which extracts the articles

    articles = []
    for text in text_matches:
        title_match = re.search(title_pattern, text, re.DOTALL | re.IGNORECASE) # find title
        body_match = re.search(body_pattern, text, re.DOTALL | re.IGNORECASE) # find body
        title = title_match.group(1) if title_match else '' # get title text
        body = body_match.group(1) if body_match else '' # get body text
        lewis_split = re.search(r'LEWISSPLIT="(.*?)"', text, re.IGNORECASE) # get lewis split
        topic_bool = re.search(r'TOPICS="(.*?)"', text, re.IGNORECASE).group(1) # get whether topics exist or not
        topics = []
        if(topic_bool == 'YES'): # if topics exist
            topic_matches = re.findall(r'<TOPICS>(.*?)</TOPICS>', text, re.DOTALL | re.IGNORECASE) # find all matches which extracts the topics
            for topic_set in topic_matches: # for each topic set
                topic_list = re.findall(r'<D>(.*?)</D>', topic_set, re.DOTALL | re.IGNORECASE) # find all topics
                topics.extend(topic_list)
            text_tokens = tokenize(title + ' ' + body) # tokenize title and body
            text_tokens = remove_stopwords(text_tokens, stopwords) # remove stopwords from title and body
        articles.append((text_tokens, topics, lewis_split.group(1) if lewis_split else None)) # append tuple of text, topics, and lewis split as article
    return articles

# get the most common n topics
def get_top_n_topics(articles, n=10):
    topic_count = {}
    for _, topics, _ in articles:
        for topic in topics:
            topic_count[topic] = topic_count.get(topic, 0) + 1
    return [topic for topic, _ in sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:n]]

# filter articles where at least one topic of the article is in the top n topics
def filter_articles_by_topics(articles, top_topics):
    return [article for article in articles if any(topic in article[1] for topic in top_topics)]

# train naive bayes model and return class probabilities and word probabilities
def train_naive_bayes(articles, alpha, model_type, top_classes):
    class_counts = {c: 0 for c in top_classes} # count of articles in each class
    class_word_counts = {c: {} for c in top_classes} # count of each word in each class
    vocab = set() # set of unique words in all articles

    for tokens, topics in articles: # for each article
        relevant_topics = [topic for topic in topics if topic in top_classes] # get topics that are in the top classes
        for topic in relevant_topics: # for each topic 
            class_counts[topic] += 1 # increment class count
            unique_tokens = set(tokens) if model_type == 'bernoulli' else tokens # get unique tokens if bernoulli model
            for token in unique_tokens: # for each token
                vocab.add(token) # add token to vocab
                class_word_counts[topic][token] = class_word_counts[topic].get(token, 0) + 1 # increment token count for class

    vocab_size = len(vocab) # size of vocab
    class_probs = {c: count / len(articles) for c, count in class_counts.items()} # probability of each class
    class_word_probs = calculate_word_probs(top_classes, class_word_counts, vocab, vocab_size, alpha, model_type) # probability of each word in each class
    return class_probs, class_word_probs 


# calculate word probabilities
def calculate_word_probs(top_classes, class_word_counts, vocab, vocab_size, alpha, model_type):
    class_word_probs = {c: {} for c in top_classes} # probability of each word in each class
    for c in top_classes: # for each class
        class_total_word_count = sum(class_word_counts[c].values()) # total number of words in class
        for token in vocab: # for each token
            token_count = class_word_counts[c].get(token, 0) if model_type == 'multinomial' else int(class_word_counts[c].get(token, 0) > 0) # get token count if multinomial model, otherwise get 1 if token exists in class
            class_word_probs[c][token] = (token_count + alpha) / (class_total_word_count + alpha * vocab_size) # calculate probability of token in class
    return class_word_probs

# predict class using naive bayes model
def predict_naive_bayes(tokens, class_probs, class_word_probs):
    log_probs = {c: np.log(prob) for c, prob in class_probs.items()} # log probability of each class
    for token in tokens: # for each token
        for c in class_probs: # for each class
            log_probs[c] += np.log(class_word_probs[c].get(token, 1e-10)) # add log probability of token in class
    return max(log_probs, key=log_probs.get)

# calculate metrics
def calculate_metrics(labels, predictions):
    unique_classes = set(labels) # unique classes
    metrics = {c: {'tp': 0, 'fp': 0, 'fn': 0} for c in unique_classes} # true positives, false positives, and false negatives for each class

    for true_label, pred_label in zip(labels, predictions): # for each true label and predicted label
        if true_label == pred_label: # if true label and predicted label are the same
            metrics[true_label]['tp'] += 1 # increment true positive
        else:
            metrics[pred_label]['fp'] += 1 # increment false positive
            metrics[true_label]['fn'] += 1 # increment false negative

    precision_recall = {
        c: {
            'precision': metrics[c]['tp'] / (metrics[c]['tp'] + metrics[c]['fp'] + 1e-6),
            'recall': metrics[c]['tp'] / (metrics[c]['tp'] + metrics[c]['fn'] + 1e-6)
        } for c in unique_classes
    } # precision and recall for each class

    macro_precision = sum([v['precision'] for v in precision_recall.values()]) / len(unique_classes) # macro precision
    macro_recall = sum([v['recall'] for v in precision_recall.values()]) / len(unique_classes) # macro recall
    macro_f_score = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) # macro f score

    total_tp = sum([m['tp'] for m in metrics.values()]) # total true positives
    total_fp = sum([m['fp'] for m in metrics.values()]) # total false positives
    total_fn = sum([m['fn'] for m in metrics.values()]) # total false negatives
    
    micro_precision = total_tp / (total_tp + total_fp) # micro precision
    micro_recall = total_tp / (total_tp + total_fn) # micro recall
    micro_f_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)   # micro f score

    return macro_precision, macro_recall, macro_f_score, micro_precision, micro_recall, micro_f_score

# load data
data = ''
for i in range(22):
    name = 'reut2-0' + str(i).zfill(2) + '.sgm'
    data += load_file('reuters21578/' + name)  

stopword_file = 'stopwords.txt' # stopwords file

stopwords = load_stopwords(stopword_file) # load stopwords

# get the articles and their topics
articles = extract_text(data, stopwords) # get articles and their topics
train_articles = [article for article in articles if article[2] == "TRAIN"] # get training articles
dev_articles, train_articles = train_articles[:len(train_articles)//2], train_articles[len(train_articles)//2:] # split training articles into training and dev articles
test_articles = [article for article in articles if article[2] == "TEST"] # get test articles

top_topics = get_top_n_topics(articles) # get top 10 topics
train_articles = filter_articles_by_topics(train_articles, top_topics) # filter training articles by top 10 topics
train_articles = [(tokens, topics) for (tokens, topics, _) in train_articles] # get tokens and topics for training articles

dev_articles = filter_articles_by_topics(dev_articles, top_topics) # filter dev articles by top 10 topics
dev_articles = [(tokens, topics) for (tokens, topics, _) in dev_articles] # get tokens and topics for dev articles

test_articles = filter_articles_by_topics(test_articles, top_topics) # filter test articles by top 10 topics
test_articles = [(tokens, topics) for (tokens, topics, _) in test_articles] # get tokens and topics for test articles

alpha_values = [1, 0.5, 2] # alpha values
model_types = ['multinomial', 'bernoulli'] # model types
 
train_labels = [next(topic for topic in topics if topic in top_topics) for _, topics in train_articles] # get training labels
dev_labels = [next(topic for topic in topics if topic in top_topics) for _, topics in dev_articles] # get dev labels
test_labels = [next(topic for topic in topics if topic in top_topics) for _, topics in test_articles] # get test labels

# Count the number of documents for each of the top 10 classes in the training and test sets
train_class_counts = {topic: 0 for topic in top_topics} 
dev_class_counts = {topic: 0 for topic in top_topics}
test_class_counts = {topic: 0 for topic in top_topics}


# Count the number of documents with more than one of the top 10 classes in the training and test sets
# Below part is commented since it is not used for the training and testing of the model. It is used for the calculation for the parameters needed in the report.

# train_multiple_labels_count = 0
# dev_multiple_labels_count = 0
# test_multiple_labels_count = 0

# for _, topics in train_articles:
#     top_topics_count = 0
#     for topic in topics:
#         if topic in top_topics:
#             train_class_counts[topic] += 1
#             top_topics_count += 1
#     if top_topics_count > 1:
#         train_multiple_labels_count += 1

# for _, topics in test_articles:
#     top_topics_count = 0
#     for topic in topics:
#         if topic in top_topics:
#             test_class_counts[topic] += 1
#             top_topics_count += 1
#     if top_topics_count > 1:
#         test_multiple_labels_count += 1

# for _, topics in dev_articles:
#     top_topics_count = 0
#     for topic in topics:
#         if topic in top_topics:
#             dev_class_counts[topic] += 1
#             top_topics_count += 1
#     if top_topics_count > 1:
#         dev_multiple_labels_count += 1

# print("Top 10 classes and the number of documents in the training and test sets:")
# for topic in top_topics:
#     print(f"{topic}: Training - {train_class_counts[topic]}, Test - {test_class_counts[topic]}, Dev - {dev_class_counts[topic]}")

# print("\nTotal number of documents in the training set:", len(train_articles))
# print("\nTotal number of documents in the dev set:", len(dev_articles))
# print("Total number of documents in the test set:", len(test_articles))
# print("Number of documents with more than one of the top 10 classes in the training set:", train_multiple_labels_count)
# print("Number of documents with more than one of the top 10 classes in the dev set:", dev_multiple_labels_count)
# print("Number of documents with more than one of the top 10 classes in the test set:", test_multiple_labels_count)


preds = {} # predictions stored for the future use for randomization test 
for model_type in model_types:
    best_macro_f_score = -1 # best macro f score
    best_alpha_for_macro_f_score = -1  # best alpha for macro f score

    for alpha in alpha_values:
        class_probs, class_priors = train_naive_bayes(dev_articles, alpha, model_type, top_topics) # train the model
        dev_predictions = [predict_naive_bayes(tokens, class_probs, class_priors) for tokens, _ in dev_articles] # predict the dev set
        macro_precision, macro_recall, macro_f_score, micro_precision, micro_recall, micro_f_score = calculate_metrics(dev_labels, dev_predictions) # calculate the metrics
        print("development f1 scores, for model ", model_type, ": ", macro_f_score, micro_f_score) # print the metrics

        if macro_f_score > best_macro_f_score: # if the macro f score is better than the previous best, update the best macro f score and the best alpha
            best_macro_f_score = macro_f_score
            best_alpha_for_macro_f_score = alpha
    
    print("best alpha for macro f score: ", best_alpha_for_macro_f_score) # print the best alpha for the model

    # Merge the train and dev sets to train on the entire training set
    combined_train_articles = train_articles + dev_articles 
    merged_train_labels = train_labels + dev_labels

    # Train the model on the entire training set with the best alpha
    final_class_probs, final_class_priors = train_naive_bayes(combined_train_articles, best_alpha_for_macro_f_score, model_type, top_topics)
    print("Vocabulary size:", len(final_class_priors[top_topics[0]]))

    # Make the prediction and calculate the metrics on the training set and the test set
    combined_predictions = [predict_naive_bayes(tokens, final_class_probs, final_class_priors) for tokens, _ in combined_train_articles]
    macro_precision, macro_recall, macro_f_score, micro_precision, micro_recall, micro_f_score = calculate_metrics(merged_train_labels, combined_predictions)
    print("train f1 score at final, for model ", model_type, ": ", macro_f_score, micro_f_score)

    # Make the prediction and calculate the metrics on the test set by using the trained model on the entire training set
    test_predictions = [predict_naive_bayes(tokens, final_class_probs, final_class_priors) for tokens, _ in test_articles]
    macro_precision, macro_recall, macro_f_score, micro_precision, micro_recall, micro_f_score = calculate_metrics(test_labels, test_predictions)
    preds[model_type] = test_predictions

    print(model_type, "Naive Bayes Classifier Test Results")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F-score: {macro_f_score:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")
    print(f"Micro F-score: {micro_f_score:.4f}")
    print("\n")

# Randomization test
def randomization_test(test_labels, model1_preds, model2_preds, num_permutations=1000):
    macro_f1_diff = abs(calculate_metrics(test_labels, model1_preds)[2] - calculate_metrics(test_labels, model2_preds)[2]) # calculate the difference between the macro f scores of the two models
    count = 0

    for _ in range(num_permutations):
        swapped_preds = [(p1 if random.random() < 0.5 else p2, p2 if random.random() < 0.5 else p1) for p1, p2 in zip(model1_preds, model2_preds)] # swap the predictions of the two models
        model1_swapped_preds, model2_swapped_preds = zip(*swapped_preds) # unzip the swapped predictions
        macro_f1_diff_swapped = abs(calculate_metrics(test_labels, model1_swapped_preds)[2] - calculate_metrics(test_labels, model2_swapped_preds)[2]) # calculate the difference between the macro f scores of the two models with swapped predictions
        if macro_f1_diff_swapped >= macro_f1_diff: # if the difference between the macro f scores of the two models with swapped predictions is greater than or equal to the difference between the macro f scores of the two models, increment the count
            count += 1

    p_value = count / num_permutations # calculate the p value
    return p_value

p_value = randomization_test(test_labels, preds['multinomial'], preds['bernoulli'])
print(f"Randomization Test P-value: {p_value:.4f}")

