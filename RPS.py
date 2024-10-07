import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def player(prev_play, opponent_history=[]):
    if prev_play != "":
        opponent_history.append(prev_play)

    # Default guess
    guess = "R"

    if len(opponent_history) > 10:
        x, y = prepare_data(opponent_history)
        if len(x) >= 3:  # Need at least 3 data to train KNN 
            knn_model = KNeighborsClassifier(n_neighbors=min(20, len(x)), weights='distance', p=1, algorithm='auto')
            knn_model.fit(x, y)
            # Make a prediction using the trained KNN model
            guess = predict_next_move(knn_model, opponent_history)
    elif len(opponent_history) > 2:
        # Use frequency analysis if not enough data for KNN
        guess = counter_strategy(max_frequency(opponent_history))
    return guess

# Frequency analysis
def frequency(opp_history):
    opp_freq = {"R": 0, "P": 0, "S": 0}
    for play in opp_history:
        opp_freq[play] += 1
    return opp_freq

def max_frequency(opp_history):
    opp_freq = frequency(opp_history)
    max_play = max(opp_freq, key=opp_freq.get)
    return max_play

def counter_strategy(play):
    if play == "R":
        return "P"
    if play == "P":
        return "S"
    if play == "S":
        return "R"

def prepare_data(opponent_history):
    mapping = {'R': 0, 'P': 1, 'S': 2}
    x = []
    y = []

    # Create sliding windows of length 10
    for i in range(len(opponent_history) - 10):
        x.append([mapping[move] for move in opponent_history[i:i+10]])
        y.append(mapping[opponent_history[i+10]])

    return np.array(x), np.array(y)

# Predict the next move using the KNN model
def predict_next_move(knn_model, opponent_history):
    mapping = {'R': 0, 'P': 1, 'S': 2}
    reverse_mapping = {0: 'R', 1: 'P', 2: 'S'}
    last_moves = np.array([[mapping[move] for move in opponent_history[-10:]]])
    prediction = knn_model.predict(last_moves)
    predicted_move = reverse_mapping[prediction[0]]
    return counter_strategy(predicted_move)