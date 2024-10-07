# Rock Paper Scissors

This is the boilerplate for the Rock Paper Scissors project. Instructions for building your project can be found at https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors

# Performance

```plaintext
python test_module.py

Testing game against abbey...
Final results: {'p1': 474, 'p2': 308, 'tie': 218}
Player 1 win rate: 60.61381074168798%
.Testing game against kris...
Final results: {'p1': 815, 'p2': 139, 'tie': 46}
Player 1 win rate: 85.42976939203353%
.Testing game against mrugesh...
Final results: {'p1': 847, 'p2': 94, 'tie': 59}
Player 1 win rate: 90.0106269925611%
.Testing game against quincy...
Final results: {'p1': 988, 'p2': 7, 'tie': 5}
Player 1 win rate: 99.2964824120603%
.
----------------------------------------------------------------------
Ran 4 tests in 20.507s

OK
```

# Method

The provided code implements a strategy for playing Rock-Paper-Scissors using a combination of frequency analysis and a K-Nearest Neighbors (KNN) machine learning model. Initially, the code tracks the opponent's previous moves and uses frequency analysis to predict the next move if there are fewer than 10 historical moves. (It's basically useless but better than just choosing a default response I guess). Once enough data is collected, the code prepares the data by creating sliding windows of the last 10 moves and their subsequent move. This data is then used to train a KNN model, which predicts the opponent's next move based on the most similar historical sequences. The predicted move is countered to decide the player's next move, aiming to maximize the chances of winning.
