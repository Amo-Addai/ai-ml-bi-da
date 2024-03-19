
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


'''
Supervised Learning: Random Forest Classifier
Unsupervised Learning: KMeans Clustering
Reinforcement Learning: None (scikit-learn does not have built-in support for reinforcement learning)
Semi-supervised Learning: LabelPropagation
Ensemble Learning: VotingClassifier
'''


# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Supervised Learning: Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Classifier Accuracy:", rf_accuracy)

# 2. Unsupervised Learning: KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
kmeans_pred = kmeans.predict(X_test)
# Since KMeans is unsupervised, we cannot calculate accuracy

# 4. Semi-supervised Learning: LabelPropagation
label_propagation = LabelPropagation()
label_propagation.fit(X_train, y_train)
lp_pred = label_propagation.predict(X_test)
lp_accuracy = accuracy_score(y_test, lp_pred)
print("LabelPropagation Accuracy:", lp_accuracy)

# 5. Ensemble Learning: VotingClassifier (Combining RandomForest and KMeans)
voting_clf = VotingClassifier([('rf', rf_clf), ('kmeans', kmeans)], voting='hard')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_pred)
print("Voting Classifier Accuracy:", voting_accuracy)


'''
Explanation:

Random Forest Classifier: A supervised learning model that constructs multiple decision trees during training and outputs the class that

'''


# todo: Supervised Learning: Support Vector Machine (SVM)

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train SVM model
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)

# Evaluate model
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Accuracy:", svm_accuracy)


# todo: Unsupervised Learning: K-Means Clustering

# Load iris dataset
X, _ = load_iris(return_X_y=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)

# Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# todo: Reinforcement Learning: Q-Learning

# Define environment (e.g., a simple grid world)
grid_world = np.array([
    [0, 0, 0, 1],
    [0, 1, 0, -1],
    [0, 0, 0, 0]
])

# Define Q-learning algorithm
def q_learning(env, learning_rate=0.1, discount_factor=0.9, num_episodes=1000):
    q_table = np.zeros_like(env)
    for _ in range(num_episodes):
        state = (0, 0)
        while True:
            action = np.random.choice([0, 1, 2, 3])  # Up, Down, Left, Right
            next_state = take_action(state, action)
            reward = env[next_state]
            q_table[state][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])
            state = next_state
            if reward != 0:
                break
    return q_table

# Define function to take action based on current state and action
def take_action(state, action):
    if action == 0:  # Up
        return (state[0] - 1, state[1])
    elif action == 1:  # Down
        return (state[0] + 1, state[1])
    elif action == 2:  # Left
        return (state[0], state[1] - 1)
    elif action == 3:  # Right
        return (state[0], state[1] + 1)

# Run Q-learning algorithm
q_table = q_learning(grid_world)
print("Q-Table:")
print(q_table)


# todo: Ensemble Learning: Random Forest

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Accuracy:", rf_accuracy)


# todo: Semi-supervised Learning: Label Propagation

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Introduce semi-supervised scenario by keeping only 10% of labels
y_labeled, y_unlabeled = train_test_split(y, test_size=0.9, random_state=42)

# Use Label Propagation to propagate labels from labeled to unlabeled instances
lp_model = LabelPropagation()
lp_model.fit(X, y_labeled)

# Predict labels for unlabeled instances
y_pred_unlabeled = lp_model.predict(X)

# Evaluate model using labeled and unlabeled data
accuracy_labeled = accuracy_score(y_labeled, lp_model.transduction_)
accuracy_unlabeled = accuracy_score(y_unlabeled, y_pred_unlabeled)
print("Label Propagation Accuracy (Labeled Data):", accuracy_labeled)
print("Label Propagation Accuracy (Unlabeled Data):", accuracy_unlabeled)


# todo: Supervised Learning: Gradient Boosting Machine (GBM)

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train GBM model
gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train, y_train)

# Evaluate model
gbm_predictions = gbm_model.predict(X_test)
gbm_accuracy = accuracy_score(y_test, gbm_predictions)
print("GBM Accuracy:", gbm_accuracy)


# todo: Unsupervised Learning: DBSCAN

# Load iris dataset
X, _ = load_iris(return_X_y=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

# Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan.labels_, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


# todo: Ensemble Learning: Voting Classifier

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create individual classifiers
log_reg = LogisticRegression()
dec_tree = DecisionTreeClassifier()
svm_clf = SVC()

# Create Voting Classifier
voting_clf = VotingClassifier(estimators=[('lr', log_reg), ('dt', dec_tree), ('svc', svm_clf)], voting='hard')

# Train Voting Classifier
voting_clf.fit(X_train, y_train)

# Evaluate model
voting_predictions = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_predictions)
print("Voting Classifier Accuracy:", voting_accuracy)


# todo: Online Learning: Passive Aggressive Classifier

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Passive Aggressive Classifier
pa_clf = PassiveAggressiveClassifier(max_iter=100, random_state=42)
pa_clf.fit(X_train, y_train)

# Evaluate model
pa_predictions = pa_clf.predict(X_test)
pa_accuracy = accuracy_score(y_test, pa_predictions)
print("Passive Aggressive Classifier Accuracy:", pa_accuracy)


# todo: Bayesian Learning: Gaussian Naive Bayes

# Load iris dataset
X, y = load_iris(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Gaussian Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)

# Evaluate model
gnb_predictions = gnb_model.predict(X_test)
gnb_accuracy = accuracy_score(y_test, gnb_predictions)
print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)

