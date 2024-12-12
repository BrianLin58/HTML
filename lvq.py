import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import tqdm

def lvq_train(X, y, learning_rate, decay_rate, max_epochs, min_learning_rate, margin):
    unique_classes, train_idx = np.unique(y, return_index=True)
    W = X[train_idx].astype(np.float64)
    classes = unique_classes

    for epoch in range(max_epochs):
        if learning_rate < min_learning_rate:
            break

        for i, x in enumerate(X):
            distances = np.linalg.norm(W - x, axis=1)
            min_1 = np.argmin(distances)
            dc = distances[min_1]
            min_2 = np.argsort(distances)[1]
            dr = distances[min_2]

            if y[i] == classes[min_1]:
                W[min_1] += learning_rate * (x - W[min_1])
            elif y[i] == classes[min_2] and dc != 0 and dr != 0:
                margin_condition = min(dc / dr, dr / dc)
                if margin_condition > (1 - margin) / (1 + margin):
                    W[min_1] -= learning_rate * (x - W[min_1])
                    W[min_2] += learning_rate * (x - W[min_2])
            else:
                W[min_1] += margin * learning_rate * (x - W[min_1])
                W[min_2] += margin * learning_rate * (x - W[min_2])

        learning_rate *= decay_rate

    return W, classes

def lvq_test(x, W):
    weights, classes = W
    # Ensure input x matches the dimensionality of weights
    x = x.reshape(weights.shape[1])  # Ensure `x` matches the feature dimension
    distances = np.linalg.norm(weights - x, axis=1)
    return classes[np.argmin(distances)]

def evaluate_model(W, X, y):
    predictions = [lvq_test(x, W) for x in X]
    return accuracy_score(y, predictions)

def print_metrics(labels, preds):
    print(f"Precision Score: {precision_score(labels, preds, average='weighted'):.4f}")
    print(f"Recall Score: {recall_score(labels, preds, average='weighted'):.4f}")
    print(f"Accuracy Score: {accuracy_score(labels, preds):.4f}")
    #print(f"F1 Score: {f1_score(labels, preds, average='weighted'):.4f}")

if __name__ == "__main__":
    train_file = "preprocess_4.csv"
    test_file = "preprocess_test_2.csv"

    # Load training data
    train_df = pd.read_csv(train_file)
    train_data = train_df.to_numpy()
    y_train_full = np.array([1 if str(yi).strip().upper() == "TRUE" else -1 for yi in train_data[:, 5]])
    train_data = np.delete(train_data, [0,3,4,5,8,9,10,11,12,39,40], axis=1)
    X_train_full = np.nan_to_num(train_data.astype(np.float32), nan=0.0)


    # Load test data
    test_df = pd.read_csv(test_file)
    test_data = test_df.to_numpy()
    #y_test = np.array([1 if str(yi).strip().upper() == "TRUE" else -1 for yi in test_data[:, 5]])
    test_data = np.delete(test_data, [0,3,6,7,8,9,10,37,48], axis=1)
    X_test = np.nan_to_num(test_data.astype(np.float32), nan=0.0)

    print("Feature Shape Before Reshaping:", X_train_full.shape, X_test.shape)
    feature_dim = X_train_full.shape[1]
    X_train_full = np.reshape(X_train_full, (-1, feature_dim))
    X_test = np.reshape(X_test, (-1, feature_dim))
    print("Feature Shape After Reshaping:", X_train_full.shape, X_test.shape)

    # Hyperparameters
    max_iterations = 10000
    learning_rate = 0.2
    decay_rate = 0.5
    max_epochs = 100
    min_learning_rate = 0.001
    margin = 0.3

    best_accuracy = 0
    best_weights = None

    # Train multiple models and select the best one
    for i in tqdm.trange(max_iterations):
        # Re-split the dataset into train and validation sets for each iteration
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=None)
        
        W = lvq_train(X_train, y_train, learning_rate, decay_rate, max_epochs, min_learning_rate, margin)
        accuracy = evaluate_model(W, X_val, y_val)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = W

        if i % 100 == 0:  # Log progress every 100 iterations
            print(f"Iteration {i}, Current Accuracy: {accuracy:.4f}, Best Accuracy: {best_accuracy:.4f}")

    print(f"Best Model Accuracy on Validation Set: {best_accuracy:.4f}")
    # print(best_weights)
    # Evaluate the best model on the test set
    y_pred = [lvq_test(x, best_weights) for x in X_test]
    # print(y_pred)

    result = pd.read_csv('same_season_sample_submission.csv')
    for idx, row in result.iterrows():
        result.at[idx, "home_team_win"] = f'True' if y_pred[idx] == 1 else f'False'
    result.to_csv("lvq.csv", index = False)
    #print_metrics(y_test, y_pred)
