import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sortedcontainers import SortedList
from collections import Counter
from datetime import datetime


class KNN:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = [None]*len(X)
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xt in enumerate(self.X):
                diff = x - xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))
            y[i] = Counter([b for a, b in sl]).most_common(1)[0][0]
        return y

    def score(self, X, y):
        p = self.predict(X)
        return np.mean(p == y)


def main():
    df = sns.load_dataset('iris')
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                        y.values,
                                                        test_size=0.3)

    for k in range(1, 20):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(X_train, y_train)
        print(f"K = {k}")
        print(f"Training time: {datetime.now() - t0}")

        t0 = datetime.now()
        print("Training accuracy:", knn.score(X_train, y_train))
        print(f"Time to compute training accuracy: {datetime.now() - t0}")

        t0 = datetime.now()
        print("Testing accuracy:", knn.score(X_test, y_test))
        print(f"Time to compute testing accuracy: {datetime.now() - t0}")


if __name__ == "__main__":
    main()
