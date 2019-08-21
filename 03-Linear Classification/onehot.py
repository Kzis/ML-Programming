from sklearn.preprocessing import LabelEncoder as LE, OneHotEncoder as OHE
import numpy as np

a = np.array([[0, 1, 100], [1, 2, 200], [2, 3, 400]])

oh = OHE(categorical_features=[0, 1])
a = oh.fit_transform(a).toarray()

a[:, 1:]

idx_to_delete = [0, 3]
indices = [i for i in range(a.shape[-1]) if i not in idx_to_delete]
a[:, indices]
