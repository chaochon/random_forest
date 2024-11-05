from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class Predictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """初始化随机森林回归模型"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )



    def load_data(self):
        """加载房价数据并拆分为训练集和测试集"""
        data = fetch_california_housing()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.3, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions


# 定义一个函数来进行正弦和余弦编码
def encode_season(season_column):
    # 定义季节
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    # 定义周期长度（因为季节有4个，所以周期为4）
    period = len(seasons)
    sin_values = season_column.apply(lambda x: np.sin(2 * np.pi * seasons.index(x) / period)).round(7)
    cos_values = season_column.apply(lambda x: np.cos(2 * np.pi * seasons.index(x) / period)).round(7)
    return sin_values, cos_values


if __name__ == "__main__":

    # load raw data
    data_train_df = pd.read_csv('../dataset/train.csv')
    data_test_df = pd.read_csv('../dataset/test.csv')

    print("------------data preprocessing----------------------------------")
    print("raw train sample length: {} features: {}".format(len(data_train_df), len(data_train_df.columns)))

    # 1.crop feature with many invalid value
    invalid_ratio = data_train_df.isnull().mean() * 100
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    print("invalid value ratio (%):\n", invalid_ratio)

    data_train_df.dropna(inplace=True, subset=['sii'])
    data_train_df.dropna(axis=1, thresh=len(data_train_df) * 0.70, inplace=True)
    print("train sample length after crop: {} features: {}".format(len(data_train_df), len(data_train_df.columns)))

    invalid_ratio = data_train_df.isnull().mean() * 100
    print("invalid value ratio (%) after crop:\n", invalid_ratio)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

    # 2.fill value with mode
    data_train_df = data_train_df.apply(lambda x: x.fillna(x.mode()[0]))

    # 3.对每一个季节特征进行编码并添加到DataFrame
    season_col_names = []
    for i, season_col in enumerate(data_train_df.columns):
        if 'Season' in season_col:  # 检查列名中是否包含 'season'
            sin_values, cos_values = encode_season(data_train_df[season_col])
            # 在当前季节列后插入 sin 和 cos 列
            data_train_df.insert(i + 1, f"{season_col}_sin", sin_values)
            data_train_df.insert(i + 2, f"{season_col}_cos", cos_values)
            # 记录原季节特征名称
            season_col_names.append(season_col)

    # 删除原来的季节列
    data_train_df = data_train_df.drop(columns=season_col_names)

    #-------------learn about data for debug------------
    # write into file for read
    # with open('output_train_clear2.txt', 'w') as f:
    #     f.write(data_train_df.to_string(index=False))

    print("train sample length after convert season: {} features: {}".format(len(data_train_df), len(data_train_df.columns)))

    print("------------divide sample into train set and valid set------------")
    # 获取特征矩阵（从第二列到倒数第二列）
    X = data_train_df.iloc[:, 1:-1]  # 选择从第二列到倒数第二列

    # 获取标签向量（最后一列）
    y = data_train_df.iloc[:, -1]  # 选择最后一列

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #random_state=42
    print("X_train len: {} y_train len: {} X_test len: {} y_test len: {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))
    print("train label: \n", y_train.value_counts(), "\ntest label: \n", y_test.value_counts())

    # random forest
    predictor = Predictor(n_estimators=100, max_depth=10)
    predictor.train(X_train, y_train)

    # 计算指标
    y_pred = predictor.test(X_test)
    y_pred, y_test = y_pred.astype(int), y_test.astype(int)

    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print("------------metric on the valid set-----------------------------")
    print(f"Accuracy: {accuracy:.9f}")
    print(f"Recall: {recall:.9f}")
    print(f"F1-score: {f1:.9f}")
    print('confusion matrix:\n', cm)

    # decision tress

    # xgboost

    # svm

    # k-nearsetNeighb


