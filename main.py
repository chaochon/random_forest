from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class HousingPricePredictor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """初始化随机森林回归模型"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            verbose=2
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
    data_train_df = pd.read_csv('./dataset/train.csv')
    data_test_df = pd.read_csv('./dataset/test.csv')

    # crop feature
    print("data_train length: {} {}".format(len(data_train_df), len(data_train_df.columns)))
    data_train_df.dropna(axis=1, thresh=len(data_train_df) * 0.5, inplace=True)
    print("data_train length after 50% dropout: {} {}".format(len(data_train_df), len(data_train_df.columns)))

    # fill value with mode
    data_train_df = data_train_df.apply(lambda x: x.fillna(x.mode()[0]))

    # 对每一个季节特征进行编码并添加到DataFrame
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

    print("data_train length after convert season features: {} {}".format(len(data_train_df), len(data_train_df.columns)))

    # random forest
    predictor = HousingPricePredictor(n_estimators=100, max_depth=10)
    # 获取特征矩阵（从第二列到倒数第二列）
    X = data_train_df.iloc[:, 1:-1]  # 选择从第二列到倒数第二列

    # 获取标签向量（最后一列）
    y = data_train_df.iloc[:, -1]  # 选择最后一列

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.4, random_state=42)

    predictor.train(X_train, y_train)

    # 计算指标
    y_pred = predictor.test(X_test)
    y_pred, y_test = y_pred.astype(int), y_test.astype(int)

    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='micro')
    # 输出结果
    print(f"Accuracy: {accuracy:.9f}")
    print(f"Recall: {recall:.9f}")
    print(f"F1-score: {f1:.9f}")
    # decision tress

    # xgboost

    # svm

    # k-nearsetNeighb


