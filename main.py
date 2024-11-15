from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

import pandas as pd
from imblearn.over_sampling import SMOTE
import os

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.optim as optim

from utils import calculate_metric, writeTxt, encode_season

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """初始化随机森林回归模型"""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_leaf = 5
        )

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        # calculate metric
        accuracy, recall, f1, precision, cm = calculate_metric(y_test, y_pred)
        print("------------metric on the valid set for random forest-----------------------------")
        print(f"Accuracy: {accuracy:.9f}")
        print(f"Recall: {recall:.9f}")
        print(f"F1-score: {f1:.9f}")
        print(f"Precision (Macro Average): {precision:.9f}")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)


    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions


class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, random_state=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        # calculate metric
        accuracy, recall, f1, precision, cm = calculate_metric(y_test, y_pred)
        print("------------metric on the valid set for decision tress----------------------------")
        print(f"Accuracy: {accuracy:.9f}")
        print(f"Recall: {recall:.9f}")
        print(f"F1-score: {f1:.9f}")
        print(f"Precision (Macro Average): {precision:.9f}")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)

    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

class XGBoost:
    def __init__(self, num_class=3, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.num_class = num_class
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',  # 多分类
            num_class = num_class,  # 类别数
            n_estimators=self.n_estimators,  # 树的数量
            learning_rate=0.1,  # 学习率
            max_depth=self.max_depth,  # 树的最大深度
            random_state=self.random_state  # 随机种子
        )

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.model.predict(X_test)
        # calculate metric
        accuracy, recall, f1, precision, cm = calculate_metric(y_test, y_pred)
        print("------------these metric on the valid set for xgboost-----------------------------")
        print(f"Accuracy: {accuracy:.9f}")
        print(f"Recall: {recall:.9f}")
        print(f"F1-score: {f1:.9f}")
        print(f"Precision (Macro Average): {precision:.9f}")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)

    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

class SVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=0.1, coef0=0.0, random_state=42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            coef0=self.coef0,
            random_state=self.random_state
        )


    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.model.predict(X_test)
        # calculate metric
        accuracy, recall, f1, precision, cm = calculate_metric(y_test, y_pred)
        print("------------these metric on the valid set for svm-----------------------------")
        print(f"Accuracy: {accuracy:.9f}")
        print(f"Recall: {recall:.9f}")
        print(f"F1-score: {f1:.9f}")
        print(f"Precision (Macro Average): {precision:.9f}")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)

    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

class KNeighbors:
    def __init__(self, n_neighbors = 5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.model.predict(X_test)
        # calculate metric
        accuracy, recall, f1, precision, cm = calculate_metric(y_test, y_pred)
        print("------------these metric on the valid set for k-meas -----------------------------")
        print(f"Accuracy: {accuracy:.9f}")
        print(f"Recall: {recall:.9f}")
        print(f"F1-score: {f1:.9f}")
        print(f"Precision (Macro Average): {precision :.9f}")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)

    def test(self, X_test):
        predictions = self.model.predict(X_test)
        return predictions

# 读取antigraphy series data
def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 3),
            nn.ReLU(),
            nn.Linear(encoding_dim * 3, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 3),
            nn.ReLU(),
            nn.Linear(input_dim * 3, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    data_tensor = torch.FloatTensor(df_scaled)

    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i: i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]')

    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()

    df_encoded = pd.DataFrame(encoded_data, columns=[f'Enc_{i + 1}' for i in range(encoded_data.shape[1])])

    return df_encoded

if __name__ == "__main__":

    # load raw data for tabular data
    train = pd.read_csv('../dataset/train.csv')
    test = pd.read_csv('../dataset/test.csv')

    # load raw data and encode data for actigraphy data
    train_ts = load_time_series('../dataset/series_train.parquet')
    test_ts = load_time_series('../dataset/series_test.parquet')

    df_train = train_ts.drop('id', axis=1)
    df_test = test_ts.drop('id', axis=1)

    train_ts_encoded = perform_autoencoder(df_train, encoding_dim=50, epochs=200, batch_size=32)
    test_ts_encoded = perform_autoencoder(df_test, encoding_dim=50, epochs=100, batch_size=32)

    time_series_cols = train_ts_encoded.columns.tolist()

    train_ts_encoded["id"] = train_ts["id"]
    test_ts_encoded['id'] = test_ts["id"]

    # train_ts_encoded = pd.merge(train_ts_encoded, train[['id', 'sii']], on='id', how='left')
    # test_ts_encoded = pd.merge(test_ts_encoded, test[['id']], on='id', how='left')

    print("------------data preprocessing----------------------------------")
    print("raw train sample length: {} features length: {}".format(len(train), len(train.columns)))

    # 1.crop feature with many invalid value
    invalid_ratio = train.isnull().mean() * 100
    pd.options.display.max_rows = None
    pd.options.display.max_columns = None
    print("invalid value ratio (%):\n", invalid_ratio)

    train.dropna(inplace=True, subset=['sii'])
    # 2.剔除PCIAT
    columns_pciat = train.columns[train.columns.str.contains('PCIAT')]
    train = train.drop(columns = columns_pciat)

    train.dropna(axis=1, thresh=len(train) * 0.7, inplace=True)
    print("train sample length after crop: {} features: {}".format(len(train), len(train.columns)))

    invalid_ratio = train.isnull().mean() * 100
    print("invalid value ratio (%) after crop:\n", invalid_ratio)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')

    # 3.fill value with model
    data_train_df = train.apply(lambda x: x.fillna(x.mode()[0]))

    # 4.对每一个季节特征进行编码并添加到DataFrame
    season_col_names = []
    for i, season_col in enumerate(data_train_df.columns):
        if 'Season' in season_col:  # 检查列名中是否包含 'season'
            sin_values, cos_values = encode_season(data_train_df[season_col])
            # 在当前季节列后插入 sin 和 cos 列
            data_train_df.insert(i + 1, f"{season_col}_sin", sin_values)
            data_train_df.insert(i + 2, f"{season_col}_cos", cos_values)
            # 记录原季节特征名称
            season_col_names.append(season_col)

    # 5.删除原来的季节列
    data_train_df = data_train_df.drop(columns=season_col_names)
    print("train sample length after preprocess: {} features length: {} feature:{}".format(len(data_train_df), len(data_train_df.columns), data_train_df.columns))

    # 6.将encoded data与train_ts拼接
    writeTxt(data_train_df, 'data_train_df.txt')
    writeTxt(train_ts_encoded, 'train_ts_encoded.txt')
    data_train_df = data_train_df.merge(train_ts_encoded, on='id', how='inner')
    writeTxt(data_train_df, 'data_train_df_merge.txt')
    print("train sample length after concat: {} feature length: {} feature:{}".format(len(data_train_df), len(data_train_df.columns), data_train_df.columns))

    print("------------divide sample into train set and valid set------------")
    # 获取特征矩阵
    X = data_train_df.drop(columns=['id', 'sii'])  # 选择从第二列到倒数第二列

    # 获取标签向量（最后一列）
    y = data_train_df.loc[:, ['sii']]  # 选择最后一列

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #random_state=42
    print("X_train len: {} y_train len: {} X_test len: {} y_test len: {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))
    print("train label: \n", y_train.value_counts(), "\ntest label: \n", y_test.value_counts())

    #todo: print("------------select feature list-----------------------------------")
    # random forest for select feature
    # random_forest_for_select = RandomForestClassifier(n_estimators=100, max_depth=10)
    # random_forest_for_select.fit(X_train, y_train)
    # 1.get feature importances
    # importances = random_forest_for_select.feature_importances_
    # 2. choose top-n feature
    # indices = importances.argsort()[::-1]
    # n = 20  # 例如，选择前20个最重要的特征
    #top_n_features = indices[:n]
    # X_train = X_train.iloc[:, top_n_features]
    # X_test = X_test.iloc[:, top_n_features]
    features_pd = X_train

    print("------------calculate for all metric-------------------------------")
    metric_values = []
    model_names = ['random forest', 'xgboost', 'decision tree', 'svm', 'k-meas']

    # random forest
    random_forest = RandomForest(n_estimators=100, max_depth=10)

    #todo: 初始化SMOTE对象
    #smote = SMOTE(random_state=42)
    # 应用SMOTE进行重采样
    #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    # 训练模型
    random_forest.train(X_train, y_train)
    # 评估模型
    metric_value = random_forest.evaluate(X_test, y_test)
    metric_values.append(metric_value)

    # xgboost
    xgboost = XGBoost(num_class=4, n_estimators=100, max_depth=3, random_state=42)
    # 训练模型
    xgboost.train(X_train, y_train)
    # 评估模型
    metric_value = xgboost.evaluate(X_test, y_test)
    metric_values.append(metric_value)

    # decision tress
    decisiontress = DecisionTree(criterion='gini', max_depth=3, random_state=42)
    # 训练模型
    decisiontress.train(X_train, y_train)
    # 评估模型
    metric_value = decisiontress.evaluate(X_test, y_test)
    metric_values.append(metric_value)

    # svm
    svm = SVM(kernel='rbf', C=1.0, gamma=0.1, coef0=0.1, random_state=42)

    # 数据标准化（SVM 对数据分布比较敏感，推荐进行标准化）
    scaler = StandardScaler()
    X_train_standard = scaler.fit_transform(X_train)
    X_test_standard = scaler.transform(X_test)

    # 训练模型
    svm.train(X_train_standard, y_train)
    # 评估模型
    metric_value = svm.evaluate(X_test_standard, y_test)
    metric_values.append(metric_value)

    # k-nearsetNeighb
    kmeas = KNeighbors(n_neighbors=4)
    # 训练模型
    kmeas.train(X_train_standard, y_train)
    # 评估模型
    metric_value = kmeas.evaluate(X_test_standard, y_test)
    metric_values.append(metric_value)

    # make static table
    metric_data = {'metric': ['accurary', 'recall', 'f1', 'precision']}
    for i, model_name in enumerate(model_names):
        metric_data[model_name] = metric_values[i]

    # 创建DataFrame
    metric_data_df = pd.DataFrame(metric_data)

    # 导出为Excel文件
    excel_path = 'model_evaluation_and_features.xlsx'
    # 使用ExcelWriter对象写入两个工作表
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        metric_data_df.to_excel(writer, sheet_name='Model Evaluation', index=False)
        features_pd.to_excel(writer, sheet_name='Features', index=False)
    print(f"Excel文件已生成并保存到 {excel_path}")

