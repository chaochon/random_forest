from nltk import precision
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
import pandas as pd


class RandomForest:
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

    # def load_data(self):
    #     """加载房价数据并拆分为训练集和测试集"""
    #     data = fetch_california_housing()
    #     X_train, X_test, y_train, y_test = train_test_split(
    #         data.data, data.target, test_size=0.3, random_state=self.random_state
    #     )
    #     return X_train, X_test, y_train, y_test

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
        print(f"Precision (Macro Average): {precision * 100:.9f}%")
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
        print(f"Precision (Macro Average): {precision * 100:.9f}%")
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
        print(f"Precision (Macro Average): {precision * 100:.9f}%")
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
        print(f"Precision (Macro Average): {precision * 100:.9f}%")
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
        print(f"Precision (Macro Average): {precision * 100:.9f}%")
        print('confusion matrix:\n', cm)
        return (accuracy, recall, f1, precision)

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

# 计算评价指标函数
def calculate_metric(y_test, y_pred):
    # 计算评价指标
    y_pred, y_test = y_pred.astype(int), y_test.astype(int)

    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, recall, f1, precision, cm

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

    # 4.删除原来的季节列
    data_train_df = data_train_df.drop(columns=season_col_names)

    print("train sample length after convert season: {} features: {}".format(len(data_train_df), len(data_train_df.columns)))

    print("------------divide sample into train set and valid set------------")
    # 获取特征矩阵（从第二列到倒数第二列）
    X = data_train_df.iloc[:, 1:-1]  # 选择从第二列到倒数第二列
    features_pd = X

    # 获取标签向量（最后一列）
    y = data_train_df.iloc[:, -1]  # 选择最后一列

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #random_state=42
    print("X_train len: {} y_train len: {} X_test len: {} y_test len: {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))
    print("train label: \n", y_train.value_counts(), "\ntest label: \n", y_test.value_counts())

    # all metric
    metric_values = []
    model_names = ['random forest', 'xgboost', 'decision tree', 'svm', 'k-meas']

    # random forest
    random_forest = RandomForest(n_estimators=100, max_depth=10)
    # 训练模型
    random_forest.train(X_train, y_train)
    # 评估模型
    metric_value = random_forest.evaluate(X_test, y_test)
    metric_values.append(metric_value)

    # xgboost
    xgboost = XGBoost(num_class=3, n_estimators=100, max_depth=3, random_state=42)
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
    kmeas = KNeighbors(n_neighbors=5)
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

