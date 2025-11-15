import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


class BPClassifier:
    """
    标准BP算法实现的神经网络二分类器，适配离散+连续特征的数据集（如西瓜数据集3.0）。

    参数:
    -----------
    n_hidden : int, 默认=5
        隐层神经元数量
    learning_rate : float, 默认=0.1
        学习率（梯度下降步长）
    n_iter : int, 默认=100
        训练迭代轮数（每轮遍历所有样本）
    random_state : int, 默认=None
        随机种子（保证参数初始化可复现）

    属性:
    -----------
    W1_, b1_ : 输入层→隐层的权重矩阵、偏置向量
    W2_, b2_ : 隐层→输出层的权重矩阵、偏置向量
    encoder_ : OneHotEncoder实例（离散特征编码）
    scaler_ : StandardScaler实例（连续特征标准化）
    le_ : LabelEncoder实例（标签二值化）
    """

    def __init__(self, n_hidden=5, learning_rate=0.1, n_iter=100, random_state=None):
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化模型参数（拟合时会根据特征维度重初始化）"""
        np.random.seed(self.random_state)
        self.W1_ = None
        self.b1_ = None
        self.W2_ = None
        self.b2_ = None
        self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.scaler_ = StandardScaler()
        self.le_ = LabelEncoder()

    def _preprocess(self, X, discrete_features, continuous_features, is_fit=True):
        """
        数据预处理：离散特征OneHot编码 + 连续特征标准化
        is_fit=True：拟合阶段（用fit_transform）；is_fit=False：预测阶段（用transform）
        """
        # 检查特征列是否存在
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X必须是pandas.DataFrame类型")
        missing_discrete = [col for col in discrete_features if col not in X.columns]
        if missing_discrete:
            raise ValueError(f"X中缺少离散特征：{missing_discrete}")
        missing_continuous = [col for col in continuous_features if col not in X.columns]
        if missing_continuous:
            raise ValueError(f"X中缺少连续特征：{missing_continuous}")

        # 离散特征编码
        X_discrete = X[discrete_features]
        X_discrete_encoded = self.encoder_.fit_transform(X_discrete) if is_fit else self.encoder_.transform(X_discrete)

        # 连续特征标准化
        X_continuous = X[continuous_features]
        X_continuous_scaled = self.scaler_.fit_transform(X_continuous) if is_fit else self.scaler_.transform(X_continuous)

        return np.hstack((X_discrete_encoded, X_continuous_scaled))

    def _sigmoid(self, x):
        """Sigmoid激活函数（输出∈(0,1)，适配二分类概率）"""
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, discrete_features=None, continuous_features=None):
        """
        训练BP神经网络

        参数:
        -----------
        X : pandas.DataFrame, 形状(n_samples, n_features)
            训练数据特征（需包含离散/连续特征列）
        y : array-like, 形状(n_samples,)
            目标标签（如西瓜数据集的“好瓜”列）
        discrete_features : list
            离散特征的列名列表（如["色泽", "根蒂"]）
        continuous_features : list
            连续特征的列名列表（如["密度", "含糖率"]）
        """
        if discrete_features is None or continuous_features is None:
            raise ValueError("必须指定discrete_features和continuous_features")

        # 数据预处理（拟合阶段）
        X_processed = self._preprocess(X, discrete_features, continuous_features, is_fit=True)
        self.le_.fit(y)
        y_encoded = self.le_.transform(y).reshape(-1, 1)  # 转为二维数组，适配矩阵运算

        # 初始化权重/偏置（根据输入特征维度）
        n_samples, n_features = X_processed.shape
        self.W1_ = np.random.randn(n_features, self.n_hidden) * 0.01  # 小随机数初始化
        self.b1_ = np.zeros(self.n_hidden)
        self.W2_ = np.random.randn(self.n_hidden, 1) * 0.01
        self.b2_ = np.zeros(1)

        # 标准BP训练（单样本随机梯度下降）
        for _ in range(self.n_iter):
            for i in range(n_samples):
                # 前向传播
                x_i = X_processed[i:i+1]  # 单个样本（形状：1×n_features）
                hidden_input = x_i @ self.W1_ + self.b1_
                hidden_output = self._sigmoid(hidden_input)
                output_input = hidden_output @ self.W2_ + self.b2_
                y_pred = self._sigmoid(output_input)

                # 反向传播（计算梯度）
                error = y_pred - y_encoded[i:i+1]
                d_W2 = hidden_output.T @ error
                d_b2 = error.sum(axis=0)
                d_hidden = (error @ self.W2_.T) * hidden_output * (1 - hidden_output)  # Sigmoid导数
                d_W1 = x_i.T @ d_hidden
                d_b1 = d_hidden.sum(axis=0)

                # 更新参数
                self.W2_ -= self.learning_rate * d_W2
                self.b2_ -= self.learning_rate * d_b2
                self.W1_ -= self.learning_rate * d_W1
                self.b1_ -= self.learning_rate * d_b1

    def predict_proba(self, X, discrete_features=None, continuous_features=None):
        """
        预测样本为正类的概率

        返回:
        -----------
        array-like, 形状(n_samples,)：每个样本的正类概率
        """
        X_processed = self._preprocess(X, discrete_features, continuous_features, is_fit=False)
        hidden_input = X_processed @ self.W1_ + self.b1_
        hidden_output = self._sigmoid(hidden_input)
        output_input = hidden_output @ self.W2_ + self.b2_
        return self._sigmoid(output_input).ravel()

    def predict(self, X, threshold=0.5, discrete_features=None, continuous_features=None):
        """
        预测样本类别（默认概率≥0.5为正类）

        返回:
        -----------
        array-like, 形状(n_samples,)：编码后的类别（0/1）
        """
        proba = self.predict_proba(X, discrete_features, continuous_features)
        return (proba >= threshold).astype(int)