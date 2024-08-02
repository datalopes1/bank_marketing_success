# %% Import das bibliotecas
# Manipulação de dados
import pandas as pd
import numpy as np
from scipy.stats import zscore

# Análise Exploratória de Dados
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning 
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, log_loss

# Pré-processamento
import category_encoders as ce
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# %% Carregamento dos dados
df = pd.read_csv("../../data/raw/bank-full.csv", sep = ';')
df
# %% Encoding do target
le = LabelEncoder()
df['y'] = le.fit_transform(df['y'])
# %% Seleção das features
features = df.drop(columns = ['y', 'pdays'], axis = 1).columns.to_list()
target = 'y'

cat_features = df[features].select_dtypes(include = 'object').columns.to_list()
num_features = df[features].select_dtypes(include = 'number').columns.to_list()
print(f"Fatures categóricas: \n {cat_features}")
print(f"Features numéricas: \n {num_features}")
# %% Divisão dos dados
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=df[target], random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(f"Taxa de resposta em treino: {y_train.mean()}")
print(f"Taxa de resposta em teste: {y_test.mean()}")
# %% Pipeline do modelo
cat_transformer = Pipeline([
    ('imput', CategoricalImputer(imputation_method='frequent')),
    ('encoder', ce.OrdinalEncoder(cols=cat_features))
])

num_transformer = Pipeline([
    ('imput', MeanMedianImputer(imputation_method='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ]
)

model = LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31)

lgb = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

lgb.fit(X_train, y_train)
# %% Previsões
y_pred = lgb.predict(X_test)
y_pred_proba = lgb.predict_proba(X_test)[:,1]
# %% Métricas
def metrics_report(y_true, y_pred_proba, threshold = 0.5):
     y_pred = (y_pred_proba >= threshold).astype(int)

     loss = log_loss(y_true, y_pred_proba)
     acc = accuracy_score(y_true, y_pred)
     roc_auc = roc_auc_score(y_true, y_pred_proba)

     return {
         'Log Loss': loss,
         'Accuracy': acc,
         'ROC AUC': roc_auc,
     }

metricas = metrics_report(y_test, y_pred_proba)
metricas
# %% Matriz de confusão
matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize = (12, 6))
sns.heatmap(matrix, annot = True, fmt = '.2f', linewidths=0.5, linecolor='white', cbar=False)
ax.set_title("Matriz de Confusão", fontsize = 15, pad = 8, loc = 'left')
plt.show()
# %% Curva ROC
curve = roc_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize = (12, 6))
plt.plot(curve[0], curve[1])
plt.plot([0,1], [0,1], '--')
ax.set_title("Curva ROC", fontsize = 15, pad = 8, loc = 'left')
ax.set_xlabel("Falsos positivos", fontsize = 8)
ax.set_ylabel("Verdadeiros positivos", fontsize = 8)
plt.show()
# %% Salvando o modelo
model_series = pd.Series({
     'model': lgb,
     'features': features,
     'metricas': metricas
})
# %%
classification_model = model_series.to_pickle("../../models/classification_model.pkl")