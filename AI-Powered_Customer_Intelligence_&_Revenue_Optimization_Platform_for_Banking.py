# =========================================
# BLOQUE 1 — BUSINESS UNDERSTANDING
# =========================================


#Objetivo del proyecto:
#---------------------
#Desarrollar una plataforma de Customer Intelligence que permita:

#1. Identificar clientes con alto riesgo de abandono (churn)
#2. Segmentar clientes según comportamiento y valor
#3. Estimar el Customer Lifetime Value (CLV)
#4. Optimizar campañas comerciales basadas en profit esperado

#Problema de negocio:
#--------------------
#La entidad bancaria busca maximizar revenue y reducir churn mediante decisiones basadas en datos, priorizando clientes con mayor impacto 
# económico.

#Enfoque:
#--------
#- Machine Learning supervisado (predicción de churn)
#- Segmentación no supervisada (clustering)
#- Optimización basada en métricas de negocio (profit-driven ML)
#- Explainable AI para interpretabilidad
#- Simulación de escenarios y A/B testing para validación

# =========================================
# BLOQUE 1.5 — DATA INGESTION
# =========================================


#Objetivo:
#---------
#Cargar y validar el dataset inicial asegurando:

#- correcta lectura de datos
#- estructura esperada
#- disponibilidad para análisis posterior

#Buenas prácticas:
#-----------------
#- rutas claras
#- validaciones básicas
#- reproducibilidad


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("dark_background")
import seaborn as sns
sns.set_theme(style="darkgrid")

df = pd.read_csv("Data Banking.csv")

df.columns.tolist()
print(df.columns)

df.columns = df.columns.str.strip()

print("Primeras filas del dataset:")
print(df.head())

print("\nShape del dataset:", df.shape)

expected_columns = [
    'Customer_ID', 'Age', 'Gender', 'Income_Level', 'Account_Type',
    'Account_Balance', 'Number_of_Transactions', 'Number_of_Products',
    'Credit_Score', 'Region', 'Tenure', 'Churned'
]

missing_cols = set(expected_columns) - set(df.columns)

if missing_cols:
    print("⚠️ Columnas faltantes:", missing_cols)
else:
    print("✅ Dataset cargado correctamente")
    
try:
    df = pd.read_csv("Data Banking.csv")
    print("✅ Dataset cargado correctamente")
except FileNotFoundError:
    print("❌ Error: archivo no encontrado")

# =========================================
# BLOQUE 2 — DATA UNDERSTANDING & EDA
# =========================================

print("Shape del dataset:", df.shape)

print("\nTipos de datos:\n", df.dtypes)

print("\nMissing values:\n", df.isnull().sum())

print("\nColumnas disponibles:\n", df.columns.tolist())

print("\nEstadísticas descriptivas:\n")
print(df.describe())


#Insights iniciales:
#- Evaluar dispersión en Account_Balance y Credit_Score
#- Analizar si existen outliers relevantes
#- Identificar posibles relaciones con churn


plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["text.color"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["axes.edgecolor"] = "white"

df.hist(figsize=(12,10))
plt.suptitle("Distribución de variables numéricas")
plt.show()

# =========================================


#Insight esperado:
#- Detectar skewness en balance o transacciones
#- Identificar posibles segmentos naturales


sns.pairplot(df, hue="Churned", palette="Set2")
plt.suptitle("Relaciones entre variables clave", y=1.02)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

target = "Churned"

features = df.select_dtypes(include=np.number).columns.tolist()

if target in features:
    features.remove(target)

features = features[:5]

sns.pairplot(
    df[features + [target]],
    hue=target,
    palette="Set2"
)

plt.suptitle("Relaciones entre variables clave", y=1.02)
plt.show()

# churn rate
churn_rate = df["Churned"].mean()
print(f"\nChurn Rate: {churn_rate:.2%}")

# =========================================
# BLOQUE 3 — FEATURE ENGINEERING
# =========================================


#Objetivo:
#---------
#Transformar variables crudas en señales que capturen:

#- comportamiento del cliente
#- nivel de engagement
#- valor económico
#- riesgo de churn

#Estrategia:
#-----------
#- ratios
#- intensidad de uso
#- valor monetario
#- interacción entre variables


df["Balance_per_Transaction"] = df["Account_Balance"] / (df["Number_of_Transactions"] + 1)

df["Transactions_per_Tenure"] = df["Number_of_Transactions"] / (df["Tenure"] + 1)

df["Products_per_Tenure"] = df["Number_of_Products"] / (df["Tenure"] + 1)

df["High_Credit_Score"] = (df["Credit_Score"] > 700).astype(int)

df["High_Income"] = (df["Income_Level"] == "High").astype(int)

# =========================================

df["EngagementScore"] = (
    df["Number_of_Transactions"] * 0.6 +
    df["Number_of_Products"] * 0.4
)

df["CLV"] = (
    df["Account_Balance"] * 0.3 +
    df["Number_of_Transactions"] * 50 +
    df["Number_of_Products"] * 500
)

df["ValueSegment"] = pd.qcut(
    df["CLV"],
    q=3,
    labels=["Low", "Medium", "High"]
)

df["Low_Engagement"] = (df["EngagementScore"] < df["EngagementScore"].median()).astype(int)

df["High_Value"] = (df["CLV"] > df["CLV"].median()).astype(int)

df["Balance_x_Products"] = df["Account_Balance"] * df["Number_of_Products"]

df["Age_x_Tenure"] = df["Age"] * df["Tenure"]

print("\nNuevas columnas creadas:\n")
print(df.columns.tolist())


#Insights esperados:
#------------------
#- Clientes con bajo engagement tienen mayor probabilidad de churn
#- Alto CLV permite priorización comercial
#- Ratios capturan comportamiento más que valores absolutos


features = [
    "Account_Balance",
    "Number_of_Transactions",
    "Number_of_Products",
    "Credit_Score",
    "Age",
    "Tenure",
    "CLV",
    "Cluster",
    "EngagementScore",
    "Balance_per_Transaction",
    "Transactions_per_Tenure",
    "Products_per_Tenure",
    "Balance_x_Products",
    "Age_x_Tenure"
]

# =========================================
# BLOQUE 4 — CUSTOMER SEGMENTATION
# =========================================


#Objetivo:
#---------
#Segmentar clientes en grupos homogéneos para:

#- identificar patrones de comportamiento
#- personalizar estrategias comerciales
#- mejorar targeting y retención

#Enfoque:
#--------
#Clustering no supervisado (KMeans) sobre variables clave


cluster_features = [
    "Account_Balance",
    "Number_of_Transactions",
    "Number_of_Products",
    "CLV",
    "EngagementScore"
]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[cluster_features])

from sklearn.cluster import KMeans

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_cluster)
    inertia.append(kmeans.inertia_)

plt.plot(range(1,10), inertia, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_cluster)

sns.scatterplot(
    x="Account_Balance",
    y="Number_of_Transactions",
    hue="Cluster",
    data=df,
    palette="Set1"
)

plt.title("Segmentación de clientes")
plt.show()

sns.boxplot(x="Cluster", y="Account_Balance", data=df)
plt.title("Balance por Cluster")
plt.show()

sns.boxplot(x="Cluster", y="Number_of_Transactions", data=df)
plt.title("Número de Transacciones por Cluster")
plt.show()

cluster_profile = df.groupby("Cluster")[cluster_features].mean()
print("\nPerfil de clusters:\n")
print(cluster_profile)


#Interpretación esperada:

# Cluster 0:
# - Bajo balance
# - Bajo engagement
# → Clientes de bajo valor (alto riesgo de churn)

# Cluster 1:
# → Clientes premium (alta prioridad)

# Cluster 2:
# → Clientes a desarrollar

# Cluster 3:
# → Potencial upselling


sns.boxplot(x="Cluster", y="Account_Balance", data=df)
plt.title("Balance por Cluster")
plt.show()

sns.boxplot(x="Cluster", y="Number_of_Transactions", data=df)
plt.title("Transacciones por Cluster")
plt.show()

sns.boxplot(x="Cluster", y="CLV", data=df)
plt.title("CLV por Cluster")
plt.show()


#Insights clave:

#- Existen segmentos claramente diferenciados por valor y comportamiento
#- Los clusters permiten priorizar campañas comerciales
#- Se pueden diseñar estrategias específicas por segmento:
    #- retención (alto churn)
    #- cross-selling
    #- fidelización


# =========================================
# BLOQUE 5 — CUSTOMER LIFETIME VALUE (CLV)
# =========================================


#Objetivo:
#---------
#Estimar el valor económico de cada cliente para:

#- priorizar acciones comerciales
#- optimizar campañas de marketing
#- maximizar revenue

#Enfoque:
#--------
#Modelo heurístico basado en:
#- valor monetario (balance)
#- actividad (transacciones)
#- relación con el banco (productos)


df["CLV"] = (
    df["Account_Balance"] * 0.3 +
    df["Number_of_Transactions"] * 50 +
    df["Number_of_Products"] * 500
)

sns.boxplot(x="Cluster", y="CLV", data=df)
plt.title("CLV por Cluster")
plt.show()

sns.histplot(df["CLV"], bins=50)
plt.title("Distribución del Customer Lifetime Value")
plt.show()

df["ValueSegment"] = pd.qcut(
    df["CLV"],
    q=3,
    labels=["Low", "Medium", "High"]
)

clv_cluster = df.groupby("Cluster")["CLV"].mean()

print("\nCLV promedio por cluster:\n")
print(clv_cluster)

sns.boxplot(x="Cluster", y="CLV", data=df)
plt.title("CLV por Cluster")
plt.show()

sns.boxplot(x="Churned", y="CLV", data=df)
plt.title("CLV vs Churn")
plt.show()

sns.countplot(x="ValueSegment", hue="Churned", data=df)
plt.title("Segmentación de valor vs Churn")
plt.show()


#Insights clave:

#- Clientes de alto CLV representan mayor impacto económico
#- El churn en clientes de alto valor genera pérdidas significativas
#- No todos los clientes deben ser tratados igual:
    #→ priorizar clientes High Value

#Estrategia:
#-----------
#- High CLV + alto riesgo → retención prioritaria
#- Low CLV → optimizar costos (no sobreinvertir)


avg_clv = df["CLV"].mean()
print(f"\nCLV promedio: ${avg_clv:,.2f}")

# =========================================
# BLOQUE 6 — CHURN PREDICTION MODEL
# =========================================


#Objetivo:
#---------
#Predecir la probabilidad de churn de cada cliente para:

#- anticipar abandono
#- activar estrategias de retención
#- optimizar decisiones comerciales

#Enfoque:
#--------
#Modelo supervisado de clasificación (XGBoost) con features de comportamiento, valor y segmentación


y = df["Churned"]

features = [
    "Account_Balance",
    "Number_of_Transactions",
    "Number_of_Products",
    "Credit_Score",
    "Age",
    "Tenure",
    "CLV",
    "Cluster",
    "EngagementScore",
    "Balance_per_Transaction",
    "Transactions_per_Tenure",
    "Products_per_Tenure",
    "Balance_x_Products",
    "Age_x_Tenure"
]

X = df[features]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

import xgboost as xgb

model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")


#Interpretación:

#- AUC > 0.80 → modelo fuerte
#- AUC ~ 0.70 → aceptable (mejorable)
#- AUC < 0.65 → revisar features

#En banca:
#→ se prioriza ranking (probabilidades) más que accuracy


xgb.plot_importance(model)
plt.title("Importancia de variables")
plt.show()

from sklearn.metrics import classification_report

y_pred = (y_proba >= 0.5).astype(int)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


#Insights esperados:

#- Variables de comportamiento (transactions, engagement) suelen ser clave
#- CLV aporta información de valor económico
#- Cluster mejora la capacidad predictiva

#El modelo permite rankear clientes por riesgo de churn


# =========================================
# BLOQUE 7 — PROFIT OPTIMIZATION
# =========================================


#Objetivo:
#---------
#Optimizar decisiones comerciales maximizando profit, en lugar de métricas tradicionales como accuracy o AUC.

#Problema:
#---------
#No todos los errores tienen el mismo costo:

#- False Positive → costo de contactar cliente
#- True Positive → beneficio de retener cliente

#Solución:
#---------
#Optimizar el threshold de decisión en función del beneficio económico


def calculate_profit(y_true, y_proba, threshold=0.5, gain=100, contact_cost=20):
    y_pred = (y_proba >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    profit = (tp * gain) - ((tp + fp) * contact_cost)
    
    return profit

thresholds = np.arange(0.10, 0.91, 0.05)
profits = []

for threshold in thresholds:
    profit = calculate_profit(y_test, y_proba, threshold=threshold)
    profits.append(profit)
    
plt.plot(thresholds, profits, marker="o")
plt.xlabel("Threshold de decisión")
plt.ylabel("Profit esperado")
plt.title("Optimización de profit vs threshold")
plt.show()

best_threshold = thresholds[np.argmax(profits)]
best_profit = max(profits)
print(f"Threshold óptimo: {best_threshold:.2f}")
print(f"Profit máximo: ${best_profit:,.2f}")

def calculate_profit_clv(y_true, y_pred, clv, contact_cost=20):

    # verdaderos positivos (clientes retenidos correctamente)
    tp = (y_true == 1) & (y_pred == 1)

    # falsos positivos (contactados innecesariamente)
    fp = (y_true == 0) & (y_pred == 1)

    profit = clv[tp].sum() - (contact_cost * fp.sum())

    return profit

thresholds = np.linspace(0, 1, 100)
profits = []

clv_test = X_test["CLV"].values

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    profit = calculate_profit_clv(y_test.values, y_pred, clv_test)
    profits.append(profit)
    
plt.plot(thresholds, profits, marker="o")
plt.xlabel("Threshold de decisión")
plt.ylabel("Profit esperado (CLV ajustado)")
plt.title("Optimización de profit ajustado por CLV")
plt.show()

best_threshold = thresholds[np.argmax(profits)]
best_profit = max(profits)
print(f"Threshold óptimo (CLV ajustado): {best_threshold:.2f}")
print(f"Profit máximo (CLV ajustado): ${best_profit:,.2f}")

best_threshold = thresholds[np.argmax(profits)]
best_profit = max(profits)

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Max Profit: ${best_profit:,.2f}")

plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Profit ($)")
plt.title("Optimización de Revenue")

plt.axvline(best_threshold, linestyle="--")
plt.show()


#Insights:

#- El threshold óptimo NO es 0.5
#- Existe un punto donde el profit es máximo
#- Optimizar por negocio cambia la decisión del modelo

#Conclusión:
#-----------
#El modelo no se usa para predecir churn directamente, sino para maximizar impacto económico


# profit con threshold estándar
y_pred_default = (y_proba >= 0.5).astype(int)

default_profit = calculate_profit_clv(
    y_test.values,
    y_pred_default,
    clv_test
)

print(f"Profit con threshold 0.5: ${default_profit:,.2f}")
print(f"Profit optimizado: ${best_profit:,.2f}")

roi = best_profit / (len(y_test) * 20)

print(f"ROI estimado: {roi:.2f}")

# =========================================
# BLOQUE 8 — CUSTOMER TARGETING STRATEGY
# =========================================


#Objetivo:
#---------
#Definir a qué clientes contactar para maximizar el retorno de campañas.

#Enfoque:
#--------
#- Usar probabilidades del modelo
#- Calcular valor esperado por cliente
#- Priorizar clientes según impacto económico
#- Aplicar restricciones de presupuesto


# probabilidades del modelo
y_proba = model.predict_proba(X_test)[:, 1]

df_targeting = X_test.copy()
df_targeting["y_true"] = y_test.values
df_targeting["proba"] = y_proba

BENEFIT_TP = 100   # ingreso por retención
COST_CONTACT = 5   # costo de contacto

df_targeting["expected_value"] = (
    df_targeting["proba"] * BENEFIT_TP - COST_CONTACT
)

df_targeting = df_targeting.sort_values(
    by="expected_value",
    ascending=False
)

TOP_N = int(len(df_targeting) * 0.3)

df_targeting["target"] = 0
df_targeting.iloc[:TOP_N, df_targeting.columns.get_loc("target")] = 1

targeted = df_targeting[df_targeting["target"] == 1]

tp = np.sum(targeted["y_true"] == 1)
fp = np.sum(targeted["y_true"] == 0)

total_profit = (tp * BENEFIT_TP) - (len(targeted) * COST_CONTACT)

print("Clientes contactados:", len(targeted))
print("True Positives:", tp)
print("False Positives:", fp)
print("Profit total:", total_profit)

BUDGET = 5000

df_targeting["cumulative_cost"] = (
    np.arange(1, len(df_targeting)+1) * COST_CONTACT
)

df_targeting["target"] = (
    df_targeting["cumulative_cost"] <= BUDGET
).astype(int)

df_targeting["segment"] = pd.qcut(
    df_targeting["proba"],
    4,
    labels=["Low","Med","High","Top"]
)

df_targeting[df_targeting["target"] == 1] \
    .to_csv("campaign_targets.csv", index=False)
    

#Insights:

#- No todos los clientes deben ser contactados
#- El ranking por expected value optimiza recursos
#- Las campañas deben ser:
    #→ dirigidas
    #→ medibles
    #→ basadas en datos

#Estrategia final:
#----------------
#- Priorizar clientes con mayor expected value
#- Ajustar targeting según budget
#- Integrar con CRM para ejecución


avg_ev = df_targeting["expected_value"].mean()
print(f"Expected Value promedio: ${avg_ev:.2f}")

# =========================================
# BLOQUE 9 — EXPLAINABILITY (SHAP)
# =========================================


#Objetivo:
#---------
#Interpretar el modelo de churn para:

#- entender drivers de abandono
#- justificar decisiones comerciales
#- mejorar confianza en el modelo

#Enfoque:
#--------
#Uso de SHAP (Shapley Values) para explicar:
#- importancia global
#- impacto individual por cliente


import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

shap.plots.bar(shap_values)

plt.title("Importancia global de variables (SHAP)")
plt.show()  

shap.plots.beeswarm(shap_values)
plt.title("Impacto de variables por cliente (SHAP)")
plt.show()

def explain_customer(i):
    row = shap_values[i]

    df_exp = pd.DataFrame({
        "feature": X_test.columns,
        "impact": row.values
    })

    df_exp["abs_impact"] = df_exp["impact"].abs()

    return df_exp.sort_values(
        by="abs_impact",
        ascending=False
    ).drop(columns="abs_impact")
    
explain_customer(0)

targeted_idx = df_targeting[df_targeting["target"] == 1].index

targeted_positions = [
    X_test.index.get_loc(idx)
    for idx in targeted_idx
]

shap_targeted = shap_values[targeted_positions]

shap.plots.beeswarm(shap_targeted)

plt.title("Impacto de variables en clientes target (SHAP)")
plt.show()

mean_impact = np.abs(shap_values.values).mean(axis=0)

pd.DataFrame({
    "feature": X_test.columns,
    "importance": mean_impact
}).sort_values(by="importance", ascending=False)

top_clients = df_targeting[df_targeting["target"] == 1].copy()

top_clients["top_driver"] = [
    explain_customer(X_test.index.get_loc(idx))["feature"].values[0]
    for idx in top_clients.index
]

top_clients.to_csv("target_with_explanations.csv", index=False)


#Insights:

#- El modelo no es una "caja negra"
#- Podemos explicar cada decisión individual
#- Identificamos drivers de churn:
    #→ bajo engagement
    #→ bajo balance
    #→ pocos productos

#Impacto en negocio:
#------------------
#- campañas personalizadas
#- mejor comunicación con clientes
#- soporte a decisiones estratégicas


def explain_customer_text(i):
    top_features = explain_customer(i).head(3)

    explanation = "Cliente con riesgo de churn debido a: "

    explanation += ", ".join(top_features["feature"].values)

    return explanation

print(explain_customer_text(0))

# =========================================
# BLOQUE 10 — MODEL MONITORING & DRIFT
# =========================================


#Objetivo:
#---------
#Monitorear la estabilidad del modelo en producción para:

#- detectar cambios en los datos (data drift)
#- identificar pérdida de performance
#- anticipar necesidad de reentrenamiento

#Enfoque:
#--------
#- Population Stability Index (PSI)
#- comparación train vs test
#- monitoreo de métricas (AUC)


train_dist = X_train.describe()

def calculate_psi(expected, actual, bins=10):

    def scale_range(input, min_val, max_val):
        input = input.copy()
        input = (input - np.min(input)) / (np.max(input) - np.min(input))
        input = input * (max_val - min_val) + min_val
        return input

    breakpoints = np.linspace(0, 1, bins + 1)

    expected_percents = np.histogram(
        scale_range(expected, 0, 1),
        bins=breakpoints
    )[0] / len(expected)

    actual_percents = np.histogram(
        scale_range(actual, 0, 1),
        bins=breakpoints
    )[0] / len(actual)

    psi = np.sum(
        (expected_percents - actual_percents) *
        np.log((expected_percents + 1e-6) / (actual_percents + 1e-6))
    )

    return psi

psi_values = {}

for col in X_train.columns:
    psi = calculate_psi(X_train[col], X_test[col])
    psi_values[col] = psi

psi_df = pd.DataFrame.from_dict(
    psi_values,
    orient='index',
    columns=['PSI']
).sort_values(by='PSI', ascending=False)

print("\nPSI por variable:\n")
print(psi_df.head())


#Interpretación PSI:

#- PSI < 0.1  → sin drift
#- PSI 0.1-0.25 → drift moderado
#- PSI > 0.25 → drift significativo (alerta)

#Acción:
#-------
#Features con alto PSI deben revisarse


from sklearn.metrics import roc_auc_score

auc_train = roc_auc_score(
    y_train,
    model.predict_proba(X_train)[:, 1]
)

auc_test = roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
)

print("AUC Train:", auc_train)
print("AUC Test:", auc_test)

if auc_test < 0.75:
    print("⚠️ ALERTA: posible caída de performance del modelo")
    
performance_log = []

performance_log.append({
    "period": "2026-01",
    "auc": auc_test
})

perf_df = pd.DataFrame(performance_log)

important_features = psi_df.head(5).index.tolist()

print("Features con mayor drift:", important_features)


#Insights:

#- El modelo puede degradarse con el tiempo
#- Cambios en comportamiento del cliente afectan performance
#- Es necesario monitoreo continuo

#Estrategia:
#-----------
#- reentrenar modelo periódicamente
#- monitorear features críticas
#- integrar alertas automáticas


drifted_features = psi_df[psi_df["PSI"] > 0.2]

print("\nFeatures con drift significativo:\n", drifted_features)

# =========================================
# BLOQUE 11 — DEPLOYMENT & MLOps
# =========================================


#Objetivo:
#---------
#Simular el uso del modelo en producción para:

#- generar predicciones en tiempo real
#- tomar decisiones automatizadas
#- integrar con sistemas de negocio (CRM, campañas)

#Enfoque:
#--------
#- funciones reutilizables
#- lógica de decisión basada en threshold óptimo
#- pipeline simple y claro


model_production = model

def predict_customer(data, threshold=best_threshold):

    proba = model_production.predict_proba(data)[:, 1]

    decision = (proba >= threshold).astype(int)

    return pd.DataFrame({
        "proba": proba,
        "decision": decision
    })
    
sample = X_test.iloc[:5]

predict_customer(sample)

def predict_and_target(data, top_n=0.3):

    preds = predict_customer(data)

    df_out = data.copy()
    df_out["proba"] = preds["proba"]

    df_out = df_out.sort_values(by="proba", ascending=False)

    cutoff = int(len(df_out) * top_n)

    df_out["target"] = 0
    df_out.iloc[:cutoff, df_out.columns.get_loc("target")] = 1

    return df_out

targeted_sample = predict_and_target(X_test)

print(targeted_sample.head())

new_customers = X_test.sample(10)

results = predict_customer(new_customers)

print(results)

final_targets = predict_and_target(X_test)

final_targets.to_csv("final_targets_production.csv", index=False)


#Uso en producción:

#- Input: datos de clientes
#- Output:
    #→ probabilidad de churn
    #→ decisión de contacto

#Integración:
#------------
#- CRM
#- campañas de marketing
#- sistemas de recomendación

#Valor:
#------
#Permite automatizar decisiones de retención y maximizar revenue


def predict_api(data_json):

    data = pd.DataFrame(data_json)

    preds = predict_customer(data)

    return preds.to_dict(orient="records")

# =========================================
# BLOQUE 12 — AI LAYER (LLM / GENERATIVE AI)
# =========================================


#Objetivo:
#---------
#Agregar una capa de Inteligencia Artificial generativa para:

#- explicar decisiones del modelo en lenguaje natural
#- generar insights automáticos para negocio
#- asistir en la toma de decisiones comerciales

#Enfoque:
#--------
#- usar resultados del modelo (SHAP, proba, CLV)
#- transformar datos en narrativa accionable


def generate_customer_explanation(i):

    proba = y_proba[i]
    top_features = explain_customer(i).head(3)["feature"].values

    explanation = f"""
Cliente con probabilidad de churn de {proba:.2%}.

Principales factores:
- {top_features[0]}
- {top_features[1]}
- {top_features[2]}

Recomendación:
Contactar cliente con estrategia personalizada de retención.
"""

    return explanation

print(generate_customer_explanation(0))

def generate_business_action(i):

    proba = y_proba[i]
    clv = X_test.iloc[i]["CLV"]

    if proba > 0.7 and clv > df["CLV"].median():
        return "Alta prioridad: ofrecer beneficios premium"
    
    elif proba > 0.7:
        return "Campaña de retención estándar"
    
    elif proba > 0.4:
        return "Seguimiento comercial"
    
    else:
        return "No contactar"
    
print(generate_business_action(0))

targeted_positions = [
    X_test.index.get_loc(idx)
    for idx in df_targeting[df_targeting["target"] == 1].index
]

df_targeting.loc[df_targeting["target"] == 1, "AI_Action"] = [
    generate_business_action(i) for i in targeted_positions
]

df_targeting.head()

def generate_executive_summary():

    churn_rate = df["Churned"].mean()
    avg_clv = df["CLV"].mean()

    summary = f"""
Resumen Ejecutivo:

- Tasa de churn: {churn_rate:.2%}
- CLV promedio: ${avg_clv:,.2f}
"""

#Insights:
#- Clientes con bajo engagement presentan mayor riesgo de abandono
#- Segmentos de alto valor requieren estrategias prioritarias

#Recomendación:
#Implementar campañas dirigidas basadas en modelos predictivos
#y optimización de profit.


    #return summary

print(generate_executive_summary())


#Extensión futura (RAG):

#- integrar base de conocimiento de clientes
#- buscar clientes similares (embeddings)
#- generar recomendaciones personalizadas

#Ejemplo:
#"Clientes similares respondieron mejor a campañas de descuento"


df_targeting.to_csv("final_ai_targeting.csv", index=False)


#Valor agregado de IA:

#- Traduce modelos complejos a lenguaje de negocio
#- Permite decisiones más rápidas y automatizadas
#- Mejora la experiencia del cliente

#Conclusión:
#-----------
#La combinación de ML + AI genera un sistema inteligente
#de toma de decisiones comerciales


