import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle
import io
from scipy.stats import gaussian_kde

# Configuración de la página
st.set_page_config(
    page_title="Credit Scoring App",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones auxiliares
@st.cache_data
def load_data():
    df = pd.read_csv('score_credit.csv')
    df['Rango_Ingreso'] = df['Ingreso'].apply(rango_ingreso)
    return df

def rango_ingreso(x):
    if x <= 1500:
        y = "1.[0-1500]"
    elif x <= 2500:
        y = "2.[1501-2500]"
    else:
        y = "3.[2501-5000]"
    return y

def plot_kde(data_mora_0, data_mora_1):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Para mora = 0
    kde_0 = gaussian_kde(data_mora_0, bw_method='scott')
    x_0 = np.linspace(data_mora_0.min(), data_mora_0.max(), 1000)
    y_0 = kde_0(x_0)
    y_0 /= y_0.sum()
    ax.plot(x_0, y_0, color='blue', label='Mora = 0')
    ax.fill_between(x_0, y_0, alpha=0.3, color='blue')
    
    # Para mora = 1
    kde_1 = gaussian_kde(data_mora_1, bw_method='scott')
    x_1 = np.linspace(data_mora_1.min(), data_mora_1.max(), 1000)
    y_1 = kde_1(x_1)
    y_1 /= y_1.sum()
    ax.plot(x_1, y_1, color='red', label='Mora = 1')
    ax.fill_between(x_1, y_1, alpha=0.3, color='red')
    
    ax.set_title('Densidad del score según Mora')
    ax.set_xlabel('Score')
    ax.set_ylabel('Densidad')
    ax.legend()
    
    return fig

def hist_fill_p(y_test, scaled_scores):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    
    bins = np.percentile(scaled_scores, np.arange(0, 100.1, 5))
    rangos = pd.cut(scaled_scores, bins=bins, duplicates='drop')
    
    df_hist = pd.DataFrame({
        'score': scaled_scores,
        'MORA': y_test,
        'rangos': rangos
    })
    
    row_sums = df_hist.pivot_table(index='rangos', columns='MORA', values='score', aggfunc='count').sum(axis=1)
    
    # Gráfico de barras apiladas
    df_hist.pivot_table(index='rangos', columns='MORA', values='score', aggfunc='count') \
           .plot(kind='bar', stacked=True, ax=ax[0])
    ax[0].set_title('Distribución de Mora por Rango de Score')
    ax[0].set_xlabel('Rango de Score')
    ax[0].set_ylabel('Cantidad')
    
    # Gráfico de proporciones normalizadas por fila
    df_hist.pivot_table(index='rangos', columns='MORA', values='score', aggfunc='count') \
           .div(row_sums, axis=0) \
           .plot(kind='bar', stacked=True, ax=ax[1])
    ax[1].set_title('Proporción de Mora por Rango de Score')
    ax[1].set_xlabel('Rango de Score')
    ax[1].set_ylabel('Proporción')
    
    plt.tight_layout()
    return fig

@st.cache_resource
def train_models(df):
    # Preparar variables
    x_vars = ['Ingreso', 'Nivel_Deuda', 'Edad', 'Nivel_Estudios']
    y_var = ['Mora']

    # Add this line to create Rango_Ingreso column
    df['Rango_Ingreso'] = df['Ingreso'].apply(rango_ingreso)
    
    # Convertir a dummies
    X = pd.get_dummies(df[x_vars])
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, df[y_var], test_size=0.3, random_state=42)
    
    # Entrenar modelos
    modelo_arbol = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X_train, y_train)
    modelo_forest = RandomForestClassifier(max_depth=8, n_estimators=10, random_state=42).fit(X_train, y_train)
    
    return modelo_arbol, modelo_forest, X_train, X_test, y_train, y_test, X.columns

def predict_new_client(modelo, cliente, feature_names):
    # Preparar los datos del cliente
    cliente_df = pd.DataFrame([cliente])
    
    # Convertir a dummies asegurando que tiene las mismas columnas que el modelo
    cliente_dummies = pd.get_dummies(cliente_df)
    
    # Asegurar que todas las columnas del modelo estén presentes
    for col in feature_names:
        if col not in cliente_dummies.columns:
            cliente_dummies[col] = 0
    
    # Seleccionar solo las columnas que el modelo espera, en el mismo orden
    cliente_input = cliente_dummies[feature_names]
    
    # Predecir probabilidad
    prob = modelo.predict_proba(cliente_input)[0, 1]
    score = int((1 - prob) * 1000)
    
    return prob, score

# Título principal
st.title("🏦 Aplicación de Credit Scoring")

# Cargar datos
try:
    df = load_data()
    
    # Sidebar
    st.sidebar.header("Navegación")
    page = st.sidebar.radio("Ir a:", ["Inicio", "Exploración de Datos", "Modelado", "Evaluación", "Predicción"])
    
    if page == "Inicio":
        st.markdown("""
        # Modelo de Credit Scoring
        
        Esta aplicación permite predecir la probabilidad de mora de un cliente basado en su perfil crediticio.
        
        ## Objetivo
        
        Desarrollar un modelo de credit scoring que permita predecir la probabilidad de mora de un cliente basado en su ingreso, nivel de deuda, edad y nivel de estudios. El análisis busca identificar patrones y relaciones entre estas variables y el historial de mora para mejorar la precisión en la evaluación del riesgo crediticio.
        
        ## Metodología
        
        1. **Exploración de Datos**: Realizar un análisis descriptivo de las variables para entender su distribución y correlación.
        2. **Preprocesamiento**: Limpiar y preparar los datos, manejando valores faltantes y normalizando variables si es necesario.
        3. **Modelado**: Aplicar técnicas de modelado como árboles de decisión y random forest para predecir la probabilidad de mora.
        4. **Evaluación**: Evaluar el modelo utilizando métricas de desempeño como precisión, recall y AUC-ROC.
        
        ---
        
        ### Muestra de Datos
        """)
        
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Información del Dataset")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with col2:
            st.markdown("### Estadísticas Descriptivas")
            st.write(df.describe())
    
    elif page == "Exploración de Datos":
        st.header("Análisis Exploratorio de Datos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribución de Ingreso")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.hist(df['Ingreso'], bins=30, color='skyblue', edgecolor='black')
            plt.title('Distribución de Ingreso', fontsize=16)
            plt.xlabel('Ingreso', fontsize=14)
            plt.ylabel('Frecuencia', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Distribución de Nivel de Deuda")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.hist(df['Nivel_Deuda'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
            plt.title('Distribución de Nivel de Deuda', fontsize=16)
            plt.xlabel('Nivel de Deuda (%)', fontsize=14)
            plt.ylabel('Frecuencia', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Distribución de Edad")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.hist(df['Edad'], bins=30, color='red', edgecolor='black', alpha=0.7)
            plt.title('Distribución de Edad', fontsize=16)
            plt.xlabel('Edad', fontsize=14)
            plt.ylabel('Frecuencia', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(fig)
        
        with col4:
            st.subheader("Distribución de Nivel de Estudios")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Nivel_Estudios'].value_counts().sort_index().plot(
                kind='pie', 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=sns.color_palette('pastel'),
                wedgeprops={'edgecolor': 'black', 'linewidth': 1}
            )
            plt.title('Distribución de Nivel de Estudios', fontsize=16)
            plt.ylabel('')
            st.pyplot(fig)
        
        st.subheader("Relación entre Variables")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_table = df.pivot_table(index='Rango_Ingreso', columns='Mora', values='Ingreso', aggfunc='count')
        pivot_table.plot(
            kind='bar', 
            stacked=True, 
            figsize=(14, 8), 
            color=sns.color_palette('pastel', len(pivot_table.columns)),
            ax=ax
        )
        plt.title('Distribución de Ingreso por Rango de Ingreso y Mora', fontsize=18)
        plt.xlabel('Rango de Ingreso', fontsize=14)
        plt.ylabel('Cantidad de Clientes', fontsize=14)
        plt.legend(title='Mora', fontsize=12, title_fontsize=14)
        plt.xticks(fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
    
    elif page == "Modelado":
        st.header("Modelado de Riesgo de Mora")
        
        # Entrenar modelos
        modelo_arbol, modelo_forest, X_train, X_test, y_train, y_test, feature_names = train_models(df)
        
        # Visualización del árbol
        st.subheader("Visualización del Árbol de Decisión")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(modelo_arbol, feature_names=feature_names, filled=True, rounded=True, class_names=['No Mora', 'Mora'])
        st.pyplot(fig)
        
        st.markdown("""
        ### 🎯 Interpretación general
        * **Edad es el criterio más importante**: los jóvenes (<30.5) tienen mayor probabilidad de estar en mora.
        * Entre los jóvenes, el ingreso bajo y la deuda moderada aumentan el riesgo de mora.
        * En adultos, el ingreso alto y baja deuda están fuertemente relacionados con no caer en mora.
        """)
        
        # Optimización de parámetros
        st.subheader("Optimización de Parámetros")
        
        lac = []
        lac_2 = []
        
        for i in range(1, 15):
            arbol = DecisionTreeClassifier(max_depth=i, random_state=42)
            arbol.fit(X_train, y_train)
            y_pred = arbol.predict(X_test)
            accuracy = arbol.score(X_test, y_test)
            y_pred2 = arbol.predict(X_train)
            accuracy2 = arbol.score(X_train, y_train)
            lac_2.append(accuracy2)
            lac.append(accuracy)
        
        result = pd.DataFrame({'accuracy_test': lac, 'accuracy_train': lac_2}, index=range(1, 15))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        result.plot(figsize=(12, 8), ax=ax)
        plt.axvline(x=3, color='red', linestyle='--', label='Profundidad Óptima')
        plt.text(1.5, 0.9, 'Aprendizaje', fontsize=12, color='blue', ha='center')
        plt.text(5, 0.9, 'Memoria', fontsize=12, color='blue', ha='center')
        plt.xlabel('Profundidad del Árbol', fontsize=14)
        plt.ylabel('Precisión', fontsize=14)
        plt.title('Precisión del Modelo vs Profundidad del Árbol', fontsize=16)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        st.markdown("""
        ### Interpretación
        - Se observa un comportamiento de aprendizaje hasta la profundidad 3.
        - Más allá de esa profundidad, el modelo tiende a memorizar los datos de entrenamiento.
        - La profundidad óptima para este modelo es 3, donde hay un balance entre sesgo y varianza.
        """)
    
    elif page == "Evaluación":
        st.header("Evaluación de Modelos")
        
        # Entrenar modelos
        modelo_arbol, modelo_forest, X_train, X_test, y_train, y_test, feature_names = train_models(df)
        
        tab1, tab2 = st.tabs(["Árbol de Decisión", "Random Forest"])
        
        with tab1:
            st.subheader("Evaluación del Árbol de Decisión")
            
            # Métricas de clasificación
            y_pred = modelo_arbol.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Métricas de Clasificación")
                metrics_df = pd.DataFrame(report).transpose()
                st.dataframe(metrics_df.style.highlight_max(axis=0))
            
            with col2:
                st.markdown("### Matriz de Confusión")
                conf_matrix = pd.crosstab(y_test['Mora'], y_pred, rownames=['Real'], colnames=['Predicción'], margins=True)
                st.dataframe(conf_matrix)
            
            # Curva ROC
            y_pred_proba = modelo_arbol.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            specificity = 1 - fpr
            sensitivity = tpr
            
            sum_sensitivity_specificity = sensitivity + specificity
            best_threshold_index = np.argmax(sum_sensitivity_specificity)
            optimal_threshold = thresholds[best_threshold_index]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_roc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='green', label=f'Umbral óptimo: {optimal_threshold:.2f}')
            plt.title('Curva ROC - Árbol de Decisión', fontsize=16)
            plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
            plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
            plt.legend(loc='lower right', fontsize=12)
            plt.grid()
            st.pyplot(fig)
        
        with tab2:
            st.subheader("Evaluación del Random Forest")
            
            # Métricas de clasificación
            y_pred = modelo_forest.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Métricas de Clasificación")
                metrics_df = pd.DataFrame(report).transpose()
                st.dataframe(metrics_df.style.highlight_max(axis=0))
            
            with col2:
                st.markdown("### Matriz de Confusión")
                conf_matrix = pd.crosstab(y_test['Mora'], y_pred, rownames=['Real'], colnames=['Predicción'], margins=True)
                st.dataframe(conf_matrix)
            
            # Curva ROC
            y_pred_proba = modelo_forest.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            specificity = 1 - fpr
            sensitivity = tpr
            
            sum_sensitivity_specificity = sensitivity + specificity
            best_threshold_index = np.argmax(sum_sensitivity_specificity)
            optimal_threshold = thresholds[best_threshold_index]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_roc:.2f})')
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')
            plt.scatter(fpr[best_threshold_index], tpr[best_threshold_index], color='green', label=f'Umbral óptimo: {optimal_threshold:.2f}')
            plt.title('Curva ROC - Random Forest', fontsize=16)
            plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
            plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
            plt.legend(loc='lower right', fontsize=12)
            plt.grid()
            st.pyplot(fig)
            
            # Distribución del Score
            score = (1 - modelo_forest.predict_proba(X_test)[:, 1]) * 100
            result3 = pd.DataFrame({'Score': score, 'Mora': y_test['Mora'].values})
            
            # KDE Plot
            data_mora_0 = result3[result3['Mora'] == 0]['Score']
            data_mora_1 = result3[result3['Mora'] == 1]['Score']
            
            fig = plot_kde(data_mora_0, data_mora_1)
            st.pyplot(fig)
            
            # Distribucion por rangos
            st.subheader("Distribución por Rangos de Score")
            
            df['score'] = (1 - modelo_forest.predict_proba(pd.get_dummies(df[['Ingreso', 'Nivel_Deuda', 'Edad', 'Nivel_Estudios']]))[:, 1]) * 1000
            
            fig = hist_fill_p(df['Mora'], df['score'])
            st.pyplot(fig)
    
    elif page == "Predicción":
        st.header("Predicción de Riesgo de Mora para Nuevos Clientes")
        
        # Entrenar modelos
        modelo_arbol, modelo_forest, X_train, X_test, y_train, y_test, feature_names = train_models(df)
        
        # Formulario para ingreso de datos de cliente
        with st.form("client_form"):
            st.subheader("Ingrese los datos del cliente")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ingreso = st.number_input("Ingreso", min_value=500, max_value=5000, value=2000)
                nivel_deuda = st.number_input("Nivel de Deuda (%)", min_value=0, max_value=100, value=30)
            
            with col2:
                edad = st.number_input("Edad", min_value=18, max_value=80, value=35)
                nivel_estudios = st.selectbox("Nivel de Estudios", options=df['Nivel_Estudios'].unique())
            
            modelo_seleccionado = st.radio("Seleccione el modelo a utilizar:", ["Árbol de Decisión", "Random Forest"])
            
            submitted = st.form_submit_button("Predecir")
        
        if submitted:
            # Crear diccionario con datos del cliente
            cliente = {
                'Ingreso': ingreso,
                'Nivel_Deuda': nivel_deuda,
                'Edad': edad,
                'Nivel_Estudios': nivel_estudios
            }
            
            # Seleccionar modelo
            if modelo_seleccionado == "Árbol de Decisión":
                modelo = modelo_arbol
            else:
                modelo = modelo_forest
            
            # Predecir
            prob_mora, score = predict_new_client(modelo, cliente, feature_names)
            
            # Mostrar resultados
            st.subheader("Resultados de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probabilidad de Mora", f"{prob_mora:.2%}")
            
            with col2:
                st.metric("Score Crediticio", score)
            
            with col3:
                if score >= 750:
                    riesgo = "Bajo Riesgo"
                    color = "green"
                elif score >= 600:
                    riesgo = "Riesgo Moderado"
                    color = "orange"
                else:
                    riesgo = "Alto Riesgo"
                    color = "red"
                
                st.markdown(f"<h3 style='color:{color};'>{riesgo}</h3>", unsafe_allow_html=True)
            
            # Recomendación
            st.subheader("Interpretación y Recomendación")
            
            if score >= 750:
                st.success("""
                El cliente presenta un perfil de bajo riesgo crediticio. Se recomiendan las siguientes acciones:
                - Aprobar solicitudes de crédito con condiciones favorables
                - Ofrecer productos crediticios premium con tasas preferenciales
                - Establecer límites de crédito elevados
                """)
            elif score >= 600:
                st.warning("""
                El cliente presenta un perfil de riesgo moderado. Se recomiendan las siguientes acciones:
                - Evaluar cuidadosamente la capacidad de pago
                - Aprobar créditos con condiciones estándar
                - Establecer límites de crédito moderados
                - Implementar seguimiento periódico
                """)
            else:
                st.error("""
                El cliente presenta un perfil de alto riesgo. Se recomiendan las siguientes acciones:
                - Solicitar garantías adicionales
                - Aprobar montos reducidos con tasas más altas
                - Implementar seguimiento frecuente
                - Considerar rechazar la solicitud si el score es extremadamente bajo
                """)
            
            # Añadir una visualización del score en un gauge
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Crear un gauge simple
            plt.barh([0], [1000], color='lightgray', height=0.5)
            plt.barh([0], [score], color='green' if score >= 750 else 'orange' if score >= 600 else 'red', height=0.5)
            
            # Añadir marcas
            plt.axvline(x=600, color='orange', linestyle='--', alpha=0.7)
            plt.axvline(x=750, color='green', linestyle='--', alpha=0.7)
            
            # Etiquetas
            plt.text(0, -0.5, "Alto Riesgo", fontsize=12, ha='left')
            plt.text(600, -0.5, "Riesgo Moderado", fontsize=12, ha='center')
            plt.text(1000, -0.5, "Bajo Riesgo", fontsize=12, ha='right')
            
            # Estilizar
            plt.xlim(0, 1000)
            plt.ylim(-1, 1)
            plt.axis('off')
            
            st.pyplot(fig)

except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.markdown("""
    ## Instrucciones para cargar el dataset
    
    Para usar esta aplicación, debe cargar el archivo 'score_credit.csv' en el mismo directorio donde ejecuta la aplicación Streamlit.
    
    El archivo debe contener las siguientes columnas:
    - Ingreso: Ingreso mensual del cliente
    - Nivel_Deuda: Porcentaje de deuda respecto al ingreso
    - Edad: Edad del cliente
    - Nivel_Estudios: Nivel educativo del cliente
    - Mora: Variable objetivo (0: No mora, 1: Mora)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Desarrollado como proyecto de ML para Credit Scoring por Federico Martinez | 2025</p>
</div>
""", unsafe_allow_html=True)