import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import numpy as np
import plotly.express as px

st.cache_resource.clear()
st.cache_data.clear()


# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Aviron 2000m avec Modèle de Prédiction",
    page_icon="🚣",
    layout="wide"
)

# Introduction du dashboard
st.sidebar.markdown("""
    ## Bienvenue au Dashboard de Performance Aviron 2000m
    
    Ce tableau de bord vous permet d'explorer les performances des participants sur une distance de 2000m. 
    Vous pouvez visualiser les vitesses moyennes, les longueurs de coups, et plus encore.
    
    De plus, un modèle de prédiction basé sur la régression Lasso est inclus, vous permettant de prédire la **vitesse moyenne** 
    en fonction des paramètres tels que le SPM (coups par minute), le nombre de coups, et les splits.
"""
)

# Charger le modèle depuis un fichier .pkl
@st.cache_resource
def load_model():
    return joblib.load('best_lasso_model.pkl')

# Charger le modèle
best_lasso_model = load_model()

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('final_df.csv')
    return df

# Chargement des données
try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur lors du chargement des données: {str(e)}")
    st.stop()

# Titre du dashboard
st.title("🚣 Dashboard Performance Aviron 2000m avec Modèle de Prédiction")

# Fonction pour convertir un temps 'mm:ss.S' en secondes
def time_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + float(seconds)

# Appliquer la conversion au moment du calcul sans créer de nouvelle colonne

# Calculer le meilleur temps (en secondes), puis formater en mm:ss.S
best_time_in_seconds = df['Total Time (2000m)'].apply(time_to_seconds).min()
best_time_minutes = int(best_time_in_seconds // 60)
best_time_seconds = best_time_in_seconds % 60
best_time_formatted = f"{best_time_minutes}:{best_time_seconds:.1f}"

# Calculer le temps moyen (en secondes), puis formater en mm:ss.S
avg_time_in_seconds = df['Total Time (2000m)'].apply(time_to_seconds).mean()
avg_time_minutes = int(avg_time_in_seconds // 60)
avg_time_seconds = avg_time_in_seconds % 60
avg_time_formatted = f"{avg_time_minutes}:{avg_time_seconds:.1f}"

# Mise à jour de ton code avec les bonnes valeurs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_speed = df['Avg Speed (2000m) km/h'].mean()
    st.metric(
        "Vitesse Moyenne",
        f"{avg_speed:.2f} km/h",
        "Générale"
    )

with col2:
    avg_spm = df['SPM (2000m)'].mean()
    st.metric(
        "SPM Moyen",
        f"{avg_spm:.1f}",
        "Coups/min"
    )

with col3:
    st.metric(
        "Nombre d'Athlètes",
        len(df),
        "Participants"
    )

with col4:
    st.metric(
        "Meilleur Temps",
        best_time_formatted,
        "2000m"
    )

with col5:
    st.metric(
        "Temps Moyen",
        avg_time_formatted,
        "2000m"
    )

    
# Tabs pour différentes visualisations
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Relation Vitesse et Longueur de Coup", "📈 Heatmap des Corrélations", "🎯 Comparaison des Participants", "🔮 Prédire sa vitesse", " 🔮 Prédire la vitesse du Participant"])

# Tab 1: Scatter plot - Vitesse Moyenne vs Longueur de Coup avec SPM
with tab1:
    
    # Image URL
    image_url = "image2.png"

    # Display the image
    st.image(image_url, use_column_width=True)
    
    # Scatter plot SPM vs Vitesse with colorbar, increased marker size, and adjusted figure size
    fig_correlation = px.scatter(
        df,
        x='SPM (2000m)',
        y='Avg Speed (2000m) km/h',
        color='Avg Speed (2000m) km/h',  # Adding color based on Avg Speed
        size='Avg Speed (2000m) km/h',   # Set size based on speed (or another variable)
        size_max=20,                     # Max size of the markers
        title="Corrélation SPM vs Vitesse",
        labels={
            'SPM (2000m)': 'Coups par minute',
            'Avg Speed (2000m) km/h': 'Vitesse moyenne (km/h)'
        },
        text='Participant',
        color_continuous_scale='Plasma'  # You can choose any color scale you like
    )

    # Adjusting the figure size
    fig_correlation.update_layout(
        height=800,  # Set figure height
        width=1600,  # Set figure width
        plot_bgcolor='#F5F5F5' 
    )

    # Adding a color bar and adjusting the text position
    fig_correlation.update_traces(textposition='top center')

    # Show the plot
    st.plotly_chart(fig_correlation, use_container_width=False)


# Tab 2: Heatmap des Corrélations entre Longueur de Coup et Vitesse Moyenne
with tab2:
    st.header("Heatmap des Longueurs de Coup et de la Vitesse Moyenne")
    heatmap_data = df[['Avg Speed (2000m) km/h', 'Stroke Length Split 1 (m)', 
                       'Stroke Length Split 2 (m)', 'Stroke Length Split 3 (m)', 
                       'Stroke Length Split 4 (m)']]

    fig = px.imshow(round(heatmap_data.corr(), 2), text_auto=True, 
                    title="Heatmap des Longueurs de Coup et de la Vitesse Moyenne",
                    labels=dict(color="Corrélation"),
                    color_continuous_scale='Plasma'  # Applying the Plasma color scale
                   )

    fig.update_layout(
        height=600, width=800, plot_bgcolor='#F5F5F5' 
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Comparaison des Participants et Prédiction
with tab3:
    st.header("Comparaison des Participants et Modèle de Prédiction")
    
    # Comparaison des Vitesses Moyennes
    fig = px.bar(df, x='Participant', y='Avg Speed (2000m) km/h', 
                 title="Comparaison des Vitesses Moyennes par Participant", 
                 labels={'Avg Speed (2000m) km/h': 'Vitesse Moyenne (km/h)', 'Participant': 'Participants'},
                 color='Participant')

    fig.update_layout(
        xaxis_title='Participants', 
        yaxis_title='Vitesse Moyenne (km/h)',
        height=600, plot_bgcolor='#F5F5F5' 
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique des Temps Totals avec annotation pour Maxime (added from original code)
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df['Participant'],
        y=df['Total Time (2000m)'],
        text=df['Total Time (2000m)'],
        textposition='outside',  # Placer le texte à l'extérieur de la barre (au-dessus)
        marker=dict(color='lightcoral'),
        hoverinfo='x+y',
        textangle=45  # Faire une rotation du texte à 45 degrés
    ))

    maxime_time = df.loc[df['Participant'] == 'Maxime', 'Total Time (2000m)'].values[0]

    fig.update_layout(
        title={
            'text': 'Temps total pour chaque participant sur 2000m',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(showgrid=False, showticklabels=True),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
        height=600, width=900, plot_bgcolor='#F5F5F5',
        annotations=[
            dict(
                x='Maxime',  
                y=maxime_time,  
                xref="x", yref="y",
                text="Maxime est le plus rapide !",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
                font=dict(
                    size=12,
                    color="purple"  
                ),
                arrowcolor="purple",
            )
        ]
    )

    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Liste des participants
    participants = df['Participant'].unique()

    # Ajouter un sélecteur de participant
    selected_participant = st.selectbox("Choisir un participant", participants)

    # Extraire les vitesses pour le participant sélectionné
    participant_data = df[df['Participant'] == selected_participant]

    # Vérifier si le participant existe
    if not participant_data.empty:
        participant_speeds = participant_data[['Avg Speed Split 1 (km/h)', 'Avg Speed Split 2 (km/h)', 'Avg Speed Split 3 (km/h)', 'Avg Speed Split 4 (km/h)']].values[0]

        # Distances correspondant à chaque split de 500m
        time_splits = ['500m', '1000m', '1500m', '2000m']

        # Créer un graphique pour visualiser la vitesse du participant en fonction de la distance
        fig = go.Figure()

        # Ajouter une courbe avec marqueurs pour le participant avec une jolie couleur
        fig.add_trace(go.Scatter(
            x=time_splits,
            y=participant_speeds,
            mode='lines+markers',
            name="Vitesse Moyenne (km/h)",
            line=dict(color='royalblue', width=3),  # Beautiful royal blue for the line
            marker=dict(size=10, color='dodgerblue', line=dict(width=2, color='darkblue'))  # Blue gradient effect for markers
        ))

        fig.update_layout(
            title=f"Vitesse de {selected_participant} en fonction de la distance",
            xaxis_title='Distance (m)',
            yaxis_title='Vitesse (km/h)',
            showlegend=True,
            height=600, width=1100,
            plot_bgcolor='#F5F5F5' 
        )

        # Afficher le graphique
        st.plotly_chart(fig)

    else:
        st.warning(f"Pas de données disponibles pour {selected_participant}.")

            
        
with tab4:
    # Modèle de Prédiction
    st.subheader("Modèle de Prédiction (Lasso)")

    # Inputs utilisateur pour faire des prédictions
    st.write("Entrez les caractéristiques pour la prédiction de la vitesse:")

    # Exemple d'inputs basés sur les caractéristiques que le modèle attend
    input_spm = st.number_input("SPM (Coups par minute)", min_value=10.0, max_value=60.0, value=30.0, step=1.0)
    input_total_strokes = st.number_input("Total Strokes (2000m)", min_value=100, max_value=400, value=220, step=10)

    input_time_split_1 = st.number_input("Time Split 1 (sec)", min_value=500, max_value=1500, value=1100, step=10)
    input_spm_split_1 = st.number_input("SPM Split 1", min_value=10.0, max_value=60.0, value=29.0, step=1.0)
    input_stroke_count_split_1 = st.number_input("Stroke Count Split 1", min_value=20, max_value=100, value=50, step=1)

    input_time_split_2 = st.number_input("Time Split 2 (sec)", min_value=500, max_value=1500, value=1120, step=10)
    input_spm_split_2 = st.number_input("SPM Split 2", min_value=10.0, max_value=60.0, value=28.0, step=1.0)
    input_stroke_count_split_2 = st.number_input("Stroke Count Split 2", min_value=20, max_value=100, value=52, step=1)

    input_time_split_3 = st.number_input("Time Split 3 (sec)", min_value=500, max_value=1500, value=1130, step=10)
    input_spm_split_3 = st.number_input("SPM Split 3", min_value=10.0, max_value=60.0, value=27.0, step=1.0)
    input_stroke_count_split_3 = st.number_input("Stroke Count Split 3", min_value=20, max_value=100, value=54, step=1)

    input_time_split_4 = st.number_input("Time Split 4 (sec)", min_value=500, max_value=1500, value=1150, step=10)
    input_spm_split_4 = st.number_input("SPM Split 4", min_value=10.0, max_value=60.0, value=26.0, step=1.0)
    input_stroke_count_split_4 = st.number_input("Stroke Count Split 4", min_value=20, max_value=100, value=55, step=1)

    input_avg_speed_split_1 = st.number_input("Avg Speed Split 1 (km/h)", min_value=0.0, max_value=30.0, value=18.0, step=0.1)
    input_stroke_length_split_1 = st.number_input("Stroke Length Split 1 (m)", min_value=0.0, max_value=20.0, value=10.5, step=0.1)

    input_avg_speed_split_2 = st.number_input("Avg Speed Split 2 (km/h)", min_value=0.0, max_value=30.0, value=17.5, step=0.1)
    input_stroke_length_split_2 = st.number_input("Stroke Length Split 2 (m)", min_value=0.0, max_value=20.0, value=10.7, step=0.1)

    input_avg_speed_split_3 = st.number_input("Avg Speed Split 3 (km/h)", min_value=0.0, max_value=30.0, value=17.0, step=0.1)
    input_stroke_length_split_3 = st.number_input("Stroke Length Split 3 (m)", min_value=0.0, max_value=20.0, value=10.3, step=0.1)

    input_avg_speed_split_4 = st.number_input("Avg Speed Split 4 (km/h)", min_value=0.0, max_value=30.0, value=16.8, step=0.1)
    input_stroke_length_split_4 = st.number_input("Stroke Length Split 4 (m)", min_value=0.0, max_value=20.0, value=10.1, step=0.1)

    input_avg_stroke_length = st.number_input("Avg Stroke Length (m)", min_value=0.0, max_value=20.0, value=10.4, step=0.1)

    # Convertir les inputs en DataFrame pour les passer au modèle
    user_input_data = pd.DataFrame({
        'SPM (2000m)': [input_spm],
        'Total Strokes (2000m)': [input_total_strokes],
        'Time Split 1': [input_time_split_1],
        'SPM Split 1': [input_spm_split_1],
        'Stroke Count Split 1': [input_stroke_count_split_1],
        'Time Split 2': [input_time_split_2],
        'SPM Split 2': [input_spm_split_2],
        'Stroke Count Split 2': [input_stroke_count_split_2],
        'Time Split 3': [input_time_split_3],
        'SPM Split 3': [input_spm_split_3],
        'Stroke Count Split 3': [input_stroke_count_split_3],
        'Time Split 4': [input_time_split_4],
        'SPM Split 4': [input_spm_split_4],
        'Stroke Count Split 4': [input_stroke_count_split_4],
        'Avg Speed Split 1 (km/h)': [input_avg_speed_split_1],
        'Stroke Length Split 1 (m)': [input_stroke_length_split_1],
        'Avg Speed Split 2 (km/h)': [input_avg_speed_split_2],
        'Stroke Length Split 2 (m)': [input_stroke_length_split_2],
        'Avg Speed Split 3 (km/h)': [input_avg_speed_split_3],
        'Stroke Length Split 3 (m)': [input_stroke_length_split_3],
        'Avg Speed Split 4 (km/h)': [input_avg_speed_split_4],
        'Stroke Length Split 4 (m)': [input_stroke_length_split_4],
        'Avg Stroke Length (m)': [input_avg_stroke_length]
    })

    # Normalisation des données comme dans le modèle d'origine
    scaler = StandardScaler()
    features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Avg Speed (2000m) km/h'])
    scaler.fit(features)  # Fit scaler to the original feature set
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Sélection de la variable cible
    target_column = 'Avg Speed (2000m) km/h'

    # Sélection des features et suppression des colonnes non numériques et de la cible
    features = df.select_dtypes(include=['float64', 'int64']).drop(columns=[target_column])
    target = df[target_column]

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Normalisation des données
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features.columns)

   # Prédictions avec les modèle Lasso
    y_pred_lasso = best_lasso_model.predict(X_test_scaled)
    
    user_input_data_scaled = pd.DataFrame(scaler.transform(user_input_data), columns=user_input_data.columns)

    # Bouton pour lancer la prédiction
    if st.button("Prédire"):
        try:
            prediction = best_lasso_model.predict(user_input_data_scaled)
            st.success(f"Vitesse Moyenne Prédite : {prediction[0]:.2f} km/h.")
            
            # Calcul des performances après la prédiction
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            y_pred_lasso = best_lasso_model.predict(X_test_scaled)
            lasso_mae = mean_absolute_error(y_test, y_pred_lasso)
            lasso_mse = mean_squared_error(y_test, y_pred_lasso)
            lasso_r2 = best_lasso_model.score(X_test_scaled, y_test)

            # Comparaison des modèles
            performance_comparison = pd.DataFrame({
                'Model': ['Lasso (L1)'],
                'Mean Absolute Error (MAE)': [lasso_mae],
                'Mean Squared Error (MSE)': [lasso_mse],
                'R-squared (R²)': [lasso_r2]
            })

            # Afficher les métriques de performance dans un visuel
            st.subheader("Performance du Modèle Lasso")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{lasso_mae:.6f}")
            with col2:
                st.metric("Mean Squared Error (MSE)", f"{lasso_mse:.6f}")
            with col3:
                st.metric("R-squared (R²)", f"{lasso_r2:.6f}")

        
            # Modifier train_sizes pour éviter de très petits ensembles de validation
            train_sizes, train_scores, test_scores = learning_curve(
                best_lasso_model, X_train_scaled, y_train, cv=5, scoring='r2', train_sizes=np.linspace(0.2, 1.0, 8)
            )

            # Calculer les moyennes et écarts-types des scores de train et de test
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_mean,
                mode='lines',
                name='Training score',
                line=dict(color='royalblue'),
                fill='tonexty',
                fillcolor='rgba(65, 105, 225, 0.3)' 
            ))

            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_mean,
                mode='lines',
                name='Validation score',
                line=dict(color='firebrick'),
                fill='tonexty',
                fillcolor='rgba(178, 34, 34, 0.3)'  # Rouge brique avec transparence
            ))

            fig.update_layout(
                title='Learning Curve for Lasso Model',
                xaxis_title='Training Set Size',
                yaxis_title='R-squared Score',
                legend_title='Score Type',
                template='plotly_dark',
                showlegend=True,
                height=500, width=800,
                plot_bgcolor='#F5F5F5' 
            )


            st.plotly_chart(fig, use_container_width=True)
            
            # Créer un scatter plot pour les résidus
            fig = go.Figure()

            # Calculer les résidus
            residuals = y_test - y_pred_lasso

            # Ajouter les résidus dans un scatter plot
            fig.add_trace(go.Scatter(
                x=y_pred_lasso,
                y=residuals, 
                mode='markers', 
                marker=dict(color='orange'),
                name='Résidus'
            ))

            # Ajouter une ligne horizontale à y=0 (résidu parfait)
            fig.add_shape(type='line', x0=min(y_pred_lasso), x1=max(y_pred_lasso), y0=0, y1=0, line=dict(color='red', dash='dash'))

            fig.update_layout(
                title="Plot des Résidus",
                xaxis_title="Valeurs Prédites",
                yaxis_title="Résidus",
                #template="plotly_dark",
                height=500,
                width=800, 
                plot_bgcolor='#F5F5F5' 
            )

            st.plotly_chart(fig, use_container_width=True)
            
            # Extract feature importance from Lasso model
            importance = best_lasso_model.coef_

            # Create a DataFrame to show the feature importance
            feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            # Filtrer les features avec une importance différente de 0
            feature_importance_filtered = feature_importance[feature_importance['Importance'] != 0]


            # Create two columns
            col1, col2 = st.columns(2)

            # Display the feature importance table in the first column
            with col1:
                 st.markdown("<h4 style='font-size:16px;'>Importance des Features</h4>", unsafe_allow_html=True)
                 st.dataframe(feature_importance)

            # Plot feature importance in the second column
            with col2:
                # Créer le graphique en utilisant les données filtrées
                fig = px.bar(feature_importance_filtered, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h', 
                            title='Feature Importance from Lasso Model',
                            labels={'Importance': 'Importance', 'Feature': 'Features'},
                            color_discrete_sequence=['#2C7C7E']  # Soft teal color
                            )

                fig.update_layout(
                    xaxis_title='Importance',
                    yaxis_title='Features',
                    height=500,
                    width=900,
                    plot_bgcolor='#F5F5F5'
                )

                # Afficher le graphique dans Streamlit
                st.plotly_chart(fig)

            # Comparer les valeurs réelles et prédites avec des légendes au-dessus de chaque point
            fig_comparison = go.Figure()

            # Valeurs réelles (ligne diagonale pour comparaison)
            fig_comparison.add_trace(go.Scatter(
                x=y_test,
                y=y_test,
                mode='lines',
                name='Valeurs Réelles',
                line=dict(color='#4CAF50')  # Un vert doux
            ))

            # Valeurs prédites avec légendes au-dessus de chaque point
            fig_comparison.add_trace(go.Scatter(
                x=y_test,
                y=y_pred_lasso,
                mode='markers+text',
                name='Valeurs Prédites',
                marker=dict(color='#9C27B0', size=10, line=dict(color='#673AB7', width=2)),  # Violet avec bordures plus foncées
                text=[f"{pred:.2f}" for pred in y_pred_lasso],  # Ajouter les valeurs prédites comme légende
                textposition='top center',  # Positionner le texte au-dessus de chaque point
                textfont=dict(color='#00BCD4')  # Texte cyan
            ))

            fig_comparison.update_layout(
                title="Comparaison des Valeurs Réelles et Prédites avec Légende",
                xaxis_title="Valeurs Réelles",
                yaxis_title="Valeurs Prédites",
                height=600,
                width=800,
                plot_bgcolor='#F5F5F5'  # Contexte de graphique en gris clair
            )
        
            # Affichage du graphique
            st.plotly_chart(fig_comparison)
            

        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            
with tab5:      
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    # List of columns that were used for training the model (replace with the actual feature names)
    required_columns = [
        'SPM (2000m)', 'Total Strokes (2000m)', 'Time Split 1', 'SPM Split 1', 
        'Stroke Count Split 1', 'Time Split 2', 'SPM Split 2', 'Stroke Count Split 2', 
        'Time Split 3', 'SPM Split 3', 'Stroke Count Split 3', 'Time Split 4', 
        'SPM Split 4', 'Stroke Count Split 4', 'Avg Speed Split 1 (km/h)', 
        'Stroke Length Split 1 (m)', 'Avg Speed Split 2 (km/h)', 
        'Stroke Length Split 2 (m)', 'Avg Speed Split 3 (km/h)', 
        'Stroke Length Split 3 (m)', 'Avg Speed Split 4 (km/h)', 
        'Stroke Length Split 4 (m)', 'Avg Stroke Length (m)'
    ]

    # Modèle de Prédiction
    st.subheader("Modèle de Prédiction (Lasso)")

    # Sélectionner un participant
    participants = df['Participant'].unique()
    selected_participant = st.selectbox("Choisir un participant pour prédiction", participants)

    # Extraire les données du participant sélectionné
    participant_data = df[df['Participant'] == selected_participant]

    # Afficher la vitesse réelle du participant
    if not participant_data.empty:
        real_speed = participant_data['Avg Speed (2000m) km/h'].values[0]
        st.metric(f"Vitesse réelle de {selected_participant}", f"{real_speed:.2f} km/h")

        # Filtrer uniquement les colonnes nécessaires pour la prédiction
        user_input_data = participant_data[required_columns]
        user_input_data_scaled = pd.DataFrame(scaler.transform(user_input_data), columns=user_input_data.columns)

        if st.button("Prédire la vitesse"):
            try:
                # Prédire la vitesse du participant sélectionné
                predicted_speed = best_lasso_model.predict(user_input_data_scaled)
                st.success(f"Vitesse prédite pour {selected_participant}: {predicted_speed[0]:.2f} km/h")

                # Calcul des métriques de performance
                mae = mean_absolute_error([real_speed], [predicted_speed])
                mse = mean_squared_error([real_speed], [predicted_speed])
    

                # Afficher les métriques dans Streamlit
                
                
                # Formule MSE
                st.title("MAE et MSE")

                # Formule MAE
                st.latex(r'''
                \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
                ''')

                # Formule MSE
                st.latex(r'''
                \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
                ''')

                st.write(f"Erreur Absolue Moyenne (MAE): {mae:.6f} km/h")
                st.write(f"Erreur Quadratique Moyenne (MSE): {mse:.6f} km/h²")

                # Comparer la vitesse réelle et prédite dans un graphique en lignes
                comparison_fig = go.Figure()

                # Ajouter une courbe pour la vitesse réelle
                comparison_fig.add_trace(go.Scatter(
                    x=['Vitesse réelle', 'Vitesse prédite'],
                    y=[real_speed, predicted_speed[0]],
                    mode='lines+markers',
                    name='Vitesse Comparaison',
                    line=dict(color='royalblue', width=4),
                    marker=dict(size=12, color='dodgerblue', line=dict(width=2, color='darkblue'))
                ))

                comparison_fig.update_layout(
                    title=f"Comparaison des vitesses pour {selected_participant}",
                    xaxis_title="Type de Vitesse",
                    yaxis_title="Vitesse (km/h)",
                    plot_bgcolor='#F5F5F5',
                    showlegend=False,
                    height=500,
                    width=800
                )

            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {str(e)}")
    else:
        st.warning(f"Aucune donnée disponible pour {selected_participant}.")