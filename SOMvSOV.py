# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:06:34 2025

@author: piotr.kociszewski
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:27:31 2025

@author: piotr.kociszewski
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

def transform_data(df):
    # Check that required columns SOV and SOM exist
    if "SOV" not in df.columns or "SOM" not in df.columns:
        st.warning("Plik musi zawierać kolumny 'SOV' oraz 'SOM'.")
        return df, None, None, None, None, None

    # Compute the initial correlation between SOV and SOM
    initial_corr = df["SOV"].corr(df["SOM"])
    
    # Set initial values: best correlation is the initial one,
    # and best SOV transformed is the original SOV column
    best_corr = initial_corr
    best_som_transformed = df["SOV"]
    best_alpha = 0
    best_gamma = 0

    # Iterate over all combinations of alpha and gamma
    for alpha in np.arange(0.5, 3.0 + 0.001, 0.1):
        for gamma in np.arange(0.3, 1.0 + 0.001, 0.1):
            # Compute new variable: new_var = 1 / (1 + (gamma / SOV)^alpha)
            new_var = 1 / (1 + (gamma / df["SOV"])**alpha)
            
            # Compute correlation of new variable with SOM
            new_corr = df["SOM"].corr(new_var)
            
            # If new correlation is better, update best values
            if new_corr > best_corr:
                best_corr = new_corr
                best_som_transformed = new_var
                best_alpha = alpha
                best_gamma = gamma

    # Add the best transformation to the DataFrame
    df["SOV transformed"] = best_som_transformed

    # Compute linear regression: y = SOM, x = SOV transformed
    reg_result = linregress(df["SOV transformed"], df["SOM"])
    intercept = reg_result.intercept
    slope = reg_result.slope

    return df, best_corr, intercept, slope, best_alpha, best_gamma

st.title('Aplikacja do liczenie zależności pomiędzy Share of Voice i Share of Market')

# File uploader with multiline prompt
uploaded_file = st.file_uploader(
    """Prześlij plik CSV - pamiętaj, że powinien zwierać kolumny 
Date (data pomiaru), 
Brand (dla jakiej marki jest pomiar), 
SOV (dokładnie taka nazwa zawierająca Share of Voice w formacie 0.00), 
SOM (dokładnie taka nazwa zawierająca Share of Market w formacie 0.00)""",
    type="csv"
)

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ["Date", "Brand", "SOV", "SOM"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error("Plik nie zawiera wymaganych kolumn: " + ", ".join(missing_columns) +
                     ". Upewnij się, że plik zawiera kolumny Date, Brand, SOV oraz SOM zgodnie z wymaganiami.")
            st.stop()
            
        st.write("Podgląd danych:", df.head())

        # Data transformation and regression
        transformed_df, corr_value, intercept, slope, best_alpha, best_gamma = transform_data(df)
        if corr_value is not None:
            st.write("Najlepszy współczynnik korelacji między 'SOM' a 'SOV transformed':", corr_value)
            st.write("Model regresji liniowej:")
            st.write("Stała (intercept):", intercept)
            st.write("Współczynnik przy 'SOV transformed' (beta):", slope)
        else:
            st.warning("Nie udało się obliczyć korelacji. Upewnij się, że plik zawiera kolumny 'SOV' oraz 'SOM'.")
        st.write("Dane po transformacji:", transformed_df.head())
        
        # Add predicted SOM column (using the regression model) so that the downloaded CSV reflects the regression plot
        transformed_df["Predicted SOM"] = intercept + slope * transformed_df["SOV transformed"]

        # Interactive Plotly regression chart
        if st.button("Pokaż interaktywny wykres regresji"):
            # Set axis ranges:
            # X-axis: 0 to 1.2 * max(SOV)
            # Y-axis: 0 to 1.2 * max(SOM)
            x_max = 1.2 * df["SOV"].max()
            y_max = 1.2 * df["SOM"].max()

            # Prepare x-values for regression line; avoid division by zero by starting with a very small value
            x_values = np.linspace(1e-6, x_max, 100)
            # Transform x: SOV transformed = 1 / (1 + (best_gamma / x)^best_alpha)
            transformed_x = 1 / (1 + (best_gamma / x_values)**best_alpha)
            y_line = intercept + slope * transformed_x

            fig = go.Figure()

            # Prepare hover data if available
            if "Brand" in df.columns and "Date" in df.columns:
                customdata = np.stack([df["Brand"], df["Date"]], axis=-1)
                hovertemplate = (
                    "SOV: %{x}<br>" +
                    "SOM: %{y}<br>" +
                    "Brand: %{customdata[0]}<br>" +
                    "Date: %{customdata[1]}<extra></extra>"
                )
            else:
                customdata = None
                hovertemplate = "SOV: %{x}<br>SOM: %{y}<extra></extra>"

            # Add scatter plot for original data
            fig.add_trace(go.Scatter(
                x = df["SOV"],
                y = df["SOM"],
                mode = "markers",
                marker = dict(color='#17becf', size=10),
                name = "Dane oryginalne",
                customdata = customdata,
                hovertemplate = hovertemplate
            ))

            # Add regression line
            fig.add_trace(go.Scatter(
                x = x_values,
                y = y_line,
                mode = "lines",
                line = dict(color='#1f77b4'),
                name = f"Linia regresji: y = {round(intercept,2)} + {round(slope,2)} * SOV_transformed"
            ))

            fig.update_layout(
                title = dict(text="Interaktywny wykres regresji", font=dict(size=20)),
                xaxis_title = dict(text="SOV", font=dict(size=18)),
                yaxis_title = dict(text="SOM", font=dict(size=18)),
                xaxis = dict(range=[0, x_max], tickfont=dict(size=16)),
                yaxis = dict(range=[0, y_max], tickfont=dict(size=16))
            )
            
            st.plotly_chart(fig)

        # Optionally: plot line chart for "SOV transformed"
        if st.button("Pokaż wykres SOV transformed"):
            st.line_chart(transformed_df["SOV transformed"])

        # Download button: CSV now includes the regression plot data (with "SOV", "SOM", "SOV transformed" and "Predicted SOM")
        csv = transformed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Pobierz przekształcony plik CSV",
            data=csv,
            file_name='przeksztalcony.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {e}")
