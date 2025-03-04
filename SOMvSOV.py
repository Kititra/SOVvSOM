# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:12:43 2025

@author: piotr.kociszewski
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 09:11:20 2025

@author: piotr.kociszewski
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

def transform_data(df, x_col, y_col):
    # Obliczamy wstępną korelację między wybranymi kolumnami x i y
    initial_corr = df[x_col].corr(df[y_col])
    initial_r2 = initial_corr**2
    
    best_r2 = initial_r2
    best_x_transformed = df[x_col]
    best_alpha = 0
    best_gamma = 0
    
    # Wyznacz zakres dla gamma – na podstawie wartości x
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    gamma_lower = round(x_min + 0.3*(x_max - x_min), 4)
    gamma_upper = round(x_max, 4)
    
    # Szukamy najlepszej transformacji zmiennej x poprzez iterację
    for alpha in np.arange(0.5, 3.0 + 0.001, 0.1):
        for gamma in np.arange(gamma_lower, gamma_upper + 0.001, (gamma_upper - gamma_lower) / 100):
            new_var = 1 / (1 + (gamma / df[x_col])**alpha)
            new_corr = df[y_col].corr(new_var)
            new_r2 = new_corr**2
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_x_transformed = new_var
                best_alpha = alpha
                best_gamma = gamma

    # Dodajemy do DataFrame kolumnę z transformacją zmiennej niezależnej
    transformed_col_name = f"{x_col} transformed"
    df[transformed_col_name] = best_x_transformed

    # Obliczamy model liniowy: y = y_col, x = przetransformowanej zmiennej x
    linreg = linregress(df[transformed_col_name], df[y_col])
    lin_intercept = linreg.intercept
    lin_slope = linreg.slope
    lin_r2 = linreg.rvalue**2

    # Testujemy modele wielomianowe o stopniach od 2 do 10
    best_poly_r2 = -np.inf
    best_poly_coeffs = None
    best_poly_deg = None
    for deg in range(2, 21):
        poly_coeffs = np.polyfit(df[transformed_col_name], df[y_col], deg=deg)
        y_poly = np.polyval(poly_coeffs, df[transformed_col_name])
        ss_res = np.sum((df[y_col] - y_poly)**2)
        ss_tot = np.sum((df[y_col] - np.mean(df[y_col]))**2)
        poly_r2 = 1 - ss_res/ss_tot
        if poly_r2 > best_poly_r2:
            best_poly_r2 = poly_r2
            best_poly_coeffs = poly_coeffs
            best_poly_deg = deg

    # Wybieramy ostateczny model: jeśli model wielomianowy (najlepszy z testowanych) ma wyższe R² niż model liniowy,
    # wybieramy model wielomianowy; w przeciwnym razie model liniowy.
    if best_poly_r2 > lin_r2:
        chosen_model = "poly"
        chosen_r2 = best_poly_r2
        model_params = best_poly_coeffs  # tablica współczynników wielomianu
        chosen_poly_deg = best_poly_deg
    else:
        chosen_model = "linear"
        chosen_r2 = lin_r2
        model_params = (lin_intercept, lin_slope)
        chosen_poly_deg = None

    return df, chosen_r2, model_params, best_alpha, best_gamma, transformed_col_name, chosen_model, chosen_poly_deg

st.title('Aplikacja do liczenia zależności między zmiennymi')

uploaded_file = st.file_uploader(
    """Prześlij plik CSV - pamiętaj, że powinien zwierać kolumny:
Date (data pomiaru),
Brand (dla jakiej marki jest pomiar),
oraz dwie dodatkowe kolumny, spośród których wybierzesz:
- jedną jako zmienną niezależną (oś X)
- jedną jako zmienną zależną (prognozowaną, oś Y)""",
    type="csv"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required_cols = ["Date", "Brand"]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error("Plik nie zawiera wymaganych kolumn: " + ", ".join(missing_required))
            st.stop()
        
        available_cols = [col for col in df.columns if col not in ["Date", "Brand"]]
        if len(available_cols) < 2:
            st.error("Plik musi zawierać co najmniej dwie kolumny oprócz 'Date' i 'Brand'.")
            st.stop()
        
        x_col = st.selectbox("Wybierz kolumnę jako zmienną niezależną (oś X):", available_cols)
        y_col = st.selectbox("Wybierz kolumnę jako zmienną zależną (prognozowaną, oś Y):", available_cols, index=1)
        
        st.write("Podgląd danych:", df.head())
        df = df.dropna(subset=[x_col, y_col])
        
        (transformed_df, chosen_r2, model_params, best_alpha, best_gamma,
         transformed_col_name, chosen_model, chosen_poly_deg) = transform_data(df, x_col, y_col)
        
        if chosen_r2 is not None:
            if chosen_model == "linear":
                st.write(f"Wybrany model (regresja liniowa) osiąga R² = {chosen_r2}")
                st.write("Model regresji liniowej:")
                st.write("Stała (intercept):", model_params[0])
                st.write(f"Współczynnik przy '{transformed_col_name}' (beta):", model_params[1])
            else:
                st.write(f"Wybrany model (regresja wielomianowa stopnia {chosen_poly_deg}) osiąga R² = {chosen_r2}")
                st.write("Współczynniki modelu wielomianowego:", model_params)
        else:
            st.warning("Nie udało się obliczyć współczynnika R².")
        
        st.write("Dane po transformacji:", transformed_df.head())
        
        # Dodajemy kolumnę z prognozowanymi wartościami dla y w zależności od wybranego modelu
        if chosen_model == "linear":
            transformed_df[f"Predicted {y_col}"] = model_params[0] + model_params[1] * transformed_df[transformed_col_name]
        else:
            transformed_df[f"Predicted {y_col}"] = np.polyval(model_params, transformed_df[transformed_col_name])
        
        if st.button("Pokaż interaktywny wykres regresji"):
            x_max_plot = 1.2 * df[x_col].max()
            y_max_plot = 1.2 * df[y_col].max()
            x_values = np.linspace(1e-6, x_max_plot, 100)
            transformed_x_line = 1 / (1 + (best_gamma / x_values)**best_alpha)
            if chosen_model == "linear":
                y_line = model_params[0] + model_params[1] * transformed_x_line
                model_label = f"Linia regresji: y = {round(model_params[0],2)} + {round(model_params[1],2)} * {transformed_col_name}"
            else:
                y_line = np.polyval(model_params, transformed_x_line)
                model_label = f"Regresja wielomianowa (stopnia {chosen_poly_deg})"
            
            fig = go.Figure()
            if "Brand" in df.columns and "Date" in df.columns:
                customdata = np.stack([df["Brand"], df["Date"]], axis=-1)
                hovertemplate = (
                    f"{x_col}: %{{x}}<br>" +
                    f"{y_col}: %{{y}}<br>" +
                    "Brand: %{customdata[0]}<br>" +
                    "Date: %{customdata[1]}<extra></extra>"
                )
            else:
                customdata = None
                hovertemplate = f"{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
            
            fig.add_trace(go.Scatter(
                x = df[x_col],
                y = df[y_col],
                mode = "markers",
                marker = dict(color='#17becf', size=10),
                name = "Dane oryginalne",
                customdata = customdata,
                hovertemplate = hovertemplate
            ))
            fig.add_trace(go.Scatter(
                x = x_values,
                y = y_line,
                mode = "lines",
                line = dict(color='#1f77b4'),
                name = model_label
            ))
            fig.update_layout(
                title = dict(text="Interaktywny wykres regresji", font=dict(size=20)),
                xaxis_title = dict(text=x_col, font=dict(size=18)),
                yaxis_title = dict(text=y_col, font=dict(size=18)),
                xaxis = dict(range=[0, x_max_plot], tickfont=dict(size=16)),
                yaxis = dict(range=[0, y_max_plot], tickfont=dict(size=16))
            )
            st.plotly_chart(fig)
        
        st.subheader("Prognoza dla nowej wartości zmiennej niezależnej")
        user_input = st.number_input(f"Wprowadź wartość {x_col}:", 
                                     min_value=float(df[x_col].min()),
                                     max_value=float(df[x_col].max()),
                                     value=float(df[x_col].mean()))
        transformed_input = 1 / (1 + (best_gamma / user_input)**best_alpha)
        if chosen_model == "linear":
            predicted_y = model_params[0] + model_params[1] * transformed_input
        else:
            predicted_y = np.polyval(model_params, transformed_input)
        st.write(f"Prognozowana wartość {y_col} dla {x_col} = {user_input}: {predicted_y}")
        
        csv = transformed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Pobierz przekształcony plik CSV",
            data=csv,
            file_name='przeksztalcony.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {e}")
