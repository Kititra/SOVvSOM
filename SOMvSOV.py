# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:12:43 2025

@author: piotr.kociszewski
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go

def transform_data(df, x_col, y_col, lag=0):
    # Apply lag only within each Brand group
    if lag != 0:
        # Shift the independent variable within each Brand group based on Date order
        x_series = df.groupby("Brand")[x_col].shift(lag)
    else:
        x_series = df[x_col]
    
    # Store the lagged x values in a new column for further processing
    df["_lagged_x_"] = x_series
    
    # Calculate initial correlation and R² between the lagged independent variable and y_col
    initial_corr = df["_lagged_x_"].corr(df[y_col])
    initial_r2 = initial_corr**2

    best_r2 = initial_r2
    best_x_transformed = df["_lagged_x_"]
    best_alpha = 0
    best_gamma = 0

    # Determine gamma range from the lagged x variable (ignoring NaNs)
    x_min = df["_lagged_x_"].min()
    x_max = df["_lagged_x_"].max()
    gamma_lower = round(x_min + 0.3*(x_max - x_min), 4)
    gamma_upper = round(x_max, 4)

    # Search for the best transformation of the lagged x variable using:
    # new_var = 1 / (1 + (gamma / x)^alpha)
    for alpha in np.arange(0.5, 3.0 + 0.001, 0.1):
        for gamma in np.arange(gamma_lower, gamma_upper + 0.001, (gamma_upper - gamma_lower) / 100):
            new_var = 1 / (1 + (gamma / df["_lagged_x_"])**alpha)
            new_corr = df[y_col].corr(new_var)
            new_r2 = new_corr**2
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_x_transformed = new_var
                best_alpha = alpha
                best_gamma = gamma

    # Create a new column name that reflects the applied lag.
    lag_label = f" (lag={lag})" if lag != 0 else ""
    transformed_col_name = f"{x_col}{lag_label} transformed"
    df[transformed_col_name] = best_x_transformed

    # Drop rows with NaN in the transformed variable (which might have resulted from the lag)
    df = df.dropna(subset=[transformed_col_name, y_col])
    
    # Compute linear regression on the transformed variable
    linreg = linregress(df[transformed_col_name], df[y_col])
    lin_intercept = linreg.intercept
    lin_slope = linreg.slope
    lin_r2 = linreg.rvalue**2

    # Test polynomial models (degrees 2 to 20) on the transformed variable
    best_poly_r2 = -np.inf
    best_poly_coeffs = None
    best_poly_deg = None
    for deg in range(2, 5):
        poly_coeffs = np.polyfit(df[transformed_col_name], df[y_col], deg=deg)
        y_poly = np.polyval(poly_coeffs, df[transformed_col_name])
        ss_res = np.sum((df[y_col] - y_poly)**2)
        ss_tot = np.sum((df[y_col] - np.mean(df[y_col]))**2)
        poly_r2 = 1 - ss_res/ss_tot
        if poly_r2 > best_poly_r2:
            best_poly_r2 = poly_r2
            best_poly_coeffs = poly_coeffs
            best_poly_deg = deg

    # Choose the final model based on the highest R²
    if best_poly_r2 > lin_r2:
        chosen_model = "poly"
        chosen_r2 = best_poly_r2
        model_params = best_poly_coeffs  # polynomial coefficients array
        chosen_poly_deg = best_poly_deg
    else:
        chosen_model = "linear"
        chosen_r2 = lin_r2
        model_params = (lin_intercept, lin_slope)
        chosen_poly_deg = None

    return df, chosen_r2, model_params, best_alpha, best_gamma, transformed_col_name, chosen_model, chosen_poly_deg

st.title('Aplikacja do liczenia zależności między zmiennymi')

# File uploader: CSV must contain at least Date, Brand, and one or more additional columns.
uploaded_file = st.file_uploader(
    """Prześlij plik CSV - pamiętaj, że powinien zawierać kolumny:
Date (data pomiaru),
Brand (dla jakiej marki jest pomiar),
oraz co najmniej jedną dodatkową kolumnę.
Na podstawie pozostałych kolumn wybierzesz zmienną niezależną (oś X) oraz zmienną zależną (prognozowaną, oś Y).""",
    type="csv"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Check that required columns exist
        required_cols = ["Date", "Brand"]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error("Plik nie zawiera wymaganych kolumn: " + ", ".join(missing_required))
            st.stop()
        
        # Allow user to filter by brand
        unique_brands = df["Brand"].unique().tolist()
        selected_brands = st.multiselect("Wybierz marki do analizy:", unique_brands, default=unique_brands)
        df = df[df["Brand"].isin(selected_brands)]
        
        # From the remaining columns, let the user choose the independent and dependent variables.
        available_cols = [col for col in df.columns if col not in ["Date", "Brand"]]
        if len(available_cols) < 2:
            st.error("Plik musi zawierać co najmniej dwie kolumny oprócz 'Date' i 'Brand'.")
            st.stop()
        
        x_col = st.selectbox("Wybierz kolumnę jako zmienną niezależną (oś X):", available_cols)
        y_col = st.selectbox("Wybierz kolumnę jako zmienną zależną (prognozowaną, oś Y):", available_cols, index=1)
        
        st.write("Podgląd danych:", df.head())
        df = df.dropna(subset=[x_col, y_col])
        
        # Allow the user to specify a lag value (applied by brand, using the Date order)
        lag = st.number_input(f"Wprowadź wartość lag dla zmiennej {x_col} (liczba całkowita, 0 = brak opóźnienia):", 
                              min_value=-100, max_value=100, value=0, step=1)
        
        (transformed_df, chosen_r2, model_params, best_alpha, best_gamma,
         transformed_col_name, chosen_model, chosen_poly_deg) = transform_data(df, x_col, y_col, lag)
        
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
        
        # Add a column with predicted y values using the chosen model
        if chosen_model == "linear":
            transformed_df[f"Predicted {y_col}"] = model_params[0] + model_params[1] * transformed_df[transformed_col_name]
        else:
            transformed_df[f"Predicted {y_col}"] = np.polyval(model_params, transformed_df[transformed_col_name])
        
        if st.button("Pokaż interaktywny wykres regresji"):
            x_max_plot = 1.2 * df["_lagged_x_"].max()
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
            # For the scatter plot, use the lagged x values stored in "_lagged_x_"
            if "Brand" in df.columns and "Date" in df.columns:
                customdata = np.stack([df["Brand"], df["Date"]], axis=-1)
                hovertemplate = (
                    f"{x_col} (lagged): %{{x}}<br>" +
                    f"{y_col}: %{{y}}<br>" +
                    "Brand: %{customdata[0]}<br>" +
                    "Date: %{customdata[1]}<extra></extra>"
                )
            else:
                customdata = None
                hovertemplate = f"{x_col} (lagged): %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
            
            fig.add_trace(go.Scatter(
                x = df["_lagged_x_"],
                y = df[y_col],
                mode = "markers",
                marker = dict(color='#17becf', size=10),
                name = "Dane oryginalne (lagged X)",
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
                xaxis_title = dict(text=f"{x_col} (lagged)", font=dict(size=18)),
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
        # Note: For a single value, lag is not applied. We only apply the transformation.
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
