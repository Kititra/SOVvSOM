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
    # Ustal dolną granicę jako x_min + 0.3*(x_max - x_min) i górną jako x_max
    gamma_lower = round(x_min + 0.3*(x_max - x_min), 4)
    gamma_upper = round(x_max, 4)
    
    # Iteracja przez wszystkie kombinacje alpha i gamma
    # Używamy kroku dla gamma równym (gamma_upper - gamma_lower) / 100
    for alpha in np.arange(0.5, 3.0 + 0.001, 0.1):
        for gamma in np.arange(gamma_lower, gamma_upper + 0.001, (gamma_upper - gamma_lower) / 1000):
            # new_var = 1 / (1 + (gamma / x)^alpha)
            new_var = 1 / (1 + (gamma / df[x_col])**alpha)
            new_corr = df[y_col].corr(new_var)
            new_r2 = new_corr**2
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_x_transformed = new_var
                best_alpha = alpha
                best_gamma = gamma

    # Dodajemy do DataFrame nową kolumnę z transformacją zmiennej niezależnej
    transformed_col_name = f"{x_col} transformed"
    df[transformed_col_name] = best_x_transformed

    # Obliczamy regresję liniową: y = y_col, x = przetransformowanej zmiennej x
    reg_result = linregress(df[transformed_col_name], df[y_col])
    intercept = reg_result.intercept
    slope = reg_result.slope

    return df, best_r2, intercept, slope, best_alpha, best_gamma, transformed_col_name

st.title('Aplikacja do liczenia zależności między zmiennymi')

# Umożliwienie przesłania pliku CSV z wieloliniowym opisem
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
        # Wczytanie danych z pliku CSV
        df = pd.read_csv(uploaded_file)
        
        # Sprawdzenie, czy plik zawiera kolumny Date i Brand
        required_cols = ["Date", "Brand"]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error("Plik nie zawiera wymaganych kolumn: " + ", ".join(missing_required))
            st.stop()
        
        # Pozostałe kolumny – użytkownik wybiera spośród nich zmienne X i Y
        available_cols = [col for col in df.columns if col not in ["Date", "Brand"]]
        if len(available_cols) < 2:
            st.error("Plik musi zawierać co najmniej dwie kolumny oprócz 'Date' i 'Brand'.")
            st.stop()
        
        x_col = st.selectbox("Wybierz kolumnę jako zmienną niezależną (oś X):", available_cols)
        y_col = st.selectbox("Wybierz kolumnę jako zmienną zależną (prognozowaną, oś Y):", available_cols, index=1)
        
        st.write("Podgląd danych:", df.head())
        
        # Usuwamy obserwacje, w których brakuje wartości w wybranych zmiennych
        df = df.dropna(subset=[x_col, y_col])
        
        # Transformacja danych i obliczenia regresji – kryterium wyboru to maksymalne R²
        (transformed_df, best_r2, intercept, slope,
         best_alpha, best_gamma, transformed_col_name) = transform_data(df, x_col, y_col)
        
        if best_r2 is not None:
            st.write(f"Najlepszy współczynnik R² między '{y_col}' a '{transformed_col_name}':", best_r2)
            st.write("Model regresji liniowej:")
            st.write("Stała (intercept):", intercept)
            st.write(f"Współczynnik przy '{transformed_col_name}' (beta):", slope)
        else:
            st.warning("Nie udało się obliczyć współczynnika R².")
        
        st.write("Dane po transformacji:", transformed_df.head())
        
        # Dodaj kolumnę z prognozowanymi wartościami dla y
        transformed_df[f"Predicted {y_col}"] = intercept + slope * transformed_df[transformed_col_name]
        
        # Interaktywny wykres regresji z Plotly
        if st.button("Pokaż interaktywny wykres regresji"):
            # Ustalenie zakresów osi: wykorzystujemy wartości oryginalnej zmiennej x i y
            x_max_plot = 1.2 * df[x_col].max()
            y_max_plot = 1.2 * df[y_col].max()
            
            # Przygotowanie punktów do wykresu linii regresji
            x_values = np.linspace(1e-6, x_max_plot, 100)
            transformed_x_line = 1 / (1 + (best_gamma / x_values)**best_alpha)
            y_line = intercept + slope * transformed_x_line
            
            fig = go.Figure()
            
            # Przygotowanie danych do hover – wyświetlamy również Brand i Date
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
            
            # Dodanie wykresu punktowego oryginalnych danych
            fig.add_trace(go.Scatter(
                x = df[x_col],
                y = df[y_col],
                mode = "markers",
                marker = dict(color='#17becf', size=10),
                name = "Dane oryginalne",
                customdata = customdata,
                hovertemplate = hovertemplate
            ))
            
            # Dodanie linii regresji
            fig.add_trace(go.Scatter(
                x = x_values,
                y = y_line,
                mode = "lines",
                line = dict(color='#1f77b4'),
                name = f"Linia regresji: y = {round(intercept,2)} + {round(slope,2)} * {transformed_col_name}"
            ))
            
            fig.update_layout(
                title = dict(text="Interaktywny wykres regresji", font=dict(size=20)),
                xaxis_title = dict(text=x_col, font=dict(size=18)),
                yaxis_title = dict(text=y_col, font=dict(size=18)),
                xaxis = dict(range=[0, x_max_plot], tickfont=dict(size=16)),
                yaxis = dict(range=[0, y_max_plot], tickfont=dict(size=16))
            )
            st.plotly_chart(fig)
        
        # Sekcja do prognozowania – użytkownik wprowadza wartość zmiennej niezależnej
        st.subheader("Prognoza dla nowej wartości zmiennej niezależnej")
        user_input = st.number_input(f"Wprowadź wartość {x_col}:", 
                                     min_value=float(df[x_col].min()),
                                     max_value=float(df[x_col].max()),
                                     value=float(df[x_col].mean()))
        # Przekształcamy wartość wprowadzoną przez użytkownika według znalezionych parametrów
        transformed_input = 1 / (1 + (best_gamma / user_input)**best_alpha)
        predicted_y = intercept + slope * transformed_input
        st.write(f"Prognozowana wartość {y_col} dla {x_col} = {user_input}: {predicted_y}")
        
        # Pobieranie przekształconego pliku CSV – zawiera on oryginalne kolumny, kolumnę transformacji oraz prognozowane wartości
        csv = transformed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Pobierz przekształcony plik CSV",
            data=csv,
            file_name='przeksztalcony.csv',
            mime='text/csv'
        )
        
    except Exception as e:
        st.error(f"Błąd przetwarzania pliku: {e}")
