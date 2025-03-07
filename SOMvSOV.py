# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:15:58 2025

@author: piotr.kociszewski
"""

import streamlit as st
import pandas as pd
import numpy as np
import itertools
from scipy.stats import linregress
import plotly.graph_objects as go


def transform_data(df, x_col, y_col, lag=0):
    # Compute the lagged X variable (per Brand)
    if lag != 0:
        x_series = df.groupby("Brand")[x_col].shift(lag)
    else:
        x_series = df[x_col]
    df["_lagged_x_"] = x_series
    df = df.dropna(subset=["_lagged_x_", y_col])
    
    # --- Scurve Model (nonlinear transformation using gamma/alpha) ---
    best_r2 = -np.inf
    best_x_transformed = df["_lagged_x_"]
    best_alpha = 0
    best_gamma = 0

    x_min = df["_lagged_x_"].min()
    x_max = df["_lagged_x_"].max()
    gamma_lower = round(x_min + 0.3 * (x_max - x_min), 4)
    gamma_upper = round(x_max, 4)

    for alpha in np.arange(0.5, 3.0+0.001, 0.1):
        for gamma in np.arange(gamma_lower, gamma_upper+0.001, (gamma_upper-gamma_lower)/100):
            new_var = 1 / (1 + (gamma / df["_lagged_x_"]) ** alpha)
            new_corr = df[y_col].corr(new_var)
            new_r2 = new_corr ** 2
            if new_r2 > best_r2:
                best_r2 = new_r2
                best_x_transformed = new_var
                best_alpha = alpha
                best_gamma = gamma

    scurve_transformed_col_name = f"{x_col} (lag={lag}) transformed"
    df[scurve_transformed_col_name] = best_x_transformed
    df = df.dropna(subset=[scurve_transformed_col_name, y_col])
    
    scurve_result = linregress(df[scurve_transformed_col_name], df[y_col])
    scurve_model = {"intercept": scurve_result.intercept, "slope": scurve_result.slope, "r2": scurve_result.rvalue**2}
    
    # --- Poly Regression Model (using strictly the original lagged X) ---
    best_poly_r2 = -np.inf
    best_poly_coeffs = None
    best_poly_deg = None
    for deg in range(2, 5):
        poly_coeffs = np.polyfit(df["_lagged_x_"], df[y_col], deg=deg)
        y_poly = np.polyval(poly_coeffs, df["_lagged_x_"])
        ss_res = np.sum((df[y_col] - y_poly) ** 2)
        ss_tot = np.sum((df[y_col] - np.mean(df[y_col])) ** 2)
        poly_r2 = 1 - ss_res / ss_tot
        if poly_r2 > best_poly_r2:
            best_poly_r2 = poly_r2
            best_poly_coeffs = poly_coeffs
            best_poly_deg = deg
    poly_model = {"coeffs": best_poly_coeffs, "deg": best_poly_deg, "r2": best_poly_r2}
    
    # --- Auto Model Selection (choose best between poly and Scurve) ---
    if best_poly_r2 > scurve_model["r2"]:
        auto_model = {"type": "poly", "params": best_poly_coeffs, "r2": best_poly_r2, "deg": best_poly_deg}
    else:
        auto_model = {"type": "scurve", "params": (scurve_model["intercept"], scurve_model["slope"]), "r2": scurve_model["r2"]}
    
    # --- Strictly Linear Regression Model (using original lagged X) ---
    strict_linreg_result = linregress(df["_lagged_x_"], df[y_col])
    strict_linear_model = {"intercept": strict_linreg_result.intercept, "slope": strict_linreg_result.slope, "r2": strict_linreg_result.rvalue**2}
    
    return df, best_r2, scurve_model, poly_model, auto_model, best_alpha, best_gamma, scurve_transformed_col_name, strict_linear_model

# ---------------- Helper functions for Brand Correlation Table with Lags ----------------
def compute_brand_correlations_with_lags(df, exclude_pairs=[], lags=[0, 1, 2]):
    rows = []
    # For Column1 we use the original numeric columns (no lag)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in ["Date", "Brand"]]
    for brand in df["Brand"].unique():
        brand_df = df[df["Brand"] == brand]
        for lag in lags:
            if lag > 0:
                lagged_df = brand_df[numeric_cols].shift(lag)
            else:
                lagged_df = brand_df[numeric_cols]
            # Append suffix "_lagX" to lagged columns
            lagged_df = lagged_df.add_suffix(f"_lag{lag}")
            temp_df = pd.concat([brand_df[numeric_cols], lagged_df], axis=1).dropna()
            if temp_df.empty:
                continue
            for col1 in numeric_cols:
                for col2 in [c for c in temp_df.columns if c.endswith(f"_lag{lag}")]:
                    base_col2 = col2.split("_lag")[0]
                    # Skip if same column (comparing a column with its own lagged version)
                    if col1 == base_col2:
                        continue
                    # Create an order-insensitive pair identifier (sorted alphabetically)
                    sorted_pair = " - ".join(sorted([col1, base_col2]))
                    r = temp_df[col1].corr(temp_df[col2])
                    if pd.notnull(r):
                        rows.append({"Brand": brand, "Lag": lag, "Column1": col1, "Column2": col2, "Pair": sorted_pair, "Correlation": r})
    if rows:
        corr_table_df = pd.DataFrame(rows)
        corr_table_df["AbsCorrelation"] = corr_table_df["Correlation"].abs()
        corr_table_df = corr_table_df.sort_values(by="AbsCorrelation", ascending=False).reset_index(drop=True)
        corr_table_df["Rank"] = corr_table_df["AbsCorrelation"].rank(ascending=False, method="min").astype(int)
        if exclude_pairs:
            # Convert exclude_pairs to sorted pairs as well.
            exclude_pairs_sorted = [" - ".join(sorted(pair.split(" - "))) for pair in exclude_pairs]
            corr_table_df = corr_table_df[~corr_table_df["Pair"].isin(exclude_pairs_sorted)]
        return corr_table_df.drop(columns=["AbsCorrelation"])
    else:
        return pd.DataFrame()

def highlight_rows(row):
    if row["Rank"] <= 5:
        return ["background-color: lightgreen"] * len(row)
    elif row["Rank"] <= 10:
        return ["background-color: orange"] * len(row)
    elif row["Rank"] <= 15:
        return ["background-color: yellow"] * len(row)
    else:
        return [""] * len(row)
# ---------------- End Helper functions ----------------

st.title('Aplikacja do liczenia zależności między zmiennymi')

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
        required_cols = ["Date", "Brand"]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            st.error("Plik nie zawiera wymaganych kolumn: " + ", ".join(missing_required))
            st.stop()
        
        unique_brands = df["Brand"].unique().tolist()
        selected_brands = st.multiselect("Wybierz marki do analizy:", unique_brands, default=unique_brands)
        df = df[df["Brand"].isin(selected_brands)]
        
        available_cols = [col for col in df.columns if col not in ["Date", "Brand"]]
        if len(available_cols) < 2:
            st.error("Plik musi zawierać co najmniej dwie kolumny oprócz 'Date' i 'Brand'.")
            st.stop()
        
        x_col = st.selectbox("Wybierz kolumnę jako zmienną niezależną (oś X):", available_cols)
        y_col = st.selectbox("Wybierz kolumnę jako zmienną zależną (prognozowaną, oś Y):", available_cols, index=1)
        
        st.write("Podgląd danych:", df.head())
        df = df.dropna(subset=[x_col, y_col])
        
        # ---------------- Display Brand Correlation Table with Lags ----------------
        # Build list of all possible pairs (order-insensitive) for exclusion.
        available_numeric = [col for col in available_cols if pd.api.types.is_numeric_dtype(df[col])]
        all_pairs = sorted(set(" - ".join(sorted(pair)) for pair in itertools.combinations(available_numeric, 2)))
        exclude_pairs = st.multiselect("Wybierz pary kolumn do pominięcia przy obliczaniu korelacji (np. 'Radio - TV'):", options=all_pairs, default=[])
        
        corr_table_df = compute_brand_correlations_with_lags(df, exclude_pairs=exclude_pairs, lags=[0, 1, 2])
        st.markdown("### Tabela korelacji dla wybranych kolumn (Column1 bez lag, Column2 z lagami 0, 1 i 2)")
        if not corr_table_df.empty:
            st.dataframe(corr_table_df.style.apply(highlight_rows, axis=1))
        else:
            st.write("Brak wystarczających danych do wygenerowania tabeli korelacji.")
        # ---------------- End Correlation Table ----------------
        
        lag = st.number_input(f"Wprowadź wartość lag dla zmiennej {x_col} (np. 1 = opóźnienie o jeden okres):", 
                              min_value=-100, max_value=100, value=0, step=1)
        
        (transformed_df, best_r2_val, scurve_model, poly_model, auto_model, 
         best_alpha, best_gamma, scurve_transformed_col_name, strict_linear_model) = transform_data(df, x_col, y_col, lag)
        
        model_choice = st.radio(
            "Wybierz model do wyświetlenia:",
            options=["Auto (najlepszy)", "Scurve", "Regresja liniowa", "Regresja wielomianowa"],
            index=0
        )
        if model_choice == "Auto (najlepszy)":
            if auto_model["type"] == "scurve":
                chosen_model = "scurve"
                chosen_r2 = auto_model["r2"]
                model_params = auto_model["params"]
            else:
                chosen_model = "poly"
                chosen_r2 = poly_model["r2"]
                model_params = poly_model["coeffs"]
                chosen_poly_deg = poly_model["deg"]
        elif model_choice == "Scurve":
            chosen_model = "scurve"
            chosen_r2 = scurve_model["r2"]
            model_params = (scurve_model["intercept"], scurve_model["slope"])
        elif model_choice == "Regresja liniowa":
            chosen_model = "strict"
            chosen_r2 = strict_linear_model["r2"]
            model_params = (strict_linear_model["intercept"], strict_linear_model["slope"])
        else:
            chosen_model = "poly"
            chosen_r2 = poly_model["r2"]
            model_params = poly_model["coeffs"]
            chosen_poly_deg = poly_model["deg"]
        
        if chosen_r2 is not None:
            if chosen_model == "scurve":
                st.write(f"Wybrany model (Scurve) osiąga R² = {chosen_r2:.3f}")
                st.write("Model Scurve:")
                st.write("Stała (intercept):", model_params[0])
                st.write("Współczynnik (beta):", model_params[1])
            elif chosen_model == "strict":
                st.write(f"Wybrany model (Regresja liniowa) osiąga R² = {chosen_r2:.3f}")
                st.write("Model regresji liniowej (strict):")
                st.write("Stała (intercept):", model_params[0])
                st.write("Współczynnik (beta):", model_params[1])
            else:
                st.write(f"Wybrany model (Regresja wielomianowa stopnia {chosen_poly_deg}) osiąga R² = {chosen_r2:.3f}")
                st.write("Współczynniki modelu wielomianowego:", model_params)
        else:
            st.warning("Nie udało się obliczyć współczynnika R².")
            
        st.write("Dane po transformacji:", transformed_df.head())
        
        # ---------------- Predictions and Regression Line ----------------
        if chosen_model == "strict":
            transformed_df[f"Predicted {y_col}"] = model_params[0] + model_params[1] * df["_lagged_x_"]
            x_for_line = np.linspace(1e-6, 1.2 * df["_lagged_x_"].max(), 100)
            y_line = model_params[0] + model_params[1] * x_for_line
            model_label = f"Regresja liniowa: y = {round(model_params[0],2)} + {round(model_params[1],2)} * {x_col}"
        elif chosen_model == "scurve":
            transformed_df[f"Predicted {y_col}"] = model_params[0] + model_params[1] * transformed_df[scurve_transformed_col_name]
            x_for_line = np.linspace(1e-6, 1.2 * df["_lagged_x_"].max(), 100)
            transformed_x_line = 1 / (1 + (best_gamma / x_for_line) ** best_alpha)
            y_line = model_params[0] + model_params[1] * transformed_x_line
            model_label = f"Scurve: y = {round(model_params[0],2)} + {round(model_params[1],2)} * {scurve_transformed_col_name}"
        else:  # Poly regression uses strictly the original lagged X.
            transformed_df[f"Predicted {y_col}"] = np.polyval(model_params, df["_lagged_x_"])
            x_for_line = np.linspace(1e-6, 1.2 * df["_lagged_x_"].max(), 100)
            y_line = np.polyval(model_params, x_for_line)
            model_label = f"Regresja wielomianowa (stopnia {chosen_poly_deg})"
        
        x_max_plot = 1.2 * df["_lagged_x_"].max()
        y_max_plot = 1.2 * df[y_col].max()
        # ---------------- End Predictions and Regression Line ----------------
        
        # ---------------- Interactive Plot with Color Coding for Brands ----------------
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if "Brand" in df.columns and "Date" in df.columns:
            brands = sorted(df["Brand"].unique())
            for i, brand in enumerate(brands):
                brand_df = df[df["Brand"] == brand]
                customdata = np.stack([brand_df["Brand"], brand_df["Date"]], axis=-1)
                fig.add_trace(go.Scatter(
                    x = brand_df["_lagged_x_"],
                    y = brand_df[y_col],
                    mode = "markers",
                    marker = dict(color=colors[i % len(colors)], size=10),
                    name = brand,
                    customdata = customdata,
                    hovertemplate = (
                        f"{x_col} (lagged): %{{x}}<br>{y_col}: %{{y}}<br>" +
                        "Brand: %{customdata[0]}<br>Date: %{customdata[1]}<extra></extra>"
                    )
                ))
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
            x = x_for_line,
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
        # ---------------- End Interactive Plot ----------------
        

        
        st.subheader("Prognoza dla nowej wartości zmiennej niezależnej")
        user_input = st.number_input(f"Wprowadź wartość {x_col}:", 
                                     min_value=float(df[x_col].min()),
                                     max_value=float(df[x_col].max()),
                                     value=float(df[x_col].mean()))
        if chosen_model == "strict":
            predicted_y = model_params[0] + model_params[1] * user_input
        elif chosen_model == "scurve":
            transformed_user_input = 1 / (1 + (best_gamma / user_input)**best_alpha)
            predicted_y = model_params[0] + model_params[1] * transformed_user_input
        else:
            predicted_y = np.polyval(model_params, user_input)
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
