# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 14:15:58 2025

@author: piotr.kociszewski

Modifications:
- Ensured pairs_df always has the expected columns, even if empty, to avoid KeyError.
- Added guard to skip plotting if no pairs were found.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go



def transform_data(df, x_col, y_col, lag=0):
    # Compute lagged spend per Brand
    if lag != 0:
        x = df.groupby('Brand')[x_col].shift(lag)
    else:
        x = df[x_col]
    df['_lagged_x_'] = x
    df = df.dropna(subset=['_lagged_x_', y_col])

    # S-curve transform (alpha/gamma grid search)
    best_r2 = -np.inf
    best_trans = df['_lagged_x_']
    best_alpha = 1.0
    best_gamma = df['_lagged_x_'].max()
    x_min, x_max = df['_lagged_x_'].min(), df['_lagged_x_'].max()
    gamma_lower = x_min + 0.3*(x_max - x_min)
    gamma_upper = x_max
    for alpha in np.arange(0.5, 3.01, 0.1):
        for gamma in np.linspace(gamma_lower, gamma_upper, 50):
            trans = 1/(1+(gamma/df['_lagged_x_'])**alpha)
            corr = df[y_col].corr(trans)
            if pd.notna(corr) and corr**2 > best_r2:
                best_r2 = corr**2
                best_trans = trans
                best_alpha = alpha
                best_gamma = gamma

    col_name = f"{x_col} (lag={lag}) transformed"
    df[col_name] = best_trans
    df = df.dropna(subset=[col_name, y_col])
    lr = linregress(df[col_name], df[y_col])
    return df, {'intercept': lr.intercept, 'slope': lr.slope}, best_alpha, best_gamma, col_name

st.title('Aplikacja: Automatyczna analiza korelacji zmiennych spend')

uploaded_file = st.file_uploader(
    """Prześlij plik CSV – pamiętaj, że powinien zawierać kolumny:
Date (data pomiaru),
Brand (dla jakiej marki jest pomiar),
oraz co najmniej jedną dodatkową kolumnę.
""",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')
    if not all(c in df.columns for c in ['Date', 'Brand']):
        st.error("Plik nie zawiera wymaganych kolumn: Date i Brand")
        st.stop()

    # Brand filter
    brands = df['Brand'].unique().tolist()
    selected_brands = st.multiselect("Wybierz marki do analizy:", brands, default=brands)
    df = df[df['Brand'].isin(selected_brands)]
    st.write(df.head())

    # Identify media spend columns and KPI candidates
    all_cols = [c for c in df.columns if c not in ['Date', 'Brand']]
    spend_cols = [c for c in all_cols if 'spend' in c.lower()]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_y = [
        c for c in numeric
        if c in all_cols
        and c not in spend_cols
        and 'spend' not in c.lower()
        and 'impressions' not in c.lower()
        and 'grp' not in c.lower()
    ]

    if not spend_cols or not candidate_y:
        st.error("Brak kolumn spend lub KPI (wyłączone spends, impressions, grp)")
        st.stop()

    # Compute best correlations over lags 0,1,2 for each brand and media spend
    pairs = []
    for brand in selected_brands:
        bdf = df[df['Brand'] == brand]
        for sp in spend_cols:
            best = {'corr': -np.inf}
            for y in candidate_y:
                for lag in [0, 1, 2]:
                    series = bdf.groupby('Brand')[sp].shift(lag) if lag else bdf[sp]
                    valid = pd.concat([series, bdf[y]], axis=1).dropna()
                    if valid.empty:
                        continue
                    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                    if pd.notna(corr) and (corr) > (best['corr']):
                        best = {'brand': brand, 'spend': sp, 'y': y, 'lag': lag, 'corr': corr}
            if best['corr'] != -np.inf:
                pairs.append({
                    'Brand': best['brand'],
                    'Spend Column': best['spend'],
                    'Best Y Column': best['y'],
                    'Lag': best['lag'],
                    'Correlation': best['corr']
                })

    # Ensure DataFrame has the expected columns even if empty
    pairs_df = pd.DataFrame(pairs, columns=['Brand', 'Spend Column', 'Best Y Column', 'Lag', 'Correlation'])

    st.markdown("### Lista par: marki, media spend vs KPI z najlepszym lagem")
    if pairs_df.empty:
        st.write("Brak dostępnych par do wyświetlenia.")
    else:
        st.dataframe(pairs_df)
        st.download_button("Pobierz listę par jako CSV", pairs_df.to_csv(index=False), file_name="lista_par.csv")

        
        # Regression charts with annotated lag
        st.markdown("## Wykresy regresji z wybranym lagem")
        for brand in selected_brands:
            st.subheader(f"Analiza dla marki: {brand}")
            bdf = df[df['Brand'] == brand]
            for sp in spend_cols:
                row = pairs_df[(pairs_df['Brand'] == brand) & (pairs_df['Spend Column'] == sp)]
                if row.empty:
                    continue
                y = row['Best Y Column'].iloc[0]
                lag = int(row['Lag'].iloc[0])
                dft, model, alpha, gamma, col_trans = transform_data(bdf.copy(), sp, y, lag)
                if dft.empty:
                    st.write(f"Brak danych dla {sp} (lag={lag}) vs {y}.")
                    continue
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dft['_lagged_x_'],
                    y=dft[y],
                    mode='markers',
                    name='Data'
                ))
                x_line = np.linspace(1e-6, 1.2 * dft['_lagged_x_'].max(), 100)
                trans_line = 1 / (1 + (gamma / x_line) ** alpha)
                y_line = model['intercept'] + model['slope'] * trans_line
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode='lines',
                    name=f'Scurve fit (lag={lag})'
                ))
                fig.update_layout(
                    title=f"{sp} vs {y} (lag={lag}, corr={row['Correlation'].iloc[0]:.2f})",
                    xaxis_title=f"{sp} (lag={lag})",
                    yaxis_title=y
                )
                st.plotly_chart(fig)
