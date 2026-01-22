import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import plotly.graph_objects as go # type: ignore
import pdfplumber
import re

st.set_page_config(
    page_title="Sistema de análisis",
    layout="wide"
)

st.title(" Sistema de gestión de procesos - Guamito S.A.S.")
st.markdown("<p style='opacity:0.7; font-size:14px;'>Sistema de gestión y programación de procesos en campo para la optimización de decisiones</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Producción", 
    "Comparación", 
    "Descartes", 
    "Incidencias", 
    "Proyección"
])

with tab1:

    st.header("Producción")

    # =========================
    # CARGA ARCHIVO INDEPENDIENTE
    # =========================
    prod_file = st.file_uploader(
        "Sube el archivo de producción",
        type=["xlsx", "xls", "csv"],
        key="prod_file"
    )

    if prod_file is None:
        st.info(" Sube un archivo para ver el dashboard de producción")
    else:
        # =========================
        # LECTURA DEL ARCHIVO
        # =========================
        df_full = (
            pd.read_csv(prod_file) if prod_file.name.endswith(".csv")
            else pd.read_excel(prod_file)
        )

        columnas = ["Semana", "Lote", "Variedad", "T Cortados"]
        faltantes = [c for c in columnas if c not in df_full.columns]

        if faltantes:
            st.error(f"Faltan columnas: {faltantes}")
        else:
            df_full = df_full[columnas].copy()
            df_full = df_full[df_full["Lote"].notna()]

            df_full["T Cortados"] = (
                df_full["T Cortados"]
                .astype(str)
                .str.replace(r"[^0-9.]", "", regex=True)
                .astype(float)
            )

            df_full["Semana"] = pd.to_numeric(
                df_full["Semana"], errors="coerce"
            ).astype("Int64")

            semanas_disponibles = sorted(df_full["Semana"].dropna().unique())

            # =========================
            # LAYOUT FILTROS
            # =========================
            col_filtros, col_main = st.columns([1, 4])

            with col_filtros:
                st.subheader("Filtros")

                modo = st.radio(
                    "Modo de consulta",
                    ["Todas", "Una semana", "Rango"]
                )

                if modo == "Una semana":
                    semana = st.selectbox("Semana", semanas_disponibles)
                elif modo == "Rango":
                    inicio = st.selectbox("Semana inicio", semanas_disponibles)
                    fin = st.selectbox("Semana fin", semanas_disponibles)

                st.markdown("---")
                umbral_caida = st.slider(
                    "Alerta caída (%)",
                    5, 50, 15, 5,
                    help="Caídas mayores a este porcentaje se marcan en rojo"
                )

            # =========================
            # FILTRADO SEGÚN MODO
            # =========================
            if modo == "Todas":
                df = df_full.copy()
                label = "Todas las semanas"
            elif modo == "Una semana":
                df = df_full[df_full["Semana"] == semana]
                label = f"Semana {semana}"
            else:
                df = df_full[
                    (df_full["Semana"] >= inicio) &
                    (df_full["Semana"] <= fin)
                ]
                label = f"Semanas {inicio}–{fin}"

            if df.empty:
                st.warning("No hay datos para el filtro seleccionado")
            else:
                # =========================
                # CONTENIDO PRINCIPAL
                # =========================
                with col_main:
                    total_tallos = int(df["T Cortados"].sum())

                    st.markdown(
                        f"""
                        <div style="
                            background:#f5f7fa;
                            padding:24px;
                            border-radius:12px;
                            text-align:center;
                            font-size:34px;
                            font-weight:700;
                            margin-bottom:20px;">
                             {total_tallos:,.0f} tallos
                            <div style="font-size:16px;color:#555;">
                                Producción total ({label})
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # =========================
                    # TABLAS BASE
                    # =========================
                    col_t1, col_t2 = st.columns([1, 1.4])

                    tabla_lote = (
                        df.groupby("Lote", as_index=False)["T Cortados"]
                        .sum()
                        .sort_values("T Cortados", ascending=False)
                    )

                    with col_t1:
                        st.subheader("Producción por lote")
                        st.caption("Total de tallos por lote (ordenado de mayor a menor)")
                        st.dataframe(tabla_lote, hide_index=True, use_container_width=True)

                    with col_t2:
                        st.subheader("Producción por lote y variedad")
                        st.caption("Distribución interna de cada lote")

                        tabla_var = df.pivot_table(
                            index="Lote",
                            columns="Variedad",
                            values="T Cortados",
                            aggfunc="sum",
                            fill_value=0
                        )
                        tabla_var["TOTAL"] = tabla_var.sum(axis=1)
                        tabla_var = tabla_var.sort_values("TOTAL", ascending=False)

                        st.dataframe(tabla_var, use_container_width=True)

                    # =========================
                    # GRÁFICO APILADO
                    # =========================
                    st.markdown("---")
                    st.subheader("Distribución de producción por lote")
                    st.caption("Cada barra es un lote. Colores = variedades.")

                    graf = tabla_var.drop(columns=["TOTAL"])

                    fig = go.Figure()
                    for variedad in graf.columns:
                        fig.add_bar(
                            x=graf.index,
                            y=graf[variedad],
                            name=variedad,
                            hovertemplate=(
                                "<b>Lote:</b> %{x}<br>"
                                f"<b>Variedad:</b> {variedad}<br>"
                                "<b>Tallos:</b> %{y:,.0f}<extra></extra>"
                            )
                        )

                    fig.update_layout(
                        barmode="stack",
                        height=450,
                        xaxis_title="Lote",
                        yaxis_title="Tallos",
                        legend_title="Variedad"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # =========================
                    # COMPARACIÓN SEMANA VS SEMANA
                    # =========================
                    st.markdown("---")
                    st.subheader("Comparación semana vs semana")
                    st.caption(
                        "Se comparan semanas consecutivas dentro del rango seleccionado.\n\n"
                        "🟢 Verde: mejora o estabilidad\n"
                        "🟡 Amarillo: caída leve\n"
                        "🔴 Rojo: caída fuerte (alerta)"
                    )

                    semanas_rango = sorted(df["Semana"].dropna().unique())
                    if len(semanas_rango) < 2:
                        st.info("Se necesitan al menos dos semanas para realizar la comparación.")
                    else:
                        modo_comp = st.radio(
                            "Modo de comparación",
                            ["Automática (recomendada)", "Elegir semanas manualmente"],
                            horizontal=True,
                            key="modo_comp_prod"
                        )

                        s0, s1 = None, None

                        if modo_comp == "Automática (recomendada)":
                            if modo == "Una semana":
                                s1 = semana
                                if s1 in semanas_rango:
                                    idx = semanas_rango.index(s1)
                                    if idx > 0:
                                        s0 = semanas_rango[idx - 1]
                            else:
                                s1 = semanas_rango[-1]
                                s0 = semanas_rango[-2]
                        else:
                            col_s0, col_s1 = st.columns(2)
                            with col_s0:
                                s0 = st.selectbox(
                                    "Semana base",
                                    semanas_rango,
                                    index=len(semanas_rango) - 2
                                )
                            with col_s1:
                                s1 = st.selectbox(
                                    "Semana a comparar",
                                    semanas_rango,
                                    index=len(semanas_rango) - 1
                                )
                            if s1 <= s0:
                                st.warning("La semana a comparar debe ser mayor que la semana base.")
                                s0 = None

                        if s0 is None or s1 is None:
                            st.info("No se encontró una semana anterior para comparar.")
                        else:
                            st.markdown(
                                f"**Comparación activa:**  Semana base: **{s0}**, Semana comparada: **{s1}**"
                            )

                            w0 = df_full[df_full["Semana"] == s0].groupby("Lote")["T Cortados"].sum()
                            w1 = df_full[df_full["Semana"] == s1].groupby("Lote")["T Cortados"].sum()

                            comp = pd.concat([w0, w1], axis=1).fillna(0)
                            comp.columns = [f"Semana {s0}", f"Semana {s1}"]
                            comp["Δ Tallos"] = comp.iloc[:, 1] - comp.iloc[:, 0]
                            comp["Δ %"] = np.where(
                                comp.iloc[:, 0] == 0,
                                0,
                                (comp["Δ Tallos"] / comp.iloc[:, 0]) * 100
                            )

                            def color_fila(row):
                                if row["Δ %"] <= -umbral_caida:
                                    return ["background-color:#f8d7da"] * len(row)
                                elif row["Δ %"] < 0:
                                    return ["background-color:#fff3cd"] * len(row)
                                else:
                                    return ["background-color:#d4edda"] * len(row)

                            st.dataframe(
                                comp.reset_index().style.apply(color_fila, axis=1).format({
                                    f"Semana {s0}": "{:,.0f}",
                                    f"Semana {s1}": "{:,.0f}",
                                    "Δ Tallos": "{:+,.0f}",
                                    "Δ %": "{:+.1f}%"
                                }),
                                use_container_width=True
                            )

                    # =========================
                    # EVOLUCIÓN SEMANAL
                    # =========================
                    # =========================
                    # EVOLUCIÓN SEMANAL (Corregido)
                    # =========================
                    if modo != "Una semana":
                        st.markdown("---")
                        st.subheader(" Evolución de Producción")
                        trend = df.groupby("Semana")["T Cortados"].sum().reset_index()

                        fig = go.Figure()
                        
                        # Añadimos la línea con área rellena
                        fig.add_scatter(
                            x=trend["Semana"],
                            y=trend["T Cortados"],
                            mode="lines+markers",
                            fill='tozeroy', 
                            line=dict(color='#1f77b4', width=3), 
                            # Corregido: 'line' dentro de 'marker' controla el borde del punto
                            marker=dict(
                                size=10, 
                                color='white', 
                                line=dict(color='#1f77b4', width=2) 
                            ),
                            hovertemplate="<b>Semana %{x}</b><br>Tallos: %{y:,.0f}<extra></extra>"
                        )

                        fig.update_layout(
                            height=350,
                            margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Semana",
                            yaxis_title="Tallos",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=True, gridcolor='#e5e5e5')
                        )

                        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Encabezado original con fondo oscuro
    st.header(" Análisis de Eficiencia: Producción vs. Descarte")

    # --- 1. CARGA DE ARCHIVOS ---
    c_u1, c_u2 = st.columns(2)
    with c_u1:
        archivo_prod = st.file_uploader("Archivo producción", type=["csv", "xlsx"], key="prod_csv_comp")
    with c_u2:
        archivo_desc = st.file_uploader("Archivo descartes", type=["csv", "xlsx"], key="desc_csv_comp")

    if archivo_prod and archivo_desc:
        # Lectura de datos
        def leer_archivo(archivo):
            if archivo.name.endswith(".csv"): return pd.read_csv(archivo)
            return pd.read_excel(archivo)

        df_p_raw = leer_archivo(archivo_prod)
        df_d_raw = leer_archivo(archivo_desc)
        df_p_raw.columns = df_p_raw.columns.str.strip()
        df_d_raw.columns = df_d_raw.columns.str.strip()

        # Limpieza (Lógica original)
        df_p = df_p_raw[["Semana", "Lote", "T Cortados"]].copy()
        df_p["T Cortados"] = pd.to_numeric(df_p["T Cortados"].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        df_p = df_p.dropna(subset=["Lote", "T Cortados", "Semana"])
        df_p["Semana"] = df_p["Semana"].astype(int)

        df_d = df_d_raw[["Semana", "Lote", "Total"]].copy()
        df_d["Total"] = pd.to_numeric(df_d["Total"].astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        df_d = df_d.dropna(subset=["Lote", "Total", "Semana"])
        df_d["Semana"] = df_d["Semana"].astype(int)

        semanas_comunes = sorted(list(set(df_p["Semana"].unique()) & set(df_d["Semana"].unique())))

        # ==========================================
        # 2. LAYOUT DE FILTROS (IZQUIERDA)
        # ==========================================
        col_filtros, col_main = st.columns([1, 4])

        with col_filtros:
            st.subheader("Filtros")
            modo = st.radio("Modo de consulta", ["Todas", "Una semana", "Rango"], key="modo_vfinal")

            if modo == "Una semana":
                sem_sel = st.selectbox("Semana", semanas_comunes, key="s_vfinal")
                f_ini = f_fin = sem_sel
                label_t = f"Semana {sem_sel}"
            elif modo == "Rango":
                f_ini = st.selectbox("Inicio", semanas_comunes, index=0, key="si_vfinal")
                f_fin = st.selectbox("Fin", semanas_comunes, index=len(semanas_comunes)-1, key="sf_vfinal")
                label_t = f"Semanas {f_ini} a {f_fin}"
            else:
                f_ini, f_fin = semanas_comunes[0], semanas_comunes[-1]
                label_t = "Histórico Completo"

        # Filtrado
        df_p_filt = df_p[(df_p["Semana"] >= f_ini) & (df_p["Semana"] <= f_fin)]
        df_d_filt = df_d[(df_d["Semana"] >= f_ini) & (df_d["Semana"] <= f_fin)]

        # ==========================================
        # 3. CONTENIDO PRINCIPAL (DERECHA)
        # ==========================================
        with col_main:
            # Cálculos
            prod_lote = df_p_filt.groupby("Lote")["T Cortados"].sum().reset_index()
            desc_lote = df_d_filt.groupby("Lote")["Total"].sum().reset_index()
            df_union = pd.merge(prod_lote, desc_lote, on="Lote", how="outer").fillna(0)
            df_union.columns = ["Lote", "Producción", "Descartes"]
            
            t_prod = df_union["Producción"].sum()
            t_desc = df_union["Descartes"].sum()
            pct_total = (t_desc / t_prod * 100) if t_prod > 0 else 0

            # Métricas con tus colores
            m1, m2, m3 = st.columns(3)
            m1.metric("Tallos cortados", f"{t_prod:,.0f}")
            m2.metric("Total descarte", f"{t_desc:,.0f}", f"{pct_total:.1f}% Ratio", delta_color="inverse")
            m3.metric("Eficiencia", f"{100-pct_total:.1f}%")

            st.markdown("---")
            st.subheader(f"Detalle por lote - {label_t}")
            
            # --- GRÁFICA 1: BARRAS COMPARATIVAS ---
            import plotly.graph_objects as go
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=df_union["Lote"], y=df_union["Producción"], name="Producción", marker_color="#3b82f6"))
            fig_bar.add_trace(go.Bar(x=df_union["Lote"], y=df_union["Descartes"], name="Descartes", marker_color="#ef4444"))
            
            fig_bar.update_layout(
                barmode='group', height=400, 
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- GRÁFICA 2: EVOLUCIÓN TEMPORAL (Recuperada) ---
            if modo != "Una semana":
                st.markdown("---")
                st.subheader("Evolución de calidad en el tiempo")
                trend_p = df_p_filt.groupby("Semana")["T Cortados"].sum().reset_index()
                trend_d = df_d_filt.groupby("Semana")["Total"].sum().reset_index()
                df_trend = pd.merge(trend_p, trend_d, on="Semana")
                
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(x=df_trend["Semana"], y=df_trend["T Cortados"], name="Producción", line=dict(color="#3b82f6", width=3)))
                fig_line.add_trace(go.Scatter(x=df_trend["Semana"], y=df_trend["Total"], name="Descartes", line=dict(color="#ef4444", width=3, dash='dot')))
                
                fig_line.update_layout(height=400, xaxis_title="Semana", yaxis_title="Tallos", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_line, use_container_width=True)

            # --- TABLA CON ESTILO ---
            with st.expander("Ver tabla de datos consolidada"):
                df_union["% Descarte"] = (df_union["Descartes"] / df_union["Producción"] * 100).replace([float('inf')], 0).fillna(0)
                st.dataframe(
                    df_union.style.format({
                        "Producción": "{:,.0f}", "Descartes": "{:,.0f}", "% Descarte": "{:.2f}%"
                    }).background_gradient(subset=["% Descarte"], cmap="Reds"),
                    use_container_width=True, hide_index=True
                )
    else:
        st.info("Por favor, sube ambos archivos para activar el análisis comparativo.")

with tab3:
    st.header("Descartes")
    
    # =========================
    # CARGA ARCHIVO
    # =========================
    desc_file = st.file_uploader(
        "Sube el archivo de descartes",
        type=["xlsx", "xls", "csv"],
        key="desc_uploader_vfinal_ultra_clean"
    )

    if desc_file is not None:
        # LECTURA
        df_d = pd.read_csv(desc_file) if desc_file.name.endswith(".csv") else pd.read_excel(desc_file)
        df_d["Total"] = pd.to_numeric(df_d["Total"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
        df_d["Semana"] = pd.to_numeric(df_d["Semana"], errors="coerce").astype("Int64")
        df_d = df_d.dropna(subset=["Semana", "Total", "Lote"])

        semanas_disponibles = sorted(df_d["Semana"].unique())

        # =========================
        # LAYOUT FILTROS
        # =========================
        col_filtros, col_main = st.columns([1, 4])

        with col_filtros:
            st.subheader("Filtros")
            tipo_analisis = st.radio("Nivel de detalle", ["Global", "Por lote", "Por concepto"], key="tipo_vf")
            modo = st.radio("Modo de consulta", ["Todas", "Una semana", "Rango"], key="modo_vf")

            if modo == "Una semana":
                semana_sel = st.selectbox("Semana", semanas_disponibles, key="s_vf")
                f_inicio = f_fin = semana_sel
            elif modo == "Rango":
                f_inicio = st.selectbox("Semana inicio", semanas_disponibles, key="si_vf")
                f_fin = st.selectbox("Semana fin", semanas_disponibles, key="sf_vf")
            else:
                f_inicio, f_fin = semanas_disponibles[0], semanas_disponibles[-1]

            lote_sel = None
            concepto_sel = None
            if tipo_analisis == "Por lote":
                lote_sel = st.selectbox("Seleccionar Lote", sorted(df_d["Lote"].unique()), key="l_vf")
            elif tipo_analisis == "Por concepto":
                concepto_sel = st.selectbox("Seleccionar Concepto", sorted(df_d["Concepto"].unique()), key="c_vf")

        # FILTRADO DE DATOS
        mask = (df_d["Semana"] >= f_inicio) & (df_d["Semana"] <= f_fin)
        if lote_sel: mask &= (df_d["Lote"] == lote_sel)
        if concepto_sel: mask &= (df_d["Concepto"] == concepto_sel)
        df_view = df_d[mask]

        # =========================
        # CONTENIDO PRINCIPAL
        # =========================
        with col_main:
            if df_view.empty:
                st.warning("No hay datos para el filtro seleccionado")
            else:
                # 1. BLOQUE DE KPI (Siempre sale si hay datos)
                total_t = int(df_view["Total"].sum())
                label_p = f"Semana {f_inicio}" if modo == "Una semana" else f"Semanas {f_inicio}-{f_fin}"
                
                st.markdown(f"""
                    <div style="background:#f5f7fa; padding:24px; border-radius:12px; text-align:center; margin-bottom:20px;">
                        <h1 style="margin:0; font-size:40px;">{total_t:,.0f} tallos</h1>
                        <div style="font-size:16px;color:#555;">Descartes ({label_p})</div>
                    </div>
                """, unsafe_allow_html=True)

                # 2. MÉTRICAS SECUNDARIAS
                c1, c2 = st.columns(2)
                with c1:
                    if tipo_analisis == "Por concepto":
                        peor_lote = df_view.groupby("Lote")["Total"].sum().idxmax()
                        st.info(f"**Lote más afectado:** {peor_lote}")
                    else:
                        peor_causa = df_view.groupby("Concepto")["Total"].sum().idxmax()
                        st.info(f"**Causa crítica:** {peor_causa}")
                with c2:
                    st.info(f"**Impacto en:** {df_view['Lote'].nunique()} lotes")

                # 3. BLOQUE DE DISTRIBUCIÓN (TABLA + BARRAS)
                st.markdown("---")
                if tipo_analisis == "Por concepto":
                    st.subheader(f" Ranking de Lotes: {concepto_sel}")
                    df_plot = df_view.groupby("Lote")["Total"].sum().sort_values(ascending=False).reset_index()
                    col_t, col_g = st.columns([1, 1.4])
                    with col_t:
                        st.dataframe(df_plot.style.background_gradient(cmap="Reds", subset=["Total"]), hide_index=True, use_container_width=True)
                    with col_g:
                        fig = go.Figure(go.Bar(x=df_plot["Lote"], y=df_plot["Total"], marker_color='#ef553b'))
                        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.subheader(" Distribución por Concepto")
                    df_plot = df_view.groupby("Concepto")["Total"].sum().sort_values(ascending=False).reset_index()
                    col_t, col_g = st.columns([1, 1.4])
                    with col_t:
                        st.dataframe(df_plot.style.background_gradient(cmap="Reds", subset=["Total"]), hide_index=True, use_container_width=True)
                    with col_g:
                        fig = go.Figure(go.Bar(y=df_plot["Concepto"], x=df_plot["Total"], orientation='h', marker_color='#ef553b'))
                        fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                # 4. BLOQUE DINÁMICO (TENDENCIA Y DONA)
                # Solo se crea este bloque si hay más de una semana y no es un lote individual
                if modo != "Una semana":
                    st.markdown("---")
                    st.subheader(" Análisis de Tendencia e Impacto")
                    c_izq, c_der = st.columns(2)
                    
                    with c_izq:
                        trend = df_view.groupby("Semana")["Total"].sum().reset_index()
                        fig_trend = go.Figure(go.Scatter(x=trend["Semana"], y=trend["Total"], fill='tozeroy', mode='lines+markers', line=dict(color='#ef553b', width=3)))
                        fig_trend.update_layout(height=300, title="Evolución Semanal", margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_trend, use_container_width=True)

                    with c_der:
                        # Solo mostrar el círculo si estamos en Global o Concepto (donde hay varios lotes)
                        if tipo_analisis != "Por lote":
                            lotes_pie = df_view.groupby("Lote")["Total"].sum().sort_values(ascending=False).head(5).reset_index()
                            fig_pie = go.Figure(go.Pie(labels=lotes_pie["Lote"], values=lotes_pie["Total"], hole=.4, marker=dict(colors=['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7'])))
                            fig_pie.update_layout(height=300, title="Top 5 Lotes Críticos", margin=dict(l=0,r=0,t=30,b=0), legend=dict(orientation="h", y=-0.2))
                            st.plotly_chart(fig_pie, use_container_width=True)

                # 5. BLOQUE DE INSIGHT (REGLA 80/20)
                # Solo sale si estamos viendo causas (Global o Por Lote)
                if tipo_analisis != "Por concepto":
                    st.markdown("---")
                    causas_sum = df_view.groupby("Concepto")["Total"].sum().sort_values(ascending=False)
                    acumulado = causas_sum.cumsum() / total_t
                    principales = causas_sum[acumulado <= 0.81].index.tolist()
                    if principales:
                        st.success(f" **Insight:** El 80% de las pérdidas se concentra en: **{', '.join(principales)}**.")

    else:
        st.info(" Sube un archivo para ver el dashboard de descartes")

with tab4:
    st.header("Incidencias")
 
    # --- 1. CARGA DE ARCHIVOS ---
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        incidencias_file = st.file_uploader("Subir Incidencias (CSV)", type=["csv"], key="u_inc")
    with col_u2:
        lotes_file = st.file_uploader("Subir Lotes y Semanas (Excel)", type=["xlsx", "xls"], key="u_lot")

    if incidencias_file and lotes_file:
        # --- PROCESAMIENTO ---
        try:
            df_incidencias = pd.read_csv(incidencias_file, encoding="utf-8")
        except:
            incidencias_file.seek(0)
            df_incidencias = pd.read_csv(incidencias_file, encoding="latin-1")
        df_lotes = pd.read_excel(lotes_file)

        for df in [df_incidencias, df_lotes]:
            df.columns = df.columns.str.strip()
            if "category" in df.columns: df.rename(columns={"category": "Lote"}, inplace=True)
            df["Lote"] = df["Lote"].astype(str).str.upper().str.strip()

        plagas_disponibles = [c for c in df_incidencias.columns if c != "Lote"]
        
        # --- 2. REPORTES POR EQUIPO (Distribución Horizontal) ---
        st.divider()
        st.subheader(" Distribución por Equipos de Aplicación")
        
        plagas_eq_default = [p for p in ["Ceniza", "Thrips", "Acaro Blanco", "Acaro Rojo"] if p in plagas_disponibles]
        plagas_eq = st.multiselect("Seleccionar plagas para visualizar en equipos:", plagas_disponibles, default=plagas_eq_default)

        rangos = {
            "AQ1": {"blanca": (8, 16), "azul": (8, 26), "shocking": (8, 26), "verde": (8, 16)},
            "AQ3": {"blanca": (17, 33), "azul": (27, 43), "shocking": (27, 43), "verde": (17, 29)},
            "AQ5": {"blanca": (19, 34), "azul": (29, 44), "shocking": (29, 44), "verde": (19, 30)},
        }

        def get_color_grupo(lote_str):
            l = lote_str.lower()
            if "blanca" in l: return "blanca"
            if "azul" in l: return "azul"
            if "shocking" in l: return "shocking"
            if any(x in l for x in ["lima", "esmeralda", "verde"]): return "verde"
            return "otros"

        df_lotes["color_grupo"] = df_lotes["Lote"].apply(get_color_grupo)

        # Creación de 3 columnas para AQ1, AQ3 y AQ5
        col_aq1, col_aq3, col_aq5 = st.columns(3)
        cols_equipos = {"AQ1": col_aq1, "AQ3": col_aq3, "AQ5": col_aq5}

        for eq, col_ui in cols_equipos.items():
            with col_ui:
                st.markdown(f"### {eq}")
                
                def check_pertenencia(row):
                    grupo = row["color_grupo"]
                    if grupo in rangos[eq]:
                        inicio, fin = rangos[eq][grupo]
                        return inicio <= row["Semana"] <= fin
                    return False

                lotes_filtrados = df_lotes[df_lotes.apply(check_pertenencia, axis=1)].copy()
                df_merge = pd.merge(lotes_filtrados, df_incidencias, on="Lote", how="left").fillna(0)

                if not df_merge.empty:
                    for color in ["blanca", "azul", "shocking", "verde"]:
                        sub_df = df_merge[df_merge["color_grupo"] == color]
                        if not sub_df.empty:
                            with st.expander(f"Bloque {color.upper()}"):
                                st.dataframe(
                                    sub_df[["Lote"] + plagas_eq]
                                    .sort_values(by=plagas_eq[0] if plagas_eq else "Lote", ascending=False)
                                    .style.format({p: "{:.0f}%" for p in plagas_eq})
                                    .background_gradient(subset=plagas_eq, cmap="YlOrRd"),
                                    use_container_width=True, hide_index=True
                                )
                else:
                    st.caption("Sin lotes programados.")

        # --- 3. RANKING VISUAL DE INCIDENCIA TOTAL (GRÁFICO VERTICAL) ---
        st.divider()
        st.subheader(" Panorama General de Incidencia")
        
        plagas_rank_sel = st.multiselect(
            "Seleccione plaga(s) para evaluar todos los lotes:",
            options=plagas_disponibles,
            default=[] 
        )
        
        if plagas_rank_sel:
            import plotly.express as px
            
            # Preparar datos de TODOS los lotes
            df_plot = df_incidencias.copy()
            # Calculamos la carga total para ordenar de mayor a menor
            df_plot["Carga Total"] = df_plot[plagas_rank_sel].sum(axis=1)
            df_plot = df_plot.sort_values("Carga Total", ascending=False)

            if not df_plot.empty:
                # Ancho dinámico: si hay muchos lotes, el gráfico se estira horizontalmente
                ancho_dinamico = max(800, len(df_plot) * 35)

                # Gráfico de Barras Vertical
                fig = px.bar(
                    df_plot, 
                    x="Lote", 
                    y=plagas_rank_sel,
                    title="Presión Sanitaria por Lote (Ordenado de Mayor a Menor)",
                    labels={"value": "Porcentaje Acumulado (%)", "variable": "Plaga"},
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                
                fig.update_layout(
                    barmode='stack', 
                    height=550,
                    width=ancho_dinamico,
                    margin=dict(l=50, r=150, t=60, b=150), # R=150 da espacio a la leyenda a la derecha
                    xaxis_title="Lotes",
                    yaxis_title="Incidencia Total (%)",
                    # CONFIGURACIÓN DE LEYENDA A LA DERECHA
                    showlegend=True,
                    legend=dict(
                        orientation="v",      # Vertical
                        yanchor="top",        # Anclado arriba
                        y=1,                  # Al tope del gráfico
                        xanchor="left",       # Anclado a la izquierda de su posición
                        x=1.02                # Justo a la derecha del gráfico (1 es el final del gráfico)
                    ),
                    xaxis={'categoryorder':'total descending', 'tickangle': -90}, # Nombres de lotes verticales
                    plot_bgcolor='rgba(240,242,246,0.5)'
                )
                
                # Para que se pueda ver todo si hay muchos lotes, usamos un contenedor con scroll
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos positivos para las plagas seleccionadas.")
        else:
            st.info("Seleccione plagas para generar el análisis visual.")


    st.header("Mapa de lotes según incidencia")

    # =========================================================
    # 1. COORDENADAS REALES (AutoCAD)
    # =========================================================

    lotes_reales = {
        "L01 H Blanca": [
            (2940.6253, 1506.1139),(2954.5914, 1515.2522),(2982.0976, 1511.1529),(2982.0976, 1562.3825),(2980.8440, 1565.3833),(2980.8440, 1595.2818),
            (2976.3065, 1602.4818),(2976.3065, 1609.5245),(2976.3065, 1613.0826),(2973.4880, 1615.1670),(2976.4302, 1620.5644),(2972.3429, 1636.4529),
            (2956.8885, 1625.7896),(2956.8885, 1622.9168),(2950.0599, 1618.2485),(2947.1846, 1618.2485),(2928.9449, 1610.6176),(2918.7917, 1610.6176),
            (2917.7016, 1536.3908),(2919.1759, 1533.3965)
        ]
        ,
        "L02 H Blanca": [
            (2918.78, 1610.03),(2917.70, 1536.39),(2919.17, 1533.39),(2940.62, 1506.11), (2896.45, 1486.34),(2876.44, 1519.97),
            (2873.85, 1530.13),(2873.85, 1541.91),(2873.85, 1544.80),(2876.63, 1590.60), (2886.69, 1594.77),(2893.58, 1594.77),
            (2901.55, 1601.69),(2905.74, 1606.70),(2918.79, 1610.62),(2917.70, 1536.39), (2918.78, 1610.03)
        ]
        ,
        "L03 H Blanca": [
            (2973.5677, 1403.7200),(2973.5677, 1416.4243),(2971.4337, 1430.3780),(2969.9983, 1442.6216),(2964.9938, 1459.6527),(2953.8603, 1483.3190),
            (2940.6253, 1506.1139),(2896.4533, 1486.3429),(2900.3233, 1481.6427),(2903.3682, 1477.9445),(2914.3773, 1459.0618),(2916.6536, 1453.6934),
            (2934.4249, 1427.6024),(2939.7231, 1419.8239),(2943.5075, 1416.4748),(2947.8325, 1416.4748),(2951.5477, 1411.3190),(2953.2339, 1386.7349),
            (2957.4234, 1388.9292),(2970.2635, 1396.9052),(2972.2056, 1400.9106),
        ]
        ,
        "L04 H Blanca": [
            (2878.4757, 1430.9183),(2899.5742, 1444.7546),(2908.1139, 1449.2240),(2916.6536, 1453.6934),(2934.4249, 1427.6024),(2939.7231, 1419.8239),
            (2943.5075, 1416.4748),(2947.8325, 1416.4748),(2951.5477, 1411.3190),(2953.2339, 1386.7349),(2946.7846, 1391.1898),(2940.6636, 1391.1898),
            (2933.7097, 1387.7158),(2927.8864, 1380.3117),(2892.2583, 1397.1493),(2887.7662, 1406.8660),(2884.8670, 1413.1371),(2884.8670, 1419.6242),
        ]
        ,
        "L05 H Blanca": [
            (2896.1531, 1357.0212),(2891.1955, 1339.7116),(2894.1575, 1335.6088),(2908.4332, 1335.6088),(2916.0537, 1336.6237),(2917.3533, 1345.8314),
            (2922.9063, 1345.8314),(2922.9063, 1347.2480),(2927.6912, 1347.2480),(2928.6112, 1369.1219),(2927.8864, 1380.3117),(2892.2583, 1397.1493),
            (2892.2583, 1383.2116),(2891.1662, 1375.2445),(2894.3745, 1367.2677)
        ]
        ,
        "L06 H Lima": [
            (2887.1216, 1340.5188),(2891.1955, 1339.7116),(2896.1531, 1357.0212),(2894.3745, 1367.2677),(2891.1662, 1375.2445),(2892.2583, 1383.2116),
            (2892.2583, 1397.1493),(2887.7662, 1406.8660),(2884.8670, 1413.1371),(2884.8670, 1419.6242),(2878.4757, 1430.9183),(2867.5970, 1427.5242),
            (2861.2231, 1418.1781),(2854.1127, 1411.7591),(2850.5995, 1405.5080),(2859.1991, 1400.6748),(2862.6295, 1398.8374),(2865.9303, 1394.6382),
            (2868.4148, 1388.7539)
        ]
        ,
        "L07 H Azul": [
            (2836.4845, 1442.1773),(2847.1264, 1415.6856),(2854.1127, 1411.7591),(2861.2231, 1418.1781),(2867.5970, 1427.5242),(2855.4220, 1449.7846)
        ]
        ,
        "L7B H Blanca": [
            (2871.0812, 1457.5661),(2878.4757, 1430.9183),(2867.5970, 1427.5242),(2855.4220, 1449.7846),(2867.5448, 1454.6543)
        ]
        ,
        "L07A H Azul": [
            (2867.5448, 1481.2275),(2857.6724, 1478.6050),(2855.3240, 1475.4304),(2843.7549, 1475.0247), (2841.4052, 1505.9109),(2829.3090, 1501.9207),
            (2829.3090, 1498.9465),(2835.8292, 1443.8086),(2836.4845, 1442.1773),(2855.4220, 1449.7846),(2867.5448, 1454.6543),(2871.0812, 1457.5661)
        ]
        ,
        "L08 H Esmeralda": [
            (2859.4102, 1511.8502),(2841.4052, 1505.9109),(2843.7549, 1475.0247),(2855.3240, 1475.4304),(2857.6724, 1478.6050),(2867.5448, 1481.2275),
            (2865.8750, 1487.5134)
        ]
        ,
        "L09 H Esmeralda": [
            (2841.1226, 1575.0386),(2850.8170, 1537.9005),(2859.4102, 1511.8502),(2841.4052, 1505.9109),(2829.3090, 1501.9207),(2826.8177, 1517.7274),
            (2821.5313, 1530.5268),(2818.8529, 1536.7585),(2813.5561, 1545.6319),(2826.2427, 1553.3783),(2830.8606, 1556.1980),(2833.5544, 1565.6824),
            (2833.5544, 1571.0654)
        ]
        ,
        "Lote 10": [
            (2876.6338, 1590.5979),(2869.9037, 1590.5979),(2857.5445, 1588.5692),(2849.9736, 1585.9328),(2841.1226, 1575.0386),(2850.8170, 1537.9005),
            (2873.8540, 1541.9053),(2873.8540, 1544.7990),
        ]
        ,
        "L11 H Shocking": [
            (2908.1139, 1449.2240),(2892.7796, 1475.4316),(2871.0812, 1457.5661),(2878.4757, 1430.9183),(2899.5742, 1444.7546),
        ]
        ,
        "Espacio Blanco": [
            (2892.7796, 1475.4316),(2900.3233, 1481.6427),(2903.3682, 1477.9445),(2914.3773, 1459.0618),(2916.6536, 1453.6934),(2908.1139, 1449.2240),
        ]
        ,
        "L11A H Shocking": [
            (2890.4677, 1496.3994),(2865.8750, 1487.5134),(2867.5448, 1481.2275),(2871.0812, 1457.5661),(2900.3233, 1481.6427),(2896.4533, 1486.3429),
        ]
        ,
        "L11B H Shocking": [
            (2850.8170, 1537.9005),(2873.8540, 1541.9053),(2873.8540, 1530.1335),(2876.4398, 1519.9676),(2890.4677, 1496.3994),(2865.8750, 1487.5134),
            (2859.4102, 1511.8502),
        ]
        ,
        "L13 H Blanca": [
            (2909.8991, 1324.1231),(2969.5879, 1351.2548),(2974.5975, 1282.6577),(3008.2703, 1243.7518),(2904.7243, 1228.7246),(2919.0370, 1261.9678),
            (2908.4645, 1320.3725)
        ]
        ,
        "L14 H Esmeralda": [
            (2825.5473, 1514.0253),(2821.6577, 1525.7596),(2817.4022, 1530.0107),(2811.6104, 1530.0107),(2807.7832, 1530.0107),(2806.3003, 1527.2487),
            (2803.8999, 1522.7779),(2805.3436, 1510.2323),(2804.8666, 1500.0510),(2804.8666, 1477.9503),(2810.2980, 1477.9503),(2818.6500, 1455.0772),
            (2823.0577, 1444.2502),(2827.9015, 1443.9116),(2829.4794, 1466.4812),(2828.2310, 1477.3126),(2826.9414, 1488.5013)
        ]
        ,
        "L15 H Esmeralda": [
            (2798.3792, 1476.4070),(2790.4817, 1491.5574),(2790.4817, 1503.3936),(2794.4305, 1511.7578),(2803.8999, 1522.7779),(2805.3436, 1510.2323),
            (2804.8666, 1500.0510),(2804.8666, 1477.9503)
        ]
        ,
        "L16 H Blanca": [
            [
                (2779.7862, 1411.0371),(2786.1004, 1391.8980),(2787.7235, 1393.3717),(2787.7235, 1395.5010),(2790.0829, 1396.2615),(2790.6538, 1397.9726),
                (2795.8293, 1399.6076),(2795.8293, 1400.5201),(2812.7074, 1399.8499),(2809.3595, 1420.3665),(2781.6088, 1431.5762)
            ],
            [
                (2728.9561, 1453.1901),(2735.7782, 1458.5247),(2738.2995, 1463.8593),(2749.0919, 1424.6672),(2732.8656, 1426.8136),(2725.6933, 1447.2628)
            ]
        ]
        ,
        "L16A H Blanca": [
            (2786.1004, 1391.8980),(2779.7862, 1411.0371),(2781.6088, 1431.5762),(2761.8045, 1433.9865),(2755.0313, 1442.1639),(2752.2591, 1450.7203),(2745.5968, 1467.0868),
            (2738.2995, 1463.8593),(2749.0919, 1424.6672),(2732.8656, 1426.8136),(2740.0379, 1406.3643),(2736.7112, 1390.5785),(2740.6318, 1386.2024),(2750.8578, 1388.1245),
            (2755.3306, 1385.9740),(2753.1065, 1393.9690),(2756.1640, 1394.8195),(2756.7043, 1392.8771),(2759.1825, 1393.5665),(2759.5773, 1392.1472),(2761.8817, 1392.7882),
            (2766.3467, 1381.6589),(2766.9179, 1381.5581),(2764.8654, 1386.6751),(2767.1688, 1388.9272),(2770.6061, 1381.3943),(2774.0996, 1382.4209),(2778.1335, 1384.6643),
            (2779.8467, 1386.2198),(2777.1472, 1391.4643),(2778.7496, 1392.8821),(2777.6846, 1394.7499),(2779.9880, 1397.0020),(2783.4253, 1389.4691)
        ]
        ,
        "L16B H Blanca": [
        [
            (2763.1552, 1429.1327),(2763.1552, 1420.1778),(2770.8107, 1420.1778),(2770.8107, 1429.1327)
        ],
        [
            (2756.1640, 1394.8195),(2756.7043, 1392.8771),(2759.1825, 1393.5665),(2759.5773, 1392.1472),(2761.8817, 1392.7882),(2766.3467, 1381.6589),
            (2763.1225, 1382.2277), (2755.3306, 1385.9740),(2753.1065, 1393.9690)
        ],
        [
            (2763.9151, 1389.0428),(2766.2184, 1391.2950),(2765.2839, 1393.6244),(2762.9805, 1391.3723)
        ],
        [
            (2764.8654, 1386.6751),(2767.1688, 1388.9272),(2770.6061, 1381.3943),(2769.5710, 1381.0901),(2766.9179, 1381.5581)
        ],
        [
            (2736.7112, 1390.5785),(2735.5892, 1385.2546),(2740.6318, 1386.2024)
        ],
        [
            (2783.4253, 1389.4691),(2779.8467, 1386.2198),(2777.1472, 1391.4643),(2778.7496, 1392.8821),(2777.6846, 1394.7499),(2779.9880, 1397.0020)
        ],
        [
            (2795.0090, 1380.1318),(2790.5306, 1385.0858),(2785.7669, 1380.7794),(2790.1003, 1375.8321)
        ]
        ]
        ,
        "L26 H Blanca": [
        [
            (2790.1003, 1375.8321),(2795.0090, 1380.1318),(2790.5306, 1385.0858),(2795.4490, 1389.5320),(2801.6091, 1391.7399),(2812.0988, 1391.7399),
            (2824.7991, 1390.8304),(2827.4432, 1390.8304),(2830.8242, 1392.2163),(2835.4216, 1392.9340),(2836.4104, 1370.4789),(2833.8193, 1367.3903),
            (2829.5007, 1366.7090),(2825.9763, 1365.4996),(2820.1682, 1356.1543),(2811.5377, 1346.7154),(2797.6590, 1344.6179),(2759.4553, 1343.1273),
            (2764.0153, 1371.4858),(2774.3292, 1374.0001),(2784.1923, 1379.3559),(2785.7669, 1380.7794)
        ],
        [
            (2754.3151, 1366.5801),(2739.0186, 1365.4689),(2739.0186, 1378.0665),(2741.6820, 1378.8332),(2756.7188, 1373.8863)
        ]
        ]
        ,
        "L26C H Blanca": [
            (2733.6748, 1320.3725),(2727.5702, 1343.4487),(2727.5702, 1350.0893),(2734.3441, 1350.0893),(2739.0186, 1365.4689),(2739.0186, 1378.0665),
            (2661.3367, 1355.7024),(2673.1564, 1338.7396),(2682.9693, 1320.3725)
        ]
        ,
        "L26A H Blanca": [
            (2762.0716, 1284.8437),(2772.5313, 1250.3477),(2797.2249, 1249.5679),(2793.8774, 1288.0142),(2790.3428, 1328.0930),(2790.3428, 1337.7399),
            (2797.6590, 1344.6179),(2759.4553, 1343.1273),(2748.0589, 1342.6827),(2753.4640, 1320.3725)
        ]
        ,
        "L26B H Blanca": [
            (2691.7448, 1303.9470),(2682.9693, 1320.3725),(2733.6748, 1320.3725),(2753.4640, 1320.3725),(2762.0716, 1284.8437),(2772.5313, 1250.3477),
            (2709.3376, 1252.3431)
        ]
        ,
        "L26D": [
            (2727.5702, 1343.4487),(2748.0589, 1342.6827),(2753.4640, 1320.3725),(2733.6748, 1320.3725)
        ]
        ,
        "L25A": [
            (2697.8931, 1180.1568),(2690.0319, 1180.1568),(2690.0319, 1168.8351),(2713.8068, 1172.3007),(2713.8068, 1202.0875),(2697.8931, 1199.7786)
        ]
        ,
        "L25 H Blanca": [
            (2666.2570, 1165.3694),(2690.0319, 1168.8351),(2690.0319, 1180.1568),(2697.8931, 1180.1568),(2697.8931, 1199.7786),(2592.9465, 1184.5518),
            (2554.8238, 1149.2628),(2569.3977, 1131.3594),(2597.9237, 1155.7048),(2632.8704, 1163.1870)
        ]
        ,
        "L24A": [
            (2678.0222, 1224.9386),(2682.2077, 1197.5028),(2713.8068, 1202.0875),(2713.8068, 1212.9191),(2709.0640, 1229.6741)
        ]
        ,
        "L24 H Blanca": [
            (2682.2077, 1197.5028),(2678.0222, 1224.9386),(2586.6251, 1210.9955),(2541.9018, 1169.3736),(2554.4569, 1149.7135),(2554.8238, 1149.2628),
            (2592.9465, 1184.5518),(2682.2251, 1197.3884)
        ]
        ,
        "L23B": [
            (2698.2926, 1264.2511),(2679.7157, 1261.6312),(2684.7459, 1225.9643),(2709.0640, 1229.6741),(2704.3212, 1246.4292)
        ]
        ,
        "L23 H Blanca": [
            (2684.7459, 1225.9643),(2679.7157, 1261.6312),(2596.2562, 1249.8608),(2584.9601, 1221.9101),(2586.6251, 1210.9955),(2678.0222, 1224.9386),
            (2684.7566, 1225.8883)
        ]
        ,
        "L23A H Blanca": [
            (2541.5067, 1169.7982),(2532.7050, 1177.8181),(2529.6666, 1183.9144),(2508.7591, 1204.4175),(2549.8557, 1243.3228),(2596.2562, 1249.8608),
            (2584.9601, 1221.9101),(2586.6251, 1210.9955),(2541.9018, 1169.3736)
        ]
        ,
        "L22A H Blanca": [
            (2624.1998, 1269.4369),(2636.3695, 1287.4790),(2686.8950, 1295.9488),(2689.5473, 1290.1038),(2698.2926, 1264.2511),(2679.7157, 1261.6312),
            (2624.1998, 1253.8017)
        ]
        ,
        "L22 H Blanca": [
            (2508.7591, 1204.4175),(2481.5998, 1239.0692),(2497.7444, 1251.7230),(2551.9916, 1273.3004),(2549.8557, 1243.3228)
        ]
        ,
        "L21A": [
            (2536.0496, 1266.9593),(2516.7770, 1301.4488),(2530.6810, 1305.7233),(2551.9916, 1273.3004)
        ]
        ,
        "L21 H Blanca": [
            (2536.0496, 1266.9593),(2516.7770, 1301.4488),(2483.8875, 1283.5734),(2483.8875, 1280.4904),(2479.4322, 1271.0749),(2466.1197, 1258.8198),
            (2481.5998, 1239.0692),(2497.7444, 1251.7230)
        ]
        ,
        "L19 H Blanca": [
            (2467.9188, 1304.6638),(2483.8875, 1283.5734),(2516.7770, 1301.4488),(2530.6810, 1305.7233),(2513.8952, 1340.2450)
        ]
        ,
        "19B H Blanca": [
            (2530.6810, 1305.7233),(2513.8952, 1340.2450),(2546.6609, 1332.0385),(2555.4134, 1314.3798),(2581.1708, 1323.3951),(2577.2764, 1307.8461),
            (2556.3773, 1303.7134),(2549.9359, 1312.4627)
        ]
        ,
        "L18 H Blanca": [
            (2513.8952, 1340.2450),(2546.6609, 1332.0385),(2531.6496, 1362.3249),(2501.6211, 1365.4880),(2442.5898, 1350.6549),(2457.2596, 1331.9357),
            (2467.9188, 1304.6638)
        ]
        ,
        "L18A H Blanca": [
            (2531.6496, 1362.3249),(2555.4134, 1314.3798),(2581.1708, 1323.3951),(2624.1879, 1335.3897),(2601.8382, 1362.3249)
        ]
        ,
        "L18B H Blanca": [
            (2601.8382, 1362.3249),(2641.9436, 1379.1605),(2657.3480, 1351.1327),(2651.1762, 1347.4279),(2652.8599, 1333.6442),(2627.2198, 1317.7337),
            (2624.1879, 1335.3897)
        ]
        ,
        "L17 H Blanca": [
            (2409.9871, 1392.2571),(2433.7933, 1361.8796),(2442.5898, 1350.6549),(2501.6211, 1365.4880),(2531.6496, 1362.3249),(2601.8382, 1362.3249),
            (2595.4759, 1385.8393),(2578.4147, 1385.8393),(2465.5289, 1383.7129),(2446.1667, 1400.4666)
        ]
        ,
        "L17A": [
            (2628.5735, 1403.7995),(2641.9436, 1379.1605),(2601.8382, 1362.3249),(2595.4759, 1385.8393),(2610.4706, 1393.9760)
        ]
        ,
        "L20 H Blanca": [
            (2632.4030, 1286.8141),(2627.2198, 1317.7337),(2652.8599, 1333.6442),(2651.1762, 1347.4279),(2657.3480, 1351.1327),(2675.8123, 1320.3725),
            (2686.8950, 1295.9488),(2636.3695, 1287.4790),(2632.3882, 1286.9021)
        ]
        ,
        "L20A H BLANCA": [
            (2632.3882, 1286.9021),(2636.3695, 1287.4790),(2624.1998, 1269.4369),(2624.1998, 1253.8017),(2596.2562, 1249.8608),(2549.8557, 1243.3228),
            (2551.9916, 1273.3004),(2565.2002, 1277.1663),(2612.7507, 1284.0566),(2610.8430, 1312.3232),(2627.2198, 1317.7337),(2632.4030, 1286.8141)
        ]
        ,
        "L20B H Blanca": [
            (2627.2198, 1317.7337),(2610.8430, 1312.3232),(2612.7507, 1284.0566),(2565.2002, 1277.1663),(2551.9916, 1273.3004),(2530.6810, 1305.7233),
            (2549.9359, 1312.4627),(2556.3773, 1303.7134),(2577.2764, 1307.8461),(2581.1708, 1323.3951),(2624.1879, 1335.3897)
        ]
    }

    # --- CABECERA ESTILIZADA ---
    
    # 1. NORMALIZAR LOTES (desde tu diccionario lotes_reales)
    lotes_normalizados = {}
    for lote, datos in lotes_reales.items():
        if isinstance(datos[0][0], (int, float)):
            lotes_normalizados[lote] = [datos]
        else:
            lotes_normalizados[lote] = datos

    # 2. SUBIR ARCHIVO
    archivo = st.file_uploader(" Sube el archivo CSV de incidencias", type=["csv"], key="uploader_mapa")

    if archivo is not None:
        df = pd.read_csv(archivo)
        df.columns = df.columns.str.strip()

        if "category" not in df.columns:
            st.error(" El archivo debe tener la columna 'category'")
        else:
            # 3. SELECCIÓN DE PLAGA (En una fila limpia)
            plagas = [c for c in df.columns if c != "category"]
            col_sel, col_empty = st.columns([1, 1])
            with col_sel:
                plaga_seleccionada = st.selectbox(" Seleccione plaga para mapear", plagas)

            incidencias = dict(zip(df["category"], df[plaga_seleccionada]))

            # 4. ESCALAR COORDENADAS
            all_x, all_y = [], []
            for polys in lotes_normalizados.values():
                for poly in polys:
                    for x, y in poly:
                        all_x.append(x); all_y.append(y)

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            def escalar(poligono):
                return [((x - min_x) / (max_x - min_x) * 100, (y - min_y) / (max_y - min_y) * 100) for x, y in poligono]

            lotes_escalados = {lote: [escalar(p) for p in polys] for lote, polys in lotes_normalizados.items()}

            # 5. LOGICA DE COLORES PROFESIONAL
            def get_style(valor):
                if valor is None or pd.isna(valor):
                    return "#E5E7EB", "Sin datos" # Gris suave
                elif valor <= 30:
                    return "#22C55E", "Baja"     # Verde vibrante
                elif valor <= 60:
                    return "#FACC15", "Media"    # Amarillo
                else:
                    return "#EF4444", "Alta"     # Rojo

            # 6. GRAFICAR MAPA
            fig = go.Figure()

            for lote, poligonos in lotes_escalados.items():
                valor = incidencias.get(lote)
                color_hex, estado = get_style(valor)
                
                # Hover dinámico
                txt = f"<b>LOTE {lote}</b><br>Incidencia: {valor:.1f}%<br>Estado: {estado}" if estado != "Sin datos" else f"<b>LOTE {lote}</b><br>Sin monitoreo"

                for coords in poligonos:
                    x_c = [p[0] for p in coords]; y_c = [p[1] for p in coords]
                    # Cerrar polígono
                    x_c.append(x_c[0]); y_c.append(y_c[0])

                    fig.add_trace(go.Scatter(
                        x=x_c, y=y_c, fill="toself", fillcolor=color_hex,
                        line=dict(color="#1F2937", width=1),
                        name=lote, text=txt, hoverinfo="text", mode="lines",
                        showlegend=False
                    ))

            fig.update_layout(
                width=900, height=700,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10)
            )

            # Mostrar mapa en un contenedor
            st.plotly_chart(fig, use_container_width=True)

            # 7. LEYENDA TIPO "KPI CARDS" (Mucho más visual)
            st.markdown("---")
            st.markdown("###  Resumen de escala")
            l1, l2, l3, l4 = st.columns(4)
            
            with l1:
                st.markdown("""<div style='border-radius:8px; border-left:8px solid #22C55E; padding:10px; background:#f0fdf4'>
                    <p style='margin:0; font-size:12px; color:#166534'>INCIDENCIA</p>
                    <h5 style='margin:0'>BAJA (0-30%)</h5></div>""", unsafe_allow_html=True)
            with l2:
                st.markdown("""<div style='border-radius:8px; border-left:8px solid #FACC15; padding:10px; background:#fefce8'>
                    <p style='margin:0; font-size:12px; color:#854d0e'>INCIDENCIA</p>
                    <h5 style='margin:0'>MEDIA (31-60%)</h5></div>""", unsafe_allow_html=True)
            with l3:
                st.markdown("""<div style='border-radius:8px; border-left:8px solid #EF4444; padding:10px; background:#fef2f2'>
                    <p style='margin:0; font-size:12px; color:#991b1b'>INCIDENCIA</p>
                    <h5 style='margin:0'>ALTA (> 60%)</h5></div>""", unsafe_allow_html=True)
            with l4:
                st.markdown("""<div style='border-radius:8px; border-left:8px solid #E5E7EB; padding:10px; background:#f9fafb'>
                    <p style='margin:0; font-size:12px; color:#374151'>ESTADO</p>
                    <h5 style='margin:0'>SIN DATOS</h5></div>""", unsafe_allow_html=True)

    else:
        st.info(" Sube un archivo CSV para visualizar el mapa de lotes.")
        
with tab5:
    st.header(" Control de producción por ciclo")

    # 1. DICCIONARIOS BASE
    porcentajes_estimados = {
        "BLANCA": {
            22: 0.0033, 23: 0.0094, 24: 0.0178, 25: 0.0338, 26: 0.0614, 27: 0.0979, 
            28: 0.1168, 29: 0.1386, 30: 0.1431, 31: 0.1331, 32: 0.0997, 33: 0.0963, 34: 0.0488
        },
        "AZUL": {
            32: 0.0539, 33: 0.0757, 34: 0.0686, 35: 0.0506, 36: 0.0622, 37: 0.0719, 
            38: 0.1000, 39: 0.0979, 40: 0.1000, 41: 0.0884, 42: 0.0900, 43: 0.0900, 44: 0.0500
        },
        "VERDE": {
            18: 0.0200, 19: 0.0400, 20: 0.0800, 21: 0.1000, 22: 0.1000, 23: 0.1200, 
            24: 0.1200, 25: 0.1100, 26: 0.1000, 27: 0.0900, 28: 0.0500, 29: 0.0400, 30: 0.0300
        }
    }

    valores_predeterminados = {
        "L01 H Blanca": {"area": 6071.0, "prod": 18.0}, "L02 H Blanca": {"area": 3903.0, "prod": 16.0},
        "L03 H Blanca": {"area": 3354.0, "prod": 16.0}, "L04 H Blanca": {"area": 2605.0, "prod": 16.5},
        "L05 H Blanca": {"area": 1433.0, "prod": 17.0}, "L13 H Blanca": {"area": 5984.0, "prod": 16.0},
        "L16 H Blanca": {"area": 993.0, "prod": 14.0}, "L16A H BLANCA": {"area": 1723.0, "prod": 11.0},
        "L17 H Blanca": {"area": 3344.0, "prod": 14.0}, "L18 H Blanca": {"area": 1662.0, "prod": 19.0},
        "L18A H Blanca": {"area": 2345.0, "prod": 8.0}, "L18B H Blanca": {"area": 1246.0, "prod": 18.0},
        "L19 H Blanca": {"area": 1123.0, "prod": 20.0}, "19B H Blanca": {"area": 863.0, "prod": 16.0},
        "L20 H Blanca": {"area": 1703.0, "prod": 8.0}, "L20A H BLANCA": {"area": 1857.0, "prod": 18.0},
        "L20B H Blanca": {"area": 2093.0, "prod": 10.0}, "L21 H Blanca": {"area": 1953.0, "prod": 17.0},
        "L22 H Blanca": {"area": 1550.0, "prod": 15.0}, "L22A H Blanca": {"area": 2060.0, "prod": 6.0},
        "L23 H Blanca": {"area": 3150.0, "prod": 10.0}, "L23A H Blanca": {"area": 3898.0, "prod": 14.0},
        "L24 H Blanca": {"area": 2240.0, "prod": 12.5}, "L25 H Blanca": {"area": 3617.0, "prod": 13.0},
        "L26 H Blanca": {"area": 2193.0, "prod": 16.0}, "L26A H Blanca": {"area": 3139.0, "prod": 10.0},
        "L26B H Blanca": {"area": 3167.0, "prod": 7.0}, "L26C H Blanca": {"area": 2575.0, "prod": 2.0},
        "L11 H Shocking": {"area": 822.5, "prod": 11.0}, "L11A H Shocking": {"area": 847.0, "prod": 14.0},
        "L11B H Shocking": {"area": 356.5, "prod": 23.0}, "L07 H Azul": {"area": 528.0, "prod": 15.0},
        "L07A H Azul": {"area": 1266.0, "prod": 15.0}, "L06 H Lima": {"area": 1445.0, "prod": 6.0},
        "L08 H Esmeralda": {"area": 460.0, "prod": 11.0}, "L09 H Esmeralda": {"area": 1015.0, "prod": 10.0},
        "L14 H Esmeralda": {"area": 1295.0, "prod": 9.0}, "L15 H Esmeralda": {"area": 184.0, "prod": 10.0}
    }

    archivo_curva = st.file_uploader(" Sube el archivo de producción", type=["xlsx", "csv"], key="final_v_curva")

    if archivo_curva:
        df_c = pd.read_excel(archivo_curva) if archivo_curva.name.endswith(".xlsx") else pd.read_csv(archivo_curva)
        df_c.columns = [str(c).strip() for c in df_c.columns]
        
        # Limpieza y Unificación
        mapeo_lotes = {"26B": "L26B H Blanca", "26C": "L26C H Blanca", "16A": "L16A H Blanca", "19B": "19B H Blanca"}
        for clave, valor in mapeo_lotes.items():
            df_c.loc[df_c["Lote"].astype(str).str.contains(clave, case=False, na=False), "Lote"] = valor

        df_c["T Cortados"] = pd.to_numeric(df_c["T Cortados"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors='coerce').fillna(0)
        df_c["Año"] = pd.to_numeric(df_c["Año"], errors='coerce').fillna(0).astype(int)
        df_c["Semana"] = pd.to_numeric(df_c["Semana"], errors='coerce').fillna(0).astype(int)
        df_c = df_c[df_c["Año"] > 0]

        # --- SELECCIÓN ---
        col_l, col_a, col_p = st.columns(3)
        with col_l:
            lote_sel = st.selectbox("Lote:", sorted(df_c["Lote"].unique().astype(str)))
        
        config_lote = valores_predeterminados.get(lote_sel, {"area": 1000.0, "prod": 10.0})
        with col_a: area = st.number_input("Área (m²):", value=float(config_lote["area"]))
        with col_p: prod_esp = st.number_input("Meta m²:", value=float(config_lote["prod"]))

        df_lote = df_c[df_c["Lote"].astype(str) == lote_sel].copy()
        
        # Variedad
        v_primer = str(df_lote["Variedad"].iloc[0]).upper() if not df_lote.empty else ""
        if any(x in v_primer for x in ["AZUL", "SHOKING"]): tipo, def_ini, def_fin = "AZUL", 32, 44
        elif any(x in v_primer for x in ["GREEN", "KIWI", "ESMERALDA"]): tipo, def_ini, def_fin = "VERDE", 18, 30
        else: tipo, def_ini, def_fin = "BLANCA", 22, 34

        # --- SINCRONIZACIÓN ---
        st.subheader(" Sincronización")
        df_semanas_unicas = df_lote.groupby(["Año", "Semana"]).agg({"T Cortados": "sum"}).reset_index().sort_values(["Año", "Semana"])
        df_semanas_unicas["Etiqueta"] = df_semanas_unicas["Semana"].astype(str) + " (" + df_semanas_unicas["Año"].astype(str) + ")"
        opciones_semanas = df_semanas_unicas[df_semanas_unicas["T Cortados"] > 0].copy()
        
        c1, c2, c3 = st.columns(3)
        with c1:
            idx_inicio = st.selectbox("Semana inicio Excel:", range(len(opciones_semanas)), format_func=lambda x: opciones_semanas.iloc[x]["Etiqueta"])
            sel_inicio = opciones_semanas.iloc[idx_inicio]
        with c2: sem_punto_ciclo = st.number_input("Semana ciclo (inicio):", value=def_ini)
        with c3: ajuste_manual = st.number_input("Ajuste inicio (+/-):", value=0)

        c4, c5, c6 = st.columns(3)
        with c4:
            idx_fin = st.selectbox("Semana final Excel:", range(len(opciones_semanas)), index=len(opciones_semanas)-1, format_func=lambda x: opciones_semanas.iloc[x]["Etiqueta"])
            sel_fin = opciones_semanas.iloc[idx_fin]
        with c5: sem_ciclo_fin = st.number_input("Semana ciclo (final):", value=def_fin)
        with c6: ajuste_final_manual = st.number_input("Ajuste final (+/-):", value=0)

        # --- CÁLCULOS ---
        dict_pct = porcentajes_estimados[tipo]
        rango_teorico = list(range(int(sem_punto_ciclo), int(sem_ciclo_fin) + 1))
        df_teorico = pd.DataFrame({
            "Semana Ciclo": rango_teorico,
            "Estimado": [round(area * prod_esp * (dict_pct.get(s, 0) / 100 if dict_pct.get(s, 0) > 1 else dict_pct.get(s, 0))) for s in rango_teorico]
        })

        id_inicio, id_fin = sel_inicio["Año"] * 100 + sel_inicio["Semana"], sel_fin["Año"] * 100 + sel_fin["Semana"]
        df_lote["ID_T"] = df_lote["Año"] * 100 + df_lote["Semana"]
        df_campaña = df_lote[(df_lote["ID_T"] >= id_inicio) & (df_lote["ID_T"] <= id_fin)].copy()

        def calcular_eje_x(fila):
            distancia = ((fila["Año"] - sel_inicio["Año"]) * 52) + (fila["Semana"] - sel_inicio["Semana"])
            return distancia + sem_punto_ciclo + ajuste_manual + (ajuste_final_manual if fila["ID_T"] != id_inicio else 0)

        df_campaña["Semana Ciclo"] = df_campaña.apply(calcular_eje_x, axis=1).apply(lambda s: max(sem_punto_ciclo, min(sem_ciclo_fin, s)))
        df_plot_real = df_campaña.groupby("Semana Ciclo")["T Cortados"].sum().reset_index().rename(columns={"T Cortados": "Real"})

        # --- GRÁFICO ---
        fig = go.Figure()
        
        # Meta Teórica
        fig.add_trace(go.Scatter(
            x=df_teorico["Semana Ciclo"], 
            y=df_teorico["Estimado"], 
            name="Estimado", 
            line=dict(color='#cfcfcf', width=2, dash='dot')
        ))
        
        # Producción Real
        fig.add_trace(go.Scatter(
            x=df_plot_real["Semana Ciclo"], 
            y=df_plot_real["Real"], 
            name="Producción real", 
            fill='tozeroy', 
            fillcolor='rgba(255, 127, 14, 0.15)', 
            line=dict(color='#ff7f0e', width=4), 
            mode='lines+markers'
        ))

        fig.update_layout(
            hovermode="x unified", 
            height=450, 
            plot_bgcolor="white", # Fondo del área del gráfico blanco
            paper_bgcolor="white", # Fondo de todo el componente blanco
            margin=dict(l=10, r=10, t=50, b=50),
            title={
                'text': f"Curva de producción: {lote_sel}",
                'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
            },
            xaxis_title="Semana del ciclo",
            yaxis_title="Tallos cortados",
            font=dict(size=12)
        )

        # --- QUITAR LAS LÍNEAS DE LA CUADRÍCULA ---
        fig.update_xaxes(
            showgrid=False,       # Quita líneas verticales
            zeroline=True,        # Muestra la línea base del eje X
            zerolinecolor='#ccc',
            linecolor='#ccc',     # Color de la línea del eje
            mirror=False
        )
        fig.update_yaxes(
            showgrid=False,       # Quita líneas horizontales
            zeroline=True,        # Muestra la línea base del eje Y
            zerolinecolor='#ccc',
            linecolor='#ccc',     # Color de la línea del eje
            mirror=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- INSPECTOR DE DETALLE ---
        st.markdown("###  Inspector de detalle semanal")
        df_comp = pd.merge(df_teorico, df_plot_real, on="Semana Ciclo", how="left").fillna(0)
        df_comp["Diferencia"] = df_comp["Real"] - df_comp["Estimado"]

        sems = df_comp["Semana Ciclo"].astype(int).tolist()
        sem_sel = st.select_slider("Analizar semana específica:", options=sems)
        det = df_comp[df_comp["Semana Ciclo"] == sem_sel].iloc[0]

        color_t = "#28a745" if det['Diferencia'] >= 0 else "#dc3545"
        st.markdown(f"""
            <div style="background-color:#ffffff; padding:20px; border-radius:15px; border: 1px solid #e6e9ef; box-shadow: 2px 2px 10px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-around; align-items: center; text-align: center;">
                    <div><p style="color:#666; margin:0; font-size:14px;">Meta Estimada</p><h3 style="margin:0;">{det['Estimado']:,.0f}</h3></div>
                    <div style="border-left: 1px solid #eee; height: 40px;"></div>
                    <div><p style="color:#666; margin:0; font-size:14px;">Real Cortado</p><h3 style="margin:0;">{det['Real']:,.0f}</h3></div>
                    <div style="border-left: 1px solid #eee; height: 40px;"></div>
                    <div><p style="color:#666; margin:0; font-size:14px;">Diferencia</p><h3 style="margin:0; color:{color_t};">{det['Diferencia']:,.0f}</h3></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
# --- 9. BALANCE GENERAL (INCLUYE PRODUCTIVIDAD REAL) ---
        st.markdown("---")
        st.markdown("###  Estado de cumplimiento y rendimiento")
        
        t_real_t = df_plot_real["Real"].sum()
        t_meta_t = df_teorico["Estimado"].sum()
        cumplimiento = (t_real_t / t_meta_t * 100) if t_meta_t > 0 else 0
        
        # CÁLCULO DE PRODUCTIVIDAD
        # Prod. Meta es la que el usuario ingresó (prod_esp)
        prod_real_actual = t_real_t / area if area > 0 else 0
        
        # Cálculo de balance a la fecha real cortada
        meta_hoy = df_teorico[df_teorico["Semana Ciclo"] <= df_plot_real["Semana Ciclo"].max()]["Estimado"].sum()
        bal_hoy = t_real_t - meta_hoy

        col_graf, col_stats = st.columns([1.2, 1])

        with col_graf:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = cumplimiento,
                number = {'suffix': "%", 'font': {'size': 40, 'color': "#ff7f0e"}},
                title = {'text': "Avance de meta total", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [0, max(100, cumplimiento)], 'tickwidth': 1},
                    'bar': {'color': "#ff7f0e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#eeeeee",
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_stats:
            # Tarjeta 1: Productividad (Rendimiento)
            color_prod = "#28a745" if prod_real_actual >= prod_esp else "#31333F"
            st.markdown(f"""
                <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border: 1px solid #d1d5db; margin-bottom:10px;">
                    <p style="margin:0; color:#555; font-size:12px; font-weight:bold;">Productividad actual (m²)</p>
                    <h2 style="margin:0; color:{color_prod};">{prod_real_actual:.2f}</h2>
                    <p style="margin:0; color:#666; font-size:13px;">Meta esperada: {prod_esp:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

            # Tarjeta 2: Tallos Totales
            st.markdown(f"""
                <div style="background-color:#ffffff; padding:15px; border-radius:10px; border: 1px solid #ff7f0e; margin-bottom:10px;">
                    <p style="margin:0; color:#ff7f0e; font-size:12px; font-weight:bold;">Tallos cortados</p>
                    <h2 style="margin:0; color:#31333F;">{t_real_t:,.0f}</h2>
                    <p style="margin:0; color:#666; font-size:13px;">Proyectados: {t_meta_t:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)

            # Tarjeta 3: Balance a la fecha
            color_bal = "#28a745" if bal_hoy >= 0 else "#dc3545"
            bg_bal = "#eafaf1" if bal_hoy >= 0 else "#fdf2f2"
            st.markdown(f"""
                <div style="background-color:{bg_bal}; padding:15px; border-radius:10px; border: 1px solid {color_bal};">
                    <p style="margin:0; color:{color_bal}; font-size:12px; font-weight:bold;">Diferencia a hoy</p>
                    <h2 style="margin:0; color:#31333F;">{"+" if bal_hoy >= 0 else ""}{bal_hoy:,.0f}</h2>
                </div>
            """, unsafe_allow_html=True)

        if cumplimiento > 100:
            st.success(f" Meta de producción superada.")
            
def extraer_tareas_pro(archivo_pdf):
    datos_extraidos = []
    try:
        with pdfplumber.open(archivo_pdf) as pdf:
            for pagina in pdf.pages:
                tablas = pagina.extract_tables()
                for tabla in tablas:
                    # Validamos que la tabla tenga contenido y las columnas mínimas
                    if tabla and len(tabla[0]) >= 6:
                        for fila in tabla:
                            lote_raw = str(fila[0]) if fila[0] else ""
                            detalles_lote = str(fila[1]) if fila[1] else ""
                            tareas_raw = str(fila[-1]) if fila[-1] else ""

                            # Filtro de seguridad para lotes
                            if (lote_raw.startswith("L") or "19B" in lote_raw) and "-" in tareas_raw:
                                lote_limpio = lote_raw.split('\n')[0].strip()
                                
                                # --- EXTRAER ÁREA PARA CÁLCULOS ---
                                area_match = re.search(r"Area-(\d+)m2", detalles_lote)
                                area_total = float(area_match.group(1)) if area_match else 0
                                area_efectiva = area_total * 0.62

                                lineas = tareas_raw.split('-')
                                semana_detectada = "N/A"

                                for linea in lineas:
                                    texto = linea.strip().replace('\n', ' ')
                                    if not texto: continue

                                    # --- DETECTAR SEMANA ---
                                    if texto.startswith(("S", "$")) and any(char.isdigit() for char in texto):
                                        semana_detectada = texto.split(":")[0].replace("$", "S")
                                        continue 

                                    if len(texto) > 3:
                                        # --- LÓGICA DE GALLINAZA Y CAL ---
                                        if "gallinaza" in texto.lower() or "cal " in texto.lower():
                                            dosis_match = re.search(r"(\d+\.?\d*)", texto)
                                            if dosis_match and area_efectiva > 0:
                                                dosis = float(dosis_match.group(1))
                                                # Cálculo: (Dosis * Area Efectiva) / 50kg
                                                bultos = (dosis * area_efectiva) / 50
                                                texto = f"{texto} | 📦 SOLICITAR: {round(bultos, 1)} Bultos"

                                        datos_extraidos.append({
                                            "Lote": lote_limpio,
                                            "Semana": semana_detectada,
                                            "Actividad": texto,
                                            "Hecho": False
                                        })
    except Exception as e:
        st.error(f"Error al leer el PDF: {e}")
    
    return pd.DataFrame(datos_extraidos).drop_duplicates()