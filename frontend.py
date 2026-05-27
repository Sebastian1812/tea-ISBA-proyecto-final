"""
STREAMLIT FRONTEND - TEA Clinical Agent System
VERSIÓN COMPLETA CON TODAS LAS HERRAMIENTAS
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcp_server import MCPServer
from graph_agent import ClinicalAgent
from config import model_config

st.set_page_config(
    page_title="TEA Clinical Agent - Sistema con IA",
    page_icon="🧠",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #1B4965 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .tool-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2E86AB;
    }
    .report-box {
        background-color: white;
        border: 1px solid #ddd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-height: 500px;
        overflow-y: auto;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_mcp_server():
    return MCPServer()

@st.cache_resource
def get_agent():
    mcp = MCPServer()
    return ClinicalAgent(mcp)

def check_ollama_status():
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()
        return result == 0
    except:
        return False

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🧠 TEA Clinical Agent System</h1>
        <p>Sistema Basado en Agentes con IA para Trastorno del Espectro Autista</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Estado")
        ollama_ok = check_ollama_status()
        if ollama_ok:
            st.success(f"✅ Ollama activo\nModelo: {model_config.default_model}")
        else:
            st.error("❌ Ollama no disponible\nEjecuta: ollama serve")
        
        st.divider()
        st.header("🔧 Herramientas")
        tool_option = st.radio(
            "Seleccionar herramienta",
            ["📊 Reporte Clínico", "📈 Simulación Intervención", "📉 Detección Outliers"]
        )
    
    mcp = get_mcp_server()
    
    # Obtener pacientes
    try:
        patients = mcp.get_resource("patients://list")
        patient_options = {f"{p['name']} ({p['id']})": p['id'] for p in patients}
    except:
        patient_options = {"Paciente de prueba (PAT-001)": "PAT-001"}
    
    # ============================================
    # HERRAMIENTA 1: REPORTE CLÍNICO
    # ============================================
    
    if tool_option == "📊 Reporte Clínico":
        st.header("📋 Generación de Reporte Clínico con IA")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.selectbox("Seleccionar Paciente", list(patient_options.keys()))
            patient_id = patient_options[selected]
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            generate = st.button("🤖 Generar Reporte con IA", type="primary", use_container_width=True)
        
        # Mostrar métricas actuales
        try:
            metrics = mcp.calculate_progress_metrics(patient_id)
            if metrics and 'error' not in metrics:
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Total Sesiones", metrics.get('total_sessions', 0))
                with col_b:
                    st.metric("Progreso Promedio", f"{metrics.get('average_progress_score', 0):.2f}")
                with col_c:
                    st.metric("Avance General", f"{metrics.get('overall_advancement_percentage', 0):.1f}%")
                with col_d:
                    trend = metrics.get('progress_trend', 'N/A')
                    st.metric("Tendencia", "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️")
        except:
            pass
        
        # Generar reporte
        if generate and ollama_ok:
            agent = get_agent()
            
            with st.spinner("🔄 Agente IA analizando datos..."):
                try:
                    result = agent.run(patient_id, thread_id=patient_id)
                    
                    if result.get('report_generated'):
                        st.success("✅ Reporte generado exitosamente")
                        
                        # Mostrar reporte
                        for key, value in result.get('memory_context', {}).items():
                            if key.startswith('report_'):
                                st.markdown("#### 📋 Reporte Clínico")
                                st.markdown(f'<div class="report-box">{value}</div>', unsafe_allow_html=True)
                                
                                st.download_button(
                                    label="📥 Descargar Reporte",
                                    data=value,
                                    file_name=f"reporte_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                                )
                                break
                    else:
                        error_msg = result.get('error', 'Error desconocido')
                        st.error(f"❌ No se generó reporte: {error_msg}")
                        
                        if "sanitize_user_prompt" in error_msg:
                            st.info("🔧 Error de método: Reinicia la aplicación para aplicar la corrección.")
                    
                    # Mostrar sugerencias diagnósticas
                    diagnostic = result.get('diagnostic_suggestion', {})
                    if diagnostic.get('requires_review'):
                        st.warning("⚠️ Sugerencia de Revisión Diagnóstica")
                        for sugg in diagnostic.get('suggestions', []):
                            st.info(f"**{sugg.get('type')}**: {sugg.get('message')}")
                            
                except Exception as e:
                    st.error(f"Error: {e}")
        
        elif generate and not ollama_ok:
            st.error("Ollama no está disponible. Ejecuta 'ollama serve' en una terminal.")
    
    # ============================================
    # HERRAMIENTA 2: SIMULACIÓN DE INTERVENCIÓN
    # ============================================
    
    elif tool_option == "📈 Simulación Intervención":
        st.header("📈 Simulador de Resultados de Intervención")
        
        st.markdown("""
        <div class="tool-card">
        Esta herramienta simula los resultados esperados de diferentes intervenciones 
        terapéuticas basándose en coeficientes de efectividad de la literatura clínica.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            intervention = st.selectbox(
                "Tipo de Intervención",
                ["ABA", "Speech Therapy", "Occupational Therapy", "Social Skills Group", "Combined Intensive"]
            )
        
        with col2:
            baseline_severity = st.slider(
                "Severidad Basal (0-10)", 
                min_value=0.0, max_value=10.0, value=7.5, step=0.5
            )
        
        with col3:
            duration_weeks = st.slider(
                "Duración (semanas)", 
                min_value=4, max_value=52, value=24, step=4
            )
        
        if st.button("🔬 Simular Outcome", type="primary"):
            with st.spinner("Calculando simulación..."):
                result = mcp.simulate_intervention_outcome(intervention, baseline_severity, duration_weeks)
                
                st.markdown("### 📊 Resultados de la Simulación")
                
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Mejora Esperada", f"{result['improvement_percentage']:.1f}%")
                with col_r2:
                    st.metric("Severidad Final", f"{result['final_severity']:.2f}")
                with col_r3:
                    st.metric("Intervalo Confianza", f"[{result['confidence_interval'][0]:.2f} - {result['confidence_interval'][1]:.2f}]")
                
                st.info(f"ℹ️ {result['note']}")
                
                # Gráfico de mejora
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Basal', 'Post-intervención'],
                    y=[baseline_severity, result['final_severity']],
                    text=[f"{baseline_severity:.1f}", f"{result['final_severity']:.2f}"],
                    textposition='auto',
                    marker_color=['#ff6b6b', '#4CAF50']
                ))
                fig.update_layout(
                    title="Reducción de Severidad",
                    yaxis_title="Nivel de Severidad",
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ============================================
    # HERRAMIENTA 3: DETECCIÓN DE OUTLIERS
    # ============================================
    
    else:
        st.header("📉 Detección de Valores Atípicos (Outliers)")
        
        st.markdown("""
        <div class="tool-card">
        Esta herramienta detecta valores atípicos en listas de puntuaciones utilizando 
        el método estadístico Z-score. Valores con |z| > 2.0 se consideran outliers.
        </div>
        """, unsafe_allow_html=True)
        
        # Entrada de datos
        input_method = st.radio("Método de entrada", ["Usar datos de ejemplo", "Ingresar datos manualmente"])
        
        scores_list = []
        
        if input_method == "Usar datos de ejemplo":
            example_data = st.selectbox(
                "Ejemplo predefinido",
                ["Progreso de sesiones", "Puntuaciones con outlier", "Datos normales"]
            )
            
            if example_data == "Progreso de sesiones":
                scores_list = [65, 68, 70, 72, 68, 70, 69, 71, 95, 70]
                st.write("Datos: [65, 68, 70, 72, 68, 70, 69, 71, 95, 70]")
            elif example_data == "Puntuaciones con outlier":
                scores_list = [65, 68, 70, 72, 68, 95, 70, 69, 71, 200]
                st.write("Datos: [65, 68, 70, 72, 68, 95, 70, 69, 71, 200] (contiene outlier)")
            else:
                scores_list = [70, 72, 71, 69, 70, 71, 72, 70, 71, 70]
                st.write("Datos: [70, 72, 71, 69, 70, 71, 72, 70, 71, 70]")
        else:
            input_text = st.text_input("Ingrese números separados por comas", "65, 68, 70, 72, 68, 95, 70, 69, 71, 200")
            try:
                scores_list = [float(x.strip()) for x in input_text.split(',')]
            except:
                st.error("Formato inválido. Use números separados por comas.")
                scores_list = []
        
        threshold = st.slider("Umbral Z-score", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
        
        if scores_list and st.button("🔍 Detectar Outliers", type="primary"):
            with st.spinner("Calculando..."):
                result = mcp.filter_statistical_outliers(scores_list, threshold)
                
                st.markdown("### 📊 Resultados")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Originales", result['original_count'])
                with col_s2:
                    st.metric("Outliers", result['outliers_count'])
                with col_s3:
                    st.metric("Media", f"{result['mean']:.2f}")
                
                # Tabla de resultados
                df = pd.DataFrame({
                    'Índice': list(range(len(scores_list))),
                    'Valor': scores_list,
                    'Z-score': [f"{z:.2f}" for z in result.get('z_scores', [0]*len(scores_list))] if 'z_scores' in result else ['N/A'] * len(scores_list)
                })
                
                # Marcar outliers
                outlier_indices = [o['index'] for o in result['outliers']]
                df['Estado'] = ['⚠️ Outlier' if i in outlier_indices else '✓ Normal' for i in range(len(df))]
                
                st.dataframe(df, use_container_width=True)
                
                if result['outliers']:
                    st.warning(f"⚠️ Se detectaron {result['outliers_count']} outlier(s)")
                    for o in result['outliers']:
                        st.write(f"  • Índice {o['index']}: valor {o['value']} (z-score = {o['z_score']:.2f})")
                else:
                    st.success("✅ No se detectaron outliers significativos")
                
                st.info(f"💡 Método: Z-score | Umbral: {threshold} | Desv. Estándar: {result['std']:.2f}")
    
    # Footer
    st.markdown("---")
    st.caption("🏥 TEA Clinical Agent System | IA con Ollama | Herramientas clínicas integradas")

if __name__ == "__main__":
    main()