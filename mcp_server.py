"""
MCP SERVER - TEA Clinical Agent
================================
NO ES UN SERVIDOR DE RED - Es una clase que implementa el patrón MCP

¿Qué es MCP?
Es un patrón de diseño (no un servidor web) que define:
- RESOURCES: Datos que se pueden consultar (como una base de datos)
- TOOLS: Funciones que transforman datos (como servicios)

Este patrón permite que LangGraph interactúe con los datos de forma ESTANDARIZADA
"""


import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from security import validate_input, InputValidator


class MCPServer:
    """
    Servidor MCP (implementado como clase para el acceso a datos clínicos)
    
    clase que sigue el patrón MCP:
    - Expone RESOURCES (datos) a través del método get_resource()
    - Expone TOOLS (operaciones) como métodos públicos
    """
    
    def __init__(self, data_path: str = "./data"):
        """
            Inicializa el servidor MCP
            Complexity: O(1) - carga inicial de datos

            Args:
            data_path: Ruta a la carpeta con datos JSON/CSV
        """
        self.data_path = Path(data_path)
        # Cargar datos al iniciar (se hace una sola vez)
        self.patients_data = self._load_patients()
        self.sessions_df = self._load_sessions()
        # Validador para seguridad
        self.validator = InputValidator()
    
    # CARGA DE DATOS (INTERNA)
    def _load_patients(self) -> Dict:
        """
            Carga datos de pacientes desde archivo JSON
            Complexity: O(n) donde n = número de pacientes (carga inicial)
        """
        json_path = self.data_path / "sample_patients.json"
        if not json_path.exists():
            print(f"ADVERTENCIA: {json_path} no encontrado. Creación de datos de muestra...")
            return self._create_sample_patients()

        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_sessions(self) -> pd.DataFrame:
        """
            Carga datos de sesiones desde archivo CSV
            Complexity: O(n) donde n = número de sesiones (carga inicial)
        """
        return pd.read_csv(self.data_path / "sessions_data.csv")
    
    # ===== RESOURCES =====
    
    def get_resource(self, uri: str) -> Any:
        """
            Obtiene un recurso por su URI
            Complexity: O(1) para acceso por ID, O(n) para listados

            Este método implementa el protocolo MCP para RESOURCES
        """
        if uri == "patients://list":
            return self._list_patients()
        elif uri.startswith("patients://") and "/sessions" in uri:
            patient_id = uri.split("//")[1].split("/")[0]
            return self._get_patient_sessions(patient_id)
        elif uri.startswith("assessments://"):
            patient_id = uri.split("//")[1]
            return self._get_assessments(patient_id)
        elif uri.startswith("reports://"):
            patient_id = uri.split("//")[1]
            return self._get_reports(patient_id)
        else:
            raise ValueError(f"URI de recurso desconocido: {uri}")
    
    def _list_patients(self) -> List[Dict]:
        """
            Lista todos los pacientes activos
            Complexity: O(n) donde n es número de pacientes
            RETURN: Lista de diccionarios con información básica
        """
        patients = self.patients_data.get('patients', [])

        # Devolver solo información necesaria (no todos los datos clínicos)
        return [{
            'id': p['id'],
            'name': p['name'],
            'age': p['age'],
            'active_therapies': p['active_therapies']
        } for p in patients]
    
    def _get_patient_sessions(self, patient_id: str) -> Dict:
        """
            Obtiene historial de sesiones de un paciente
            Complexity: O(n) donde n es número de sesiones (filtrado con pandas)

            IMPORTANTE: Esta operación es O(n) porque debe filtrar todas las sesiones
            para encontrar las del paciente específico. Si tuviéramos 1,000,000 de sesiones,
            sería lenta. En producción se usaría una base de datos con índice.
        """
        if not self.validator.validate_patient_id(patient_id):
            raise ValueError(f"Formato de ID de paciente no válido: {patient_id}")
        
        # FILTRADO (O(n) - tiene que revisar todas las filas)
        patient_sessions = self.sessions_df[self.sessions_df['patient_id'] == patient_id]
        
        if patient_sessions.empty:
            return {'patient_id': patient_id, 'sessions': [], 'total_sessions': 0}
        # ESTADÍSTICAS (O(1) después del filtrado)
        return {
            'patient_id': patient_id,
            'sessions': patient_sessions.to_dict('records'),
            'total_sessions': len(patient_sessions),
            'domains': patient_sessions['domain'].unique().tolist(),
            'date_range': {
                'first': patient_sessions['session_date'].min(),
                'last': patient_sessions['session_date'].max()
            }
        }
    
    def _get_assessments(self, patient_id: str) -> Dict:
        """
            Obtiene resultados de escalas de evaluación
            Complexity: O(1) - búsqueda en lista (average case O(1) con next())

            En el peor caso (paciente no existe) es O(n) porque recorre toda la lista
        """
        if not self.validator.validate_patient_id(patient_id):
            raise ValueError(f"Invalid patient ID format: {patient_id}")
        
        # Búsqueda lineal (O(n) en el peor caso, pero n es pequeño)
        patient = next((p for p in self.patients_data['patients'] if p['id'] == patient_id), None)
        if not patient:
            return {'error': f'Paciente {patient_id} no encontrado'}
        
        return {
            'patient_id': patient_id,
            'assessments': patient.get('assessments', {}),
            'baseline_severity': patient.get('baseline_severity')
        }
    
    def _get_reports(self, patient_id: str) -> List[Dict]:
        """
            Obtiene reportes previos
            Complexity: O(1) - en esta simulación, siempre devuelve lista vacía
            En producción, esto leería de una base de datos o sistema de archivos
        """
        # In production, this would read from persistent storage
        # For simulation, return empty list
        return []
    
    # ===== TOOLS =====
    
    @validate_input
    def calculate_progress_metrics(self, patient_id: str, date_range: Optional[str] = None) -> Dict:
        """
                TOOL 1: Calcula métricas de progreso clínico

        Complexity Analysis:

            O(n) donde n = número de sesiones del paciente

            El filtrado por fecha_range es O(n)

            Los cálculos de promedio son O(n)

            El cálculo de tendencia (regresión) es O(n) con numpy

        Parámetros:

            patient_id: ID del paciente (ej: "PAT-001")

            date_range: Rango opcional "YYYY-MM-DD,YYYY-MM-DD"

        Retorna:

            Diccionario con métricas detalladas de progreso
        """
        # Obtener sesiones (esto ya es O(n))
        sessions_data = self._get_patient_sessions(patient_id)
        sessions = sessions_data.get('sessions', [])
        
        if not sessions:
            return {'error': f'No hay sesiones encontradas para el paciente {patient_id}'}
        
        # Filtrar por fecha si se especifica (O(n))
        if date_range:
            start_date, end_date = date_range.split(',')
            sessions = [s for s in sessions if start_date <= s['session_date'] <= end_date]
        
        # Convertir a DataFrame para cálculos eficientes (O(n))
        df = pd.DataFrame(sessions)
        
        # Cálculos de métricas (todos O(n) individualmente)
        metrics = {
            'patient_id': patient_id,
            'total_sessions': int(len(df)),  # int() explícito
            'average_positive_responses': float(df['positive_responses'].mean()) if len(df) > 0 else 0.0,
            'average_progress_score': float(df['progress_score'].mean()) if len(df) > 0 else 0.0,
            'progress_trend': self._calculate_trend(df['progress_score'].tolist()),
            'domain_progress': {}
        }

        # Métricas por dominio (O(d*n) donde d=dominios, n=sesiones) y como d es pequeño (3-4 dominios), esto es O(n)
        # Calcular progreso por dominio
        for domain in df['domain'].unique():
            domain_df = df[df['domain'] == domain]
            metrics['domain_progress'][domain] = {
                'average_progress': float(domain_df['progress_score'].mean()),
                'sessions_count': len(domain_df),
                'improvement_rate': self._calculate_improvement_rate(domain_df['progress_score'].tolist())
            }
        
        # Calcular avance general (O(1))
        if len(df) > 1:
            first_score = df.iloc[0]['progress_score']
            last_score = df.iloc[-1]['progress_score']
            metrics['overall_advancement_percentage'] = ((last_score - first_score) / first_score) * 100
        else:
            metrics['overall_advancement_percentage'] = 0
        
        return metrics
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """
            Calcula tendencia usando regresión lineal simple
            Complexity: O(n) con numpy (polyfit es O(n))

            Coeficiente de regresión:
                slope > 0.05 → "improving" (mejorando)
                slope < -0.05 → "declining" (empeorando)
                else → "stable" (estable)
        """
        if len(scores) < 2:
            return "insufficient_data"
        
        # O(n) - numpy calcula la pendiente en una pasada
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """
        Calcula tasa de mejora porcentual
        Complexity: O(1) - solo usa primer y último elemento
        """
        if len(scores) < 2:
            return 0
        return ((scores[-1] - scores[0]) / scores[0]) * 100 if scores[0] > 0 else 0
    
    @validate_input
    def generate_diagnostic_suggestion(self, patient_id: str, assessment_data: Optional[Dict] = None) -> Dict:
        """
            TOOL 2: Genera sugerencias diagnósticas basadas en escalas

            Complexity: O(1) - operaciones aritméticas simples

                No itera sobre colecciones grandes

                Solo hace comparaciones de números

            Algoritmo:

                Obtener scores CARS y ADOS

                Comparar con umbrales clínicos

                Generar sugerencias según umbrales superados
        """
        from config import clinical_thresholds
        
        # Obtener datos de evaluación si no se proporcionan
        if not assessment_data:
            assessment_data = self._get_assessments(patient_id)
        
        assessments = assessment_data.get('assessments', {})
        cars_score = assessments.get('CARS', {}).get('score', 0)
        ados_score = assessments.get('ADOS', {}).get('score', 0)
        
        # COMPLEJIDAD: Todas estas operaciones son O(1)
        suggestions = {
            'patient_id': patient_id,
            'current_scores': {
                'CARS': cars_score,
                'ADOS': ados_score
            },
            'suggestions': [],
            'requires_review': False
        }
        
        # Verificar umbrales críticos (O(1) - comparaciones simples)
        if cars_score > clinical_thresholds.cars_threshold_critical:
            suggestions['suggestions'].append({
                'type': 'diagnostic_review',
                'severity': 'high',
                'message': f"CARS score of {cars_score} exceeds critical threshold ({clinical_thresholds.cars_threshold_critical}). Consider comprehensive diagnostic review.",
                'recommendation': 'Schedule multidisciplinary evaluation'
            })
            suggestions['requires_review'] = True
        
        if ados_score > clinical_thresholds.ados_threshold_critical:
            suggestions['suggestions'].append({
                'type': 'diagnostic_review',
                'severity': 'high',
                'message': f"ADOS score of {ados_score} exceeds clinical threshold ({clinical_thresholds.ados_threshold_critical}).",
                'recommendation': 'Review diagnostic criteria and consider additional assessments'
            })
            suggestions['requires_review'] = True
        
        # Diagnóstico diferencial (O(1))
        if cars_score > 25 and cars_score < 30:
            suggestions['suggestions'].append({
                'type': 'differential',
                'severity': 'moderate',
                'message': 'Score in borderline range. Consider ruling out other developmental conditions.',
                'differentials': ['Language disorder', 'Intellectual disability', 'ADHD']
            })
        
        return suggestions
    
    @validate_input
    def filter_statistical_outliers(self, scores_list: List[float], threshold: float = 2.0) -> Dict:
        """
            TOOL 3: Detecta valores atípicos usando método Z-score

            Complexity Analysis: O(n) donde n = número de scores

                Cálculo de media: O(n)

                Cálculo de desviación estándar: O(n)

                Cálculo de z-scores: O(n)

                Filtrado: O(n)

            Total: 4 pasadas O(n) → O(n)

            Método Z-score:
            z = |x - μ| / σ
            Donde μ = media, σ = desviación estándar
            Valores con |z| > threshold son outliers
        """
        if not scores_list:
            return {'outliers': [], 'filtered_scores': [], 'method': 'z-score'}
        
        # Convertir a numpy array para cálculos eficientes
        scores_array = np.array(scores_list)

        # PASO 1: Calcular media (O(n))
        mean = np.mean(scores_array)
        # PASO 2: Calcular desviación estándar (O(n))
        std = np.std(scores_array)
        
        if std == 0:
            return {'outliers': [], 'filtered_scores': scores_list, 'method': 'z-score', 'note': 'No variance in data'}
        
        # PASO 3: Calcular z-scores (O(n))
        z_scores = np.abs((scores_array - mean) / std)

        # PASO 4: Identificar outliers (O(n))
        outlier_indices = np.where(z_scores > threshold)[0]
        
        # Construir resultado (O(k) donde k = número de outliers)
        outliers = [{'index': int(i), 'value': float(scores_list[i]), 'z_score': float(z_scores[i])} 
                   for i in outlier_indices]
        
        # Filtrar scores (O(n))
        filtered_scores = [scores_list[i] for i in range(len(scores_list)) if i not in outlier_indices]
        
        return {
            'outliers': outliers,
            'filtered_scores': filtered_scores,
            'method': 'z-score',
            'threshold': threshold,
            'original_count': len(scores_list),
            'outliers_count': len(outliers),
            'mean': float(mean),
            'std': float(std)
        }
    
    @validate_input
    def simulate_intervention_outcome(self, intervention_type: str, baseline_severity: float, duration_weeks: int) -> Dict:
        """
            TOOL 4: Simula resultados de intervención

            Complexity: O(1) - todas las operaciones son aritméticas simples

                No hay loops sobre colecciones

                Fórmulas matemáticas directas

            Modelo de simulación:
            improvement = baseline × effectiveness × (duration / 12)
            Con rendimientos decrecientes después de 24 semanas
        """

        # Validaciones (O(1))
        if baseline_severity < 0 or baseline_severity > 10:
            raise ValueError("Baseline severity must be between 0 and 10")
        
        if duration_weeks < 1 or duration_weeks > 52:
            raise ValueError("Duration must be between 1 and 52 weeks")
        
        # Coeficientes de efectividad basados en literatura (O(1))
        effectiveness = {
            'ABA': 0.25,
            'Speech Therapy': 0.20,
            'Occupational Therapy': 0.18,
            'Social Skills Group': 0.22,
            'Combined Intensive': 0.35
        }
        
        coeff = effectiveness.get(intervention_type, 0.15)
        
        # Rendimientos decrecientes (ley de los rendimientos marginales) Después de 24 semanas, cada semana adicional es menos efectiva
        if duration_weeks > 24:
            effective_weeks = 24 + (duration_weeks - 24) * 0.3
        else:
            effective_weeks = duration_weeks
        
        # Calcular mejora esperada (O(1))
        expected_improvement = baseline_severity * coeff * (effective_weeks / 12)
        expected_improvement = min(expected_improvement, baseline_severity * 0.7)  # Cap at 70% improvement
        
        final_severity = max(0, baseline_severity - expected_improvement)
        
        return {
            'intervention_type': intervention_type,
            'baseline_severity': baseline_severity,
            'duration_weeks': duration_weeks,
            'expected_improvement': round(expected_improvement, 2),
            'final_severity': round(final_severity, 2),
            'improvement_percentage': round((expected_improvement / baseline_severity) * 100, 1),
            'confidence_interval': [round(final_severity - 0.5, 2), round(final_severity + 0.5, 2)],
            'note': 'Simulation based on clinical literature averages. Individual results may vary.'
        }
    
if __name__ == "__main__":
    """
    Ejecuta: python mcp_server.py
    
    Esto DEMUESTRA cómo funciona el MCP sin necesidad de un servidor externo.
    NO abre puertos de red, solo muestra la funcionalidad.
    """
    
    # Mostrar encabezado de la demostración
    print("="*70)
    print("MCP SERVER DEMONSTRATION")
    print("="*70)
    print("\n⚠️ NOTA: Este NO es un servidor de red.")
    print(" Es una demostración de la funcionalidad MCP como clase Python.\n")
    
    # Paso 1: Crear instancia del servidor MCP
    server = MCPServer()
    
    print("\n" + "="*70)
    print("1. DEMOSTRACIÓN DE RESOURCES (Recursos)")
    print("="*70)
    
    # Recurso 1: Obtener y mostrar lista de pacientes
    print("\n📁 Resource: patients://list")
    patients = server.get_resource("patients://list")
    for p in patients:
        print(f" - {p['name']} ({p['id']}), Age: {p['age']}")
    
    # Recurso 2: Obtener y mostrar sesiones de un paciente específico
    print("\n📁 Resource: patients://PAT-001/sessions")
    sessions = server.get_resource("patients://PAT-001/sessions")
    print(f" Total sessions: {sessions['total_sessions']}")
    
    # Recurso 3: Obtener y mostrar evaluaciones de un paciente
    print("\n📁 Resource: assessments://PAT-001")
    assessments = server.get_resource("assessments://PAT-001")
    print(f" CARS: {assessments['assessments'].get('CARS', {}).get('score', 'N/A')}")
    print(f" ADOS: {assessments['assessments'].get('ADOS', {}).get('score', 'N/A')}")
    
    print("\n" + "="*70)
    print("2. DEMOSTRACIÓN DE TOOLS (Herramientas)")
    print("="*70)
    
    # Tool 1: Calcular métricas de progreso para un paciente
    print("\n🔧 Tool: calculate_progress_metrics('PAT-001')")
    metrics = server.calculate_progress_metrics("PAT-001")
    print(f" Average progress: {metrics['average_progress_score']:.2f}")
    print(f" Trend: {metrics['progress_trend']}")
    
    # Tool 2: Generar sugerencia diagnóstica basada en datos del paciente
    print("\n🔧 Tool: generate_diagnostic_suggestion('PAT-001')")
    suggestion = server.generate_diagnostic_suggestion("PAT-001")
    print(f" Requires review: {suggestion['requires_review']}")
    
    # Tool 3: Filtrar valores atípicos estadísticos en un conjunto de datos
    print("\n🔧 Tool: filter_statistical_outliers([65,68,70,72,95,70,200])")
    outliers = server.filter_statistical_outliers([65, 68, 70, 72, 95, 70, 200])
    print(f" Outliers found: {outliers['outliers_count']}")
    
    # Tool 4: Simular resultado de una intervención terapéutica
    print("\n🔧 Tool: simulate_intervention_outcome('ABA', 7.5, 24)")
    simulation = server.simulate_intervention_outcome("ABA", 7.5, 24)
    print(f" Expected improvement: {simulation['improvement_percentage']}%")
    
    # Paso final: Mostrar mensaje de finalización y ejemplos de integración
    print("\n" + "="*70)
    print("✅ MCP Server demonstration complete")
    print("\n📌 CÓMO SE INTEGRA CON LANGGRAPH:")
    print(" from mcp_server import MCPServer")
    print(" mcp = MCPServer()")
    print(" data = mcp.get_resource('patients://list')")
    print(" result = mcp.calculate_progress_metrics('PAT-001')")
    print("="*70)