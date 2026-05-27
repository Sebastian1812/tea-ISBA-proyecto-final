"""
GRAPH AGENT - TEA Clinical Agent
================================

Permite que la IA decida QUÉ hacer basado en el contexto.

Este archivo CONECTA:
1. El MCP Server (datos y herramientas)
2. El LLM (Ollama - inteligencia)
3. La memoria (recordar lo que hizo)
"""
import numpy as np

from typing import TypedDict, Annotated, List, Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime
from mcp_server import MCPServer
from security import PromptSanitizer
from config import clinical_thresholds, model_config


# ============================================
# FUNCIÓN UTILITARIA: Convertir numpy types a Python nativos
# ============================================

def convert_numpy_to_python(obj: Any) -> Any:
    """
    Convierte objetos numpy a tipos nativos de Python para serialización.
    
    CRÍTICO: LangGraph usa msgpack para serializar el estado cuando usa MemorySaver.
    msgpack NO puede serializar numpy.float64, numpy.int64, etc.
    Esta función convierte recursivamente cualquier objeto numpy a tipos Python nativos.
    
    Complexity: O(n) donde n es la profundidad del objeto
    
    Ejemplo:
        numpy.float64(3.14) -> 3.14 (float)
        numpy.int64(42) -> 42 (int)
        numpy.array([1,2,3]) -> [1,2,3] (list)
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def sanitize_state_for_serialization(state: Dict) -> Dict:
    """
    Aplica la conversión a todo el estado para asegurar que sea serializable.
    Útil antes de guardar en memoria.
    """
    return convert_numpy_to_python(state)


# ============================================
# PARTE 1: DEFINICIÓN DEL ESTADO
# ============================================
# El "estado" es como la memoria a corto plazo del agente
# Viaja a través de todos los nodos del grafo


class ClinicalAgentState(TypedDict):
    """
    ESTADO DEL AGENTE
    
    Esto es como el "expediente" que se pasa entre los médicos.
    Cada nodo puede leer y modificar este estado.
    
    Tipado: TypedDict permite autocompletado y validación
    """
    patient_id: str
    patient_data: Dict[str, Any]
    sessions_data: Dict[str, Any]
    metrics: Dict[str, Any]
    report_generated: bool
    diagnostic_suggestion: Dict[str, Any]
    should_report: bool
    should_suggest_diagnostic: bool
    error: str
    retry_count: int
    memory_context: Dict[str, Any]


# ============================================
# PARTE 2: LA CLASE AGENTE PRINCIPAL
# ============================================
class ClinicalAgent:
    """
        EL AGENTE DE IA
        
        Este es el ORQUESTADOR principal. Decide qué hacer y cuándo.
        
        Responsabilidades:
        1. Conectar MCP (datos) con LLM (inteligencia)
        2. Ejecutar el flujo de decisión (grafo)
        3. Mantener memoria entre ejecuciones
        4. Manejar errores gracefulmente
    """
    
    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.llm = self._initialize_llm()
        self.sanitizer = PromptSanitizer()
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        print("🤖 Clinical Agent initialized (with numpy serialization fix)")
        print(f"   LLM Model: {model_config.default_model}")
        print(f"   Memory: Enabled with serialization protection")
        
    def _initialize_llm(self):
        """
            Inicializa el modelo de lenguaje (IA)
            
            ¿Por qué es importante?
            - Este es el componente que GENERA inteligencia
            - Todo lo demás (MCP, grafo, memoria) SOLO organiza
            - El LLM es el único que "crea" texto nuevo
            
        """
        try:
            return ChatOllama(
                model=model_config.default_model,
                temperature=model_config.temperature,
                num_predict=model_config.max_tokens
            )
        except Exception as e:
            print(f"Warning: Could not initialize {model_config.default_model}. Falling back to tinyllama. Error: {e}")
            return ChatOllama(
                model="tinyllama:1.1b-chat",
                temperature=0.3
            )
    
    def _build_graph(self) -> StateGraph:
        """
            CONSTRUYE EL GRAFO DE DECISIÓN
            
            El grafo define:
            - Los NODOS (qué funciones ejecutar)
            - Las CONEXIONES (órden de ejecución)
            - Las CONDICIONES (cuándo ir a cada nodo)
            
            Esto es como un mapa de flujo: "si pasa A, ve a B, si no, ve a C"
        """
        # Crear nuevo grafo con el tipo de estado definido
        workflow = StateGraph(ClinicalAgentState)
        
        # REGISTRAR NODOS (las funciones del agente)
        workflow.add_node("analyze_patient_data", self.analyze_patient_data)
        workflow.add_node("should_generate_report", self.should_generate_report)
        workflow.add_node("generate_progress_report", self.generate_progress_report)
        workflow.add_node("suggest_diagnostic_review", self.suggest_diagnostic_review)
        workflow.add_node("save_and_memorize", self.save_and_memorize)
        workflow.add_node("error_handler", self.error_handler)
        
        # DEFINIR PUNTO DE ENTRADA
        workflow.set_entry_point("analyze_patient_data")
        
        # DEFINIR CONEXIONES CONDICIONALES

        # CONEXIÓN 1: Después de analizar, ¿hubo error?
        workflow.add_conditional_edges(
            "analyze_patient_data",
            self.route_after_analysis,
            {
                "continue": "should_generate_report",
                "error": "error_handler"
            }
        )
        
        # CONEXIÓN 2: Decisión principal del reporte
        # CORRECCIÓN: "generate_report" y "suggest_diagnostic" ya no son mutuamente
        # excluyentes. Si hay datos suficientes Y alertas clínicas, el flujo pasa
        # primero por generate_progress_report y luego por suggest_diagnostic_review.
        workflow.add_conditional_edges(
            "should_generate_report",
            self.route_report_decision,
            {
                "generate_report": "generate_progress_report",
                "insufficient_data": END,
                "suggest_diagnostic_only": "suggest_diagnostic_review"
            }
        )
        
        # CONEXIÓN 3: Después de generar reporte, verificar si TAMBIÉN hay alerta diagnóstica
        workflow.add_conditional_edges(
            "generate_progress_report",
            self.route_after_report,
            {
                "suggest_diagnostic": "suggest_diagnostic_review",
                "save": "save_and_memorize"
            }
        )
        # CONEXIÓN 4: Después de sugerir diagnóstico, siempre guardar
        workflow.add_edge("suggest_diagnostic_review", "save_and_memorize")
        # CONEXIÓN 5: Después de guardar, terminar
        workflow.add_edge("save_and_memorize", END)
        # CONEXIÓN 6: Error handler termina
        workflow.add_edge("error_handler", END)
        
        return workflow
    

    # PARTE 3: NODOS DEL GRAFO (LA LÓGICA)
    def analyze_patient_data(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """
            NODO 1: ANALIZAR DATOS DEL PACIENTE
            
            ¿Qué hace?
            1. Toma el patient_id del estado
            2. Llama al MCP para obtener datos
            3. Calcula métricas usando tools del MCP
            4. Actualiza el estado con los resultados
            
        """
        try:
            patient_id = state['patient_id']
            
            print(f"\n📊 NODE: analyze_patient_data")
            print(f"   Patient ID: {patient_id}")
            
            # CONSULTAR RECURSOS DEL MCP (datos)
            patient_sessions = self.mcp_server.get_resource(f"patients://{patient_id}/sessions")
            assessments = self.mcp_server.get_resource(f"assessments://{patient_id}")
            
            # Calcula métricas usando tools del MCP
            metrics = self.mcp_server.calculate_progress_metrics(patient_id)
            
            # ============================================
            # CRÍTICO: Convertir numpy types a Python nativos
            # ============================================
            # Los cálculos de pandas/numpy pueden devolver numpy.float64
            # Los convertimos a float nativo de Python para evitar errores de serialización
            
            if metrics:
                # Convertir métricas principales
                if 'average_progress_score' in metrics:
                    metrics['average_progress_score'] = float(metrics['average_progress_score'])
                if 'overall_advancement_percentage' in metrics:
                    metrics['overall_advancement_percentage'] = float(metrics['overall_advancement_percentage'])
                if 'average_positive_responses' in metrics:
                    metrics['average_positive_responses'] = float(metrics['average_positive_responses'])
                
                # Convertir domain_progress
                if 'domain_progress' in metrics and metrics['domain_progress']:
                    for domain, domain_data in metrics['domain_progress'].items():
                        if 'average_progress' in domain_data:
                            domain_data['average_progress'] = float(domain_data['average_progress'])
                        if 'improvement_rate' in domain_data:
                            domain_data['improvement_rate'] = float(domain_data['improvement_rate'])
                        if 'sessions_count' in domain_data:
                            domain_data['sessions_count'] = int(domain_data['sessions_count'])
            
            # Actualiza el estado con los resultados (ya convertidos)
            state['patient_data'] = assessments
            state['sessions_data'] = patient_sessions
            state['metrics'] = metrics
            state['error'] = ""
            state['retry_count'] = state.get('retry_count', 0)
            
            # Verificar memoria existente
            memory_key = f"patient_{patient_id}_analyzed"
            if memory_key in state.get('memory_context', {}):
                print(f"   📝 Uso de la memoria caché para el paciente {patient_id}")
            else:
                print(f"   📝 Primera vez analizando al paciente {patient_id}")
            
            print(f"   ✅ Análisis completado: {metrics.get('total_sessions', 0)} sesiones encontradas")
            
            return state
            
        except Exception as e:
            state['error'] = str(e)
            state['retry_count'] = state.get('retry_count', 0) + 1
            print(f"   ❌ Error en analyze_patient_data: {e}")
            return state
    
    def route_after_analysis(self, state: ClinicalAgentState) -> Literal["continue", "error"]:
        """Ruta basada en el éxito del análisis"""
        if state.get('error') and state.get('retry_count', 0) < 3:
            # Retry logic
            print(f"   ⚠️ Error en el análisis, reintentando ({state['retry_count']}/3)")
            return "error"
        elif state.get('error'):
            print(f"   ❌ Error fatal después de 3 reintentos")
            return "error"
        print(f"   ✅ Análisis exitoso, continuando...")
        return "continue"
    
    def should_generate_report(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """
        Nodo de decisión: evaluar si hay suficientes datos para el informe.
        Complexity: O(1)
        """
        print(f"\n🧠 NODE: should_generate_report")

        # Obtener datos necesarios
        total_sessions = state.get('metrics', {}).get('total_sessions', 0)
        
        # Asegurar que los scores son números Python nativos
        cars_score_raw = state.get('patient_data', {}).get('assessments', {}).get('CARS', {}).get('score', 0)
        ados_score_raw = state.get('patient_data', {}).get('assessments', {}).get('ADOS', {}).get('score', 0)
        
        # Convertir a float por si vienen como numpy types
        cars_score = float(cars_score_raw) if cars_score_raw else 0.0
        ados_score = float(ados_score_raw) if ados_score_raw else 0.0

        # DECISIÓN 1: ¿Hay suficientes datos para reporte?
        state['should_report'] = total_sessions >= clinical_thresholds.min_sessions_for_report
        
        # DECISIÓN 2: ¿Sugerir revisión diagnóstica?
        state['should_suggest_diagnostic'] = (
            cars_score > float(clinical_thresholds.cars_threshold_critical) or 
            ados_score > float(clinical_thresholds.ados_threshold_critical)
        )
        
        print(f"   Sesiones: {total_sessions} / {clinical_thresholds.min_sessions_for_report} requeridas")
        print(f"   → Datos suficientes para reporte: {state['should_report']}")
        print(f"   CARS: {cars_score} (umbral: {clinical_thresholds.cars_threshold_critical})")
        print(f"   ADOS: {ados_score} (umbral: {clinical_thresholds.ados_threshold_critical})")
        print(f"   → Sugerir revisión diagnóstica: {state['should_suggest_diagnostic']}")
        
        return state
    
    def route_report_decision(self, state: ClinicalAgentState) -> Literal["generate_report", "insufficient_data", "suggest_diagnostic_only"]:
        """
        Enrutamiento condicional basado en la disponibilidad de datos.
        
        CORRECCIÓN: Si hay suficientes datos, SIEMPRE genera el reporte primero,
        sin importar si hay alertas diagnósticas. Las alertas se procesan DESPUÉS
        del reporte en route_after_report(). Solo si NO hay suficientes datos
        pero SÍ hay alertas, se va directo a suggest_diagnostic_only.
        """
        if state['should_report']:
            # Siempre genera reporte si hay datos suficientes (≥3 sesiones)
            print(f"   🔀 Redirigiendo a: generate_progress_report (datos suficientes)")
            return "generate_report"
        elif state['should_suggest_diagnostic']:
            # Sin datos suficientes para reporte, pero sí hay alertas clínicas
            print(f"   🔀 Redirigiendo a: suggest_diagnostic_review (alerta sin datos suficientes)")
            return "suggest_diagnostic_only"
        else:
            print(f"   🔀 Redirigiendo a: END (datos insuficientes)")
            return "insufficient_data"
    
    def route_after_report(self, state: ClinicalAgentState) -> Literal["suggest_diagnostic", "save"]:
        """
        NUEVO: Después de generar el reporte, decide si TAMBIÉN debe
        ejecutar la revisión diagnóstica (si hay alertas CARS/ADOS).
        Esto permite que ambos nodos corran en el mismo flujo.
        """
        if state.get('should_suggest_diagnostic'):
            print(f"   🔀 Post-reporte: también activando revisión diagnóstica")
            return "suggest_diagnostic"
        print(f"   🔀 Post-reporte: guardando directamente")
        return "save"
    
    def generate_progress_report(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """
        Generar informe clínico mediante LLM
        """
        print(f"\n📝 generate_progress_report - LLAMA AL LLM")
        
        # Verificar que el LLM existe
        if self.llm is None:
            state['error'] = "LLM no disponible. Verifica que Ollama esté corriendo."
            print(f"   ❌ {state['error']}")
            return state
        
        try:
            metrics = state['metrics']
            patient_data = state['patient_data']
            
            # Extraer datos con valores por defecto
            total_sessions = metrics.get('total_sessions', 0)
            avg_progress = float(metrics.get('average_progress_score', 0))
            advancement = float(metrics.get('overall_advancement_percentage', 0))
            trend = metrics.get('progress_trend', 'unknown')
            cars = patient_data.get('assessments', {}).get('CARS', {}).get('score', 'N/A')
            ados = patient_data.get('assessments', {}).get('ADOS', {}).get('score', 'N/A')
            
            # Obtener progreso por dominio
            domain_progress = metrics.get('domain_progress', {})
            comm_progress = domain_progress.get('communication', {}).get('average_progress', 0)
            social_progress = domain_progress.get('social_interaction', {}).get('average_progress', 0)
            
            # Crear prompt SEGURO usando el método correcto
            system_prompt = self.sanitizer.create_safe_system_prompt()
            
            # Prompt simplificado para tinyllama
            user_prompt = f"""Generate a brief clinical report for autism patient.

    Data:
    - Patient: {state['patient_id']}
    - Sessions: {total_sessions}
    - CARS: {cars}, ADOS: {ados}
    - Progress score: {avg_progress:.2f}
    - Trend: {trend}
    - Communication progress: {comm_progress:.2f}
    - Social interaction progress: {social_progress:.2f}
    - Overall advancement: {advancement:.1f}%

    Write a report with:
    1. Current status
    2. Progress summary
    3. 2-3 recommendations

    Be professional but concise."""

            # ✅ CORREGIDO: usar sanitize_user_input (no sanitize_user_prompt)
            safe_user_prompt = self.sanitizer.sanitize_user_input(user_prompt)
            
            print(f"   🤖 Calling LLM with prompt ({len(user_prompt)} chars)...")
            
            messages = [
                ("system", system_prompt),
                ("human", safe_user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            if response and response.content:
                state['report_generated'] = True
                state['memory_context'] = {
                    **state.get('memory_context', {}),
                    f"report_{state['patient_id']}_{datetime.now().isoformat()}": response.content
                }
                print(f"   ✅ Report generated ({len(response.content)} chars)")
                print(f"   Preview: {response.content[:200]}...")
            else:
                print(f"   ⚠️ LLM returned empty response")
                
        except Exception as e:
            state['error'] = f"Report generation failed: {str(e)}"
            print(f"   ❌ {state['error']}")
        
        return state
    
    def suggest_diagnostic_review(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """
        Generar sugerencias de diagnóstico
        Complexity: O(1) - llamada directa a la herramienta

        Diferencia del reporte: este es más corto y enfocado en diagnóstico
        """
        print(f"\n⚠️ NODE: suggest_diagnostic_review")
        
        try:
            # Utilice la herramienta MCP para sugerencias de diagnóstico.
            suggestion = self.mcp_server.generate_diagnostic_suggestion(
                state['patient_id'],
                state['patient_data']
            )
            
            # Aquí podrías llamar al LLM para expandir la sugerencia
            state['diagnostic_suggestion'] = suggestion
            state['memory_context'] = {
                **state.get('memory_context', {}),
                f"diagnostic_{state['patient_id']}_{datetime.now().isoformat()}": suggestion
            }
            
            print(f"   ✅ Diagnóstico generado y revisión sugerida para el paciente {state['patient_id']}")
            if suggestion.get('requires_review'):
                print(f"   ⚠️ Se requiere revisión diagnóstica inmediata")
            
        except Exception as e:
            state['error'] = f"Diagnostic suggestion failed: {str(e)}"
            print(f"   ❌ Error: {state['error']}")
        
        return state
    
    def save_and_memorize(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """
        Conservar los resultados y actualizar la memoria.
        Complexity: O(1)
        """
        print(f"\n💾 NODE: save_and_memorize")
        
        # Actualizar memoria con timestamp de análisis
        memory_key = f"patient_{state['patient_id']}_last_analysis"
        
        # Asegurar que las métricas guardadas son serializables
        serializable_metrics = convert_numpy_to_python(state.get('metrics', {}))
        
        state['memory_context'] = {
            **state.get('memory_context', {}),
            memory_key: datetime.now().isoformat(),
            f"patient_{state['patient_id']}_metrics": serializable_metrics
        }
        
        print(f"   💾 Análisis guardado en memoria para paciente {state['patient_id']}")
        print(f"   📊 Total de memorias: {len(state.get('memory_context', {}))}")
        
        return state
    
    def error_handler(self, state: ClinicalAgentState) -> ClinicalAgentState:
        """MANEJAR ERRORES"""
        print(f"\n❌ NODE: error_handler")
        
        error_msg = f"Error al procesar el paciente {state['patient_id']}: {state.get('error', 'Unknown error')}"
        print(f"   {error_msg}")
        
        if state.get('retry_count', 0) < 3:
            current_retry = state.get('retry_count', 0) + 1
            print(f"   🔄 Reintentando con parámetros ajustados... ({current_retry}/3)")
            # Reiniciar error para reintentar
            state['error'] = ""
            state['retry_count'] = current_retry
        else:
            print(f"   🛑 Se ha superado el número máximo de reintentos (3). No se puede recuperar.")
            print(f"   📝 Por favor, verifica los datos del paciente y los servicios MCP.")

        return state
    
    def run(self, patient_id: str, thread_id: str = "default") -> Dict:
        """
            EJECUTAR EL AGENTE
            
            Este es el método principal que se llama desde main.py o frontend.py.
            
            Args:
                patient_id: ID del paciente a analizar (ej: "PAT-001")
                thread_id: Identificador para la memoria (diferentes hilos = diferente memoria)
            
            Returns:
                Estado final del agente (con resultados, memoria, etc.)
            
            Ejemplo:
                agent = ClinicalAgent(mcp)
                resultado = agent.run("PAT-001")
                print(resultado['report_generated'])  # True si generó reporte
        """
        print(f"\n{'='*60}")
        print(f"🚀 EJECUTANDO AGENTE CLÍNICO")
        print(f"   Paciente: {patient_id}")
        print(f"   Hilo de memoria: {thread_id}")
        print(f"{'='*60}")
        
        initial_state: ClinicalAgentState = {
            'patient_id': patient_id,
            'patient_data': {},
            'sessions_data': {},
            'metrics': {},
            'report_generated': False,
            'diagnostic_suggestion': {},
            'should_report': False,
            'should_suggest_diagnostic': False,
            'error': "",
            'retry_count': 0,
            'memory_context': {}
        }
        
        # Configure with thread for memory
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Run the graph
            final_state = self.compiled_graph.invoke(initial_state, config)
            
            # Sanitizar el estado final para asegurar serialización (por si acaso)
            final_state = sanitize_state_for_serialization(final_state)
            
            print(f"\n{'='*60}")
            print(f"✅ EJECUCIÓN DEL AGENTE COMPLETADA")
            print(f"   Reporte generado: {final_state.get('report_generated', False)}")
            print(f"   Revisión diagnóstica sugerida: {final_state.get('diagnostic_suggestion', {}).get('requires_review', False)}")
            print(f"{'='*60}\n")
            
            return final_state
            
        except TypeError as e:
            # Error específico de serialización de numpy
            if "not msgpack serializable" in str(e) or "numpy.float" in str(e):
                print(f"\n❌ ERROR DE SERIALIZACIÓN DETECTADO")
                print(f"   Este error ocurre cuando hay tipos numpy en el estado.")
                print(f"   Aplicando limpieza de estado y reintentando...")
                
                # Limpiar completamente el estado inicial
                clean_state = sanitize_state_for_serialization(initial_state)
                final_state = self.compiled_graph.invoke(clean_state, config)
                final_state = sanitize_state_for_serialization(final_state)
                
                return final_state
            else:
                raise e
    
    def get_memory_status(self, thread_id: str = "default") -> Dict:
        """
        Obtener el estado actual de la memoria
        
        Útil para debugging y para el frontend
        """
        # Esto es simplificado - en producción tendrías acceso al estado guardado
        return {
            "thread_id": thread_id,
            "memory_enabled": True,
            "note": "MemorySaver stores state between runs automatically"
        }
    

if __name__ == "__main__":
    """
    Ejecutar: python graph_agent.py
    
    Esto demuestra cómo funciona el agente IA.
    """
    
    print("="*70)
    print("GRAPH AGENT DEMONSTRATION (VERSION CORREGIDA)")
    print("="*70)
    
    # Crear MCP Server (provee datos)
    from mcp_server import MCPServer
    mcp = MCPServer()
    
    # Crear AGENTE IA (orquestador)
    agent = ClinicalAgent(mcp)
    
    print("\n" + "="*70)
    print("DEMOSTRACIÓN DEL FLUJO COMPLETO")
    print("="*70)
    
    # Ejecutar para cada paciente
    for patient_id in ["PAT-001", "PAT-002", "PAT-003"]:
        try:
            result = agent.run(patient_id, thread_id="demo_thread")
            
            print(f"\n--- RESULTADO PARA {patient_id} ---")
            print(f"📊 Total sessions: {result['metrics'].get('total_sessions', 0)}")
            
            avg_progress = result['metrics'].get('average_progress_score', 0)
            if avg_progress:
                print(f"📈 Average progress: {float(avg_progress):.2f}")
            
            if result.get('report_generated'):
                print(f"✅ Reporte generado por IA")
            
            if result.get('diagnostic_suggestion', {}).get('requires_review'):
                print(f"⚠️ Revisión diagnóstica sugerida")
            
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Error con paciente {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("="*70)