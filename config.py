"""
Configuration module for TEA Clinical Agent System
Este archivo centraliza todas las configuraciones del sistema
"""

import os
from dataclasses import dataclass
from typing import Optional, List

# DATACLASS: Una forma elegante de crear clases que solo almacenan datos
# Python automáticamente genera los métodos __init__, __repr__, etc.
@dataclass
class ModelConfig:
    """
    Configuración del modelo de lenguaje (LLM)
    Los dataclasses en Python permiten definir estructuras de datos
    con anotaciones de tipo de forma muy limpia.
    """
    
    # Atributos con valores por defecto
    default_model: str = "tinyllama:1.1b-chat"  # Modelo principal
    alternative_models: Optional[List[str]] = None  # Modelos alternativos
    temperature: float = 0.3  # Temperatura baja = respuestas más consistentes (menos creativas)
    max_tokens: int = 2048  # Longitud máxima de respuesta
    
    def __post_init__(self):
        """
        Este método se ejecuta DESPUÉS del constructor __init__
        Es útil para inicializar valores dinámicos
        """
        if self.alternative_models is None:
            self.alternative_models = ["mistral", "llama3"]


@dataclass
class ClinicalThresholds:
    """
    Umbrales clínicos para diagnóstico de TEA
    Estos valores deben basarse en la literatura médica validada
    """
    
    cars_threshold_critical: float = 30.0  
    # CARS = Childhood Autism Rating Scale
    # > 30 indica autismo moderado-severo
    
    ados_threshold_critical: float = 7.0   
    # ADOS = Autism Diagnostic Observation Schedule
    # > 7 indica espectro autista
    
    min_sessions_for_report: float = 3
    # Número mínimo de sesiones para generar un reporte confiable
    
    # === CORRECCIÓN DEL ERROR ===
    # El error estaba aquí: el comentario debe empezar con #
    # No se puede usar % como comentario en Python
    progress_percentage_threshold: float = 15.0  # Porcentaje para mejora significativa
    # El símbolo # indica comentario hasta el final de la línea


@dataclass
class SecurityConfig:
    """
    Configuraciones de seguridad y validación
    Protege el sistema contra abusos e inyecciones maliciosas
    """
    
    max_tool_calls_per_minute: int = 30
    # Límite de llamadas por minuto para evitar sobrecarga
    
    allowed_patient_id_pattern: str = r'^PAT-\d{3}$'
    # Expresión regular para validar IDs de pacientes
    # ^PAT-XXX$ donde X son dígitos: Ej: PAT-001, PAT-999
    
    max_input_length: int = 5000
    # Límite de caracteres en inputs para prevenir ataques
    
    forbidden_commands: Optional[List[str]] = None
    # Comandos prohibidos que podrían intentar inyectar
    
    def __post_init__(self):
        if self.forbidden_commands is None:
            self.forbidden_commands = [
                "change diagnosis", "modify score", "delete patient",
                "override", "force", "ignore thresholds", "bypass"
            ]
            # Lista negra de comandos peligrosos


# Instancias globales (singletons)
# Se crean una sola vez y se importan en otros módulos
model_config = ModelConfig()
clinical_thresholds = ClinicalThresholds()
security_config = SecurityConfig()


# ============================================
# EXPLICACIÓN DE CONCEPTOS CLAVE
# ============================================

"""
1. DATACLASSES (decorador @dataclass):
   ------------------------------------
   Los dataclasses son una forma moderna y limpia de definir clases que
   principalmente almacenan datos. Antes de Python 3.7, tenías que escribir:
   
   class ModelConfig:
       def __init__(self, default_model="tinyllama"):
           self.default_model = default_model
   
   Con dataclass, Python genera automáticamente:
   - __init__ (constructor)
   - __repr__ (representación en string)
   - __eq__ (comparación de igualdad)
   - Y otros métodos útiles

2. TIPO DE DATOS (type hints):
   ---------------------------
   default_model: str = "tinyllama"
   ^^^^^^^^^^^^^^^ ^^^
   nombre          tipo de dato esperado
   
   Beneficios:
   - Mejor documentación
   - Autocompletado en editores
   - Detección temprana de errores

3. CONSTANTES vs CONFIGURACIÓN:
   ----------------------------
   - CONSTANTE: No cambia nunca (ej: PI = 3.14159)
   - CONFIGURACIÓN: Puede cambiar según el entorno (desarrollo, producción, pruebas)

4. ¿POR QUÉ USAR DATACLASSES EN VEZ DE DICCIONARIOS?
   -------------------------------------------------
   # Con diccionario (propenso a errores):
   config = {'model': 'llama', 'temp': 0.3}
   print(config['temperatura'])  # Error: typo, no existe la clave
   
   # Con dataclass (seguro y autocompletable):
   config = ModelConfig()
   print(config.temperature)  # Funciona, y el editor te ayuda
   print(config.temperatura)  # Error: AttributeError inmediato

5. MÉTODO __post_init__:
   ---------------------
   Es un "gancho" (hook) que se ejecuta después del constructor.
   Útil para:
   - Validar valores
   - Inicializar listas/diccionarios vacíos
   - Convertir tipos de datos
   - Calcular valores derivados

Ejemplo de uso en otros archivos:
---------------------------------
from config import model_config, clinical_thresholds, security_config

# Usar la configuración en tu código
if cars_score > clinical_thresholds.cars_threshold_critical:
    print("¡Revisión diagnóstica necesaria!")

# Cambiar temporalmente para pruebas
model_config.temperature = 0.8  # Más creativo para pruebas
"""

# ============================================
# EJEMPLOS PRÁCTICOS ADICIONALES
# ============================================

if __name__ == "__main__":
    # Este bloque se ejecuta solo si ejecutas directamente este archivo
    # Sirve para probar la configuración
    
    print("=== TESTING CONFIGURATION MODULE ===\n")
    
    # Probar ModelConfig
    print("1. Model Configuration:")
    m = ModelConfig()
    print(f"   Default model: {m.default_model}")
    print(f"   Temperature: {m.temperature}")
    print(f"   Alternative models: {m.alternative_models}")
    
    # Probar ClinicalThresholds (con la corrección)
    print("\n2. Clinical Thresholds:")
    c = ClinicalThresholds()
    print(f"   CARS threshold: {c.cars_threshold_critical}")
    print(f"   ADOS threshold: {c.ados_threshold_critical}")
    print(f"   Min sessions: {c.min_sessions_for_report}")
    print(f"   Progress % threshold: {c.progress_percentage_threshold}%")
    
    # Probar SecurityConfig
    print("\n3. Security Configuration:")
    s = SecurityConfig()
    print(f"   Rate limit: {s.max_tool_calls_per_minute} calls/minute")
    print(f"   Patient ID pattern: {s.allowed_patient_id_pattern}")
    print(f"   Forbidden commands: {s.forbidden_commands}")
    
    # Demostrar validación con regex
    print("\n4. Patient ID Validation Demo:")
    test_ids = ["PAT-001", "PAT-999", "PAT-1000", "INV-001", "pat-001"]
    import re
    for pid in test_ids:
        is_valid = bool(re.match(s.allowed_patient_id_pattern, pid))
        print(f"   {pid}: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Demostrar cómo se accede desde otros módulos
    print("\n5. How to import in other files:")
    print("   from config import model_config, clinical_thresholds")
    print("   if score > clinical_thresholds.cars_threshold_critical:")
    print("       print('Threshold exceeded')")