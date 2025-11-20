import numpy as np
import math
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ALGORITMO K-MEANS MANUAL
# =============================================================================

def calcular_distancia_euclidiana(punto1, punto2):
    """Calcula la distancia euclidiana entre dos puntos"""
    try:
        return math.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)
    except (TypeError, ValueError) as e:
        logger.error(f"Error calculando distancia: {e}")
        return float('inf')

def inicializar_centroides(datos, k, random_state=None):
    """Selecciona k puntos aleatorios como centroides iniciales"""
    if random_state:
        np.random.seed(random_state)
    indices = np.random.choice(len(datos), k, replace=False)
    return np.array([datos[i].copy() for i in indices])

def asignar_clusters(datos, centroides):
    """Asigna cada punto al centroide más cercano"""
    etiquetas = []
    for punto in datos:
        distancias = [calcular_distancia_euclidiana(punto, centroide) for centroide in centroides]
        etiquetas.append(np.argmin(distancias))
    return np.array(etiquetas)

def actualizar_centroides(datos, etiquetas, k):
    """Calcula nuevas posiciones para los centroides"""
    nuevos_centroides = []
    for i in range(k):
        puntos_cluster = [datos[j] for j in range(len(datos)) if etiquetas[j] == i]
        if len(puntos_cluster) > 0:
            nuevo_centro = np.mean(puntos_cluster, axis=0)
            nuevos_centroides.append(nuevo_centro)
        else:
            # Si cluster vacío, elegir punto aleatorio
            nuevos_centroides.append(datos[np.random.randint(0, len(datos))])
    return np.array(nuevos_centroides)

def calcular_inercia(datos, centroides, etiquetas):
    """Calcula la inercia total del clustering"""
    inercia = 0
    for i in range(len(datos)):
        distancia = calcular_distancia_euclidiana(datos[i], centroides[etiquetas[i]])
        inercia += distancia ** 2
    return inercia

def kmeans_robusto(datos, k, max_iter=100, random_state=None):
    """K-Means robusto que asegura exactamente k clusters"""
    if random_state is not None:
        np.random.seed(random_state)

    # Asegurar que k sea válido
    k = max(1, min(k, len(datos)))

    centroides = inicializar_centroides(datos, k, random_state)
    historial_centroides = [centroides.tolist()]

    for iteracion in range(max_iter):
        etiquetas = asignar_clusters(datos, centroides)

        # Verificar que todos los clusters tengan al menos un punto
        clusters_vacios = []
        for i in range(k):
            if sum(etiquetas == i) == 0:
                clusters_vacios.append(i)

        # Si hay clusters vacíos, reinicializarlos
        if clusters_vacios:
            for cluster_id in clusters_vacios:
                # Elegir un punto aleatorio lejos de los centroides existentes
                distancias = []
                for punto in datos:
                    dist_min = min(calcular_distancia_euclidiana(punto, c) for c in centroides if not np.isnan(c).any())
                    distancias.append(dist_min)

                if distancias:
                    punto_lejano = datos[np.argmax(distancias)]
                    centroides[cluster_id] = punto_lejano

        nuevos_centroides = actualizar_centroides(datos, etiquetas, k)

        # Verificar convergencia
        if np.allclose(centroides, nuevos_centroides, rtol=1e-4, atol=1e-6):
            break

        centroides = nuevos_centroides
        historial_centroides.append(centroides.tolist())

    # Asignación final
    etiquetas = asignar_clusters(datos, centroides)
    inercia = calcular_inercia(datos, centroides, etiquetas)

    return centroides, etiquetas.tolist(), float(inercia), historial_centroides

# =============================================================================
# MÉTRICAS DE CALIDAD
# =============================================================================

def calcular_silhouette_score(datos, etiquetas, centroides):
    """Calcula el Silhouette Score manualmente"""
    if len(set(etiquetas)) <= 1:
        return 0

    silhouette_scores = []
    for i, punto in enumerate(datos):
        cluster_actual = etiquetas[i]

        # Cohesión
        puntos_mismo_cluster = [datos[j] for j in range(len(datos))
                              if etiquetas[j] == cluster_actual and j != i]
        a = np.mean([calcular_distancia_euclidiana(punto, p) for p in puntos_mismo_cluster]) if puntos_mismo_cluster else 0

        # Separación
        b_values = []
        for cluster_id in set(etiquetas):
            if cluster_id != cluster_actual:
                puntos_otro_cluster = [datos[j] for j in range(len(datos)) if etiquetas[j] == cluster_id]
                if puntos_otro_cluster:
                    dist_promedio = np.mean([calcular_distancia_euclidiana(punto, p) for p in puntos_otro_cluster])
                    b_values.append(dist_promedio)

        b = min(b_values) if b_values else 0

        if max(a, b) > 0:
            silhouette = (b - a) / max(a, b)
        else:
            silhouette = 0

        silhouette_scores.append(silhouette)

    return np.mean(silhouette_scores) if silhouette_scores else 0

def calcular_davies_bouldin(datos, etiquetas, centroides):
    """Calcula Davies-Bouldin Index (menor es mejor)"""
    if len(set(etiquetas)) <= 1:
        return 0

    k = len(centroides)
    # Calcular diámetro promedio de cada cluster
    diametros = []
    for i in range(k):
        puntos_cluster = [datos[j] for j in range(len(datos)) if etiquetas[j] == i]
        if len(puntos_cluster) > 1:
            diametro = np.mean([calcular_distancia_euclidiana(p, centroides[i]) for p in puntos_cluster])
        else:
            diametro = 0
        diametros.append(diametro)

    # Calcular DB Index
    db_values = []
    for i in range(k):
        if diametros[i] == 0:
            continue
        max_ratio = 0
        for j in range(k):
            if i != j and diametros[j] > 0:
                distancia_centros = calcular_distancia_euclidiana(centroides[i], centroides[j])
                if distancia_centros > 0:
                    ratio = (diametros[i] + diametros[j]) / distancia_centros
                    max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)

    return np.mean(db_values) if db_values else 0

def evaluar_calidad_k(silhouette, db_index, k_actual, k_optimo):
    """Evalúa la calidad del k elegido por el usuario"""
    evaluacion = {
        'puntuacion_total': 0,
        'recomendacion': '',
        'es_optimo': bool(k_actual == k_optimo),
        'detalles': []
    }

    # Puntos por silhouette
    if silhouette > 0.7:
        evaluacion['puntuacion_total'] += 3
        evaluacion['detalles'].append("✅ Silhouette Score excelente (> 0.7)")
    elif silhouette > 0.5:
        evaluacion['puntuacion_total'] += 2
        evaluacion['detalles'].append("⚠️ Silhouette Score aceptable (> 0.5)")
    elif silhouette > 0.25:
        evaluacion['puntuacion_total'] += 1
        evaluacion['detalles'].append("🔶 Silhouette Score débil (> 0.25)")
    else:
        evaluacion['detalles'].append("❌ Silhouette Score pobre (≤ 0.25)")

    # Puntos por DB Index
    if db_index < 0.5:
        evaluacion['puntuacion_total'] += 3
        evaluacion['detalles'].append("✅ Davies-Bouldin excelente (< 0.5)")
    elif db_index < 1.0:
        evaluacion['puntuacion_total'] += 2
        evaluacion['detalles'].append("⚠️ Davies-Bouldin aceptable (< 1.0)")
    else:
        evaluacion['detalles'].append("❌ Davies-Bouldin pobre (≥ 1.0)")

    # Evaluación del k elegido
    if k_actual == k_optimo:
        evaluacion['puntuacion_total'] += 2
        evaluacion['recomendacion'] = f"🎯 Excelente elección! k={k_actual} es el número óptimo de hospitales"
    elif abs(k_actual - k_optimo) == 1:
        evaluacion['recomendacion'] = f"⚠️ k={k_actual} es cercano al óptimo (k={k_optimo}), considerá cambiar"
    else:
        evaluacion['recomendacion'] = f"❌ k={k_actual} no es óptimo. Se recomienda k={k_optimo}"

    # Evaluación final
    if evaluacion['puntuacion_total'] >= 5:
        evaluacion['recomendacion'] += " - Calidad: EXCELENTE 🌟"
    elif evaluacion['puntuacion_total'] >= 3:
        evaluacion['recomendacion'] += " - Calidad: BUENA 👍"
    else:
        evaluacion['recomendacion'] += " - Calidad: MEJORABLE 💡"

    return evaluacion

def calcular_metricas_completas(datos, centroides, etiquetas, k_actual, k_optimo):
    """Calcula todas las métricas de calidad"""
    try:
        inercia = calcular_inercia(datos, centroides, etiquetas)
        silhouette = calcular_silhouette_score(datos, etiquetas, centroides)
        db_index = calcular_davies_bouldin(datos, etiquetas, centroides)

        # Métricas de distancia
        distancias = []
        for i, punto in enumerate(datos):
            dist = calcular_distancia_euclidiana(punto, centroides[etiquetas[i]])
            distancias.append(dist)

        # Evaluación de calidad
        evaluacion = evaluar_calidad_k(silhouette, db_index, k_actual, k_optimo)

        return {
            'inercia': float(inercia),
            'silhouette_score': float(silhouette),
            'davies_bouldin_index': float(db_index),
            'distancia_promedio': float(np.mean(distancias)),
            'distancia_maxima': float(np.max(distancias)),
            'distancia_minima': float(np.min(distancias)),
            'evaluacion_calidad': evaluacion
        }
    except Exception as e:
        logger.error(f"Error calculando métricas: {e}")
        return {
            'inercia': 0,
            'silhouette_score': 0,
            'davies_bouldin_index': 0,
            'distancia_promedio': 0,
            'distancia_maxima': 0,
            'distancia_minima': 0,
            'evaluacion_calidad': {'puntuacion_total': 0, 'recomendacion': 'Error en cálculo', 'detalles': []}
        }

def encontrar_k_optimo(datos, k_max=None):
    """Encuentra el k óptimo usando método del codo.

    Si k_max es None la función calcula automáticamente un k_max
    adecuado basado en el tamaño del dataset.
    """
    n = len(datos)
    if n <= 2:
        return 1

    # Heurística automática para k_max si no se pasó explícitamente
    if k_max is None:
        # sugerir hasta 2 * sqrt(n), con tope en 30 y como máximo n-1
        sugerido = int(max(2, min(30, np.sqrt(n) * 2)))
        k_max_calculado = min(sugerido, n - 1)
    else:
        # asegurar enteros y que no exceda n-1
        k_max_calculado = min(int(k_max), n - 1)
        k_max_calculado = max(2, k_max_calculado)

    inercias = []

    for k in range(1, k_max_calculado + 1):
        try:
            centroides, etiquetas, inercia, _ = kmeans_robusto(datos, k, random_state=42)
            inercias.append(inercia)
        except Exception as e:
            logger.error(f"Error con k={k}: {e}")
            inercias.append(float('inf'))

    if len(inercias) < 3:
        # Si hay pocas muestras, elegir k por Silhouette (más robusto que forzar 2)
        best_k = 1
        best_score = -1.0
        # probar k desde 2 hasta un pequeño tope (no más que n-1)
        for kk in range(2, min(5, n - 1) + 1):
            try:
                centroides, etiquetas, _, _ = kmeans_robusto(datos, kk, random_state=42)
                score = calcular_silhouette_score(datos, etiquetas, centroides)
                if score > best_score:
                    best_score = score
                    best_k = kk
            except Exception as e:
                logger.debug(f"Silhouette error for k={kk}: {e}")
        return int(best_k)

    # Encontrar el codo usando la segunda derivada
    diferencias = np.diff(inercias)
    if len(diferencias) > 1:
        segundas_diff = np.diff(diferencias)
        if len(segundas_diff) > 0:
            k_optimo = int(np.argmax(segundas_diff) + 2)
            # asegurar que k_optimo esté en el rango válido
            k_optimo = max(1, min(k_optimo, k_max_calculado))
            return int(k_optimo)

    # Fallback razonable
    return min(3, k_max_calculado)

# =============================================================================
# ENDPOINTS DE LA API
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'K-Means API funcionando'})

@app.route('/api/generar-vecindarios', methods=['POST'])
def generar_vecindarios():
    """Genera vecindarios aleatorios no simétricos"""
    try:
        data = request.json
        n_vecindarios = data.get('n_vecindarios', 50)
        tamano_espacio = data.get('tamano_espacio', 100)

        logger.info(f"Generando {n_vecindarios} vecindarios en espacio {tamano_espacio}x{tamano_espacio}")

        vecindarios = []
        n_clusters = max(3, n_vecindarios // 15)

        for _ in range(n_clusters):
            centro_cluster = [
                np.random.uniform(20, tamano_espacio - 20),
                np.random.uniform(20, tamano_espacio - 20)
            ]
            puntos_en_cluster = max(3, n_vecindarios // n_clusters)

            for _ in range(puntos_en_cluster):
                x = centro_cluster[0] + np.random.normal(0, 10)
                y = centro_cluster[1] + np.random.normal(0, 10)
                x = max(0, min(tamano_espacio, x))
                y = max(0, min(tamano_espacio, y))
                vecindarios.append([float(x), float(y)])

        while len(vecindarios) < n_vecindarios:
            vecindarios.append([
                float(np.random.uniform(0, tamano_espacio)),
                float(np.random.uniform(0, tamano_espacio))
            ])

        logger.info(f"Generados {len(vecindarios)} vecindarios")

        return jsonify({
            'vecindarios': vecindarios[:n_vecindarios],
            'total_puntos': n_vecindarios,
            'tamano_espacio': tamano_espacio
        })

    except Exception as e:
        logger.error(f"Error generando vecindarios: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/calcular-hospitales', methods=['POST'])
def calcular_hospitales():
    """Calcula ubicaciones óptimas para hospitales con el k especificado por el usuario"""
    try:
        data = request.json or {}
        vecindarios = data.get('vecindarios', [])
        k_usuario = int(data.get('k', 3))
        tamano_espacio = float(data.get('tamano_espacio', 100.0))

        logger.info(f"Calculando hospitales: k={k_usuario}, vecindarios={len(vecindarios)}")

        if not vecindarios:
            return jsonify({'error': 'No hay vecindarios'}), 400

        datos = np.array(vecindarios, dtype=float)

        # Validar k
        k_usuario = max(1, min(k_usuario, len(datos)))

        # Encontrar k óptimo
        k_optimo = encontrar_k_optimo(datos)

        # Usar K-Means robusto
        centroides, etiquetas, inercia, historial = kmeans_robusto(datos, k_usuario, random_state=42)

        metricas = calcular_metricas_completas(datos, centroides, etiquetas, k_usuario, k_optimo)

        logger.info(f"Hospitales calculados: {len(centroides)} centroides")

        return jsonify({
            'hospitales': [list(map(float, c)) for c in centroides.tolist()],
            'etiquetas': list(map(int, etiquetas)),
            'metricas': metricas,
            'historial_centroides': historial,
            'k_optimo_calculado': int(k_optimo),
            'resumen': {
                'k_usuario': int(k_usuario),
                'k_optimo': int(k_optimo),
                'total_vecindarios': int(len(vecindarios)),
                'inercia': float(inercia),
                'clusters_creados': len(centroides)
            },
            'tamano_espacio': float(tamano_espacio)
        })

    except Exception as e:
        logger.error(f"Error calculando hospitales: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analizar-k-optimo', methods=['POST'])
def analizar_k_optimo():
    try:
        data = request.json or {}
        vecindarios = data.get('vecindarios', [])
        # Si el cliente envía 'k_max' positivo lo usamos; si no, lo dejamos en None
        k_max_input = data.get('k_max', None)
        tamano_espacio = float(data.get('tamano_espacio', 100.0))

        logger.info(f"Analizando k óptimo: k_max_input={k_max_input}, vecindarios={len(vecindarios)}")

        if not vecindarios:
            return jsonify({'error': 'No hay vecindarios'}), 400

        datos = np.array(vecindarios, dtype=float)

        # calcular k_max automáticamente dentro de encontrar_k_optimo si k_max_input es None
        # pero también queremos usar un k_max para el bucle de resultados: calculamos el valor sugerido
        n = len(datos)
        if k_max_input is None:
            sugerido = int(max(2, min(30, np.sqrt(n) * 2)))
            k_max_usado = min(sugerido, n - 1)
        else:
            k_max_usado = min(int(k_max_input), n - 1)
            k_max_usado = max(2, k_max_usado)

        resultados = []
        for k in range(1, k_max_usado + 1):
            centroides, etiquetas, inercia, _ = kmeans_robusto(datos, k, random_state=42)
            silhouette = calcular_silhouette_score(datos, etiquetas, centroides)
            db_index = calcular_davies_bouldin(datos, etiquetas, centroides)

            resultados.append({
                'k': int(k),
                'inercia': float(inercia),
                'silhouette': float(silhouette),
                'davies_bouldin': float(db_index)
            })

        k_optimo = encontrar_k_optimo(datos, k_max=k_max_usado)

        logger.info(f"K óptimo encontrado: {k_optimo} (k_max_usado={k_max_usado})")

        return jsonify({
            'resultados': resultados,
            'k_optimo': int(k_optimo),
            'k_max_usado': int(k_max_usado),
            'recomendacion': f"K óptimo recomendado: {int(k_optimo)} hospitales",
            'tamano_espacio': float(tamano_espacio)
        })

    except Exception as e:
        logger.error(f"Error analizando k óptimo: {e}")
        return jsonify({'error': str(e)}), 500
    
    
#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug=False)
