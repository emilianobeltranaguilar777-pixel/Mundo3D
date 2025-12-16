import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random

import cv2
import mediapipe as mp
import numpy as np

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WINDOW_TITLE = "Proyecto 3: Ciudad Entorno 3D - Equipo 5"

# Dimensiones del Terreno
ISLAND_SIZE = 40  # Radio del mundo (Total 80x80 unidades)
STEP = 0.5  # Resolución de la malla (0.5 = Alta calidad)
DIRT_DEPTH = 10  # Profundidad de la base de tierra

# Variable global para almacenar la geometría generada
vertex_data = []

# ==========================================
# CARGADOR DE MODELOS OBJ
# ==========================================

# Cache para modelos cargados (evita recargar el mismo archivo)
loaded_models = {}


def load_obj(filepath):
    """
    Carga un archivo .obj y retorna los vértices y caras.
    Ignora materiales (.mtl) y normales para mantener simplicidad.
    """
    if filepath in loaded_models:
        return loaded_models[filepath]

    vertices = []
    faces = []

    try:
        with open(filepath, 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Vértices
                if parts[0] == 'v':
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    vertices.append((x, y, z))

                # Caras (pueden ser triángulos o quads)
                elif parts[0] == 'f':
                    face_vertices = []
                    for p in parts[1:]:
                        # Formato puede ser: v, v/vt, v/vt/vn, v//vn
                        vertex_index = int(p.split('/')[0]) - 1  # OBJ usa índice base 1
                        face_vertices.append(vertex_index)
                    faces.append(face_vertices)
    except Exception as e:
        print(f"Error cargando {filepath}: {e}")
        return None

    model_data = {'vertices': vertices, 'faces': faces}
    loaded_models[filepath] = model_data
    return model_data


def draw_obj_model(model_data, position=(0, 0, 0), scale=1.0, color=(0.7, 0.7, 0.7)):
    """
    Dibuja un modelo OBJ cargado en una posición específica.
    """
    if model_data is None:
        return

    vertices = model_data['vertices']
    faces = model_data['faces']

    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glScalef(scale, scale, scale)
    glColor3f(*color)

    glBegin(GL_TRIANGLES)
    for face in faces:
        if len(face) >= 3:
            # Triangular el polígono (fan triangulation)
            for i in range(1, len(face) - 1):
                idx0 = face[0]
                idx1 = face[i]
                idx2 = face[i + 1]

                if idx0 < len(vertices) and idx1 < len(vertices) and idx2 < len(vertices):
                    glVertex3f(*vertices[idx0])
                    glVertex3f(*vertices[idx1])
                    glVertex3f(*vertices[idx2])
    glEnd()

    glPopMatrix()


# ==========================================
# CONFIGURACIÓN DE OBJETOS EN LA ESCENA
# ==========================================

# Y Offsets por tipo de objeto (evita que queden enterrados)
Y_OFFSETS = {
    "tree": 1.5,  # Árboles necesitan offset alto (origen en centro)
    "house": 2.5,  # Casas con origen bajo
    "car": 0.8,  # Coches justo sobre el asfalto
    "snowman": 2.0,  # Muñecos de nieve (se suma a altura de montaña)
    "monkey": 1.5,  # Monos decorativos
}

# Lista de objetos a renderizar: (modelo, posición, escala, color)
scene_objects = []


def setup_scene_objects():
    """
    Configura 48 objetos distribuidos en la escena.
    Los modelos se reutilizan (instancias) en diferentes posiciones.
    """
    global scene_objects
    scene_objects = []

    # Rutas de los modelos
    import os
    base_path = os.path.dirname(os.path.abspath(__file__))

    tree_path = os.path.join(base_path, "MODELS", "Tree", "tree.obj")
    house_path = os.path.join(base_path, "MODELS", "Small House", "small_house.obj")
    car_path = os.path.join(base_path, "MODELS", "Car", "1377 Car.obj")
    snowman_path = os.path.join(base_path, "MODELS", "Snow Man", "snowman.obj")
    monkey_path = os.path.join(base_path, "MODELS", "Monkey", "monkey.obj")

    # Cargar modelos
    tree_model = load_obj(tree_path)
    house_model = load_obj(house_path)
    car_model = load_obj(car_path)
    snowman_model = load_obj(snowman_path)
    monkey_model = load_obj(monkey_path)

    # =============================================
    # ÁRBOLES - 20 instancias en clusters
    # =============================================
    tree_positions_xz = [
        # Cluster izquierdo (bosque)
        (-20, -15), (-22, -12), (-18, -18), (-24, -10), (-19, -8),
        # Cluster derecho
        (15, 20), (18, 22), (12, 18), (20, 25), (17, 28),
        # Dispersos por el terreno
        (-30, 10), (-28, -5), (8, -25), (-15, 30), (-35, 15),
        # Cerca de casas
        (-32, -22), (-25, 28), (8, -28), (15, 30), (-10, -30)
    ]
    for x, z in tree_positions_xz:
        pos = (x, Y_OFFSETS["tree"], z)
        scene_objects.append((tree_model, pos, 3.0, (0.15, 0.45, 0.15)))

    # =============================================
    # CASAS - 8 instancias en las orillas
    # =============================================
    house_positions_xz = [
        (-30, -25), (-32, 20), (-28, 0), (-35, -10),
        (8, -32), (10, 32), (12, -15), (6, 25)
    ]
    for x, z in house_positions_xz:
        pos = (x, Y_OFFSETS["house"], z)
        scene_objects.append((house_model, pos, 4.0, (0.75, 0.55, 0.35)))

    # =============================================
    # COCHES - 8 instancias SOBRE la carretera
    # =============================================
    car_x_positions = [-25, -18, -10, -3, 4, 10, 18, 25]
    car_colors = [
        (0.8, 0.2, 0.2),  # Rojo
        (0.2, 0.4, 0.8),  # Azul
        (0.9, 0.9, 0.2),  # Amarillo
        (0.3, 0.7, 0.3),  # Verde
        (0.6, 0.3, 0.6),  # Morado
        (0.9, 0.5, 0.2),  # Naranja
        (0.2, 0.2, 0.2),  # Negro
        (0.9, 0.9, 0.9),  # Blanco
    ]
    for i, car_x in enumerate(car_x_positions):
        road_z = get_road_center(car_x)
        z_offset = 1.5 if i % 2 == 0 else -1.5
        pos = (car_x, Y_OFFSETS["car"], road_z + z_offset)
        scene_objects.append((car_model, pos, 0.05, car_colors[i]))

    # =============================================
    # MUÑECOS DE NIEVE - 6 instancias en la montaña
    # =============================================
    snowman_base_positions = [
        (28, 8.0, -28), (32, 10.0, -32), (25, 6.0, -25),
        (30, 12.0, -35), (35, 9.0, -30), (27, 7.0, -33)
    ]
    for x, mountain_y, z in snowman_base_positions:
        pos = (x, mountain_y + Y_OFFSETS["snowman"], z)
        scene_objects.append((snowman_model, pos, 8.0, (0.95, 0.95, 1.0)))

    # =============================================
    # MONOS - 6 instancias decorativas
    # =============================================
    monkey_positions_xz = [
        (-35, 5), (35, -5), (-8, 35), (5, -35), (-38, -15), (38, 20)
    ]
    for x, z in monkey_positions_xz:
        pos = (x, Y_OFFSETS["monkey"], z)
        scene_objects.append((monkey_model, pos, 1.5, (0.55, 0.35, 0.2)))

    print(f"Escena configurada con {len(scene_objects)} objetos.")


# ==========================================
# LÓGICA MATEMÁTICA DEL TERRENO
# ==========================================

def get_road_center(x):
    """
    Define la trayectoria de la carretera.
    Retorna la posición Z para un X dado, creando una curva suave.
    """
    return x + 10 * math.sin(x / 12.0)


def interpolate_color(color1, color2, factor):
    """
    Función auxiliar para crear degradados suaves entre dos colores.
    factor: 0.0 (color1) a 1.0 (color2)
    """
    r = color1[0] + (color2[0] - color1[0]) * factor
    g = color1[1] + (color2[1] - color1[1]) * factor
    b = color1[2] + (color2[2] - color1[2]) * factor
    return (r, g, b)


def generate_terrain_geometry():
    """
    Genera la malla de vértices, calcula alturas y asigna colores estáticos.
    Se ejecuta una sola vez al inicio para optimizar rendimiento.
    """
    global vertex_data
    vertex_data = []

    # Configuración de la Montaña (Esquina Superior Derecha)
    mount_x, mount_z = 32, -32
    mount_radius = 40

    # Iteramos sobre el área total con la resolución definida
    range_limit = int(ISLAND_SIZE * 2)

    for i in range(-range_limit, range_limit + 1):
        x = i / 2.0  # Ajuste por el STEP de 0.5
        row = []

        for j in range(-range_limit, range_limit + 1):
            z = j / 2.0

            # --- 1. CÁLCULO DE ALTURA ---
            base_noise = random.uniform(0.0, 0.3)  # Pequeña variación natural
            y = base_noise

            road_z = get_road_center(x)
            dist_to_road = abs(z - road_z)
            road_width = 3.5

            is_road = False
            is_mount = False
            mount_height_factor = 0  # Qué tan alto estamos en la montaña (0 a 1)

            # Zona Carretera
            if dist_to_road < road_width:
                y = 0.0
                is_road = True

            # Zona Terreno / Montaña
            else:
                dist_to_mount = math.sqrt((x - mount_x) ** 2 + (z - mount_z) ** 2)

                if dist_to_mount < mount_radius:
                    # Elevación usando Coseno para suavidad
                    factor = (dist_to_mount / mount_radius) * (math.pi / 2)
                    elevation = 15 * math.cos(factor)
                    if elevation < 0: elevation = 0

                    y += elevation
                    is_mount = True
                    # Guardamos el factor de altura para el color (0 = base, 15 = pico)
                    mount_height_factor = elevation

            # --- 2. ASIGNACIÓN DE COLOR ---

            if is_road:
                # Gris Asfalto
                r, g, b = 0.25, 0.25, 0.25

            elif is_mount:
                # Lógica de Degradado de Montaña

                # Definición de Colores Clave
                col_pasto = (0.1, 0.6, 0.1)  # Base (Igual al plano)
                col_bosque = (0.05, 0.25, 0.05)  # Verde Intenso Oscuro
                col_roca = (0.4, 0.35, 0.25)  # Café Roca
                col_nieve = (0.95, 0.95, 1.0)  # Blanco

                # Transiciones
                if y > 13:
                    # Nieve (Pico)
                    r, g, b = col_nieve
                elif y > 10:
                    # Roca (Parte alta)
                    r, g, b = col_roca
                else:
                    # DEGRADADO: De Pasto a Bosque Intenso
                    # Normalizamos la altura de 0 a 10 para el degradado
                    grad_factor = min(y / 10.0, 1.0)
                    r, g, b = interpolate_color(col_pasto, col_bosque, grad_factor)

                    # Agregamos un mínimo de ruido para mantener textura
                    noise = random.uniform(-0.02, 0.02)
                    g += noise

            else:
                # Zona Plana (Ciudad)
                # Verde pasto con variación ligera
                noise = random.uniform(-0.05, 0.05)
                r, g, b = 0.1, 0.6 + noise, 0.1

            # Guardamos el vértice completo
            row.append({
                "coords": (x, y, z),
                "color": (r, g, b)
            })
        vertex_data.append(row)


# ==========================================
# FUNCIONES DE DIBUJADO
# ==========================================

def draw_terrain_surface():
    """Dibuja la superficie superior (pasto, carretera, montaña)."""
    rows = len(vertex_data)
    cols = len(vertex_data[0])

    glBegin(GL_QUADS)
    for r in range(rows - 1):
        for c in range(cols - 1):
            p1 = vertex_data[r][c]
            p2 = vertex_data[r + 1][c]
            p3 = vertex_data[r + 1][c + 1]
            p4 = vertex_data[r][c + 1]

            # Usamos el color pre-calculado del vértice
            col = p1["color"]
            glColor3f(*col)

            glVertex3f(*p1["coords"])
            glVertex3f(*p2["coords"])
            glVertex3f(*p3["coords"])
            glVertex3f(*p4["coords"])
    glEnd()


def draw_dirt_walls():
    """
    Dibuja los muros laterales de tierra.
    Se adaptan dinámicamente a la altura del terreno para evitar huecos.
    """
    rows = len(vertex_data)
    cols = len(vertex_data[0])

    glColor3f(0.35, 0.2, 0.05)  # Color Marrón Tierra

    glBegin(GL_QUADS)

    # Recorremos los 4 bordes del mapa

    # 1. Borde Norte (z mínima)
    for c in range(cols - 1):
        p1 = vertex_data[0][c]["coords"]
        p2 = vertex_data[0][c + 1]["coords"]
        # Pared conecta la superficie con la profundidad
        glVertex3f(p1[0], p1[1], p1[2])  # Arriba Izq
        glVertex3f(p2[0], p2[1], p2[2])  # Arriba Der
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])  # Abajo Der
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])  # Abajo Izq

    # 2. Borde Sur (z máxima)
    for c in range(cols - 1):
        p1 = vertex_data[rows - 1][c]["coords"]
        p2 = vertex_data[rows - 1][c + 1]["coords"]
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])

    # 3. Borde Oeste (x mínima)
    for r in range(rows - 1):
        p1 = vertex_data[r][0]["coords"]
        p2 = vertex_data[r + 1][0]["coords"]
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])

    # 4. Borde Este (x máxima)
    for r in range(rows - 1):
        p1 = vertex_data[r][cols - 1]["coords"]
        p2 = vertex_data[r + 1][cols - 1]["coords"]
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])

    # Tapa Inferior (Fondo plano para cerrar el bloque)
    glVertex3f(-ISLAND_SIZE, -DIRT_DEPTH, -ISLAND_SIZE)
    glVertex3f(ISLAND_SIZE, -DIRT_DEPTH, -ISLAND_SIZE)
    glVertex3f(ISLAND_SIZE, -DIRT_DEPTH, ISLAND_SIZE)
    glVertex3f(-ISLAND_SIZE, -DIRT_DEPTH, ISLAND_SIZE)

    glEnd()


def draw_road_lines():
    """Dibuja las líneas blancas discontinuas de la carretera."""
    glLineWidth(2)
    glColor3f(1, 1, 1)
    glBegin(GL_LINES)

    step_check = 0.5
    range_val = int(ISLAND_SIZE / step_check)

    for i in range(-range_val, range_val):
        x = i * step_check
        # Patrón discontinuo
        if i % 8 < 4:
            z = get_road_center(x)
            if -ISLAND_SIZE < z < ISLAND_SIZE:
                glVertex3f(x, 0.05, z)  # Elevación mínima para evitar Z-fighting

                next_x = x + step_check
                next_z = get_road_center(next_x)
                glVertex3f(next_x, 0.05, next_z)
    glEnd()


# ==========================================
# ZONA DE COLABORACIÓN (EQUIPO)
# ==========================================
def draw_team_objects():
    """
    Dibuja todos los objetos 3D configurados en la escena.
    Total: 22 objetos (8 árboles + 4 casas + 4 coches + 3 muñecos + 3 monos)
    """
    for obj in scene_objects:
        model, position, scale, color = obj
        draw_obj_model(model, position, scale, color)


# ----------------------------
#   MEDIAPIPE (MANOS)
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

cap = cv2.VideoCapture(0)

# Centro al que va a mirar la camara
target_x, target_y, target_z = 0.0, 0.0, 0.0

# posicion "real" de la camara
cam_x, cam_y, cam_z = 0.0, 40.0, 70.0

# limite de zoom para no meternos a el entorno
CAM_Z_MIN = 12.0
CAM_Z_MAX = 140.0

yaw = 0.0  # angulo horizontal
YAW_SPEED = 0.06  # rapidez del giro
DEAD_ZONE = 0.08  # zona muerta para evitar vibracion
SMOOTH = 0.15  # suavizado para que no tiemble


# CONTROL DE LA CAMARA
def update_camera_from_hands():
    global cam_x, cam_z, yaw

    ret, frame = cap.read()
    if not ret:
        return True

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    h, w, _ = frame.shape
    indice_der = None
    indice_izq = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            idx = hand_landmarks.landmark[8]  # PUNTO DEL INDICE
            x, y = int(idx.x * w), int(idx.y * h)

            label = handedness.classification[0].label
            if label == "Right":
                indice_der = (x, y)
            else:
                indice_izq = (x, y)

            cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

    if indice_der and indice_izq:
        cv2.line(frame, indice_der, indice_izq, (255, 0, 0), 3)

        # PARTE DEL ZOOM
        distancia = np.linalg.norm(np.array(indice_der) - np.array(indice_izq))
        radius = np.interp(distancia, [50, 400], [CAM_Z_MAX, CAM_Z_MIN])
        radius = np.clip(radius, CAM_Z_MIN, CAM_Z_MAX)

        # midpoint
        centro_x = (indice_der[0] + indice_izq[0]) / 2.0
        centro_y = (indice_der[1] + indice_izq[1]) / 2.0

        mid_x = int(centro_x)
        mid_y = int(centro_y)
        cv2.circle(frame, (mid_x, mid_y), 9, (0, 255, 0), -1)

        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)

        x_centered = (centro_x / w) * 2.0 - 1.0

        if abs(x_centered) > DEAD_ZONE:
            yaw += x_centered * YAW_SPEED

        # la orbita
        desired_x = target_x + radius * np.sin(yaw)
        desired_z = target_z + radius * np.cos(yaw)

        cam_x = (1 - SMOOTH) * cam_x + SMOOTH * desired_x
        cam_z = (1 - SMOOTH) * cam_z + SMOOTH * desired_z

    cv2.imshow("Control con manos usando landmarks y midpoint", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True


# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def main():
    pygame.init()
    display = (SCREEN_WIDTH, SCREEN_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(WINDOW_TITLE)

    # Configuración OpenGL
    glEnable(GL_DEPTH_TEST)  # Habilitar buffer de profundidad
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

    # 1. Generar el mundo (Una sola vez)
    print("Generando entorno...")
    generate_terrain_geometry()
    print("Entorno listo.")

    # 2. Cargar y configurar objetos 3D
    print("Cargando modelos 3D...")
    setup_scene_objects()
    print("Modelos cargados.")

    # Variables de Cámara
    look_x, look_y, look_z = 0, 0, 0

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ok = update_camera_from_hands()
        if not ok:
            running = False

        # Renderizado
        glClearColor(0.53, 0.81, 0.92, 1)  # Cielo azul claro
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Actualizar Cámara
        gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 1, 0)

        # Dibujar Escena
        draw_terrain_surface()
        draw_dirt_walls()
        draw_road_lines()

        # Objetos del equipo
        draw_team_objects()

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


if __name__ == "__main__":
    main()
