import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random

# ==========================================
# CONFIGURACIÓN DEL PROYECTO
# ==========================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WINDOW_TITLE = "Proyecto 3: Ciudad Entorno 3D - Equipo 5"

# Dimensiones del Terreno
ISLAND_SIZE = 40       # Radio del mundo (Total 80x80 unidades)
STEP = 0.5             # Resolución de la malla (0.5 = Alta calidad)
DIRT_DEPTH = 10        # Profundidad de la base de tierra

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

# Lista de objetos a renderizar: (modelo, posición, escala, color)
scene_objects = []

def setup_scene_objects():
    """
    Configura los 20+ objetos que se colocarán en la escena.
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

    # ÁRBOLES - 8 instancias distribuidas en el terreno
    # Tree: ~1.7 unidades de alto, escala 3.0 = ~5 unidades final
    tree_positions = [
        (-15, 0.5, -10), (-20, 0.5, -5), (-25, 0.5, 5), (-18, 0.5, 15),
        (15, 0.5, -20), (20, 0.5, -15), (25, 0.5, 10), (18, 0.5, 20)
    ]
    for pos in tree_positions:
        scene_objects.append((tree_model, pos, 3.0, (0.2, 0.5, 0.2)))  # Verde

    # CASAS - 4 instancias
    # House: ~1.2 unidades de alto, escala 4.0 = ~5 unidades final
    house_positions = [
        (-30, 0.5, -25), (-28, 0.5, 25), (10, 0.5, -30), (12, 0.5, 28)
    ]
    for pos in house_positions:
        scene_objects.append((house_model, pos, 4.0, (0.8, 0.6, 0.4)))  # Marrón claro

    # COCHES - 4 instancias cerca de la carretera
    # Car: ~89 unidades de largo! escala 0.05 = ~4.5 unidades final
    car_positions = [
        (-10, 0.3, -3), (5, 0.3, 8), (-5, 0.3, 0), (0, 0.3, 5)
    ]
    for pos in car_positions:
        scene_objects.append((car_model, pos, 0.05, (0.8, 0.2, 0.2)))  # Rojo

    # MUÑECOS DE NIEVE - 3 instancias en zona de montaña
    # Snowman: ~0.6 unidades de alto, escala 8.0 = ~5 unidades final
    snowman_positions = [
        (25, 10, -25), (28, 8, -28), (30, 12, -30)
    ]
    for pos in snowman_positions:
        scene_objects.append((snowman_model, pos, 8.0, (0.95, 0.95, 1.0)))  # Blanco

    # MONOS - 3 instancias decorativas
    # Monkey: ~2 unidades de alto, escala 1.5 = ~3 unidades final
    monkey_positions = [
        (-35, 0.5, 0), (35, 0.5, -10), (-10, 0.5, 35)
    ]
    for pos in monkey_positions:
        scene_objects.append((monkey_model, pos, 1.5, (0.6, 0.4, 0.2)))  # Marrón

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
            base_noise = random.uniform(0.0, 0.3) # Pequeña variación natural
            y = base_noise
            
            road_z = get_road_center(x)
            dist_to_road = abs(z - road_z)
            road_width = 3.5
            
            is_road = False
            is_mount = False
            mount_height_factor = 0 # Qué tan alto estamos en la montaña (0 a 1)

            # Zona Carretera
            if dist_to_road < road_width:
                y = 0.0 
                is_road = True
            
            # Zona Terreno / Montaña
            else:
                dist_to_mount = math.sqrt((x - mount_x)**2 + (z - mount_z)**2)
                
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
                col_pasto = (0.1, 0.6, 0.1)       # Base (Igual al plano)
                col_bosque = (0.05, 0.25, 0.05)   # Verde Intenso Oscuro
                col_roca = (0.4, 0.35, 0.25)      # Café Roca
                col_nieve = (0.95, 0.95, 1.0)     # Blanco
                
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
            p2 = vertex_data[r+1][c]
            p3 = vertex_data[r+1][c+1]
            p4 = vertex_data[r][c+1]
            
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
    
    glColor3f(0.35, 0.2, 0.05) # Color Marrón Tierra

    glBegin(GL_QUADS)
    
    # Recorremos los 4 bordes del mapa
    
    # 1. Borde Norte (z mínima)
    for c in range(cols - 1):
        p1 = vertex_data[0][c]["coords"]
        p2 = vertex_data[0][c+1]["coords"]
        # Pared conecta la superficie con la profundidad
        glVertex3f(p1[0], p1[1], p1[2])           # Arriba Izq
        glVertex3f(p2[0], p2[1], p2[2])           # Arriba Der
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])     # Abajo Der
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])     # Abajo Izq

    # 2. Borde Sur (z máxima)
    for c in range(cols - 1):
        p1 = vertex_data[rows-1][c]["coords"]
        p2 = vertex_data[rows-1][c+1]["coords"]
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])

    # 3. Borde Oeste (x mínima)
    for r in range(rows - 1):
        p1 = vertex_data[r][0]["coords"]
        p2 = vertex_data[r+1][0]["coords"]
        glVertex3f(p2[0], p2[1], p2[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glVertex3f(p1[0], -DIRT_DEPTH, p1[2])
        glVertex3f(p2[0], -DIRT_DEPTH, p2[2])

    # 4. Borde Este (x máxima)
    for r in range(rows - 1):
        p1 = vertex_data[r][cols-1]["coords"]
        p2 = vertex_data[r+1][cols-1]["coords"]
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
                glVertex3f(x, 0.05, z) # Elevación mínima para evitar Z-fighting
                
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

# ==========================================
# BUCLE PRINCIPAL
# ==========================================
def main():
    pygame.init()
    display = (SCREEN_WIDTH, SCREEN_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(WINDOW_TITLE)

    # Configuración OpenGL
    glEnable(GL_DEPTH_TEST) # Habilitar buffer de profundidad
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0]/display[1]), 0.1, 1000.0)
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
    cam_x, cam_y, cam_z = 0, 40, 70 
    look_x, look_y, look_z = 0, 0, 0

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # TODO: Integrar aquí controles finales o MediaPipe
            keys = pygame.key.get_pressed()
            # Movimiento manual para pruebas de visualización
            if keys[pygame.K_LEFT]: cam_x -= 1
            if keys[pygame.K_RIGHT]: cam_x += 1
            if keys[pygame.K_UP]: cam_z -= 1 
            if keys[pygame.K_DOWN]: cam_z += 1
            if keys[pygame.K_w]: cam_y += 1 
            if keys[pygame.K_s]: cam_y -= 1

        # Renderizado
        glClearColor(0.53, 0.81, 0.92, 1) # Cielo azul claro
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

    pygame.quit()

if __name__ == "__main__":
    main()
