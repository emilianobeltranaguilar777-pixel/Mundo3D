# Proyecto 3: Ciudad Entorno 3D

## Descripción del Proyecto
Este proyecto consiste en la implementación de una simulación de un entorno tridimensional interactivo utilizando **Python** y **OpenGL**. El objetivo es recrear una "ciudad" o mundo virtual que contenga múltiples objetos móviles y animados, controlados mediante una interfaz natural de usuario (NUI) basada en visión por computadora.

El sistema permite la exploración del entorno modificando la cámara (`gluLookAt`) a través de la detección de movimientos de manos (Landmarks) procesados con **MediaPipe**.

## Objetivos
1. **Entorno 3D:** Implementar un suelo/terreno y un sistema de coordenadas base.
2. **Población:** Integrar al menos **20 objetos móviles** (basados en primitivas, casas, snowmans, etc.).
3. **Animación:** Los objetos deben contar con transformaciones geométricas (traslación, rotación, escalado).
4. **Interacción:** Control de la cámara y visualización mediante el rastreo de manos (Hand Tracking).

## Tecnologías Utilizadas
* **Lenguaje:** Python 3.x
* **Gráficos:** PyOpenGL & PyOpenGL_accelerate
* **Ventana e Input:** Pygame
* **Visión por Computadora:** MediaPipe (Google)
* **Matemáticas:** NumPy

## Equipo de Desarrollo
* **Miguel Ángel Torres**
* **Jesús Reyes Moreno**
* **José Emiliano Beltrán Aguilar**
* **Axel Alberto Dueñas Cantero**

## Datos Académicos
* **Materia:** Graficación
* **Profesor:** Eduardo Alcaraz

---

## Poblado del Entorno 3D (Objetos)

El terreno 3D es generado de forma independiente.
Este módulo se encarga únicamente de **cargar, instanciar y posicionar objetos 3D** sobre dicho terreno.

### Modelos 3D
Los modelos se encuentran en la carpeta:
```
MODELS/
```

Formato utilizado:
- `.obj` (Wavefront)
- `.mtl` es opcional y actualmente ignorado para simplicidad

Los modelos se cargan una sola vez y se reutilizan mediante instancias.

---

### Loader de OBJ
Se implementó un loader simple que:
- Parsea vértices (`v`) y caras (`f`)
- Maneja índices base-1 del formato OBJ
- Ignora normales y materiales para estabilidad
- Utiliza cache para evitar recargar el mismo modelo

Esto permite agregar nuevos modelos OBJ sin modificar el pipeline.

---

### Instanciación de Objetos
Los objetos se instancian mediante una tupla que contiene:
- Modelo (datos de vértices y caras)
- Posición (x, y, z)
- Escala
- Color RGB

Se utilizan **offsets en Y por tipo de objeto** para evitar que queden enterrados en el terreno:

```python
Y_OFFSETS = {
    "tree": 1.5,      # Árboles necesitan offset alto (origen en centro)
    "house": 2.5,     # Casas con origen bajo
    "car": 0.8,       # Coches justo sobre el asfalto
    "snowman": 2.0,   # Muñecos de nieve (se suma a altura de montaña)
    "monkey": 1.5,    # Monos decorativos
}
```

---

### Cantidad de Objetos

Actualmente el entorno contiene **48 instancias**, incluyendo:

| Tipo | Cantidad |
|------|----------|
| Árboles | 20 |
| Casas | 8 |
| Vehículos | 8 |
| Muñecos de nieve | 6 |
| Monos (decorativos) | 6 |
| **TOTAL** | **48** |

Esto cumple y supera el requisito mínimo de 20 objetos del proyecto.
