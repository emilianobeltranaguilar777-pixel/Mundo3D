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
