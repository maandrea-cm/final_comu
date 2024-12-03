# Sistema Digital Paso Banda - Proyecto Interactivo

Este proyecto implementa un sistema de comunicaciones digitales paso banda, utilizando **modulación QPSK**, ruido AWGN, y herramientas interactivas para la simulación y visualización de señales en dominios del tiempo y frecuencia. La interfaz gráfica es desarrollada con **Streamlit** y las operaciones principales están en **Python**.

---

## **Estructura del Proyecto**

### **Archivos Principales**

1. **`app.py`**: 
   - La interfaz gráfica interactiva donde el usuario puede:
     - Generar y visualizar secuencias de bits.
     - Aplicar codificación de canal y codificación de línea.
     - Modular señales QPSK.
     - Agregar ruido AWGN y visualizar sus efectos.
     - Demodular señales y comparar los bits enviados y recibidos.
     - Graficar diagramas de ojo y constelaciones.

   - **Tablas interactivas y gráficas:** Utiliza **Plotly** para mostrar gráficas interactivas como el diagrama de ojo, la constelación y las señales en tiempo.

2. **`sistema.py`**: 
   - Contiene las funciones que implementan las operaciones DSP del sistema:
     - **Generación de bits aleatorios**: Simula una fuente binaria.
     - **Codificación de canal y línea**: Agrega redundancia y convierte bits a niveles de amplitud.
     - **Modulación QPSK**: Implementa la modulación en baseband y en paso banda.
     - **Ruido AWGN**: Agrega ruido a la señal basado en una relación \(E_b/N_0\).
     - **Demodulación**: Recupera las componentes I/Q y decodifica los bits.
     - **Gráficas PSD y señales**: Métodos para analizar y visualizar las señales.

---

## **Instalación**

1. **Clona el repositorio**
   ```bash
   git clone https://github.com/tu_usuario/final_comu.git
   cd final_comu

2. **Crea un entorno virtual (opcional pero recomendado)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En macOS/Linux
    .\venv\Scripts\activate   # En Windows

3. **Instala las dependencias**
    ```bash
    pip install -r requirements.txt

4. **Ejecuta la aplicación**
    ```bash
    streamlit run app.py


## **Uso del Sistema**

**Funciones Interactivas en app.py**

1. **Barra Lateral:**
    - Configura los parámetros:
        - Número de bits a simular.
        - Redundancia en la codificación.
        - Relación Señal a Ruido (SNR) en decibeles.
2. **Generación de Bits:**
    - Visualiza la secuencia binaria generada.
3. **Codificación:**
    - Aplica codificación de canal y línea (NRZ bipolar).
4. **Modulación QPSK:**
    - Genera la señal QPSK y muestra:
    - Componente en fase (𝐼𝑡).
    - Componente en cuadratura (𝑄𝑡).
    - Señal combinada.
5. **Agregar Ruido:**
    - Aplica ruido AWGN basado en el 𝐸𝑏/𝑁0 configurado.
    - Visualiza la señal con ruido.
6. **Demodulación:**
    - Recupera los bits originales y los compara con los enviados.
7. **Análisis de Señales:**
    - Muestra diagramas de ojo y constelaciones para evaluar la calidad.
    
**Funciones Clave en sistema.py**
- Modulación y Demodulación QPSK:
  - Implementa el proceso de modulación y recuperación de señales usando correladores y filtros.
- Ruido AWGN:
  - Calcula la potencia del ruido basada en 𝐸𝑏/𝑁0 y agrega ruido a la señal.
- Gráficas y PSD:
  - Genera visualizaciones para señales en tiempo y frecuencia.

## **Requerimientos del Sistema**
- Python 3.8+
- Bibliotecas principales:
    - numpy
    - scipy
    - streamlit
    - plotly

## **Contribución**
1. Crea una rama de trabajo
    ```bash
    git checkout -b