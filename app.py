import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sistema  # Importamos nuestras funciones definidas en sistema.py

# --- Parámetros Principales ---
Rb = 2400  # Velocidad de transmisión en bps
Fs = 19200  # Frecuencia de muestreo en Hz
Tb = 1 / Rb  # Duración de cada bit en segundos
#num_bits = 100  # Número de bits a transmitir (ajustable)

# --- Centro de Control (Barra Lateral) ---
st.sidebar.title("Centro de Control")
num_bits = st.sidebar.number_input("Número de Bits a Simular", min_value=8, max_value=5000, value=50, step=8)
redundancy = st.sidebar.slider("Redundancia de Canal (repetición)", 1, 5, 3)
snr_db = st.sidebar.slider("Relación Señal a Ruido (SNR) [dB]", 0, 30, 10)

# --- Estado Inicial ---
if "bits" not in st.session_state:
    st.session_state["bits"] = None
if "encoded_bits" not in st.session_state:
    st.session_state["encoded_bits"] = None
if "line_code" not in st.session_state:
    st.session_state["line_code"] = None
if "qpsk_signal" not in st.session_state:
    st.session_state["qpsk_signal"] = None
if "received_signal" not in st.session_state:
    st.session_state["received_signal"] = None
if "demodulated_bits" not in st.session_state:
    st.session_state["demodulated_bits"] = None

# --- Funciones Adaptadas ---
# 1. Generación de Bits Aleatorios
import plotly.graph_objects as go

def generar_bits():
    bits = sistema.generate_binary_data(num_bits)
    tiempo = np.arange(len(bits)) * Tb

    # Gráfica interactiva con escalones (rectangular)
    st.subheader("Bits Aleatorios (Gráfica Completa)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.repeat(tiempo, 2)[1:-1],  # Duplicar puntos para escalones
        y=np.repeat(bits, 2),          # Duplicar valores para escalones
        mode="lines",
        line_shape="hv",               # Escalón horizontal-vertical
        name="Bits"
    ))
    fig.update_layout(
        title="Bits Aleatorios",
        xaxis_title="Tiempo [s]",
        yaxis_title="Valor",
        xaxis=dict(rangeslider=dict(visible=True)),  # Habilitar barra deslizante
    )
    st.plotly_chart(fig)

    return bits

# 2. Codificación de Canal
def codificar_canal(bits, redundancy):
    encoded_bits = sistema.repetition_encoding(bits, redundancy)
    tiempo = np.arange(len(encoded_bits)) * Tb / redundancy
    fig, ax = plt.subplots()
    ax.step(tiempo[:100 * redundancy], encoded_bits[:100 * redundancy], where="post")
    ax.set_title("Bits Codificados con Redundancia")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Valor")
    st.pyplot(fig)
    return encoded_bits

# 3. Codificación de Línea
def codificar_linea(encoded_bits):
    line_code = sistema.polar_nrz_encoding(encoded_bits)
    tiempo = np.arange(len(line_code)) * Tb / redundancy
    fig, ax = plt.subplots()
    ax.step(tiempo[:100 * redundancy], line_code[:100 * redundancy], where="post")
    ax.set_title("Codificación de Línea NRZ")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    st.pyplot(fig)
    return line_code

# 4. Modulación QPSK
def modular_qpsk_coherente(line_code):
    qpsk_signal = sistema.qpsk_modulation(line_code, Rb * 4, Fs, 2*int(Fs / Rb))
    tiempo = np.arange(len(qpsk_signal)) / Fs
    fig, ax = plt.subplots()
    ax.plot(tiempo[:500], qpsk_signal[:500])  # Graficamos las primeras 500 muestras
    ax.set_title("Señal Modulada (QPSK Coherente)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    st.pyplot(fig)
    return qpsk_signal

# 5. Agregar Ruido AWGN
def agregar_ruido(qpsk_signal, snr_db):
    received_signal = sistema.awgn(qpsk_signal, snr_db)
    tiempo = np.arange(len(received_signal)) / Fs
    fig, ax = plt.subplots()
    ax.plot(tiempo[:500], received_signal[:500])  # Graficamos las primeras 500 muestras
    ax.set_title("Señal con Ruido (AWGN)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    st.pyplot(fig)
    return received_signal

# 6. Diagrama de Ojo
def diagrama_ojo(received_signal):
    samples_per_symbol = int(Fs / Rb)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(10):  # Dibujar 10 ojos
        ax.plot(received_signal[i * samples_per_symbol:(i + 1) * samples_per_symbol], color="blue")
    ax.set_title("Diagrama de Ojo")
    ax.set_xlabel("Muestras por símbolo")
    ax.set_ylabel("Amplitud")
    st.pyplot(fig)

# --- Interfaz de Usuario ---
st.header("Sistema Digital Paso Banda - Interactivo")

if st.sidebar.button("Generar Bits"):
    st.session_state["bits"] = generar_bits()
if st.sidebar.button("Codificar Canal"):
    if st.session_state["bits"] is not None:
        st.session_state["encoded_bits"] = codificar_canal(st.session_state["bits"], redundancy)
if st.sidebar.button("Codificar Línea"):
    if st.session_state["encoded_bits"] is not None:
        st.session_state["line_code"] = codificar_linea(st.session_state["encoded_bits"])
if st.sidebar.button("Modular QPSK"):
    if st.session_state["line_code"] is not None:
        st.session_state["qpsk_signal"] = modular_qpsk_coherente(st.session_state["line_code"])
if st.sidebar.button("Agregar Ruido"):
    if st.session_state["qpsk_signal"] is not None:
        st.session_state["received_signal"] = agregar_ruido(st.session_state["qpsk_signal"], snr_db)
if st.sidebar.button("Diagrama de Ojo"):
    if st.session_state["received_signal"] is not None:
        diagrama_ojo(st.session_state["received_signal"])
if st.sidebar.button("Demodular y Mostrar"):
    if st.session_state["received_signal"] is not None:
        samples_per_symbol = int(Fs / Rb)
        I_values, Q_values = sistema.qpsk_demodulation(st.session_state["received_signal"], Rb * 4, Fs, samples_per_symbol)
        received_bits = sistema.decision(I_values, Q_values)
        st.write("Bits Demodulados:")
        st.write(received_bits[:100])  # Mostramos los primeros 100 bits
