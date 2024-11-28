import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sistema  # Importamos nuestras funciones definidas en sistema.py

# --- Parámetros Principales ---
Rb = 2400  # Velocidad de transmisión en bps
Fs = 19200  # Frecuencia de muestreo en Hz
Tb = 1 / Rb  # Duración de cada bit en segundos
Fc = Rb * 4 # Frecuencia portadora

# --- Centro de Control (Barra Lateral) ---
st.sidebar.title("Centro de Control")
num_bits = st.sidebar.number_input("Número de Bits a Simular", min_value=8, max_value=5000, value=20, step=8)
redundancy = st.sidebar.slider("Redundancia de Canal (repetición)", 1, 5, 3)
snr_db = st.sidebar.slider("Relación Señal a Ruido (SNR) [dB]", 0, 30, 15)

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

# --- Funciones de Graficación con Plotly ---
def plot_bits(bits, tb, title):
    """Genera una gráfica de bits en formato escalonado con Plotly."""
    tiempo = np.arange(len(bits)) * tb
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.repeat(tiempo, 2)[1:-1],
        y=np.repeat(bits, 2),
        mode="lines",
        line_shape="hv",
        name="Bits"
    ))
    # Agregar una línea horizontal en y=0
    fig.add_shape(
        go.layout.Shape(
            type="line", 
            x0=min(tiempo), x1=max(tiempo),  # Las coordenadas en X para cubrir todo el rango
            y0=0, y1=0,  # Las coordenadas en Y de la línea horizontal (y=0)
            line=dict(color="red", width=2, dash="dash"),  # Color y estilo de la línea
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo [s]",
        yaxis_title="Valor",
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    return fig

def plot_signal(signal, fs, title, num_samples=500):
    """Genera una gráfica de señal continua en el tiempo."""
    tiempo = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tiempo[:num_samples],
        y=signal[:num_samples],
        mode="lines",
        name="Señal"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo [s]",
        yaxis_title="Amplitud",
    )
    return fig

# --- Interfaz de Usuario ---
st.header("Sistema Digital Paso Banda - Interactivo")
plots_container = st.container()  # Contenedor dinámico para agregar gráficas

# 1. Generar Bits Aleatorios
if st.sidebar.button("Generar Bits"):
    st.session_state["bits"] = sistema.generate_binary_data(num_bits)
    with plots_container:
        st.plotly_chart(plot_bits(st.session_state["bits"], Tb, "Bits Aleatorios"))

# 2. Codificar Canal
if st.sidebar.button("Codificar Canal"):
    if st.session_state["bits"] is not None:
        st.session_state["encoded_bits"] = sistema.repetition_encoding(st.session_state["bits"], redundancy)
        with plots_container:
            st.plotly_chart(plot_bits(st.session_state["encoded_bits"], Tb , "Bits Codificados con Redundancia"))

# 3. Codificar Línea
if st.sidebar.button("Codificar Línea"):
    if st.session_state["encoded_bits"] is not None:
        st.session_state["line_code"] = sistema.polar_nrz_encoding(st.session_state["encoded_bits"])
        with plots_container:
            st.plotly_chart(plot_bits(st.session_state["line_code"], Tb , "Codificación de Línea NRZ"))

# 4. Modular QPSK
if st.sidebar.button("Modular QPSK"):
    if st.session_state["line_code"] is not None:
        st.session_state["qpsk_signal"] = sistema.qpsk_modulation(
            st.session_state["line_code"], Fc, 4*Fs, 2*int(Fs / Rb)
        )
        with plots_container:
            st.plotly_chart(plot_signal(st.session_state["qpsk_signal"], Fs, "Señal Modulada (QPSK)"))

# 5. Agregar Ruido
if st.sidebar.button("Agregar Ruido"):
    if st.session_state["qpsk_signal"] is not None:
        st.session_state["received_signal"] = sistema.awgn(st.session_state["qpsk_signal"], snr_db)
        with plots_container:
            st.plotly_chart(plot_signal(st.session_state["received_signal"], Fs, "Señal con Ruido (AWGN)"))

# 6. Diagrama de Ojo
if st.sidebar.button("Diagrama de Ojo"):
    if st.session_state["received_signal"] is not None:
        samples_per_symbol = 2*int(Fs / Rb)
        eye_signal = st.session_state["received_signal"][:10 * samples_per_symbol]
        fig_eye = go.Figure()
        for i in range(10):
            fig_eye.add_trace(go.Scatter(
                x=np.arange(samples_per_symbol) / Fs,
                y=eye_signal[i * samples_per_symbol:(i + 1) * samples_per_symbol],
                mode="lines",
                name=f"Ojo {i+1}"
            ))
        fig_eye.update_layout(
            title="Diagrama de Ojo",
            xaxis_title="Tiempo dentro del símbolo [s]",
            yaxis_title="Amplitud",
        )
        with plots_container:
            st.plotly_chart(fig_eye)

# 7. Demodular y Mostrar
if st.sidebar.button("Demodular y Mostrar"):
    if st.session_state["received_signal"] is not None:
        samples_per_symbol = 2*int(Fs / Rb)
        I_values, Q_values = sistema.qpsk_demodulation(
            st.session_state["received_signal"], Fc, Fs, samples_per_symbol
        )
        st.session_state["demodulated_bits"] = sistema.decision(I_values, Q_values)
        with plots_container:
            st.write("Bits Demodulados:")
            st.write(st.session_state["demodulated_bits"][:100])  # Mostrar los primeros 100 bits
