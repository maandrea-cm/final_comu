import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sistema  # Importamos nuestras funciones definidas en sistema.py



# --- Parámetros Principales ---
Rb = 2400  # Velocidad de transmisión en bps
Fs = 19200  # Frecuencia de muestreo en Hz
Tb = 1 / Rb  # Duración de cada bit en segundos
Fc = Rb * 32 # Frecuencia portadora
OF = 16 # Sobremuestreo para modulacion y demod

# --- Centro de Control (Barra Lateral) ---
st.sidebar.title("Centro de Control")
num_bits = st.sidebar.number_input("Número de Bits a Simular", min_value=8, max_value=50000, value=20, step=8)
redundancy = st.sidebar.slider("Redundancia de Canal (repetición)", 1, 5, 3)
snr_db = st.sidebar.slider("Relación Señal a Ruido (SNR) [dB]", 0, 25, 15)

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
        mode="lines+markers",
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


# --- Interfaz de Usuario ---
st.header("Sistema Digital Paso Banda - Interactivo")
tab1, tab2, tab3 = st.tabs(["Dominio del tiempo", "PSD", "BER"])

with tab1:
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
                st.plotly_chart(plot_bits(st.session_state["line_code"], Tb , "Codificación de Línea BNRZ"))

    # 4. Modular QPSK
    if st.sidebar.button("Modular QPSK"):
        if st.session_state["line_code"] is not None:
            st.session_state["qpsk_signal"], st.session_state["time"], I_t, Q_t= sistema.qpsk_modulation2(
                st.session_state["encoded_bits"], Fc, OF)
            with plots_container:
                fig = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,  # Compartir eje x entre subplots
                        vertical_spacing=0.05,  # Espaciado vertical entre subplots
                        subplot_titles=("Señal QPSK", "Componente en Fase (I_t)", "Componente en Cuadratura (Q_t)")
                    )

                    # Añadir la señal QPSK
                fig.add_trace(
                    go.Scatter(x=st.session_state["time"], y=st.session_state["qpsk_signal"], mode="lines", name="QPSK"),
                    row=1, col=1
                )

                # Añadir la componente en fase (I_t)
                fig.add_trace(
                    go.Scatter(x=st.session_state["time"], y=I_t, mode="lines", name="Componente en Fase (I_t)", line=dict(color='red')),
                    row=2, col=1
                )

                # Añadir la componente en cuadratura (Q_t)
                fig.add_trace(
                    go.Scatter(x=st.session_state["time"], y=Q_t, mode="lines", name="Componente en Cuadratura (Q_t)", line=dict(color='rgb(0, 254, 129)')),
                    row=3, col=1
                )

                # Configurar el diseño general
                fig.update_layout(
                    title="Componentes de la Señal QPSK",
                    height=800,  # Altura total del gráfico
                    xaxis3=dict(  # Configurar el range slider en el último subplot
                        rangeslider=dict(visible=True),
                        title="Tiempo (s)"
                    ),
                    yaxis1=dict(title="Amplitud"),
                    yaxis2=dict(title="Amplitud"),
                    yaxis3=dict(title="Amplitud")
                )

                st.plotly_chart(fig)            

                
    # 5. Agregar Ruido
    if st.sidebar.button("Agregar Ruido"):
        if st.session_state["qpsk_signal"] is not None:
            st.session_state["received_signal"] = sistema.add_awgn(st.session_state["qpsk_signal"], snr_db, st.session_state["time"], Rb)
            max_s = np.max(np.abs(st.session_state["received_signal"]))
            with plots_container:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,  # Compartir el eje X
                    vertical_spacing=0.1,  # Espaciado vertical entre subplots
                    subplot_titles=("Señal QPSK Original", "Señal QPSK con Ruido AWGN")
                )

                # Añadir la señal QPSK original al primer subplot
                fig.add_trace(
                    go.Scatter(x=st.session_state["time"], y=st.session_state["qpsk_signal"], mode="lines", name="QPSK Original"),
                    row=1, col=1
                )

                # Añadir la señal QPSK con ruido al segundo subplot
                fig.add_trace(
                    go.Scatter(x=st.session_state["time"], y=st.session_state["received_signal"], mode="lines", name="QPSK con Ruido", line=dict(color='red')),
                    row=2, col=1
                )

                # Configurar diseño general
                fig.update_layout(
                    title="Señal QPSK Original y con Ruido AWGN",
                    height=800,  # Altura total del gráfico
                    xaxis2=dict(  # Range slider en el eje X del subplot inferior
                        rangeslider=dict(visible=True),
                        title="Tiempo (s)"
                    ),
                    yaxis1=dict(title="Amplitud", range=[-max_s, max_s]),
                    yaxis2=dict(title="Amplitud", range=[-max_s, max_s])
                )
            
                st.plotly_chart(fig)

    # 6. Diagrama de Ojo
    if st.sidebar.button("Diagrama de Ojo"):
        if st.session_state["received_signal"] is not None:
            samples_per_symbol = 2*int(Fs / Rb)
            eye_signal = st.session_state["received_signal"][:10 * samples_per_symbol]
            fig_eye = go.Figure()
            for i in range(20):
                fig_eye.add_trace(go.Scatter(
                    x=np.arange(samples_per_symbol) / Fs,
                    y=eye_signal[i * samples_per_symbol:(i + 1) * samples_per_symbol],
                    mode="lines",
                    name=""
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
            st.session_state["demodulated_bits"], I_values, Q_values = sistema.qpsk_demodulation(st.session_state["received_signal"], Fc, OF)
            st.session_state["demodulated_bits"] = st.session_state["demodulated_bits"][::1]
            with plots_container:
                # Create a subplot with 2 rows and 1 column
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,  # Share the x-axis between subplots
                    vertical_spacing=0.1,  # Vertical spacing between subplots
                    subplot_titles=("Bits Enviados", "Bits Recibidos")
                )

                # Add the transmitted bits to the first subplot
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(st.session_state["encoded_bits"])) * Tb,
                        y=st.session_state["encoded_bits"],
                        mode="lines+markers",
                        line_shape="hv",
                        name="Bits Enviados"
                    ),
                    row=1, col=1
                )

                # Add the received bits to the second subplot
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(st.session_state["demodulated_bits"])) * Tb,
                        y=st.session_state["demodulated_bits"],
                        mode="lines+markers",
                        line_shape="hv",
                        name="Bits Recibidos"
                    ),
                    row=2, col=1
                )

                # Configure the layout
                fig.update_layout(
                    title="Comparación de Bits Enviados y Recibidos",
                    height=800,  # Total height of the plot
                    xaxis2=dict(  # Range slider on the x-axis of the bottom subplot
                        rangeslider=dict(visible=True),
                        title="Tiempo [s]"
                    ),
                    yaxis1=dict(title="Valor"),
                    yaxis2=dict(title="Valor")
                )

                st.plotly_chart(fig)

    # 8. Signal Constellation Plot
    if st.sidebar.button("Signal Constellation Plot"):
        if st.session_state["received_signal"] is not None:
            st.session_state["demodulated_bits"], I_values, Q_values = sistema.qpsk_demodulation(st.session_state["received_signal"], Fc, OF)

            with plots_container:
                # Create a scatter plot for the signal constellation
                fig = go.Figure(
                    data=go.Scatter(
                        x=I_values,
                        y=Q_values,
                        mode="markers",
                        marker=dict(size=8, color="blue"),
                        name="Signal Constellation"
                    )
                )

                # Add x and y axis labels with LaTeX formatting
                fig.update_layout(
                    xaxis_title="In-phase (I)",
                    yaxis_title="Quadrature-phase (Q)",
                )

                # Add reference points in each quadrant
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=0, x1=0,  # Horizontal line at x=0
                        y0=min(Q_values), y1=max(Q_values),  # Vertical range from min(Q_values) to max(Q_values)
                        line=dict(color="white", dash="dot"),  # Dotted line style
                    )
                )
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=min(I_values), x1=max(I_values),  # Vertical line at y=0
                        y0=0, y1=0,  # Horizontal range from min(I_values) to max(I_values)
                        line=dict(color="white", dash="dot"),  # Dotted line style
                    )
                )
                # Configure the layout
                fig.update_layout(
                    title="Signal Constellation Plot",
                    height=800,  # Total height of the plot
                )

                st.plotly_chart(fig)


with tab2:
    selected_signal = st.multiselect(
        "Señal para calcular PSD",
        options=["bits", "encoded_bits", "line_code", "qpsk_signal", "received_signal", "demodulated_bits"],
        default=["qpsk_signal"]
    )
    
    if st.button("Mostrar PSD"):
        for signal_name in selected_signal:
            if signal_name in st.session_state:
                psd = sistema.calculate_psd_with_dataframe(st.session_state[signal_name], st.session_state["time"], signal_name)
                fig = px.line(psd).update_layout(
                    yaxis_title_text="PSD [dB/Hz]", 
                    xaxis_title_text="Frecuencia [Hz]", 
                    legend_title_text=f"PSD de {signal_name}",
                    xaxis=dict(rangeslider=dict(visible=True),))
                st.plotly_chart(fig)

with tab3:
    """Creates a tab in Streamlit to display the BER curve."""
    st.title("BER Curve")

    # Define Eb/No range in dB
    eb_no_db = np.linspace(0, 15, 100)

    # Calculate BER for given Eb/No values
    pe = sistema.calculate_ber(eb_no_db)

    # Graficamos BER contra Eb/No en dB
    plt.figure(figsize=(10, 6))
    plt.semilogy(eb_no_db, pe, label="Pe vs Eb/No (dB)", color='b')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.title("BER")
    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Pe")
    plt.legend()
    
    st.pyplot(plt)
