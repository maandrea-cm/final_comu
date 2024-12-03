import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.special import erfc
import endaq
import pandas as pd

# --- 1. Generación de bits aleatorios ---
def generate_binary_data(num_bits):
    """Genera una secuencia binaria aleatoria de 1s y 0s."""
    return np.random.randint(0, 2, num_bits)

# --- 2. Codificación de Canal ---
def repetition_encoding(bits, redundancy):
    """Codifica los bits repitiéndolos n veces."""
    return np.repeat(bits, redundancy)

def repetition_decoding(received_bits, n):
    """Decodifica bits repetidos usando mayoría."""
    reshaped = received_bits.reshape(-1, n)
    return (np.sum(reshaped, axis=1) > (n // 2)).astype(int)

# --- 3. Codificación de Línea Polar NRZ ---
def polar_nrz_encoding(bits):
    """Convierte 1s y 0s en amplitudes +/- sqrt(Eb)."""
    #amplitude = np.sqrt(1 / bit_duration)  # Asegura E_b = T_b
    return  (2 * bits - 1)

# --- 4. Modulación QPSK ---
def qpsk_modulation(bits, carrier_freq, fs, samples_per_symbol):
    """Modula una señal QPSK con amplitudes normalizadas."""
    num_symbols = len(bits) // 2  # Cada símbolo usa 2 bits
    t = np.linspace(0, num_symbols * samples_per_symbol / fs, num_symbols * samples_per_symbol, endpoint=False)

    # Separar bits en flujos I y Q
    I_bits = bits[0::2]  # Bits en fase (I)
    Q_bits = bits[1::2]  # Bits en cuadratura (Q)

    # NRZ: ya están en -1 y +1
    I_signal = np.repeat(I_bits, samples_per_symbol)
    Q_signal = np.repeat(Q_bits, samples_per_symbol)

    # Generar portadoras coherentes
    cos_wave = np.cos(2 * np.pi * carrier_freq * t)
    sin_wave = np.sin(2 * np.pi * carrier_freq * t)

    # Modulación QPSK
    qpsk_signal = I_signal * cos_wave - Q_signal * sin_wave
    return qpsk_signal

# --- 4.2 Modulación QPSK 2 ---
def qpsk_modulation2(bits, fc, OF):
    """
    Modulate an incoming binary stream using conventional QPSK
    Parameters:
        bits : input unipolar NRZ data stream (-1's and 1's) to modulate
        fc : carrier frequency in Hertz
        OF : oversampling factor - at least 4 is better
    """
    L = 2*OF # samples in each symbol (QPSK has 2 bits in each symbol)
    I = bits[0::2];Q = bits[1::2] #even and odd bit streams
    # even/odd streams at 1/2Tb baud
    
    from scipy.signal import upfirdn #NRZ encoder
    I = upfirdn(h=[1]*L, x=2*I-1, up = L)
    Q = upfirdn(h=[1]*L, x=2*Q-1, up = L)
        
    fs = OF*fc # sampling frequency 
    t=np.arange(0,len(I)/fs,1/fs)  #time base    
    
    I_t = I*np.cos(2*np.pi*fc*t); Q_t = -Q*np.sin(2*np.pi*fc*t)
    qpsk_signal = I_t + Q_t # QPSK modulated baseband signal 
    return qpsk_signal, t, I_t, Q_t

# --- 5. Canal AWGN ---
import numpy as np

def add_awgn(qpsk_signal, Eb_No_dB, time_vector, Rb):
    """
    Agrega ruido AWGN a una señal QPSK dado un Eb/No en decibeles.

    Parameters:
        qpsk_signal : numpy array
            Señal QPSK a la que se agregará el ruido.
        Eb_No_dB : float
            Relación Eb/No en decibeles.
        time_vector : numpy array
            Vector de tiempo correspondiente a la señal.
        Rb : float
            Tasa de bits (bps).
    
    Returns:
        signal_with_noise : numpy array
            Señal QPSK con el ruido agregado.
    """
    # Convertir Eb/No de dB a valor lineal
    Eb_No = 10**(Eb_No_dB / 10)
    
    # Calcular la duración de un bit (Tb)
    Tb = 1 / Rb
    
    # Calcular la energía por bit (Eb)
    signal_power = np.mean(qpsk_signal**2)  # Potencia promedio de la señal QPSK
    Eb = signal_power * Tb  # Energía por bit
    
    # Calcular la densidad espectral de potencia del ruido (N0)
    N0 = Eb / Eb_No

    noise_power = N0 * 2*Rb  # Potencia del ruido total
    
    # Generar ruido AWGN
    noise = np.random.normal(0, np.sqrt(noise_power), len(qpsk_signal))
    
    # Sumar el ruido a la señal QPSK
    signal_with_noise = qpsk_signal + noise
    
    return signal_with_noise


# --- 6. Filtro Pasabanda ---
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Aplica un filtro pasabanda a la señal."""
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return filtfilt(b, a, signal)

# --- 7. Demodulación QPSK ---
def qpsk_demodulation(received_signal, carrier_freq, fs, samples_per_symbol):
    """Demodula una señal QPSK en I y Q."""
    t = np.arange(len(received_signal)) / fs
    cos_wave = np.cos(2 * np.pi * carrier_freq * t)
    sin_wave = np.sin(2 * np.pi * carrier_freq * t)

    # Multiplicación por las portadoras
    I_signal = received_signal * cos_wave
    Q_signal = received_signal * -sin_wave

    # Integración numérica por símbolo
    num_symbols = len(received_signal) // samples_per_symbol
    I_values = np.zeros(num_symbols)
    Q_values = np.zeros(num_symbols)

    for i in range(num_symbols):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        I_values[i] = np.sum(I_signal[start:end]) / samples_per_symbol
        Q_values[i] = np.sum(Q_signal[start:end]) / samples_per_symbol

    return I_values, Q_values

# --- 8. Demodulacion 2 ---
def qpsk_demodulation(signal,fc,OF):

    fs = OF*fc # sampling frequency
    L = 2*OF # number of samples in 2Tb duration
    t=np.arange(0,len(signal)/fs,1/fs) # time base
    x=signal*np.cos(2*np.pi*fc*t) # Componente en fase (I)
    y=-signal*np.sin(2*np.pi*fc*t) # Q arm
    x = np.convolve(x,np.ones(L)) # Integrar sobre L muestras para I
    y = np.convolve(y,np.ones(L)) # integrate for L (Tsym=2*Tb) duration
    
    x = x[L-1::L] # Tomar valores de I al final de cada duración de símbolo
    y = y[L-1::L] # Q arm - sample at every symbol instant Tsym
    bits_r = np.zeros(2*len(x))
    bits_r[0::2] = (x>0) # even bits si x > 0 guarda 1 sino deja el 0
    bits_r[1::2] = (y>0) # odd bits

    return bits_r, x, y

# --- 8. Decisión ---
def decision(i_samples, q_samples):
    """Convierte las muestras I y Q en bits binarios."""
    bits_I = (i_samples > 0).astype(int)
    bits_Q = (q_samples > 0).astype(int)
    return np.ravel(np.column_stack((bits_I, bits_Q)))



#Graficas

#1. Graficar señal
def plot_signal(signal, time_vector, title):
    """
    Genera una gráfica de señal continua en el tiempo.

    Parameters:
        signal : numpy array
            Vector de la señal a graficar.
        time_vector : numpy array
            Vector de tiempo correspondiente a la señal.
        title : str
            Título de la gráfica.
    
    Returns:
        fig : plotly.graph_objects.Figure
            Figura interactiva de Plotly.
    """
    import plotly.graph_objects as go

    # Graficar toda la señal
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_vector,
        y=signal,
        mode="lines",
        name="Señal"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo [s]",
        yaxis_title="Amplitud",
    )
    return fig

# 2. PSDs
def calculate_psd(signal, fs, window='hann', nperseg=None):
    """
    Calcula la PSD de una señal utilizando el método de Welch y una ventana Hanning.

    Parameters:
        signal : numpy array
            Vector de la señal a analizar.
        fs : float
            Frecuencia de muestreo de la señal.
        window : str, optional
            Nombre de la ventana a utilizar (por defecto es 'hanning').
        nperseg : int, optional
            Longitud de la ventana a utilizar (por defecto es None, lo que calcula welch).
    
    Returns:
        freqs : numpy array
            Vector de frecuencias correspondiente a la PSD.
        psd : numpy array
            Vector de la PSD.
    """
    # Calcular la PSD utilizando el método de Welch
    freqs, psd = welch(signal, fs, window=window, nperseg=nperseg)
    return freqs, psd

def calculate_psd2(x, fs, fc, ax=None, color='b', label=None):
    """
    Calcula y grafica la PSD de una señal modulada utilizando el método de Welch.
    
    Parameters:
        x : numpy array
            Vector de señal para calcular la PSD.
        fs : float
            Frecuencia de muestreo.
        fc : float
            Frecuencia portadora central de la señal.
        ax : matplotlib.axes.Axes, optional
            Objeto Axes de Matplotlib donde se graficará.
        color : str, optional
            Color para la gráfica (por defecto, 'b').
        label : str, optional
            Etiqueta para la gráfica.
    
    Returns:
        f_rel : numpy array
            Frecuencias relativas centradas en fc.
        Pxx_norm : numpy array
            PSD normalizada respecto al valor en fc.
    """
    from scipy.signal import welch
    import numpy as np

    # Ajustar parámetros de Welch
    na = 16  # Factor de promediado
    nperseg = len(x) // na  # Tamaño de la ventana para Welch

    # Calcular PSD usando Welch con ventana Hann
    f, Pxx = welch(x, fs, window='hann', nperseg=nperseg, noverlap=0)

    # Filtrar frecuencias alrededor de fc (de fc a 4*fc)
    indices = (f >= fc) & (f < 4 * fc)
    f_rel = f[indices] - fc  # Frecuencias relativas centradas en fc
    Pxx_norm = Pxx[indices] / Pxx[indices][0]  # Normalizar PSD respecto a valor en fc

    # Graficar si se proporciona un objeto Axes
    if ax is not None:
        ax.plot(f_rel, 10 * np.log10(Pxx_norm), color=color, label=label)
        ax.set_xlabel("Frecuencia Relativa (Hz)")
        ax.set_ylabel("PSD Normalizada (dB)")
        ax.grid(True)
        if label:
            ax.legend()

    return f_rel, Pxx_norm

# 2.3 PSD usando endaq
import pandas as pd
import endaq.calc as calc

def calculate_psd_with_dataframe(signal, time_vector, signal_name, bin_width=1):
    """
    Calcula la PSD de una señal utilizando endaq.calc y crea un DataFrame con los valores de tiempo y señal.
    
    Parameters:
        signal : numpy array
            Vector de la señal a analizar.
        time_vector : numpy array
            Vector de tiempo correspondiente a la señal.
        signal_name : str
            Nombre de la señal (para el DataFrame).
        bin_width : float, optional
            Ancho de bin para el cálculo de la PSD (por defecto es 1).
    
    Returns:
        df : pandas.DataFrame
            DataFrame con las columnas "Time (s)" y "Signal".
        psd_freqs : numpy array
            Frecuencias calculadas en la PSD.
        psd_values : numpy array
            Valores de la PSD.
    """
    # Crear DataFrame con tiempo y señal
    df = pd.DataFrame(index=time_vector)
    df[signal_name] = signal

    # Calcular PSD usando endaq.calc.psd.welch
    psd = calc.psd.welch(df, bin_width=bin_width)
    
    return psd

def calculate_ber(eb_no_db):
    """Calculates the Bit Error Rate (BER) for given Eb/No values."""
    eb_no_linear = 10 ** (eb_no_db / 10)  # Convert to linear scale
    sqrt_eb_no = np.sqrt(eb_no_linear)
    pe = 0.5 * erfc(sqrt_eb_no)  # Pe = Q(sqrt(2*Eb/No))
    return pe
