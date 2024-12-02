import numpy as np
from scipy.signal import butter, filtfilt

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
        bits : input Bipolar NRZ data stream (-1's and 1's) to modulate
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
