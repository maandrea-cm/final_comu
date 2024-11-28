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

# --- 5. Canal AWGN ---
def awgn(signal, snr_db):
    """Agrega ruido gaussiano al canal."""
    snr_linear = 10**(snr_db / 10)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

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

# --- 8. Decisión ---
def decision(i_samples, q_samples):
    """Convierte las muestras I y Q en bits binarios."""
    bits_I = (i_samples > 0).astype(int)
    bits_Q = (q_samples > 0).astype(int)
    return np.ravel(np.column_stack((bits_I, bits_Q)))
