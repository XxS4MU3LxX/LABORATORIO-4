# LABORATORIO-4

***Luz Marina Valderrama-5600741***

***Shesly Nicole Colorado - 5600756***

***Samuel Esteban Fonseca Luna - 5600808***

```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann, hamming
from scipy.fftpack import fft
import re
import time
```

# Ruta del archivo
```
file_path = r"D:\Descargas\datos lab4.txt"  # ¡Cambia esta ruta!

with open(file_path, "r") as file:
    raw_data = file.read()

data_values = np.array([float(x) for x in re.findall(r"-?\d+\.\d+", raw_data)])

```
# Configuración de parámetros

```
fs = 10000  
window_size = 256  
overlap = window_size // 2  
ventana = "hanning"  
tiempo_espera = 0.5 

def calcular_frecuencia_media_mediana(frequencies, spectrum):

```
    Calcula la frecuencia media y mediana de un espectro de frecuencias.
  ```
  
    spectrum_energy = np.abs(spectrum)
    
    freq_media = np.sum(frequencies * spectrum_energy) / np.sum(spectrum_energy)
    
    energy_cumsum = np.cumsum(spectrum_energy)
    freq_mediana = np.interp(energy_cumsum[-1] / 2, energy_cumsum, frequencies)

    return freq_media, freq_mediana

def procesar_senal_emg(data, fs=10000, window_size=256, ventana="hanning"):

```
    Aplica ventana y calcula FFT en un segmento de la señal EMG.

    Retorna:
    - fft_resultado: Magnitud del espectro de frecuencias.
    - frecuencias: Array con las frecuencias correspondientes.
    
 ```
   if ventana == "hanning":
        window_func = hann(window_size)
    elif ventana == "hamming":
        window_func = hamming(window_size)
    else:
        raise ValueError("Solo se permiten 'hanning' o 'hamming' como ventana")

    frecuencias = np.fft.fftfreq(window_size, d=1/fs)[:window_size//2]
    fft_resultado = np.abs(fft(data * window_func))[:window_size//2]

    return fft_resultado, frecuencias

plt.ion()
fig, ax = plt.subplots()
frequencies = np.fft.fftfreq(window_size, d=1/fs)[:window_size//2]
line, = ax.plot(frequencies, np.zeros_like(frequencies))
ax.set_ylim(0, 10)
ax.set_xlim(0, fs/2)
ax.set_xlabel("Frecuencia (Hz)")
ax.set_ylabel("Magnitud")
ax.set_title("Espectro de la señal EMG en Tiempo Real")

print("Procesando datos del archivo...")

num_windows = (len(data_values) - window_size) // overlap

fft_promedio = np.zeros(len(frequencies))  

freqs_medias = []
freqs_medianas = []

for i in range(num_windows):
    start = i * overlap  
    segment = data_values[start:start + window_size]
    
    if len(segment) < window_size:
        break  

    fft_result, freqs = procesar_senal_emg(segment, fs, window_size, ventana=ventana)

    fft_promedio += fft_result

    freq_media, freq_mediana = calcular_frecuencia_media_mediana(freqs, fft_result)
    freqs_medias.append(freq_media)
    freqs_medianas.append(freq_mediana)

    print(f"Ventana {i+1}/{num_windows}: Freq Media = {freq_media:.2f} Hz, Freq Mediana = {freq_mediana:.2f} Hz")

    line.set_ydata(fft_result)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(tiempo_espera)  

fft_promedio /= num_windows  # Promedio de todas las ventanas

plt.figure(figsize=(8, 4))
plt.plot(frequencies, fft_promedio)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro Promedio de la Señal EMG")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(freqs_medias, label="Frecuencia Media")
plt.plot(freqs_medianas, label="Frecuencia Mediana")
plt.xlabel("Ventanas de tiempo")
plt.ylabel("Frecuencia (Hz)")
plt.title("Evolución de la Frecuencia Media y Mediana en el Tiempo")
plt.legend()
plt.show()

print("Procesamiento del archivo finalizado.")

plt.ioff()
plt.show(block=True)
```
