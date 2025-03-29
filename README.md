# LABORATORIO-4

***Luz Marina Valderrama-5600741***

***Shesly Nicole Colorado - 5600756***

***Samuel Esteban Fonseca Luna - 5600808***

Este laboratorio tiene como propósito procesar señales electromiográficas (EMG) previamente adquiridas en el brazo de un sujeto de prueba llamado Mrtin Alexander Torres Carvajas donde se pusieron los electrodos en la zona del músculo, generalmente sobre el área donde se encuentra la contracción del bíceps y uno en el area del hueso como elecrrodo de referencia. Realiza un análisis espectral en tiempo real, utilizando ventanas móviles sobre la señal, para estudiar cómo varía la energía en el dominio de la frecuencia, especialmente enfocado en la detección de fatiga muscular.

La fatiga se manifiesta típicamente por una disminución en la frecuencia mediana del espectro EMG con el tiempo.

![image](https://github.com/user-attachments/assets/5c6bf4db-7f4e-4ded-a4e9-fe9352ed42de)

# Importación de librerías

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal.windows import hann, hamming
    from scipy.fftpack import fft
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import median_abs_deviation
    from scipy.signal import welch
    import re
    import time

`numpy:`  cálculos numéricos eficientes con vectores/matrices.

`matplotlib.pyplot:` graficación de datos.

`hann, hamming:` ventanas para suavizar bordes antes de aplicar la FFT.

`fft:` algoritmo rápido para calcular transformada de Fourier.

`gaussian_filter1d:` suavizado tipo campana (Gauss).

`median_abs_deviation:` medida estadística robusta contra ruido.

`welch:` estima densidad espectral de potencia (PSD).

`re:` extrae números de texto con expresiones regulares.

`time:` para pausar la ejecución (simulación en tiempo real).

# Carga del archivo y extracción de la señal

    file_path = r"D:\Descargas\datos lab4.txt" 

    with open(file_path, "r") as file:
      raw_data = file.read()

    data_values = np.array([float(x) for x in re.findall(r"-?\d+\.\d+", raw_data)])

Se abre el archivo de texto y se lee como cadena.

Se extraen todos los números flotantes con regex.

Se convierte la lista de strings en un array de floats para análisis.

Resultado: `data_values` es tu señal EMG, una secuencia de voltajes registrados en el tiempo.

# Convolución para suavizar la señal

    def convolucionar_senal(data, kernel_size=5):
       kernel = np.ones(kernel_size) / kernel_size
       return np.convolve(data, kernel, mode="same")

Es un filtro promedio móvil. Crea un "kernel" (ventana) de tamaño 5 y lo desliza sobre la señal.

Reduce fluctuaciones rápidas (ruido).

Conserva la forma general de la señal.


    data_convolucionada = convolucionar_senal(data_values, kernel_size=5)

Se aplica ese suavizado a la señal EMG original.

# Visualización de señal original vs convolucionada

    plt.figure(figsize=(10, 4))
    plt.plot(data_values, label="Señal Original", alpha=0.7)
    plt.plot(data_convolucionada, label="Señal Convolucionada", linestyle="dashed", linewidth=1.5)
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")
    plt.title("Señal Completa - Antes del Procesamiento por Ventanas")
    plt.legend()
    plt.show()

Muestra ambas señales para comparar:

La original con mucho ruido.

La convolucionada más suave.

    total_muestras = len(data_values)
    tercio = total_muestras // 3

# Dividir señal en 3 segmentos

    inicio = data_values[:tercio]
    mitad = data_values[tercio:2*tercio]
    final = data_values[2*tercio:]

Permite comparar visualmente:

¿Cómo luce la señal al principio?

¿Cambia a la mitad o al final?

¿Se nota fatiga? ¿Aparecen oscilaciones distintas?

Esto refuerza la hipótesis de que la fatiga cambia el patrón de la señal EMG a lo largo del tiempo.

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(inicio, color="b")
    plt.title("Inicio de la Señal")
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")

    plt.subplot(1, 3, 2)
    plt.plot(mitad, color="g")
    plt.title("Mitad de la Señal")
    plt.xlabel("Tiempo (muestras)")

    plt.subplot(1, 3, 3)
    plt.plot(final, color="r")
    plt.title("Final de la Señal")
    plt.xlabel("Tiempo (muestras)")

    plt.tight_layout()
    plt.show()

![Imagen de WhatsApp 2025-03-28 a las 19 01 59_3cea404a](https://github.com/user-attachments/assets/0240134c-89ad-4639-8fe8-33f23bd13216)

# Suavizado adicional: filtro Gaussiano

    sigma = 20  
    señal_suavizada_gauss = gaussian_filter1d(data_values, sigma=sigma)

El suavizado gaussiano aplica una campana alrededor de cada punto:

Más suave que la convolución promedio.

Suprime aún más el ruido sin perder componentes importantes.

    plt.figure(figsize=(10, 5))
    plt.plot(data_values, label="Señal Original", alpha=0.5, color="gray")
    plt.plot(data_convolucionada, label="Convolución", alpha=0.7, color="b")
    plt.plot(señal_suavizada_gauss, label="Suavizado Gaussiano", color="r", linewidth=2)
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")
    plt.title("Señal Suavizada con Filtro Gaussiano")
    plt.legend()
    plt.show()
![Imagen de WhatsApp 2025-03-27 a las 22 38 25_ce88e509](https://github.com/user-attachments/assets/c46be012-ee00-4c1f-8eca-ad815a90db67)

# Parámetros de la FFT y PSD

    fs = 10000            # Frecuencia de muestreo (Hz)
    window_size = 256     # Tamaño de ventana para FFT
    overlap = 128         # 50% de solapamiento
    ventana = "hanning"   # Ventana usada
    tiempo_espera = 6     # Tiempo entre iteraciones (simula tiempo real)

Estos valores controlan cómo se analiza la señal:

Cuánto dura cada ventana.

Cuánto se superponen (mayor solapamiento = más suavidad temporal).

Qué función ventana se aplica para reducir artefactos en la FFT.

# Funciones para análisis espectral

    def calcular_frecuencia_media_mediana(frequencies, spectrum):
       spectrum_energy = np.abs(spectrum)
       freq_media = np.sum(frequencies * spectrum_energy) / np.sum(spectrum_energy)
       energy_cumsum = np.cumsum(spectrum_energy)
       freq_mediana = np.interp(energy_cumsum[-1] / 2, energy_cumsum, frequencies)
      return freq_media, freq_mediana

`freq_media:` promedio ponderado por energía.

`freq_mediana:` frecuencia donde se acumula el 50% de la energía.

Fatiga muscular se detecta cuando la frecuencia mediana disminuye.

# FFT y PSD en una función

     def procesar_senal_emg(data, fs=10000, window_size=256, ventana="hanning"):
        if ventana == "hanning":
          window_func = hann(window_size)
        elif ventana == "hamming":
          window_func = hamming(window_size)
        else:
              raise ValueError("Solo se permiten 'hanning' o 'hamming' como ventana")

    frecuencias = np.fft.fftfreq(window_size, d=1/fs)[:window_size//2]
    fft_resultado = np.abs(fft(data * window_func))[:window_size//2]
    
    freqs_psd, psd = welch(data, fs, window=window_func, nperseg=window_size)

    return fft_resultado, frecuencias, freqs_psd, psd

    plt.ion()

Esta función hace dos cosas:

Aplica Transformada de Fourier (FFT): vista puntual de energía por frecuencia.

Aplica Welch: estimación más estable de Densidad Espectral de Potencia (PSD).

# Gráfica interactiva en tiempo real

     fig, (ax_fft, ax_psd) = plt.subplots(2, 1, figsize=(8, 6))

`ax_fft:` gráfico para la FFT.

`ax_psd:` gráfico para la PSD en escala logarítmica (mejor para ver energía en frecuencias bajas).

Estas gráficas se actualizan en cada ventana del análisis.

     frequencies = np.fft.fftfreq(window_size, d=1/fs)[:window_size//2]
     line_fft, = ax_fft.plot(frequencies, np.zeros_like(frequencies), label="FFT")
     line_psd, = ax_psd.plot(frequencies, np.zeros_like(frequencies), label="PSD", color="r")

     ax_fft.set_ylim(0, 10)
     ax_fft.set_xlim(0, fs/2)
     ax_fft.set_xlabel("Frecuencia (Hz)")
     ax_fft.set_ylabel("Magnitud")
     ax_fft.set_title("Espectro de Frecuencia en Tiempo Real")
     ax_fft.legend()

     ax_psd.set_ylim(1e-8, 1e-2)  # Ajuste para mejor visualización
     ax_psd.set_xlim(0, fs/2)
     ax_psd.set_yscale("log")  # Escala logarítmica para la PSD
     ax_psd.set_xlabel("Frecuencia (Hz)")
     ax_psd.set_ylabel("Densidad Espectral de Potencia (V²/Hz)")
     ax_psd.set_title("Densidad Espectral de Potencia en Tiempo Real")
     ax_psd.legend()

     print("Procesando datos del archivo...")

     num_windows = (len(data_values) - window_size) // overlap

    fft_promedio = np.zeros(len(frequencies))  
    psd_promedio = np.zeros_like(frequencies)  

    freqs_medias = []
    freqs_medianas = []

![Imagen de WhatsApp 2025-03-23 a las 10 45 43_82c3e6fd](https://github.com/user-attachments/assets/1d937d3c-23b5-4591-90c4-894b32c0cbc4)
![Imagen de WhatsApp 2025-03-23 a las 10 45 44_c1c3a85a](https://github.com/user-attachments/assets/9fc57f7f-b7f1-412e-b376-0781971eca8f)
![Imagen de WhatsApp 2025-03-23 a las 10 45 44_c98c620f](https://github.com/user-attachments/assets/4a8736f7-45ab-4e62-9c5d-3cdae8775b04)
![Imagen de WhatsApp 2025-03-23 a las 10 45 44_46aabb8e](https://github.com/user-attachments/assets/377b4d2e-9fd8-47f9-9418-e9a840fd3079)
![Imagen de WhatsApp 2025-03-23 a las 10 45 44_3937efa9](https://github.com/user-attachments/assets/76ca7a59-0d07-4e9c-8df7-3e6f570c5a9e)
![Imagen de WhatsApp 2025-03-23 a las 10 45 45_051dbfd9](https://github.com/user-attachments/assets/ffbeb6be-b112-403e-bd08-1f7e8c9e6a94)
![Imagen de WhatsApp 2025-03-23 a las 10 45 45_9a5558ab](https://github.com/user-attachments/assets/e79feca1-bda0-4f61-948a-917382373d53)
![Imagen de WhatsApp 2025-03-23 a las 10 45 45_5372a8bc](https://github.com/user-attachments/assets/feec86b4-aaa9-47e4-8f6e-8eab81c4281f)
![Imagen de WhatsApp 2025-03-23 a las 10 45 45_e11c3d9e](https://github.com/user-attachments/assets/b02a9c2b-e1bb-4c97-bfa3-8d4fbdb574ba)
![Imagen de WhatsApp 2025-03-23 a las 10 45 45_54b4b5ea](https://github.com/user-attachments/assets/68bdbfaa-6ca0-4e04-aa79-ea4e59c4f569)
![Imagen de WhatsApp 2025-03-23 a las 10 45 46_2ffd11be](https://github.com/user-attachments/assets/f2362077-f0dc-44c6-b925-8eb08df86e9f)

# Análisis por ventanas deslizantes

     for i in range(num_windows):
        start = i * overlap  
         segment = data_values[start:start + window_size]
    
    if len(segment) < window_size:
        break  

    fft_result, freqs, freqs_psd, psd = procesar_senal_emg(segment, fs, window_size, ventana=ventana)

    fft_promedio += fft_result
    psd_promedio += psd[:len(frequencies)]  # Aseguramos el tamaño correcto

    freq_media, freq_mediana = calcular_frecuencia_media_mediana(freqs, fft_result)
    freqs_medias.append(freq_media)
    freqs_medianas.append(freq_mediana)

    print(f"Ventana {i+1}/{num_windows}: Freq Media = {freq_media:.2f} Hz, Freq Mediana = {freq_mediana:.2f} Hz")

    line_fft.set_ydata(fft_result)
    line_psd.set_ydata(psd[:len(frequencies)])

    ax_fft.relim()
    ax_fft.autoscale_view()
    ax_psd.relim()
    ax_psd.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(tiempo_espera)  

En cada iteración:

Toma un segmento de la señal con solapamiento.

Calcula FFT y PSD.

Suma cada resultado (para promedio final).

Calcula frecuencia media y mediana.

Actualiza las gráficas.

Imprime los valores de frecuencia.

Espera 6 segundos (simulación tiempo real).

Este bucle es clave para observar cómo evoluciona el espectro con el tiempo.

 # Promedios y gráficas finales
 
     fft_promedio /= num_windows
     psd_promedio /= num_windows

Se grafica:

Espectro FFT promedio de toda la señal.

PSD promedio (más suave, ideal para observación detallada).

Evolución temporal de la frecuencia media y mediana, lo más importante para detectar fatiga.

     plt.figure(figsize=(8, 4))
     plt.plot(frequencies, fft_promedio)
     plt.xlabel("Frecuencia (Hz)")
     plt.ylabel("Magnitud")
     plt.title("Espectro Promedio de la Señal EMG")
     plt.show()

     plt.figure(figsize=(8, 4))
     plt.semilogy(frequencies, psd_promedio, color='r')
     plt.xlabel("Frecuencia (Hz)")
     plt.ylabel("Densidad Espectral de Potencia (V²/Hz)")
     plt.title("Densidad Espectral de Potencia Promedio")
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

![image](https://github.com/user-attachments/assets/945e8db0-3479-4784-bf7e-97e25c26f9e0)

# Cargar los datos
     alpha = 0.05
     data = np.loadtxt('datos lab4.txt')

Usa `np.loadtxt()` para cargar el archivo `datos lab4.txt`, que debería contener los datos EMG que estamos analizando.

`data` es un array `numpy` que contiene todos los valores de la señal.

# Asignar el primer esfuerzo a los primeros 256 datos
     primer_esfuerzo = data[:256]  # Primeros 256 datos

`primer_esfuerzo` es un segmento de la señal que contiene los primeros 256 datos de `data`, lo que representa el primer esfuerzo muscular durante la contracción (probablemente cuando el músculo está descansando al principio de la medición).

# Asignar el último esfuerzo a los antepenúltimos 256 datos
     ultimo_esfuerzo = data[-512:-256]  # Últimos 256 datos (antepenúltimos)

`ultimo_esfuerzo` selecciona los últimos 256 datos, pero empieza en la posición antepenúltima (es decir, va desde el índice -512 hasta el -256). Esto corresponde a la parte final de la señal, posiblemente cuando el músculo está fatigado.

# Realizar la prueba t de Student con la hipótesis correcta (primer esfuerzo > último esfuerzo)
     result = stats.ttest_ind(primer_esfuerzo, ultimo_esfuerzo, alternative='greater')

La prueba t de Student se utiliza para determinar si hay una diferencia significativa entre las medias de dos grupos independientes. En este caso:

`primer_esfuerzo:` datos del primer esfuerzo.

`ultimo_esfuerzo:` datos del último esfuerzo.

El objetivo de la prueba es comprobar si el primer esfuerzo (cuando el músculo no está fatigado) tiene una media significativamente mayor que el último esfuerzo (cuando el músculo puede estar fatigado). La hipótesis alternativa se establece con `alternative='greater'`, lo que significa que estamos probando si el primer esfuerzo es mayor que el último.

# Mostrar el resultado de la prueba
     print(f"Estadístico t: {result.statistic[0]}")  # Solo tomar el primer valor del estadístico t
     print(f"Valor p: {result.pvalue[0]}")  # Solo tomar el primer valor de p

- El estadístico t indica la diferencia en las medias de los dos grupos, ponderada por la variabilidad dentro de los grupos. Un valor alto de t sugiere una gran diferencia entre los dos esfuerzos.
- El valor p es la probabilidad de obtener un resultado igual o más extremo que el observado si la hipótesis nula (que dice que no hay diferencia entre los esfuerzos) fuera cierta.
- Si el valor p es menor que el nivel de significancia α (0.05), rechazamos la hipótesis nula y concluimos que hay suficiente evidencia para afirmar que el primer esfuerzo es mayor que el último (es decir, que hay fatiga).

# Comparar el valor p con el nivel de significancia alpha (0.05)
     if result.pvalue[0] < alpha:  # Evaluar solo el primer valor de p
       print("Rechazamos la hipótesis nula. Hay evidencia suficiente para afirmar que el primer esfuerzo es mayor que el último.")
    else:
       print("No podemos rechazar la hipótesis nula. No hay evidencia suficiente para afirmar que el primer esfuerzo es mayor que el último.")
       
Si el valor p es menor que α = 0.05, significa que la diferencia entre el primer y el último esfuerzo es estadísticamente significativa. Esto nos llevaría a rechazar la hipótesis nula y concluir que efectivamente el primer esfuerzo es mayor que el último (lo que indicaría fatiga muscular).

Si el valor p es mayor que 0.05, no hay suficiente evidencia para rechazar la hipótesis nula, por lo que no se puede concluir que haya una diferencia significativa entre los dos esfuerzos.

     plt.ioff()
    plt.show(block=True)
    
`plt.ioff()` desactiva el modo interactivo en las gráficas, lo que asegura que las gráficas finales se muestren correctamente.

`plt.show(block=True)` mantiene la ventana de la gráfica abierta hasta que se cierre manualmente.

![image](https://github.com/user-attachments/assets/d80e9d2a-9ac3-4895-a623-300ba19c6dee)
.