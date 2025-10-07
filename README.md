# Laboratorio-3- Procesamiento Digital de Señales

# ANÁLISIS ESPECTRAL DE VOZ
En esta práctica se trabajaron conceptos fundamentales del procesamiento digital de señales aplicados al análisis de la voz humana. Primero, se adquirieron grabaciones de voz de hombres y mujeres, asegurando condiciones similares de muestreo y entorno. Luego, se aplicó la Transformada de Fourier para representar las señales en el dominio de la frecuencia y analizar su espectro, a partir de esto, se calcularon parámetros característicos como la frecuencia fundamental, el brillo, la intensidad, el jitter y el shimmer. Finalmente, se compararon los resultados entre voces masculinas y femeninas, identificando sus principales diferencias y comprendiendo la importancia de estas medidas en la evaluación y caracterización de la voz.
# Parte A
En primer lugar se realizaron las grabaciones de seis personas, tres hombres y tres mujeres, pronunciando la misma frase y cada archivo se guardó en formato .wav con nombres identificadores. Luego, las señales fueron importadas a Python, donde se graficaron en el dominio del tiempo para observar su forma de onda. Posteriormente, se aplicó la Transformada de Fourier a cada una de las grabaciones, obteniendo el espectro de magnitud y permitiendo identificar las componentes principales de frecuencia. Finalmente, se calcularon parámetros como la frecuencia fundamental, frecuencia media, brillo e intensidad, los cuales sirvieron como base para el análisis comparativo entre voces masculinas y femeninas.

**Graficado de la señales**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# mujeres
fs1,signal1=  wavfile.read('/Mujer-1.wav')
fs2,signal2=  wavfile.read('/Mujer-2.wav')
fs3,signal3=  wavfile.read('/Mujer-3.wav')

# Hombres
fs4,signal4=  wavfile.read('/Hombre-1.wav')
fs5,signal5= wavfile.read('/Hombre-2.wav')
fs6,signal6= wavfile.read('/Hombre-3.wav')


duracion = len(signal1)/fs1

duracion = len(signal2)/fs1

duracion = len(signal3)/fs1



duracion = len(signal4)/fs1

duracion = len(signal5)/fs1

duracion = len(signal6)/fs1


Tiempo1 = np.arange(len(signal1)) / fs1
Tiempo2 = np.arange(len(signal2)) / fs2
Tiempo3 = np.arange(len(signal3)) / fs3
Tiempo4 = np.arange(len(signal4)) / fs4
Tiempo5 = np.arange(len(signal5)) / fs5
Tiempo6 = np.arange(len(signal6)) / fs6
fig, axs = plt.subplots(6,1, figsize=(14,10),sharex=False)

print(len(Tiempo3),len(signal3))

# graficas
axs[0].plot(Tiempo1, signal1)
axs[0].set_title("Mujer 1")
axs[0].set_ylabel('Bits')
axs[0].set_xlabel('Tiempo (s)')

axs[1].plot(Tiempo2, signal2)
axs[1].set_title("Mujer 2")
axs[1].set_ylabel('Bits')
axs[1].set_xlabel('Tiempo (s)')

axs[2].plot(Tiempo3, signal3)
axs[2].set_title("Mujer 3")
axs[2].set_ylabel('Bits')
axs[2].set_xlabel('Tiempo (s)')

axs[3].plot(Tiempo4, signal4)
axs[3].set_title("Hombre 1")
axs[3].set_ylabel('Bits')
axs[3].set_xlabel('Tiempo (s)')

axs[4].plot(Tiempo5, signal5)
axs[4].set_title("Hombre 2")
axs[4].set_ylabel('Bits')
axs[4].set_xlabel('Tiempo (s)')

axs[5].plot(Tiempo6, signal6)
axs[5].set_title("Hombre 3")
axs[5].set_ylabel('Bits')
axs[5].set_xlabel('Tiempo (s)')

plt.tight_layout()
plt.show()
```

**Gráficas de las señales en el dominio del tiempo**


<img width="1391" height="669" alt="image" src="https://github.com/user-attachments/assets/aad52399-4cb8-4b81-ba97-193ee7769408" />


<img width="1391" height="324" alt="image" src="https://github.com/user-attachments/assets/2a7d231d-8984-4098-9d8f-fbae8637e094" />


**Aplicación de la Transformada de Fourier**


```python
senales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    nrows=len(senales), ncols=1,
    figsize=(12, 2.8*len(senales)),
    sharex=True
)

fny_comun = min(fs/2 for _, _, fs in senales)
ymax = 0.0
espectros = []

for (titulo, senal, fs) in senales:
    N = len(senal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(senal))
    idx = freqs <= fny_comun
    espectros.append((titulo, freqs[idx], espectro[idx]))
    ymax = max(ymax, espectro[idx].max())

for ax, (titulo, F, X) in zip(axes, espectros):
    Fp = F[1:]
    Xp = X[1:]
    ax.semilogx(Fp, Xp)
    ax.set_title(titulo)
    ax.set_ylabel('Amplitud')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
axes[-1].set_xlabel('Frecuencia (Hz)')
fig.tight_layout(); plt.show()
```

**Gráficas del espectro de Fourier de magnitudes frecuenciales**

<img width="1211" height="545" alt="image" src="https://github.com/user-attachments/assets/4abbcc6e-f149-400e-b8fa-31fe5e3b2f01" />

<img width="1197" height="274" alt="image" src="https://github.com/user-attachments/assets/7885e7fa-3e5d-434d-b58c-595f7f150f32" />

<img width="1193" height="539" alt="image" src="https://github.com/user-attachments/assets/21cd5149-28e0-4356-86be-eeae68870858" />

<img width="1221" height="322" alt="image" src="https://github.com/user-attachments/assets/9e36fb5b-35a8-472f-b4b4-1986224fdd9b" />


**Características de la señal**

```python
import numpy as np
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


def _to_mono_float(x):
    x = np.asarray(x)
    if x.ndim > 1:  
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    x = x - x.mean()  
    return x

def analizar_senal(signal, fs):
    sig = _to_mono_float(signal)
    N = len(sig)
    if N == 0 or fs <= 0:
        return 0.0, 0.0, 0.0, 0.0

    X = np.abs(rfft(sig))
    f = rfftfreq(N, d=1.0/fs)

    #Frecuencia fundamental
    mask = f >= 50.0 
    Xb = X[mask]
    fb = f[mask]
    if Xb.size and Xb.max() > 0:
        peaks, _ = find_peaks(Xb, height=0.10 * Xb.max())
        if peaks.size:
            f0 = float(fb[peaks[np.argmax(Xb[peaks])]])  #Pico más alto
        else:
            f0 = 0.0
    else:
        f0 = 0.0

    #Frecuencia media
    denom = X.sum()
    f_media = float((f * X).sum() / denom) if denom > 0 else 0.0

    #Brillo
    E_total = float((X**2).sum())
    E_altas = float((X[f > 1500.0]**2).sum())
    brillo = float(E_altas / E_total) if E_total > 0 else 0.0

    #Intensidad
    intensidad = float((sig**2).mean())

    return f0, f_media, brillo, intensidad

nombres = ["Mujer 1","Mujer 2","Mujer 3","Hombre 1","Hombre 2","Hombre 3"]
senales = [signal1, signal2, signal3, signal4, signal5, signal6]
fs_list = [fs1, fs2, fs3, fs4, fs5, fs6]

resultados = []
for nom, sig, fs in zip(nombres, senales, fs_list):
    f0, fmedia, brillo, intensidad = analizar_senal(sig, fs)
    resultados.append([nom, f0, fmedia, brillo, intensidad])

tabla = pd.DataFrame(resultados, columns=["Señal","f0 (Hz)","f_media (Hz)","Brillo","Intensidad"])
print(tabla.to_string(index=False))

```
<img width="353" height="108" alt="Captura de pantalla 2025-10-06 181855" src="https://github.com/user-attachments/assets/6c5a9a8e-ee83-4daa-b619-68c4da652f01" />

<br>

<img width="341" height="896" alt="image" src="https://github.com/user-attachments/assets/9f4021fc-ff94-4deb-bf7c-3fe221d92b46" />


# Parte B
Ahora bien, en esta parte se seleccionó una grabación de voz masculina y una femenina para realizar un análisis más detallado de estabilidad vocal. Primero, se aplicó un filtro pasa-banda en el rango correspondiente a cada género (80–400 Hz para hombres y 150–500 Hz para mujeres) con el fin de eliminar ruidos externos y conservar únicamente las frecuencias relevantes de la voz. Luego, se calculó el jitter, que representa la variación en la frecuencia fundamental entre ciclos consecutivos, y el shimmer, que mide la variación en la amplitud, para ello se detectaron los periodos y los picos de cada señal, obteniendo tanto los valores absolutos como los relativos de cada parámetro. Finalmente, se registraron los resultados para todas las grabaciones, lo que permitió comparar la estabilidad vocal entre hombres y mujeres y analizar posibles diferencias en la regularidad de sus señales.

**Filtro pasabandas**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def filtro_pasabanda(x, fs, lowcut, highcut):
  nyq = 0.5 * fs
  low= lowcut/ nyq
  high = highcut / nyq
  b, a= butter(4, [low, high], btype='band')
  return filtfilt(b, a, x)

mujer1_filtrada= filtro_pasabanda(signal1, fs3, 150, 500)

hombre3_filtrada= filtro_pasabanda(signal6, fs6, 80, 400)

plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(signal1, color='tab:pink', label="Original mujer")
plt.plot(mujer1_filtrada, color='tab:blue', label="Filtrada mujer")
plt.title("Mujer 1")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()

plt.subplot(2,1,2)
plt.plot(signal6, color='tab:cyan', label="Original hombre")
plt.plot(hombre3_filtrada, color='tab:purple', label="Filtrada hombre")
plt.title("Hombre 3")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.legend()

plt.tight_layout()
plt.show()

```

<img width="1210" height="585" alt="image" src="https://github.com/user-attachments/assets/47ece3d0-8e02-4e04-8295-856043f05066" />

**Medición de Jitter y Shimmer**

```python
mport numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

def bandpass_butter(x, fs, f_low, f_high, order=4):
    nyq = fs / 2.0
    wn = [f_low/nyq, f_high/nyq]
    sos = butter(order, wn, btype='bandpass', output='sos')
    return sosfiltfilt(sos, x)

def jitter_shimmer(x, fs, sexo='hombre'):
    if sexo.lower().startswith('h'):
        x_f = bandpass_butter(x, fs, 80, 400, order=4)
        F0_min, F0_max = 80, 200
    else:
        x_f = bandpass_butter(x, fs, 150, 500, order=4)
        F0_min, F0_max = 150, 300

    x_f = x_f - np.mean(x_f)
    sigma = np.std(x_f)


    dist_min = int(fs / F0_max)
    peaks, _ = find_peaks(x_f, distance=dist_min, prominence=0.1*sigma)

    if len(peaks) < 3:
        raise ValueError("Muy pocos picos detectados. Ajusta filtros/prominence o usa un segmento más estable.")

    t = peaks / fs
    T = np.diff(t)


    Tmin, Tmax = 1.0/F0_max, 1.0/F0_min
    mask = (T >= Tmin) & (T <= Tmax)
    T = T[mask]


    A = x_f[peaks].astype(float)
    A = A[:len(T)+1]
    dT = np.abs(np.diff(T))
    dA = np.abs(np.diff(A))

    if len(dT) == 0 or len(dA) == 0:
        raise ValueError("No hay suficientes ciclos válidos tras filtrar outliers.")

    jitter_abs = np.mean(dT)
    jitter_rel = (jitter_abs / np.mean(T)) * 100.0

    shimmer_abs = np.mean(dA)
    shimmer_rel = (shimmer_abs / np.mean(A)) * 100.0

    resultados = {
        "N_ciclos": len(T)+1,
        "Jitter_abs_s": jitter_abs,
        "Jitter_rel_%": jitter_rel,
        "Shimmer_abs": shimmer_abs,
        "Shimmer_rel_%": shimmer_rel,
        "peaks_idx": peaks,
        "x_filtrada": x_f
    }
    return resultados
# hombre:
res_h1 = jitter_shimmer(signal4, fs4, sexo='hombre')
print("Hombre 1: ", res_h1["Jitter_abs_s"], res_h1["Jitter_rel_%"], res_h1["Shimmer_abs"], res_h1["Shimmer_rel_%"])
res_h2 = jitter_shimmer(signal5, fs5, sexo='hombre')
print("Hombre 2: ", res_h2["Jitter_abs_s"], res_h2["Jitter_rel_%"], res_h2["Shimmer_abs"], res_h2["Shimmer_rel_%"])
res_h3 = jitter_shimmer(signal6, fs6, sexo='hombre')
print("Hombre 3: ", res_h3["Jitter_abs_s"], res_h3["Jitter_rel_%"], res_h3["Shimmer_abs"], res_h3["Shimmer_rel_%"])

# mujer:
res_m1 = jitter_shimmer(signal1, fs1, sexo='mujer')
print("Mujer 1: ",res_m1["Jitter_abs_s"], res_m1["Jitter_rel_%"], res_m1["Shimmer_abs"], res_m1["Shimmer_rel_%"])
res_m2 = jitter_shimmer(signal2, fs2, sexo='mujer')
print("Mujer 2: ",res_m2["Jitter_abs_s"], res_m2["Jitter_rel_%"], res_m2["Shimmer_abs"], res_m2["Shimmer_rel_%"])
res_m3 = jitter_shimmer(signal3, fs3, sexo='mujer')
print("Mujer 1: ",res_m3["Jitter_abs_s"], res_m3["Jitter_rel_%"], res_m3["Shimmer_abs"], res_m3["Shimmer_rel_%"])

```

<img width="708" height="113" alt="image" src="https://github.com/user-attachments/assets/2371db95-a71d-45ac-8cee-7187cc0dd1cf" />


# Parte C
Finalmente, en esta parte se compararon los resultados obtenidos entre las voces masculinas y femeninas, se observaron diferencias notables en la frecuencia fundamental, siendo más alta en las voces femeninas debido a la anatomía de las cuerdas vocales, mientras que las masculinas presentaron frecuencias más bajas y mayor energía en las componentes graves. También se analizaron parámetros como el brillo, la intensidad, el jitter y el shimmer, encontrando variaciones asociadas a la estabilidad y calidad vocal. Finalmente, se discutió la relevancia clínica de estos parámetros, ya que valores elevados de jitter o shimmer pueden indicar alteraciones en la voz o fatiga vocal. Esta comparación permitió comprender cómo las características espectrales reflejan diferencias fisiológicas y funcionales entre los géneros.


**¿Qué diferencias se observan en la frecuencia fundamental?**


**¿Qué otras diferencias notan en términos de brillo, media o intensidad?**

Entre los audios de los hombres, la frecuencia media tiende a ser mucho más variada que entre los audios de las mujeres. Los audios de las mujeres se quedaron dentro del rango de 3.1k - 3.8k Hz, mientras que los de los hombres tuvieron un rango mucho más grande de 2.8k - 5.4k Hz. Las voces femeninas tienden a tener una frecuencia media más alta, pero debido a diferencias en articulación o la grabación de los audios los hombre pueden presentar frecuancias medias más altas. En terminos de brillo, las mujeres tuvieron un rango de 0.029-0.042, y lo hombre tuvieron un rango de 0.012-0.053. Las voces femeninas tienden a tener un brillo más alto por tener voces más agudas, sin embargo la pronunciación más fuerte puede hacer que las voces masculinas tengan un brillo más alto. Continunado, en la intensidad las mujeres tuvieron un rango más amplio de 7500000-36000000. Por otro lado, los hombres obtuvieron un rango de 10000000-14000000. En este laboratorio, no se presencio una tendencia clara dividida por sexo, ya que esta intensidad puede depender de la distancias entre el telefono y la persona durante la grabación además del esfuerzo al hablar.

**Conclusiones sobre el comportamiento de la voz en hombres y
mujeres a partir de los análisis realizados.**

A partir de lo establecido antes, se podría decir que la frecuencia fundamental es consistentemente mayor en las voces femeninas (220-496 Hz), a comparación de las masculinas (213-234 Hz). Esto es consistente con la fisiología de las cuerdas vocales, ya que las mujeres tienen cuardas vocales más cortas y delgadas, lo cuál causa que haya una vibración más rapida, por ende tienen una frecuencia fundamental más rápida. En contraste, los hombre presentan frecuencias fundamentales más bajas, demostrando una vibración más lenta.

Además, se visualizó que la frecuancia media del espectro varía más en hombres que en mujeres, indicando una mayor variabilidad en la articulación de palabras. La mujeres, por otra parte, tuvieron valores más concentrados dentro de un rango menor, lo que indica a mayor homgeneidad vocal. 

En adición, el brillo a pesar de que las voces femeninas tienden a ser más "brillantes" por ser más agudas, en este labooratorio se observó como las voces masculinas también pueden alcanzar valores altos debido a la ponunciación más fuerte.

Por ultimo, la intensidad no presentó una tendencia clara dependiendo de los sexos, ya que esta puede depender de valores externos cómo la distancia entre el emisor y el micrófono, y el esfuerzo del habla.

Para concluir, los resultados se alinean en gran parte con las diferencias anatómicas y fisiológicas entre las voces masculinas y femeninas. Estas diferencias influyen directamente con la frecuencia fundamental y media, mientras que el brillo e intensidad muestran mayor variabilidad según las condiciones de grabación y enunciación.

**Importancia clínica del jitter y shimmer en el análisis de la voz.**

El jitter y el shimmer son parámetros acústicos que permiten evaluar la estabilidad y la calidad de la voz de manera objetiva.
Para comenzar, el jitter representa las pequeñas variaciones en la frecuencia fundamental de un ciclo a otro, mientras que el shimmer mide las variaciones en la amplitud o intensidad de la señal, estos cambios reflejan el grado de control y regularidad con que vibran las cuerdas vocales durante la fonación.

De acuerdo con Wertzner et al. (2005), estos índices son fundamentales en el diagnóstico y seguimiento de alteraciones vocales, ya que un incremento en sus valores puede estar asociado con inestabilidad en la vibración glótica, lesiones en las cuerdas vocales o disminución del control neuromuscular. Aunque en su estudio con niños con trastornos fonológicos no se hallaron diferencias significativas en jitter y shimmer respecto al grupo control, los autores destacan que estos parámetros son útiles para descartar la presencia de disfunciones laríngeas y analizar la regularidad de la voz.

En el contexto clínico y biomédico, el análisis de jitter y shimmer es esencial para evaluar la calidad vocal, detectar trastornos de fonación, monitorear procesos terapéuticos y complementar el diagnóstico de trastornos del habla o del lenguaje, ya que reflejan directamente la estabilidad vibratoria y la eficiencia de las cuerdas vocales.
