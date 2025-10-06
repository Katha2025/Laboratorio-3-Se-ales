# Laboratorio-3-Se-ales
En esta práctica se trabajaron conceptos fundamentales del procesamiento digital de señales aplicados al análisis de la voz humana. Primero, se adquirieron grabaciones de voz de hombres y mujeres, asegurando condiciones similares de muestreo y entorno. Luego, se aplicó la Transformada de Fourier para representar las señales en el dominio de la frecuencia y analizar su espectro, a partir de esto, se calcularon parámetros característicos como la frecuencia fundamental, el brillo, la intensidad, el jitter y el shimmer. Finalmente, se compararon los resultados entre voces masculinas y femeninas, identificando sus principales diferencias y comprendiendo la importancia de estas medidas en la evaluación y caracterización de la voz.
# Parte A
En primer lugar se realizaron las grabaciones de seis personas, tres hombres y tres mujeres, pronunciando la misma frase y cada archivo se guardó en formato .wav con nombres identificadores. Luego, las señales fueron importadas a Python, donde se graficaron en el dominio del tiempo para observar su forma de onda. Posteriormente, se aplicó la Transformada de Fourier a cada una de las grabaciones, obteniendo el espectro de magnitud y permitiendo identificar las componentes principales de frecuencia. Finalmente, se calcularon parámetros como la frecuencia fundamental, frecuencia media, brillo e intensidad, los cuales sirvieron como base para el análisis comparativo entre voces masculinas y femeninas.

**Graficado de la señales captadas**

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
Tiempo1 = np.arange(0,duracion,1/fs1)
duracion = len(signal2)/fs1
Tiempo2 = np.arange(0,duracion,1/fs2)
duracion = len(signal3)/fs1
Tiempo3 = np.arange(0,duracion,1/fs3)


duracion = len(signal4)/fs1
Tiempo4 = np.arange(0,duracion,1/fs4)
duracion = len(signal5)/fs1
Tiempo5 = np.arange(0,duracion,1/fs5)
duracion = len(signal6)/fs1
Tiempo6 = np.arange(0,duracion,1/fs6)

fig, axs = plt.subplots(6,1, figsize=(14,10),sharex=False)

print(len(Tiempo3),len(signal3))
# graficas
axs[0].plot(Tiempo1,signal1)
axs[0].set_title("Mujer 1")
plt.xlabel('Tiempo (s)')
plt.ylabel('Bits')

axs[1].plot(Tiempo2,signal2)
axs[1].set_title("Mujer 2")
plt.ylabel('Bits')

axs[2].plot(Tiempo3[0:-1],signal3)
axs[2].set_title("Mujer 3")
plt.ylabel('Bits')

axs[3].plot(Tiempo4,signal4)
axs[3].set_title("Hombre 1")
plt.ylabel('Bits')

axs[4].plot(Tiempo5,signal5)
axs[4].set_title("Hombre 2")
plt.ylabel('Bits')

axs[5].plot(Tiempo6,signal6)
axs[5].set_title("Hombre 3")
plt.ylabel('Bits')
```
plt.tight_layout()
plt.show()



# Parte B
Ahora bien, en esta parte se seleccionó una grabación de voz masculina y una femenina para realizar un análisis más detallado de estabilidad vocal. Primero, se aplicó un filtro pasa-banda en el rango correspondiente a cada género (80–400 Hz para hombres y 150–500 Hz para mujeres) con el fin de eliminar ruidos externos y conservar únicamente las frecuencias relevantes de la voz. Luego, se calculó el jitter, que representa la variación en la frecuencia fundamental entre ciclos consecutivos, y el shimmer, que mide la variación en la amplitud, para ello se detectaron los periodos y los picos de cada señal, obteniendo tanto los valores absolutos como los relativos de cada parámetro. Finalmente, se registraron los resultados para todas las grabaciones, lo que permitió comparar la estabilidad vocal entre hombres y mujeres y analizar posibles diferencias en la regularidad de sus señales.



# Parte C
Finalmente, en esta parte se compararon los resultados obtenidos entre las voces masculinas y femeninas. Se observaron diferencias notables en la frecuencia fundamental, siendo más alta en las voces femeninas debido a la anatomía de las cuerdas vocales, mientras que las masculinas presentaron frecuencias más bajas y mayor energía en las componentes graves. También se analizaron parámetros como el brillo, la intensidad, el jitter y el shimmer, encontrando variaciones asociadas a la estabilidad y calidad vocal. Finalmente, se discutió la relevancia clínica de estos parámetros, ya que valores elevados de jitter o shimmer pueden indicar alteraciones en la voz o fatiga vocal. Esta comparación permitió comprender cómo las características espectrales reflejan diferencias fisiológicas y funcionales entre los géneros.


**¿Qué diferencias se observan en la frecuencia fundamental?**


**¿Qué otras diferencias notan en términos de brillo, media o intensidad?**


**Conclusiones sobre el comportamiento de la voz en hombres y
mujeres a partir de los análisis realizados.**


**Importancia clínica del jitter y shimmer en el análisis de la voz.**
