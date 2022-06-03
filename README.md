## Entrega final Mineria de Datos - MIIA

Este proyecto hace parte de la entrega final de minería de datos. Por el momento no cuenta con un despliegue productivo, sin embargo, el demo se puede ralizar de manera local. Para esto es necesario tener instalado `python`, con sus respectivas librerías, en la máquina de trabajo. Para esto es posible ejecutar la instalación de los requerimientos utilizando el archivo `requirements.txt`, sin embargo, dado que se hizo sobre un ambiente `conda`, es posible que los requerimientos estén incompletos.

### Ejecución

Se recomienda utilizar pycharm y ejecutar el script `src/app/server.py`, o en su defecto, ejecutar

```sh
python src/app/server.py
```

Una vez levantado el servidor, es posible realizar peticiones en
[http://localhost:5000](http://localhost:5000).

#### Ejemplo petición

```json
{
    "topic": "Política",
    "headline": "Título del artículo",
    "article": "Cuerpo del artículo. Esto es un ejemplo."
}
```

Las categorías son predefinidas con base en la siguiente lista:

```json
[
    "Educacion",
    "Sociedad",
    "Ciencia",
    "Seguridad",
    "Salud",
    "Economia",
    "Deportes",
    "Política",
    "Entretenimiento",
    "Covid-19",
    "Internacional",
    "Deporte",
    "Ambiental"
 ]
```

#### Ejemplo respuesta.

```json
{
      probability: {
        catboost: 0.6,
        rnn: 0.55,
      },
      ngrams: {
        text_bigrams: [
          ["(1, 2)", 10],
          ["(1, 3)", 10],
          ["(1, 4)", 10],
          ["(1, 5)", 10],
          ["(1, 6)", 10],
          ["(1, 7)", 10],
          ["(1, 8)", 10],
          ["(1, 9)", 10],
          ["(1, 10)", 10],
          ["(1, 11)", 10],
        ],
        text_trigrams: [
          ["(1, 2, 1)", 10],
          ["(1, 2, 2)", 10],
          ["(1, 2, 3)", 10],
          ["(1, 2, 4)", 10],
          ["(1, 2, 5)", 10],
          ["(1, 2, 6)", 10],
          ["(1, 2, 7)", 10],
          ["(1, 2, 8)", 10],
          ["(1, 2, 9)", 10],
          ["(1, 2, 10)", 10],
        ],
      },
      sentiment: {
        headline: {
          polarity: 0.7,
          subjetivity: 0.8,
        },
        text: {
          polarity: 0.7,
          subjetivity: 0.8,
        },
      },
      variables: {
        headline_palabras: 100,
        headline_palabras_avg_len: 100,
        headline_mayusculas: 100,
        headline_numbers: 100,
        headline_especiales: 100,
        headline_stopwords: 100,
        headline_unicas: 100,
        headline_avg_subjetivity: 100,
        text_palabras: 100,
        text_palabras_avg_len: 100,
        text_mayusculas: 100,
        text_numbers: 100,
        text_especiales: 100,
        text_stopwords: 100,
        text_unicas: 100,
        text_oraciones: 100,
        text_oraciones_avg_len: 100,
      },
}
```

### Front

El front que expone la herramienta interactiva de predicción de textos, se encuentra hecha en `javascript`, haciendo uso de `node.js`. Para referirse a este proyecto ir a:
https://github.com/camtorr95/fake_news_front
