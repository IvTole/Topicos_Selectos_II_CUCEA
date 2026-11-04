import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Tema 12 · Localización e IoU")


@app.cell
def _():
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    import torch
    from torchvision.ops import box_convert, box_iou

    return Circle, Rectangle, box_convert, box_iou, math, mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tema 12 · De clasificación a detección: cajas delimitadoras e IoU

    **Deep Learning para Visión por Computadora · Licenciatura en IA y Ciencia de Datos**

    Hasta ahora hemos asignado una clase a una imagen. En una escena con varios
    peatones también queremos saber **dónde está cada persona**.

    Al terminar podrás:

    - Interpretar y convertir coordenadas de cajas delimitadoras.
    - Calcular intersección sobre unión (**IoU**) a mano, en Python y con Torchvision.
    - Explicar por qué confianza de clasificación y calidad de localización son distintas.
    - Preparar una anotación y reconocer qué salidas necesitaría una red de localización.

    **Prerrequisitos:** listas, funciones, tensores, clasificación con CNN y noción
    de regresión. Se utiliza geometría de rectángulos; no necesitas haber entrenado
    un detector. Tiempo sugerido: **100–120 minutos**, incluyendo ejercicios.

    La escena de esta práctica es un dibujo sintético creado con Matplotlib.
    Las cajas de referencia y las predicciones son ejemplos construidos a mano:
    no son salidas de un modelo. Puedes trabajar sin GPU ni descargar datos o pesos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. ¿Qué cambia respecto a clasificación?

    | Tarea | Pregunta | Salida por imagen |
    | --- | --- | --- |
    | Clasificación | ¿Hay una persona? | Clase o probabilidades |
    | Clasificación y localización de un objeto | ¿Qué es y dónde está? | Clase y una caja |
    | Detección | ¿Qué objetos hay y dónde están? | Una lista de clases, cajas y puntuaciones |
    | Segmentación | ¿Qué píxeles pertenecen a cada objeto o clase? | Máscaras |

    La clasificación de peatones del tema anterior no proporciona coordenadas.
    Una caja agrega información espacial, aunque también incluye fondo y no describe
    la silueta del objeto. Hoy estudiaremos esa representación antes de entrenar detectores.

    **Pregunta inicial:** si una foto contiene tres personas, ¿qué información pierde
    una única etiqueta `persona`?
    """)
    return


@app.cell
def _():
    IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360
    reference_box = (140.0, 65.0, 240.0, 335.0)
    second_box = (420.0, 90.0, 510.0, 335.0)
    return IMAGE_HEIGHT, IMAGE_WIDTH, reference_box, second_box


@app.cell
def _(Circle, IMAGE_HEIGHT, IMAGE_WIDTH, Rectangle):
    def draw_scene(ax):
        """Escena didáctica en coordenadas de imagen, con origen arriba a la izquierda."""
        ax.set_facecolor("#e6f1f8")
        ax.add_patch(Rectangle((0, 210), IMAGE_WIDTH, 150, color="#e1e6e9"))
        ax.axhline(210, color="#b0bec5", linewidth=1)
        for left, top, width, height, color in (
            (140, 65, 100, 270, "#435b8e"),
            (420, 90, 90, 245, "#b96745"),
        ):
            cx = left + width / 2
            ax.add_patch(Circle((cx, top + height * 0.095), height * 0.09,
                                facecolor="#deb394", edgecolor="#6e5342"))
            ax.add_patch(Rectangle((left + width * 0.16, top + height * 0.20),
                                   width * 0.68, height * 0.40, color=color))
            for offset in (0.02, 0.84):
                ax.add_patch(Rectangle((left + width * offset, top + height * 0.24),
                                       width * 0.14, height * 0.35, color=color))
            for offset in (0.20, 0.56):
                ax.add_patch(Rectangle((left + width * offset, top + height * 0.60),
                                       width * 0.24, height * 0.40, color="#37474f"))
        ax.set_xlim(0, IMAGE_WIDTH)
        ax.set_ylim(IMAGE_HEIGHT, 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x: aumenta hacia la derecha (px)")
        ax.set_ylabel("y: aumenta hacia abajo (px)")
        ax.set_xticks(range(0, IMAGE_WIDTH + 1, 80))
        ax.set_yticks(range(0, IMAGE_HEIGHT + 1, 60))
        ax.grid(alpha=0.18)

    def draw_box(ax, box, color, label, linestyle="-"):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=2.5,
                               linestyle=linestyle, label=label))

    return draw_box, draw_scene


@app.cell
def _(draw_box, draw_scene, plt, reference_box, second_box):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    draw_scene(_ax)
    draw_box(_ax, reference_box, "#167348", "Referencia: persona A")
    draw_box(_ax, second_box, "#8545a1", "Referencia: persona B", "--")
    _ax.set_title("Una imagen puede tener más de una anotación")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    _fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.90)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Coordenadas: primero define la convención

    Usaremos **xyxy**: `(x_min, y_min, x_max, y_max)`. La esquina superior izquierda
    de la imagen es `(0, 0)`. **x recorre columnas; y recorre filas**.
    Una imagen de ancho W y alto H tiene límites de caja entre 0 y W, y entre 0 y H.
    Sus índices de píxel, en cambio, llegan hasta W−1 y H−1.

    Trabajaremos con bordes continuos: `ancho = x_max − x_min` y
    `alto = y_max − y_min`. **No añadimos +1** al calcular áreas.
    Una caja con área positiva cumple `x_min < x_max` y `y_min < y_max`.

    | Formato | Coordenadas | Conversión desde xyxy |
    | --- | --- | --- |
    | xyxy | Esquinas opuestas | `(x1, y1, x2, y2)` |
    | xywh | Esquina superior izquierda y tamaño | `(x1, y1, x2−x1, y2−y1)` |
    | cxcywh | Centro y tamaño | `((x1+x2)/2, (y1+y2)/2, x2−x1, y2−y1)` |

    **Ejemplo:** `(140, 65, 240, 335)` equivale a `(140, 65, 100, 270)` en xywh
    y a `(190, 200, 100, 270)` en cxcywh. Los valores de una caja solo tienen sentido
    si conocemos su formato y las dimensiones de la imagen.
    """)
    return


@app.cell
def _(box_convert, mo, reference_box, torch):
    _box_tensor = torch.tensor([reference_box], dtype=torch.float32)
    mo.ui.table([
        {"formato": _fmt,
         "valores": str(box_convert(_box_tensor, in_fmt="xyxy", out_fmt=_fmt)[0].tolist())}
        for _fmt in ("xyxy", "xywh", "cxcywh")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. ¿Cómo medimos si dos cajas coinciden?

    **IoU** (*Intersection over Union*) compara una caja predicha P con una anotación
    de referencia G (*ground truth*). Esa referencia es una anotación humana, que
    también puede contener errores.

    \[
    x_I^{min}=\max(x_P^{min},x_G^{min}),\qquad
    x_I^{max}=\min(x_P^{max},x_G^{max})
    \]
    \[
    w_I=\max(0,x_I^{max}-x_I^{min}),\quad
    h_I=\max(0,\min(y_P^{max},y_G^{max})-\max(y_P^{min},y_G^{min}))
    \]
    \[
    A_I=w_Ih_I,\qquad A_U=A_P+A_G-A_I,\qquad
    \operatorname{IoU}(P,G)=\frac{A_I}{A_U}
    \]

    Restamos la intersección una vez porque la suma de áreas la cuenta dos veces.
    IoU vale **1** para cajas iguales de área positiva y **0** cuando no hay área
    compartida. Tocar un borde no produce intersección de área positiva.

    **Ejemplo resuelto:** G = `(10, 10, 50, 50)` y P = `(30, 30, 70, 70)`.
    Cada área vale 1600 px²; la intersección mide 20 × 20 = 400 px².
    La unión vale 1600 + 1600 − 400 = 2800 px², por tanto IoU = **1/7 ≈ 0.143**.
    """)
    return


@app.cell
def _(math):
    def validate_box(box):
        """Acepta xyxy finito y ordenado; permite área cero para estudiar el caso límite."""
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError("La caja debe tener cuatro coordenadas finitas.")
        x1, y1, x2, y2 = map(float, box)
        if x2 < x1 or y2 < y1:
            raise ValueError("Las esquinas están invertidas: se requiere x1 <= x2, y1 <= y2.")
        return x1, y1, x2, y2

    def box_metrics(box_a, box_b):
        """IoU para bordes continuos: no hay +1 y no se reordenan esquinas en silencio."""
        ax1, ay1, ax2, ay2 = validate_box(box_a)
        bx1, by1, bx2, by2 = validate_box(box_b)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = area_a + area_b - intersection
        return {
            "area_a": area_a, "area_b": area_b,
            "intersection": intersection, "union": union,
            # Convención didáctica explícita: dos cajas sin área reciben IoU 0.
            "iou": intersection / union if union > 0 else 0.0,
            "intersection_box": (ix1, iy1, ix2, iy2),
        }

    return (box_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Laboratorio: mueve la caja predicha

    La caja verde es la referencia de la **persona A**; la naranja es una predicción
    hipotética. La región amarilla muestra la intersección. La persona B es contexto:
    por ahora evaluamos únicamente la localización de A.

    Ajusta los dos extremos de cada intervalo. Empieza por predecir el resultado
    antes de moverlos: ¿qué sucede si la caja incluye a la persona completa pero
    también mucho fondo?
    """)
    return


@app.cell
def _(IMAGE_HEIGHT, IMAGE_WIDTH, mo):
    x_interval = mo.ui.range_slider(
        0, IMAGE_WIDTH, step=5, value=(120, 270), show_value=True,
        label="Predicción: [x_min, x_max] (px)", full_width=True,
    )
    y_interval = mo.ui.range_slider(
        0, IMAGE_HEIGHT, step=5, value=(80, 320), show_value=True,
        label="Predicción: [y_min, y_max] (px)", full_width=True,
    )
    iou_threshold = mo.ui.slider(
        0.05, 0.95, step=0.05, value=0.5, show_value=True, label="Umbral geométrico τ",
    )
    mo.vstack([x_interval, y_interval, iou_threshold])
    return iou_threshold, x_interval, y_interval


@app.cell
def _(box_metrics, reference_box, x_interval, y_interval):
    predicted_box = (
        float(x_interval.value[0]), float(y_interval.value[0]),
        float(x_interval.value[1]), float(y_interval.value[1]),
    )
    current_metrics = box_metrics(reference_box, predicted_box)
    return current_metrics, predicted_box


@app.cell
def _(
    Rectangle,
    current_metrics,
    draw_box,
    draw_scene,
    iou_threshold,
    plt,
    predicted_box,
    reference_box,
):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    draw_scene(_ax)
    if current_metrics["intersection"] > 0:
        _ix1, _iy1, _ix2, _iy2 = current_metrics["intersection_box"]
        _ax.add_patch(Rectangle((_ix1, _iy1), _ix2 - _ix1, _iy2 - _iy1,
                                facecolor="#f5cf47", alpha=0.50, label="Intersección"))
    draw_box(_ax, reference_box, "#167348", "Referencia A")
    draw_box(_ax, predicted_box, "#ce5b0a", "Predicción", "--")
    _ax.set_title(f"IoU = {current_metrics['iou']:.3f} · umbral = {iou_threshold.value:.2f}")
    _ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    _fig.subplots_adjust(left=0.10, right=0.97, bottom=0.24, top=0.90)
    _fig
    return


@app.cell
def _(current_metrics, iou_threshold, mo, predicted_box):
    _m = current_metrics
    _valid = _m["area_b"] > 0
    _status = (
        "Caja degenerada: no tiene área; ajusta sus extremos."
        if not _valid else
        ("Cumple el umbral geométrico." if _m["iou"] >= iou_threshold.value
         else "No cumple el umbral geométrico.")
    )
    mo.vstack([
        mo.md(f"**xyxy predicha:** `{predicted_box}` · **{_status}**"),
        mo.ui.table([
            {"cantidad": "Área referencia (px²)", "valor": _m["area_a"]},
            {"cantidad": "Área predicción (px²)", "valor": _m["area_b"]},
            {"cantidad": "Intersección (px²)", "valor": _m["intersection"]},
            {"cantidad": "Unión (px²)", "valor": _m["union"]},
            {"cantidad": "IoU (sin unidades)", "valor": _m["iou"]},
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Casos para contrastar tu intuición

    En la siguiente tabla todas las cajas se comparan con la misma referencia.
    Una caja muy grande no consigue IoU alta solo por contener al objeto.
    Una caja pequeña dentro de la referencia tampoco representa bien su extensión.

    Nuestra función devuelve 0 si ambas áreas y la unión son cero, como convención
    para este ejercicio. En un dataset de detección se deben rechazar las cajas
    degeneradas; esa convención no las convierte en anotaciones válidas.
    """)
    return


@app.cell
def _(box_metrics, mo, reference_box):
    comparison_cases = {
        "Igual a la referencia": reference_box,
        "Desplazada 50 px a la derecha": (190, 65, 290, 335),
        "Contenida en la referencia": (165, 120, 215, 280),
        "Cubre toda la imagen": (0, 0, 640, 360),
        "Solo toca el borde derecho": (240, 65, 340, 335),
        "Sin contacto": (350, 65, 450, 335),
    }
    mo.ui.table([
        {"caso": _name, "caja xyxy": str(_box), "IoU": box_metrics(reference_box, _box)["iou"]}
        for _name, _box in comparison_cases.items()
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Verificar con Torchvision

    `box_iou` recibe dos conjuntos de cajas. Si tienen formas `[N, 4]` y `[M, 4]`,
    devuelve una matriz `[N, M]`: cada celda compara una caja del primer conjunto
    con una del segundo. Usaremos `float32` y formato xyxy.

    Para nuestra única comparación, la salida es `[1, 1]`. Calculamos IoU con
    Torchvision solo cuando ambas cajas tienen área positiva.
    """)
    return


@app.cell
def _(box_iou, current_metrics, mo, predicted_box, reference_box, torch):
    if current_metrics["area_b"] > 0:
        _g = torch.tensor([reference_box], dtype=torch.float32)
        _p = torch.tensor([predicted_box], dtype=torch.float32)
        _torch_iou = box_iou(_g, _p)
        _message = (
            f"**Python:** {current_metrics['iou']:.6f} · "
            f"**Torchvision:** {_torch_iou.item():.6f} · forma: `{list(_torch_iou.shape)}`"
        )
    else:
        _message = "La caja predicha tiene área cero: corrígela para comparar con Torchvision."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Coordenadas normalizadas y cambio de tamaño

    Para representar una caja independientemente del tamaño de imagen, divide sus
    coordenadas por `(W, H, W, H)`. No es la normalización de colores de ImageNet:
    aquí transformamos **coordenadas**, no intensidades de píxeles.

    Al redimensionar una imagen de W × H a W' × H', transforma **también sus cajas**:
    multiplica x por W'/W e y por H'/H. Si aplicas el mismo cambio de escala positivo
    a ambas cajas, sus áreas cambian por el mismo factor y **IoU se conserva**.

    Un recorte requiere restar el desplazamiento y recortar las cajas al nuevo marco;
    un flip horizontal requiere transformar x. No basta con cambiar el tensor de imagen.
    """)
    return


@app.cell
def _(mo):
    resize_scale = mo.ui.slider(
        0.25, 2.0, step=0.25, value=0.5, show_value=True, label="Factor de redimensionamiento",
    )
    resize_scale
    return (resize_scale,)


@app.cell
def _(
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    box_metrics,
    mo,
    predicted_box,
    reference_box,
    resize_scale,
):
    _denominators = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT)
    _normalized = tuple(_v / _d for _v, _d in zip(predicted_box, _denominators))
    _scaled_g = tuple(_v * resize_scale.value for _v in reference_box)
    _scaled_p = tuple(_v * resize_scale.value for _v in predicted_box)
    mo.md(
        f"**Predicción xyxy normalizada:** `{tuple(round(_v, 4) for _v in _normalized)}`\n\n"
        f"**Nuevo tamaño:** {IMAGE_WIDTH * resize_scale.value:g} × "
        f"{IMAGE_HEIGHT * resize_scale.value:g} px. "
        f"**Predicción escalada:** `{_scaled_p}`.\n\n"
        f"**IoU después del cambio:** {box_metrics(_scaled_g, _scaled_p)['iou']:.6f}. "
        "Compárala con la IoU original."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Confianza e IoU responden preguntas distintas

    Un detector asigna una **puntuación de confianza** a cada predicción. No es su IoU
    ni necesariamente una probabilidad calibrada. En producción no conocemos la caja
    real; por eso no podemos consultar su IoU para decidir qué predicciones mostrar.

    Para evaluar, sí contamos con anotaciones. En este ejemplo aislado hay **una sola
    referencia, persona A, y una sola predicción**. Aceptamos la predicción si supera
    el umbral de confianza. Cuenta como acierto si además tiene la clase correcta,
    área positiva e IoU ≥ τ.

    - Predicción aceptada y correcta: **TP = 1**.
    - Predicción aceptada pero incorrecta: **FP = 1 y FN = 1**.
    - Predicción descartada: **FN = 1**.

    La persona B del dibujo queda fuera de este ejercicio de evaluación. Para evaluar
    una escena completa hay que anotar todos los objetos y emparejar predicciones y
    referencias uno a uno: varias cajas sobre el mismo objeto no son varios aciertos.
    Esta simplificación no calcula AP ni mAP.
    """)
    return


@app.cell
def _(mo):
    predicted_class = mo.ui.dropdown(
        ["persona", "bicicleta"], value="persona", label="Clase predicha",
    )
    confidence = mo.ui.slider(0.0, 1.0, step=0.05, value=0.9, show_value=True,
                              label="Confianza hipotética")
    confidence_threshold = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5,
                                        show_value=True, label="Umbral de confianza")
    mo.vstack([predicted_class, confidence, confidence_threshold])
    return confidence, confidence_threshold, predicted_class


@app.function
def evaluate_single_prediction(iou, positive_area, predicted_label, score, score_threshold, iou_threshold):
    accepted = score >= score_threshold
    matched = (accepted and positive_area and predicted_label == "persona"
               and iou >= iou_threshold)
    return {"TP": int(matched), "FP": int(accepted and not matched), "FN": int(not matched)}


@app.cell
def _(
    confidence,
    confidence_threshold,
    current_metrics,
    iou_threshold,
    mo,
    predicted_class,
):
    _counts = evaluate_single_prediction(
        current_metrics["iou"], current_metrics["area_b"] > 0,
        predicted_class.value, confidence.value, confidence_threshold.value, iou_threshold.value,
    )
    mo.vstack([
        mo.md("**Evaluación de una predicción frente a una referencia de clase persona:**"),
        mo.ui.table([_counts]),
        mo.md("Prueba una confianza de 0.95 con una caja sin contacto. Después conserva "
              "la caja exacta y cambia la clase a bicicleta. ¿Por qué fallan ambos casos?"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Conexión con una red neuronal

    Para una imagen con exactamente un objeto, podemos reutilizar un backbone de CNN
    y producir dos salidas a partir de sus características:

    ```text
    imagen → backbone → características ┬→ cabeza de clasificación → logits de clase
                                       └→ cabeza de regresión → cuatro coordenadas
    ```

    La etiqueta de entrenamiento ahora incluye `clase` y `caja`. Una posible pérdida
    combina clasificación y regresión:

    \[
    L = L_{clasificación} + \lambda L_{caja}
    \]

    Por ejemplo, entropía cruzada para la clase y Smooth L1 sobre coordenadas de caja
    en un formato acordado. λ controla su peso relativo. Normalizar coordenadas ayuda
    a manejar escalas, pero no garantiza cajas válidas: hay que asegurar ancho y alto
    positivos y documentar cómo se decodifican las salidas.

    **No conviene usar nuestra IoU de Python como pérdida:** convierte valores a
    `float` y corta el grafo de autograd. Además, IoU estándar puede no dar una señal
    útil para acercar cajas sin solapamiento. Más adelante veremos pérdidas de cajas
    implementadas con tensores y arquitecturas de detección.

    Una cabeza de cuatro números produce una sola caja: todavía no resuelve el número
    variable de objetos de una escena. El siguiente tema será inferencia con un
    detector preentrenado, filtrado por confianza y supresión de cajas redundantes (NMS).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Actividad para entregar

    Trabaja primero a mano y luego verifica con código. Incluye tus procedimientos,
    no solo capturas de los resultados.

    1. **Representación (20 %).** Convierte la referencia A a xywh, cxcywh y xyxy
       normalizado. Recupera las coordenadas en píxeles desde la versión normalizada.
    2. **Cálculo (25 %).** Para A = `(0, 0, 100, 100)` y B = `(50, 0, 150, 100)`,
       calcula las dos áreas, intersección, unión e IoU. Dibuja las cajas y verifica
       con `box_metrics` y `box_iou`.
    3. **Experimentación (25 %).** Ajusta los controles para lograr IoU ≥ 0.75.
       Registra las coordenadas. Después encuentra una caja que contenga a A completa
       pero tenga IoU < 0.25. Explica ambos resultados usando áreas.
    4. **Transformaciones (15 %).** Redimensiona conceptualmente la imagen a 320 × 180.
       Calcula la nueva caja A y explica por qué conservar las coordenadas originales
       sería un error. Comprueba que escalar ambas cajas mantiene IoU.
    5. **Interpretación (15 %).** Construye un caso de confianza alta e IoU baja y otro
       de IoU alta con clase incorrecta. Reporta TP, FP y FN en cada caso. Explica
       qué cambiaría si hubiera dos predicciones sobre una única persona.

    **Entrega:** una copia del notebook con tus celdas de solución y una conclusión
    de 150–200 palabras sobre lo que IoU mide y lo que no mide.

    **Extensión opcional:** calcula con `box_iou` una matriz para las dos referencias
    de la escena y tres predicciones diseñadas por ti. Interpreta cada fila y columna.
    No cuentes dos predicciones sobre la misma referencia como dos verdaderos positivos.
    """)
    return


@app.cell
def _(mo):
    show_answers = mo.ui.checkbox(value=False, label="Mostrar guía de comprobación (después de resolver)")
    show_answers
    return (show_answers,)


@app.cell(hide_code=True)
def _(mo, show_answers):
    mo.stop(not show_answers.value, mo.md("Resuelve la actividad antes de consultar la guía."))
    mo.md(r"""
    ### Guía de comprobación

    - Referencia A: xywh `(140, 65, 100, 270)`; cxcywh `(190, 200, 100, 270)`.
      xyxy normalizado ≈ `(0.21875, 0.18056, 0.375, 0.93056)`.
    - Ejercicio 2: áreas 10000 y 10000; intersección 5000; unión 15000;
      **IoU = 1/3**.
    - La referencia exacta logra IoU = 1. La imagen completa como caja tiene
      IoU = 27000/230400 ≈ 0.1172 aunque contiene a A.
    - En 320 × 180, la referencia A pasa a `(70, 32.5, 120, 167.5)`.
    - Una predicción aceptada sin solapamiento produce FP = 1 y FN = 1.
      Una clase incorrecta con IoU = 1 también produce FP = 1 y FN = 1.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Referencias y siguiente paso

    - [Torchvision: formatos de cajas y conversiones](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_convert.html)
    - [Torchvision: IoU entre conjuntos de cajas](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_iou.html)
    - [Tutorial de detección y fine tuning con Penn-Fudan](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

    **Siguiente práctica:** aplicar un detector preentrenado a imágenes de peatones,
    inspeccionar sus cajas y puntuaciones y estudiar qué predicciones conserva NMS.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
