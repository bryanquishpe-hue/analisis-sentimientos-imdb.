document.addEventListener("DOMContentLoaded", function () {

    // ===== MENU MOVIL =====
    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.getElementById("navLinks");

    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", function () {
            navLinks.classList.toggle("active");
            const icon = menuToggle.querySelector("i");
            if (icon) {
                icon.classList.toggle("bi-list");
                icon.classList.toggle("bi-x-lg");
            }
        });
    }

    // ===== ENTRENAR MODELO (analisis.html) =====
    const formEntrenamiento = document.getElementById("formEntrenamiento");
    const estado = document.getElementById("estadoEntrenamiento");

    if (formEntrenamiento) {
        formEntrenamiento.addEventListener("submit", async function (e) {
            e.preventDefault();

            const boton = formEntrenamiento.querySelector("button[type='submit']");
            boton.disabled = true;
            estado.innerHTML = '<i class="bi bi-hourglass-split"></i> Entrenando el modelo, espera unos minutos...';

            try {
                const resp = await fetch("/api/entrenar", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        max_features: parseInt(document.getElementById("maxFeatures").value),
                        C: parseFloat(document.getElementById("paramC").value)
                    })
                });
                const datos = await resp.json();

                if (!resp.ok) {
                    estado.innerHTML = '<i class="bi bi-x-circle"></i> ' + (datos.error || "Error al entrenar.");
                } else {
                    estado.innerHTML = '<i class="bi bi-check-circle"></i> ' + datos.mensaje;
                }
            } catch (err) {
                estado.innerHTML = '<i class="bi bi-x-circle"></i> No se pudo conectar con el servidor.';
            } finally {
                boton.disabled = false;
            }
        });
    }

    // ===== ANALIZAR RESEÑA (analisis.html) =====
    const formResena = document.getElementById("formResena");

    if (formResena) {
        formResena.addEventListener("submit", async function (e) {
            e.preventDefault();

            const texto = document.getElementById("resena").value.trim();
            const contenedor = document.getElementById("resultadoPrediccion");
            const sentimiento = document.getElementById("sentimiento");
            const confianza = document.getElementById("confianza");
            const barra = document.getElementById("confidenceBar");
            const probPos = document.getElementById("probabilidadPositiva");
            const probNeg = document.getElementById("probabilidadNegativa");

            if (!texto) return;

            contenedor.classList.remove("hidden");
            contenedor.style.display = "block";
            sentimiento.textContent = "Analizando...";
            confianza.textContent = "";
            barra.style.width = "0%";

            try {
                const resp = await fetch("/api/predecir", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ review: texto })
                });
                const datos = await resp.json();

                if (!resp.ok) {
                    sentimiento.textContent = datos.error || "Error al analizar.";
                    return;
                }

                const positivo = datos.sentimiento === "Positivo";
                const color = positivo ? "#22c55e" : "#ef4444";

                sentimiento.innerHTML = "Sentimiento: <strong style='color:" + color + "'>" +
                    (positivo ? "👍 " : "👎 ") + datos.sentimiento + "</strong>";
                confianza.textContent = "Confianza: " + datos.confianza + "%";
                barra.style.width = datos.confianza + "%";
                barra.style.height = "100%";
                barra.style.background = color;
                probPos.textContent = "Positiva: " + datos.probabilidad_positiva + "%";
                probNeg.textContent = "Negativa: " + datos.probabilidad_negativa + "%";
            } catch (err) {
                sentimiento.textContent = "No se pudo conectar con el servidor.";
            }
        });
    }

    // ===== METRICAS (resultados.html) =====
    const metricasGrid = document.getElementById("metricasGrid");
    const btnMetricas = document.getElementById("actualizarMetricas");

    function porcentaje(valor) {
        return (valor * 100).toFixed(2) + "%";
    }

    function tarjeta(etiqueta, valor) {
        return '<div class="metric-card">' +
            '<span class="metric-label">' + etiqueta + '</span>' +
            '<strong class="metric-value">' + valor + '</strong></div>';
    }

    async function cargarMetricas() {
        if (!metricasGrid) return;
        metricasGrid.innerHTML = "<p>Cargando métricas...</p>";

        try {
            const resp = await fetch("/api/metricas");
            const datos = await resp.json();

            if (!resp.ok) {
                metricasGrid.innerHTML = "<p>" + (datos.error || "No hay métricas.") + "</p>";
                return;
            }

            const m = datos.metricas || {};
            let html = "";

            html += tarjeta("Exactitud (Accuracy)", porcentaje(m.accuracy));
            html += tarjeta("Precisión", porcentaje(m.precision));
            html += tarjeta("Recall", porcentaje(m.recall));
            html += tarjeta("F1-Score", porcentaje(m.f1_score));

            if (datos.registros !== undefined) {
                html += tarjeta("Registros totales", Number(datos.registros).toLocaleString("es-EC"));
                html += tarjeta("Datos de entrenamiento", Number(datos.entrenamiento).toLocaleString("es-EC"));
                html += tarjeta("Datos de prueba", Number(datos.prueba).toLocaleString("es-EC"));
            }

            if (datos.matriz_confusion) {
                const mc = datos.matriz_confusion;
                html += '<div class="metric-card metric-matrix">' +
                    '<span class="metric-label">Matriz de confusión</span>' +
                    '<table class="matrix-table">' +
                    '<tr><th></th><th>Pred. Negativo</th><th>Pred. Positivo</th></tr>' +
                    '<tr><th>Real Negativo</th><td>' + mc[0][0] + '</td><td>' + mc[0][1] + '</td></tr>' +
                    '<tr><th>Real Positivo</th><td>' + mc[1][0] + '</td><td>' + mc[1][1] + '</td></tr>' +
                    '</table></div>';
            }

            metricasGrid.innerHTML = html;
        } catch (e) {
            metricasGrid.innerHTML = "<p>Error al cargar las métricas.</p>";
        }
    }

    if (metricasGrid) {
        cargarMetricas();
        if (btnMetricas) btnMetricas.addEventListener("click", cargarMetricas);
    }
});


