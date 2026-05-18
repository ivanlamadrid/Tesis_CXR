"""Minimal Streamlit UI for the Tesis_CXR academic demo."""

from __future__ import annotations

import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

DISCLAIMER = os.getenv(
    "DISCLAIMER",
    "Resultado preliminar de uso académico. No constituye diagnóstico clínico.",
)
DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "admin123")

st.set_page_config(
    page_title="Tesis_CXR",
    layout="wide",
)


def check_api_health(api_base_url: str) -> tuple[bool, dict[str, object] | str]:
    """Call the API health endpoint and return a display-friendly result."""
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException as exc:
        return False, str(exc)


def render_login() -> None:
    """Render a simple environment-based login form."""
    st.title("Tesis_CXR")
    st.warning(DISCLAIMER)
    st.subheader("Acceso")

    with st.form("login_form"):
        username = st.text_input("Usuario", value="")
        password = st.text_input("Contraseña", value="", type="password")
        submitted = st.form_submit_button("Ingresar")

    if submitted:
        if username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Credenciales inválidas.")


def render_app() -> None:
    """Render the Phase 1 placeholder UI."""
    st.title("Tesis_CXR")
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Configuración")
        api_base_url = st.text_input("API_BASE_URL", value=DEFAULT_API_BASE_URL)

        if st.button("Probar conexión API"):
            ok, result = check_api_health(api_base_url)
            if ok:
                st.success("API disponible.")
                st.json(result)
            else:
                st.error("No se pudo conectar con la API.")
                st.code(str(result))

        if st.button("Cerrar sesión"):
            st.session_state["authenticated"] = False
            st.rerun()

    st.subheader("Carga de radiografía")
    uploaded_file = st.file_uploader(
        "Seleccione un archivo",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
    )

    if uploaded_file is not None:
        suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
        st.write(f"Archivo cargado: `{uploaded_file.name}`")

        if suffix in {"png", "jpg", "jpeg"}:
            image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
            st.image(image, caption="Vista previa", use_container_width=True)
        else:
            st.info("La vista previa DICOM se implementará en fases posteriores.")

    if st.button("Ejecutar análisis preliminar"):
        st.info("Placeholder: la inferencia real se implementará en fases posteriores.")
        st.markdown(
            """
            En fases posteriores se mostrará:

            - 14 probabilidades por etiqueta NIH.
            - Hallazgos positivos según thresholds.
            - Grad-CAM.
            - Tiempo de procesamiento.
            - Disclaimer obligatorio.
            """
        )
        st.warning(DISCLAIMER)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if st.session_state["authenticated"]:
    render_app()
else:
    render_login()
