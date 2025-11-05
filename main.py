import fitz  # PyMuPDF
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import yaml
from pathlib import Path
import logging
import time
FORBIDDEN_PHRASES = [
    "la imagen contiene",
    "el texto dice",
    "esta imagen muestra",
    "en formato markdown",
    "lista de subtítulos",
    "bloques de texto",
    "el contenido del documento",
]


class DeepSeekOCRProcessor:
    def __init__(self):
        """Inicializa el procesador con el modelo DeepSeek-OCR."""
        self.llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        self.sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},
            ),
            skip_special_tokens=False,
        )
        logging.info("Modelo DeepSeek-OCR cargado.")

    def _is_output_valid(self, text: str) -> bool:
        """Verifica si la salida del modelo parece una alucinación."""
        text_lower = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text_lower:
                logging.warning(
                    f"Posible alucinación detectada. Frase encontrada: '{phrase}'"
                )
                return False
        return True

    def process_pdf(self, pdf_path: Path, output_path: Path) -> bool:
        """
        Procesa un PDF con DeepSeek-OCR página por página y guarda el Markdown.
        """
        try:
            logging.info(f"Procesando PDF '{pdf_path.name}' con DeepSeek-OCR...")
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            with open(output_path, "w", encoding="utf-8") as f:
                for page_number in range(total_pages):
                    page_index = page_number + 1
                    logging.info(f"Procesando página {page_index}/{total_pages}...")

                    page = pdf_document.load_page(page_number)

                    zoom = 4.0
                    matrix = fitz.Matrix(zoom, zoom)
                    pixmap = page.get_pixmap(matrix=matrix)
                    image = Image.frombytes(
                        "RGB", [pixmap.width, pixmap.height], pixmap.samples
                    )

                    prompt_template = (
                        "<image>\n"
                        "Extrae el texto en español de la imagen y formatéalo en Markdown. "
                        "Conserva la estructura original del documento (títulos, listas, tablas). "
                        "No añadas comentarios ni texto que no esté en la imagen.\n\n"
                        "--- TEXTO EXTRAÍDO ---\n"
                    )
                    prompt = prompt_template.format(
                        page_number=page_index, total_pages=total_pages
                    )

                    model_input = [
                        {"prompt": prompt, "multi_modal_data": {"image": image}}
                    ]

                    max_retries = 2
                    for attempt in range(max_retries):
                        model_outputs = self.llm.generate(
                            model_input, self.sampling_param
                        )
                        page_text = model_outputs[0].outputs[0].text.strip()

                        if self._is_output_valid(page_text):
                            break  
                        else:
                            logging.warning(
                                f"Reintentando página {page_index} (Intento {attempt + 1}/{max_retries})..."
                            )
                            time.sleep(1)

                    metadata = {
                        "page": page_index,
                        "total_pages": total_pages,
                        "source": pdf_path.name,
                        "processor": "DeepSeek-OCR",
                    }
                    frontmatter = yaml.dump(
                        metadata, default_flow_style=False, allow_unicode=True
                    )
                    f.write("---\n")
                    f.write(frontmatter)
                    f.write("---\n\n")
                    f.write(page_text)

                    if page_index < total_pages:
                        f.write("\n\n")

            pdf_document.close()
            logging.info(
                f"Procesadas exitosamente {total_pages} páginas en '{output_path}'"
            )
            return True

        except Exception as e:
            logging.error(
                f"El procesamiento con DeepSeek-OCR falló para {pdf_path.name}: {e}",
                exc_info=True,
            )
            return False


def process_directory(input_dir: Path, output_dir: Path):
    """
    Procesa todos los PDFs en un directorio y guarda los Markdown en otro.

    Args:
        input_dir: Directorio con los archivos PDF
        output_dir: Directorio donde guardar los Markdown procesados
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"No se encontraron archivos PDF en {input_dir}")
        return

    logging.info(f"Encontrados {len(pdf_files)} archivos PDF para procesar")

    processor = DeepSeekOCRProcessor()

    successful = 0
    failed = 0

    for pdf_path in pdf_files:
        try:
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Procesando: {pdf_path.name}")
            logging.info(f"{'=' * 60}")

            output_path = output_dir / f"{pdf_path.stem}.md"

            if processor.process_pdf(pdf_path, output_path):
                logging.info(f"✓ Guardado exitosamente: {output_path}")
                successful += 1
            else:
                logging.error(f"✗ Falló el procesamiento de: {pdf_path.name}")
                failed += 1

        except Exception as e:
            logging.error(f"✗ Error procesando {pdf_path.name}: {e}")
            failed += 1

    logging.info(f"\n{'=' * 60}")
    logging.info("Procesamiento completado")
    logging.info(f"Exitosos: {successful} | Fallidos: {failed}")
    logging.info(f"{'=' * 60}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    input_directory = Path("docs")
    output_directory = Path("new_docs")

    if not input_directory.exists():
        logging.error(f"El directorio '{input_directory}' no existe")
        exit(1)

    process_directory(input_directory, output_directory)

    markdown_files = list(output_directory.glob("*.md"))
    print(
        f"\n✓ Proceso finalizado: {len(markdown_files)} archivos Markdown generados en '{output_directory}'"
    )
