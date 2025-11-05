import fitz  # PyMuPDF
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image
import yaml
from pathlib import Path
import logging


class DeepSeekOCRProcessor:
    def __init__(self):
        """Inicializa el procesador con el modelo DeepSeek-OCR."""
        self.llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
        )
        logging.info("Modelo DeepSeek-OCR cargado.")

    def process_pdf(self, pdf_path: Path, output_path: Path) -> bool:
        """
        Procesa un PDF con DeepSeek-OCR y guarda directamente el Markdown.

        Args:
            pdf_path: Ruta al archivo PDF de entrada
            output_path: Ruta donde guardar el Markdown

        Returns:
            True si el procesamiento fue exitoso, False en caso contrario
        """
        try:
            logging.info(f"Processing PDF '{pdf_path.name}' with DeepSeek-OCR...")

            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            model_input = []
            prompt = (
                "<image>\n"
                "Your task is to perform Optical Character Recognition (OCR). "
                "Transcribe the text from the image verbatim. "
                "Do not add, infer, or generate any content that is not explicitly written in the image. "
                "Preserve the original structure as much as possible using Markdown. "
                "Your sole function is transcription."
            )

            for page_number in range(total_pages):
                logging.info(
                    f"Converting page {page_number + 1}/{total_pages} to image..."
                )
                page = pdf_document.load_page(page_number)

                zoom = 4.0
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix)
                image = Image.frombytes(
                    "RGB", [pixmap.width, pixmap.height], pixmap.samples
                )

                model_input.append(
                    {"prompt": prompt, "multi_modal_data": {"image": image}}
                )

            pdf_document.close()

            if not model_input:
                logging.warning("No pages extracted.")
                return False

            sampling_param = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                ),
                skip_special_tokens=False,
            )

            logging.info("Sending pages to model for OCR...")
            model_outputs = self.llm.generate(model_input, sampling_param)
            logging.info("OCR processing completed.")

            with open(output_path, "w", encoding="utf-8") as f:
                for i, output in enumerate(model_outputs, 1):
                    page_text = output.outputs[0].text.strip()

                    # Crear metadata compatible con MarkdownParser
                    metadata = {
                        "page": i,
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
        
                    if i < total_pages:
                        f.write("\n\n")

            logging.info(f"Successfully processed {total_pages} pages to {output_path}")
            return True

        except Exception as e:
            logging.error(f"DeepSeek-OCR processing failed for {pdf_path.name}: {e}")
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
