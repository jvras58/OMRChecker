import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.template import Template
from src.core import ImageInstanceOps
from src.utils.parsing import (
    open_config_with_defaults,
    get_concatenated_response,
)


# Caminho do template/config padrão — pode virar parâmetro depois
DEFAULT_SAMPLE_DIR = Path("samples/simureka")


class OMRService:
    def __init__(self, sample_dir: Path = DEFAULT_SAMPLE_DIR):
        self.sample_dir = sample_dir
        self._load_template_and_config()

    def _load_template_and_config(self):
        """Carrega template e config uma única vez (reuse entre requisições)"""
        config_path = self.sample_dir / "config.json"
        template_path = self.sample_dir / "template.json"

        self.tuning_config = open_config_with_defaults(config_path)
        # Desabilita abertura de janelas de imagem na API (headless)
        self.tuning_config.outputs.show_image_level = 0
        self.template = Template(
            template_path,
            self.tuning_config,
        )
        self.image_ops = ImageInstanceOps(self.tuning_config)
        # Injetar image_ops no template (como entry.py faz)
        self.template.image_instance_ops = self.image_ops

    def process_image(self, image_bytes: bytes) -> dict:
        """
        Recebe bytes de imagem, processa com OMR pipeline.
        Retorna dict com:
          - omr_response: dict das respostas { "q1": "A", ... }
          - final_marked: np.ndarray da imagem anotada
          - multi_marked: int (0 = ok, >0 = múltiplas marcações)
        """
        # Decodifica bytes → numpy array (grayscale)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        in_omr = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if in_omr is None:
            raise ValueError("Não foi possível decodificar a imagem enviada.")

        # Reseta estado de imagens de debug
        self.image_ops.reset_all_save_img()
        self.image_ops.append_save_img(1, in_omr)

        # Pré-processamento (CropOnMarkers, blur, etc.)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            cv2.imwrite(str(tmp_path), in_omr)

        processed = self.image_ops.apply_preprocessors(
            str(tmp_path), in_omr, self.template
        )
        tmp_path.unlink(missing_ok=True)

        if processed is None:
            raise ValueError(
                "Pré-processamento falhou: marcadores não encontrados na imagem."
            )

        # Leitura das bolhas
        response_dict, final_marked, multi_marked, _ = self.image_ops.read_omr_response(
            self.template,
            image=processed,
            name="api_request",
            save_dir=None,
        )

        # Monta resposta concatenada (ex: roll + questões)
        omr_response = get_concatenated_response(response_dict, self.template)

        return {
            "omr_response": omr_response,
            "final_marked": final_marked,  # np.ndarray BGR
            "multi_marked": multi_marked,
        }
