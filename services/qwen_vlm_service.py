import re
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from config import MAX_NEW_TOKENS, MODEL_CACHE_DIR, VLM_MODEL_QWEN
from prompts import PROMPT

# Configure 4-bit NF4 quantization to keep VRAM usage around ~2-2.5GB total
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)


class QwenService:

    def __init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_QWEN,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=MODEL_CACHE_DIR,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            VLM_MODEL_QWEN, cache_dir=MODEL_CACHE_DIR
        )

    def _query_vlm(self, frame, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        with torch.no_grad():
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output_text

    def analyze_frame(self, frame):
        response = self._query_vlm(frame, PROMPT)
        parsed = self._parse_response(response[0])
        return parsed

    def _extract_field(self, text, field):
        pattern = rf"{field}:\s*(.*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Unknown"

    def _parse_response(self, response):
        # Updated to match the 4 merged categories in your prompt
        categories = ["Blur", "Noise", "Compression", "Lighting"]

        result = {"text_description": response, "defects": {}}

        for category in categories:
            # Capture block starting from "Category:" up to the next section header or end of text
            pattern = rf"{category}:(.*?)(?=\n(?:Blur|Noise|Compression|Lighting|Overall Quality|Summary):|\Z)"
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)

            if not match:
                continue

            section = match.group(1)
            present_val = self._extract_field(section, "Present")

            result["defects"][category] = {
                "present": present_val.lower() == "yes",
                "confidence": self._extract_field(section, "Confidence"),
                "severity": self._extract_field(section, "Severity"),
                "evidence": self._extract_field(
                    section, "Evidence"
                ),  # Updated to 'Evidence'
            }

        # Extract root level metadata
        result["overall_quality"] = self._extract_field(
            response, "Overall Quality"
        )
        result["summary"] = self._extract_field(response, "Summary")

        return result