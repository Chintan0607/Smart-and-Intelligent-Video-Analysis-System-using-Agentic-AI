import re
import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

from config import MAX_NEW_TOKENS, MODEL_CACHE_DIR, VLM_MODEL_QWEN
from prompts import PROMPT, ANOMALY_PROMPT

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
            VLM_MODEL_QWEN,
            cache_dir=MODEL_CACHE_DIR,
        )

        self.processor.tokenizer.padding_side = "left"

    def _query_vlm(self, frames, prompt):

        messages = []

        for frame in frames:

            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": frame,
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ]
            )

        with torch.no_grad():

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )

            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs.input_ids,
                    generated_ids,
                )
            ]

            outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return outputs

    def analyze_frame(self, frame):
        return self.analyze_batch([frame])[0]

    def analyze_batch(self, frames):

        responses = self._query_vlm(frames, PROMPT)

        parsed = []

        for response in responses:
            parsed.append(
                self._parse_response(response)
            )

        return parsed

    def _extract_field(self, text, field):

        pattern = rf"{field}:\s*(.*)"

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

        return "Unknown"

    def _parse_response(self, response):

        categories = [
            "Blur",
            "Noise",
            "Compression",
            "Lighting",
        ]

        result = {
            "text_description": response,
            "defects": {},
        }

        for category in categories:

            pattern = (
                rf"{category}:(.*?)(?=\n(?:Blur|Noise|Compression|Lighting|Overall Quality|Summary):|\Z)"
            )

            match = re.search(
                pattern,
                response,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if not match:
                continue

            section = match.group(1)

            present_val = self._extract_field(
                section,
                "Present",
            )

            result["defects"][category] = {
                "present": present_val.lower() == "yes",
                "confidence": self._extract_field(
                    section,
                    "Confidence",
                ),
                "severity": self._extract_field(
                    section,
                    "Severity",
                ),
                "evidence": self._extract_field(
                    section,
                    "Evidence",
                ),
            }

        result["overall_quality"] = self._extract_field(
            response,
            "Overall Quality",
        )

        result["summary"] = self._extract_field(
            response,
            "Summary",
        )

        return result

    def analyze_anomaly_frame(self, frame):
        return self.analyze_anomaly_batch([frame])[0]

    def analyze_anomaly_batch(self, frames):
        responses = self._query_vlm(frames, ANOMALY_PROMPT)
        return [self._parse_anomaly_response(r) for r in responses]

    def _parse_anomaly_response(self, response):
        categories = [
            "Violence", "Weapon", "Fire", "Accident", "Fall", "Intrusion", "Theft",
        ]

        result = {"anomalies": {}}

        for category in categories:
            pattern = (
                rf"{category}:(.*?)(?=\n(?:"
                rf"Violence|Weapon|Fire|Accident|Fall|Intrusion|Theft|"
                rf"Overall Risk|Summary):|\Z)"
            )
            match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
            if not match:
                continue

            section = match.group(1)
            present_val = self._extract_field(section, "Present")

            result["anomalies"][category] = {
                "present": present_val.lower() == "yes",
                "confidence": self._extract_field(section, "Confidence"),
                "severity": self._extract_field(section, "Severity"),
                "evidence": self._extract_field(section, "Evidence"),
            }

        result["overall_risk"] = self._extract_field(response, "Overall Risk")
        result["summary"] = self._extract_field(response, "Summary")
        result["review_recommended"] = any(
            a["present"] for a in result["anomalies"].values()
        )

        return result