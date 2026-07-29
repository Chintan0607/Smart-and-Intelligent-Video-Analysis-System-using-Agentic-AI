"""
Vision Language Model Service (VLM)
"""
from config import DO_SAMPLE, MAX_NEW_TOKENS, VLM_MODEL_ID, DEVICE,MODEL_CACHE_DIR
from transformers import AutoModelForCausalLM
import torch
from prompts import *

class VLMService:
    def __init__(self):
        print("Loading local VLM model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_ID,
            revision="2025-06-21",
            trust_remote_code=True,
            device_map={"":"cuda"}
        )
        print("Model Loaded Successfully!")


    def _query_vlm(self,prompt,frame):
         response = self.model.query(frame,prompt)
         return response["answer"]

    def analyze_frame(self,frame):
        """
        Analyze an extracted frame
        
        Parameters:
        -----------
        frame : numpy.ndarray
        Image extracted from the video

        frame_number : int
        Frame Number

        Returns
        -------
        dict
        Structured quality analysis of the frame  
        """

        analysis = {
            "quality_issues" : [],
            "text_description" : self._query_vlm(prompt=PROMPT,frame=frame),
            "regeneration_recommended" : False
            }
        return analysis

