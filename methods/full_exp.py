from overrides import overrides
from typing import Dict, Any, List
import numpy as np
import torch
import spacy
from argparse import Namespace

from methods.ref_method import RefMethod
from lattice import Product as L


class Full_exp(RefMethod):
    """each isolated proposal is evaluated with the full expression."""

    nlp = spacy.load('en_core_web_sm')

    def __init__(self, args: Namespace):
        self.args = args
        self.box_area_threshold = args.box_area_threshold
        self.batch_size = args.batch_size
        self.batch = []

    @overrides
    def execute(self, caption: str, env: "Environment") -> Dict[str, Any]:
        probs = env.filter(caption, area_threshold = self.box_area_threshold, softmax=True)
        pred = np.argmax(probs)
        return {
            "probs": probs,
            "pred": pred,
            "box": env.boxes[pred],
        }
