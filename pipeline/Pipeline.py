from __future__ import annotations
from typing import List

from pipeline.PipelineStep import PipelineStep


class Pipeline:

    def __init__(self, pipeline_steps: List[PipelineStep] = None):
        if pipeline_steps is None:
            pipeline_steps = []
        self.pipeline_steps = pipeline_steps

    def addAlgorithm(self, algorithm: PipelineStep) -> Pipeline:
        self.pipeline_steps.append(algorithm)
        return self

    def getName(self):
        return '_'.join(map(lambda x: x.name, self.pipeline_steps))

    def run(self, data):
        for pipeline_step in self.pipeline_steps:
            data = pipeline_step.perform(data)
        return data
