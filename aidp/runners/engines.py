"""This module defines execution engines that will perform work"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from aidp.data.experiments import ClinicalOnlyDataExperiment, ImagingOnlyDataExperiment, \
    FullDataExperiment

class Engine(ABC):
    """Abstract Base Class for classes which execute a series of related tasks"""
    _logger = logging.getLogger(__name__)
    experiments = [
            FullDataExperiment(),
            ImagingOnlyDataExperiment(),
            ClinicalOnlyDataExperiment()
        ]

    def __init__(self, model_data):
        self.model_data = model_data

    @abstractmethod
    def start(self):
        """Abstract method for executing the engine's tasks"""

class PredictionEngine(Engine):
    """Defines tasks that will be completed as part of the prediction workflow"""
    def start(self, model_key='default'):
        for experiment in self.experiments:
            self._logger.info("Starting prediction experiment: %s", experiment)
            experiment.predict(self.model_data.data, model_key,)
            self._logger.debug("Finished prediction experiment: %s", experiment)

            results = experiment.get_results()
            self.model_data.add_results(results)

        self.model_data.write_output_file()

class TrainingEngine(Engine):
    """Defines tasks that will be completed as part of the training workflow"""
    def start(self, model_key = datetime.now().strftime("%Y-%m-%d-%H%M%S%f")):
        for experiment in self.experiments:
            self._logger.info("Starting training experiment: %s", experiment)
            experiment.train(self.model_data.data, model_key)
            self._logger.debug("Finished training experiment: %s", experiment)

def getEngine(key, model_data):
    logger = logging.getLogger(__name__)

    if key == 'predict':
        return PredictionEngine(model_data)
    if key == 'train':
        return TrainingEngine(model_data)
    else:
        logger.error("Use of unsupported Engine key: %s", key)
        raise NotImplementedError
