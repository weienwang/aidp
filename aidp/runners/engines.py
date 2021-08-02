"""This module defines execution engines that will perform work"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from aidp.data.experiments import ClinicalOnlyDataExperiment, ImagingOnlyDataExperiment, \
    FullDataExperiment
import pathlib
import os
import pandas as pd
import matplotlib.pyplot as plt

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
            experiment.predict(self.model_data.data, model_key)
            self._logger.debug("Finished prediction experiment: %s", experiment)

            results = experiment.get_results()
            self.model_data.add_results(results)
        self.model_data.write_output_file()

    def generate_report(self, model_key='default'):        
        # get the clinical data and diffusion data

        parent_path=str(pathlib.Path(__file__).parent.parent.parent)
        new_filename = '%s_out.xlsx' %os.path.splitext(os.path.basename(self.model_data.filename))[-2]
        in_table = pd.read_excel(parent_path + '/' + new_filename).drop('Unnamed: 0',axis=1)
        df1 =in_table.loc[:, 'both_park_v_control (PD/MSA/PSP Probability)':'clinical_psp_v_msa (PSP Probability)']
        df2=df1.transpose()
        df2['Matrics'] = df2.index
        df2 = df2.reset_index(drop=True)
        df2_new = df2.rename(columns={0: 'Value'})
        df2_new.plot.bar(x="Matrics", y = "Value",title="U01_Test_Run")
        plt.savefig(parent_path + '/' + 'Predict_result.png', bbox_inches='tight') 
     


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
        
class plot_result:

    def __init__(self, model_data, test_prob, circle_title_1, circle_title_2):
        self.model_data = model_data
        self.test_prob=test_prob
        self.circle_title_1=circle_title_1
        self.circle_title_2=circle_title_2
        self.output="c:\\users\\weienwang\\onedrive\\documents\\github\\aidp_BETA\\"


# generate report

    # create a donut chart
    def donut_chart(self):
        value=(float(self.model_data[self.test_prob]))
        value=value*100
        if value >= 50:
            color = [0.9569,0.6941,0.5137]
        else:
            color = [0.8, 0.9, 1]

        first_pie_value = [value, 100 - value]
    
        fig, ax = plt.subplots()  
        plt.axis("off")

        # first circle 
        ax = fig.add_subplot(1,2,1)
        x = 0
        y = 0
    
        my_circle=plt.Circle( (0,0), radius = 0.7, color='white')
        ax.add_patch(my_circle)
        str_1 = str(round(value,1)) + '%'
        label= ax.annotate(str_1, xy=(x,y-0.05), fontsize = 24, ha='center')

        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()
    
        plt.pie(first_pie_value,
                wedgeprops = { 'linewidth' : 0, 'edgecolor' : 'white' },
                colors=[color,'#d2d2d2'], startangle = 90, radius=1.2)
        #plt.title("Parkinsonism") 
        #plt.figtext(.3,.8,'PD/MSA/PSP', fontsize=20, ha='center')
        #plt.figtext(.3,.8,'PD', fontsize=20, ha='center')
        plt.figtext(.3,.8,self.circle_title_1, fontsize=24, ha='center')

        p=plt.gcf()
        p.gca().add_artist(my_circle)
    
        # second circle 

        second_pro= round(100 - value,1)
        second_pie_value = [100 - value, value] 

        if second_pro >= 50:
            color = [0.9569,0.6941,0.5137]
        else:
            color = [0.8, 0.9, 1]


        ax = fig.add_subplot(1,2,2)
        x = 0
        y = 0
    
        my_circle=plt.Circle( (0,0), radius = 0.7, color='white')
        ax.add_patch(my_circle)
        str_2 = str(round(second_pro,1)) + '%'
        label= ax.annotate(str_2, xy=(x,y-0.05), fontsize = 24, ha='center')

        ax.axis('off')
        ax.set_aspect('equal')
        ax.autoscale_view()
    
        plt.pie(second_pie_value,
                wedgeprops = { 'linewidth' : 0, 'edgecolor' : 'white' },
                colors=[color,'#d2d2d2'], startangle = 90, radius=1.2)
        #plt.title("Control") 
        #plt.figtext(.72,.8,'Control', fontsize=20, ha='center')
        #plt.figtext(.72,.8,'Atypical', fontsize=20, ha='center')
        plt.figtext(.72,.8,self.circle_title_2, fontsize=24, ha='center')
    
        plt.gca().add_artist(my_circle)
        plt.savefig(self.output +'test_v3.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)
        #plt.show()
        #plt.close('all')
        return fig   

        #plt.savefig('fig2.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)