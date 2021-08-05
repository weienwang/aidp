"""This module defines execution engines that will perform work"""

from abc import ABC, abstractmethod
from datetime import datetime
import logging
from aidp.data.experiments import ClinicalOnlyDataExperiment, ImagingOnlyDataExperiment, \
    FullDataExperiment
import pathlib
import os
import pandas as pd
pd.options.mode.chained_assignment = None  
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import datetime


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
        for experiment in self.experiments: # loops through all the experiments
            self._logger.info("Starting prediction experiment: %s", experiment)
            experiment.predict(self.model_data.data, model_key) # this through all the group comparisons
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
        df2_new.plot.bar(x="Matrics", y = "Value",title=new_filename)
        output_dir =parent_path + '/output/'
        plt.savefig(output_dir + '/' + new_filename, '/_Predict_result.png', bbox_inches='tight') 
        plt.clf()
    
     # create a donut chart
    def donut_chart(self, test_prob, circle_title_1, circle_title_2, switch):       
        parent_path=str(pathlib.Path(__file__).parent.parent.parent) 
        new_filename = '%s_out.xlsx' %os.path.splitext(os.path.basename(self.model_data.filename))[-2]
        output_model_data = pd.read_excel(parent_path + '/' + new_filename).drop('Unnamed: 0',axis=1)
        output_dir=parent_path + '/output/'
        subject_ID_list=output_model_data['Subject']
        #dignosis_report = []	


        for s in subject_ID_list:
            sub_data=output_model_data.loc[output_model_data['Subject'] == s]
            if switch == 'Match':
                value = float(sub_data[test_prob])*100

            else:
                value=100-float(sub_data[test_prob])*100
            
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
            plt.figtext(.3,.8,circle_title_1, fontsize=26, ha='center')

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
            plt.figtext(.72,.8,circle_title_2, fontsize=26, ha='center')

            plt.gca().add_artist(my_circle)

            filepath = output_dir + str(s) + '_' +str(circle_title_1) + 'vs' + str(circle_title_2) + '_prob.png'
            plt.savefig(filepath ,orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0,dpi=300)
            plt.clf()

        #plt.savefig('fig2.png',orientation='portrait',transparent=True, bbox_inches=None, pad_inches=0)

    def bar_chart(self):  
        parent_path=str(pathlib.Path(__file__).parent.parent.parent) 
        new_filename = '%s_out.xlsx' %os.path.splitext(os.path.basename(self.model_data.filename))[-2]
        output_model_data = pd.read_excel(parent_path + '/' + new_filename).drop('Unnamed: 0',axis=1)
        output_dir=parent_path + '/output/'
        subject_ID_list=output_model_data['Subject']

        # control data
        control_dir =parent_path + '/control/'
        df= pd.read_excel(control_dir+'1002_Data_Update_Sixflags_Master.xlsx')
        # select control data
        df = df.loc[df.GroupID == 0] 
        df_new =df[["GroupID", "pSN_FW", "Putamen_FW", "Cerebellar_SCP_FW", "Cerebellar_MCP_FW" ]]
        

        for s in subject_ID_list:
        
            true_sub_data=output_model_data.loc[output_model_data['Subject'] == s]
                       
            # concat control and subject data 
            sub_data=true_sub_data.copy()
            sub_data['GroupID'].iloc[0] = '0'
            sub_data =sub_data[['GroupID', "pSN_FW", "Putamen_FW", "Cerebellar_SCP_FW", "Cerebellar_MCP_FW" ]]
            combined=pd.concat([df_new, sub_data], axis=0, ignore_index=True)

            #wide to long             
            combined.set_index('GroupID')
            combined = combined.reset_index()
            long_df=pd.melt(combined, id_vars='GroupID', value_vars=[ "pSN_FW", "Putamen_FW", "Cerebellar_SCP_FW", "Cerebellar_MCP_FW" ])

          
            # set the matlab figure
            plt.figure(figsize=(7,3))
            #sns.set(rc={'figure.figsize':(7,3)})
            #sns.set_style("whitegrid")
            sns.set_style("dark")
            sns.set_context("talk")
            colors = ['#4c72b0', '#55a868'] # Set your custom color palette
            sns.set_palette(sns.color_palette(colors))

            # make a bar plot
            g= sns.barplot(x="variable",y="value", hue = "GroupID", data=long_df, capsize=.1 )
            plt.legend([],[], frameon=False)

            g.set_xticklabels(["pSN", "Putamen", "SCP" , "MCP"])
            g.set(xlabel='ROIs', ylabel='FW')
            plt.gca().set_prop_cycle(None)
            filepath = output_dir + str(s) + '_FW_barplot.png'
            plt.savefig(filepath ,orientation='landscape',transparent=True, bbox_inches='tight', pad_inches=0)
            #plt.show()
            plt.clf()
    

    def pdf_report(self):  
        parent_path=str(pathlib.Path(__file__).parent.parent.parent) 
        new_filename = '%s_out.xlsx' %os.path.splitext(os.path.basename(self.model_data.filename))[-2]
        output_model_data = pd.read_excel(parent_path + '/' + new_filename).drop('Unnamed: 0',axis=1)
        output_dir=parent_path + '/output/'
        subject_ID_list=output_model_data['Subject']
        
        for s in subject_ID_list:
            sub_data=output_model_data[output_model_data["Subject"] == s]
            
            ID=sub_data['Subject'].iloc[0]
            print(ID)
            Age=sub_data['Age'].iloc[0]
            Sex=sub_data['Sex'].iloc[0]
            UPDRS=sub_data['UPDRS'].iloc[0]   


            pdf = FPDF('P', 'mm', 'Letter')
            pdf.add_page()
            pdf.set_font('Arial', '', 16)
            parent_path=str(pathlib.Path(__file__).parent.parent.parent) 
            output_dir=parent_path + '/output/'
            template_dir=parent_path + '/resources/'
            print("Hello-1")
            #output_dir="c:\\users\\weienwang\\onedrive\\documents\\github\\aidp_BETA\\output\\"
            #template_dir="c:\\users\\weienwang\\onedrive\\documents\\github\\aidp_BETA\\resources\\"

            pdf.image(template_dir+"template_v2-600.png",x = 0, y = 0, w = 215.9, h = 279.4)
            #pdf.image(template_dir+"template-150.png",x = 0, y = 0, w = 215.9, h = 279.4)


            pdf.set_xy(77, 36.5)
            pdf.cell(25, 30, ID)

            pdf.set_xy(77, 50.5)
            date_output=datetime.datetime.now().strftime("%Y-%m-%d")
            pdf.cell(25, 30, date_output)


            clinical='Age: '+str(Age) + '   Sex: '+ str(Sex) + '   UPDRS: ' + str(UPDRS)
            pdf.set_xy(77, 65)
            pdf.cell(25, 30, clinical)

            print("Hello-2")
            filepath = output_dir + str(ID) + str('_PD · MSA · PSPvsControl_prob.png')
            pdf.image(filepath,x = 123, y = 90.71, w =65.630, h = 49.222)
            print("Hello-3")
            filepath = output_dir + str(ID) +  str('_PDvsMSA · PSP_prob.png')
            pdf.image(filepath,x = 123, y = 122.71, w = 65.630, h =49.222)

            filepath = output_dir + str(ID) + str('_MSAvsPSP_prob.png')
            pdf.image(filepath,x = 123, y = 154.71 , w = 65.630, h =49.222)


            filepath = output_dir + str(ID) + str('_FW_barplot.png')
            pdf.image(filepath,x = 12.25, y = 209.4 , w = 127.694, h = 61.918)

            file_name=output_dir + str(ID) + '_Imaging_Report.pdf'
            pdf.output(file_name, 'F')
        
        
class TrainingEngine(Engine):
    """Defines tasks that will be completed as part of the training workflow"""
    def start(self, model_key = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S%f")):
        for experiment in self.experiments:
            self._logger.info("Starting training experiment: %s", experiment)
            experiment.train(self.model_data.data, model_key)
            self._logger.debug("Finished training experiment: %s", experiment)
    def generate_report(self,model_key='default'):
        pass
    def donut_chart(self, test_prob, circle_title_1, circle_title_2, switch):
        pass
    def bar_chart(self):
        pass
    def pdf_report(self):
        pass


def getEngine(key, model_data):
    logger = logging.getLogger(__name__)

    if key == 'predict':
        return PredictionEngine(model_data)
    if key == 'train':
        return TrainingEngine(model_data)
    else:
        logger.error("Use of unsupported Engine key: %s", key)
        raise NotImplementedError
        

   