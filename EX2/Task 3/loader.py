import numpy as np
import os
import json


class loader:
    """
    This python file has three funtions: 1.read the scenarion file as JSON; 
                                         2.add pedestrain;
                                         3.save the new scenario result. 
    """

    def __init__(self, scenario_file):
        """
        read and initailize the scenario file
        """
        self.scenario_file = scenario_file
        self.load_file()

    def load_file(self):
        """
        load the JSON file as dict,
        find the dynamicElements to add pedestrain
        """
        new_scenario = open(self.scenario_file, "r")
        self.scenario_dict = json.load(new_scenario)
        

        # find the dynamic Element to add new pedestrian
        self.dynamicElements = self.scenario_dict[
            "scenario"]["topography"]["dynamicElements"]
        
        new_scenario.close()
    
    
    def add_Pedestrain(self, x=0, y=0, id=-1, speed=np.random.normal(1.34, 0.26), targetIDs=[]):
        """
        Add a pedestrian into the dynamicElements dict.

        Args:
            x (int): x coordinate of the pedestrian. Defaults to 0.
            y (int): y coordinate of the pedestrian. Defaults to 0.
            speed (float): the speed of pedestrain which is calculate by the noraml distribution
            targetIDs (list): could assign the pedstrain to which target
        """
        ped = {
            "attributes": {
                "id": id,
                "visible" : True,
                "radius": 0.2,
                "densityDependentSpeed": False,
                "speedDistributionMean": 1.34,
                "speedDistributionStandardDeviation": 0.26,
                "minimumSpeed": 0.5,
                "maximumSpeed": 2.2,
                "acceleration": 2.0,
                "footstepHistorySize": 4,
                "searchRadius": 1.0,
                "walkingDirectionCalculation": "BY_TARGET_CENTER",
                "walkingDirectionSameIfAngleLessOrEqual": 45.0
          
            },
            "source": None,
            "targetIds": targetIDs,
            "nextTargetListIndex": 0,
            "isCurrentTargetAnAgent": False,
            "position": {
                "x": x,
                "y": y
            },
            "velocity": {
                "x": 0.0,
                "y": 0.0
            },
            "freeFlowSpeed": speed,
            "followers": [],
            "idAsTarget": -1,
            "isChild": False,
            "isLikelyInjured": False,
            "psychologyStatus": {
                "mostImportantStimulus": None,
                "threatMemory": {
                    "allThreats": [],
                    "latestThreatUnhandled": False
                },
                "selfCategory": "TARGET_ORIENTED",
                "groupMembership": "OUT_GROUP",
                "knowledgeBase": {
                    "knowledge": []
                }
            },
            "groupIds": [],
            "groupSizes": [],
            "trajectory": {
                "footSteps": []
            },
            "modelPedestrianMap": None,
            "type": "PEDESTRIAN"
        }

        self.dynamicElements.append(ped)

    def save(self, output_file, scienario_name):
        """
        Save the scenario as JSON 

        Args:
            output_file (string): The directory of the output file.
            scienario_name (string): Name of the output scenario.
        """

        self.scenario_dict["name"] = scienario_name

        output_file_path = os.path.join(output_file, scienario_name+".scenario")
        
        output_scenario = open(output_file_path, "w")
        # convert the dict to JSON
        json.dump(self.scenario_dict, output_scenario, indent=2)
        print("The scenario is saved to", output_file_path)
        output_scenario.close()
        
