
# ----------------------------------
# Contoller unit
# ----------------------------------

#General Includes:
import pickle
from cv2 import imread
import numpy as np



#Project Includes:
# from View_unit.view_manager import View_manager
# from Logical_unit.TFL_manager import TFL_Manager


#A class that control the flow of the whole program:
class Controller:

    def __init__(self) -> None:
        print("Init the controller")

        #TFL_manager-Traffic lights manager,
        #manage the logical unit of finding the traffic lights
        self.__TFL_manager = TFL_Manager()

        #Manage the view of the program
        self.__viewer = View_manager()


    def parse_pickle(self,pkl_path):
        with open(pkl_path, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        self.__focal = data['flx']
        self.__pp = data['principle_point']
        self.__data=data
        self.__EM = np.eye(4)




    def parse_frames(self,frames_pathes,first_frame):
        absolute=[[]]
        for index, frame_path in enumerate( frames_pathes):
            frame = imread(frame_path)
            self.__EM=self.__data['egomotion_' + str(index) + '-' + str(index + 1)]
            EM_parameters = {'image': frame, 'focal': self.__focal, 'pp': self.__pp,'EM': self.__EM}
            view_parameters = self.__TFL_manager.parse_frame(frame,EM_parameters,absolute)
            view_parameters['image'] = frame
            self.__viewer.view_current_type('tfl').view(view_parameters)


    def run(self, file_path):
        print("Run the controller")
        with open(file_path, 'r') as paly_list_file:
            play_list = paly_list_file.read().split('\n')
        pkl_path = play_list[0]
        first_frame = int(play_list[1])
        frames_pathes = play_list[2:]
        self.parse_pickle(pkl_path)
        self.parse_frames(frames_pathes, first_frame)

