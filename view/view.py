
# ------------------------------------------
# view unit
# ------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

class TrafficLights:

    def view(self, params):
        print("View the traffic lights")
        image=params.get('image')
        fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(150, 150))
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.01)
        self.__view_Identify_light_sources(params.get('part_1'),image,ax1)
        self.__view_Selection_of_traffic_lights(params.get('part_2'), image,ax2)
        self.__view_Finding_the_distance(params.get('part_3'), image,ax3)
        plt.show(block=True)

    def __view_Identify_light_sources(self, params, image,ax):
        red_x=params.get('red_x')
        red_y = params.get('red_y')
        green_x = params.get('green_x')
        green_y = params.get('green_y')
        ax.set_title('part_1',fontsize=70)

        ax.imshow(image)
        ax.plot(red_x, red_y, 'ro', color='r', markersize=25)
        ax.plot(green_x, green_y, 'ro', color='g', markersize=25)

    def __view_Selection_of_traffic_lights(self,params, image,ax):
        red_x=params.get('red_x')
        red_y = params.get('red_y')
        green_x = params.get('green_x')
        green_y = params.get('green_y')
        ax.set_title('part_2',fontsize=70)
        ax.imshow(image)
        ax.plot(red_x, red_y, 'ro', color='r', markersize=25)
        ax.plot(green_x, green_y, 'ro', color='g', markersize=25)

    def __view_Finding_the_distance(self,params, image, curr_sec):
        focal=params.get('focal')
        pp=params.get('pp')
        curr_container=params.get('curr_container')
        prev_container = params.get('prev_container')



        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
        norm_rot_pts = rotate(norm_prev_pts, R)
        rot_pts = unnormalize(norm_rot_pts, focal, pp)
        foe = np.squeeze(unnormalize(np.array([norm_foe]), focal, pp))

        curr_p = curr_container.traffic_light


        valid=curr_container.valid
        traffic_lights_3d_location=curr_container.traffic_lights_3d_location


        curr_sec.set_title('part_3',fontsize=70)
        curr_sec.imshow(image)

        if len(curr_p):

            curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+',markersize=70)

            for i in range(len(curr_p)):
                curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b', markersize=70)
                if len(valid) and valid[i]:
                    curr_sec.text(curr_p[i, 0], curr_p[i, 1], r'{0:.1f}'.format(traffic_lights_3d_location[i, 2]), color='r',fontsize=60
                                  )
            curr_sec.plot(foe[0], foe[1], 'r+')

            curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+', markersize=70)
        plt.show()


class View_manager:
    def __init__(self):
        self.types = {
            'tfl':TrafficLights
        }

    def view_current_type(self,general_type):
        # print("view_current_type")
        try:
            return self.types.get(general_type)()
        except Exception:
            raise Exception("Command not found in factory")

