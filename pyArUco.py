__version__ = '0.1'
import numpy as np
import cv2
from cv2 import aruco
from tkinter import filedialog
from tkinter import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from celluloid import Camera
from pyfiglet import Figlet
from PIL import Image
import time
from cv2 import VideoWriter, VideoWriter_fourcc

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


def file_explorer():
    '''
    Open window for file exploring
    File type supoprted: .avi, .mov, .mp4

    Return:
        path {str} -- path file of the video
    '''
    root = Tk()
    path =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("Avi Files","*.avi"),("MOV Files","*.MOV"),("mp4 Files","*.mp4")))   
    root.destroy()

    return path

def load_video(path):
    '''
    Load video and read it as a matrix

    Arguments:
        path {str} -- path file of the video 

    Returns:
        video {array} -- selected video as matrix 
        frameRate {float} -- frame rate of the selected video 
    '''
    
    cap = cv2.VideoCapture(path)

    frameWidth=cap.get(3)
    frameHeigth=cap.get(4)
    frameRate = cap.get(5)
    frameAmount = cap.get(7)

    pbar = tqdm(total=frameAmount,desc='Loading Video...',colour='yellow') 

    video = np.zeros((int(frameAmount),int(frameHeigth),int(frameWidth)),dtype=np.uint8)

    for i in range(int(frameAmount)):

        _,frame = cap.read()

        video[i,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pbar.update(1)

    cap.release()
    pbar.close()

    return video,frameRate


def arUCO_video_detection(video, show_figure=True, figure_scale = 3, adv_param = True, binarization=True, thresh=80, otsu=False, sharpening=False, dilate=False, erode=False):
    '''
    Displacement monitoring detecting ArUco marker

    Arguments:
        video {array} -- selected video as matrix
        figure_scale {int} -- figure scaling
        adv_param {boolean} -- advance paramiters enabling
        binarization {boolean} -- binarization enabling
        thresh {int} -- thresholding
        otsu {boolean} -- Otsu filtering enabling
        sharpening {boolean} -- Sharpening filtering enabling
        dilate {boolean} -- Dilatation filteirng enabling
        erode {boolean} -- Erosion filteirng enabling

    Returns: 
        full_mrks_pos {array} -- position of the marker in each computed frame
        sample_frame {array} -- sample frame
        id_detected {array} -- id of the markers
    '''
    pbar = tqdm(total=video.shape[0],desc='Marker detection...',colour='red') 

    id_detected = []
    full_mrks_pos = {}
    mrks_count = 0

    for i in range(video.shape[0]):
        
            mrks_frame,mrks_pos,ids,frame_mod = _arUCO_detection(video[i,:,:],adv_param,binarization,thresh,otsu,sharpening,dilate,erode)
            
            for x in np.ravel(ids):
                if x not in id_detected:
                    id_detected.append(x)
            
            if len(id_detected) > mrks_count:     
                sample_frame = mrks_frame
                mrks_count +=1
                    
            full_mrks_pos['%i'%(i)] = mrks_pos

            if show_figure == True:
                cv2.namedWindow('Frame', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Frame', mrks_frame)
                cv2.resizeWindow('Frame', mrks_frame.shape[1]//figure_scale, mrks_frame.shape[0]//figure_scale)

                cv2.namedWindow('Filter Frame', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('Filter Frame', frame_mod)
                cv2.resizeWindow('Filter Frame', frame_mod.shape[1]//figure_scale, frame_mod.shape[0]//figure_scale)
                    
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            pbar.update(1)
                
    pbar.close()
    cv2.destroyAllWindows()

    return full_mrks_pos, sample_frame, id_detected


def _arUCO_detection(frame, adv_param, binarization, thresh, otsu, sharpening, dilate, erode):
    """
    Function to apply image filter and for detect aruco markers
    
    Arguments:
        frame {array} -- frame to elaborate.
        adv_param {bool} -- enable/disable advanced parameters for arUCO detection.
        thresh {int} -- threshold for image binarization.
        sharpening {bool} -- enable/disable sharpening filter (default=False).
        dilate {bool} -- enable/disable dilate filter (default=False).
        erode {bool} -- enable/disable erode filter (default=False).
    
    Returns:
        mrks_frame {array} -- frame with drawn markers and realtive IDs.
        mrks_pos {dict} -- dictionary with the coordinate of the markers center position for each markers ID
        ids --
        frame_mod --  
    """
    
    frame_mod = frame

    #SHARPENING FILTER

    if sharpening == True:
        sh_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame_mod = cv2.filter2D(frame_mod, -1, sh_kernel)
        
    #DILATE FILTER

    if dilate == True:
        dil_kernel = np.ones((3, 3), np.uint8) 
        frame_mod = cv2.dilate(frame_mod, dil_kernel,iterations=1)

    #ERODE FILTER

    if erode == True:
        dil_kernel = np.ones((3, 3), np.uint8) 
        frame_mod = cv2.erode(frame_mod, dil_kernel,iterations=1)
 
    #BINARIZATION FILTER (BINARY+OTSU)

    if binarization == True:
        if otsu == True:
            _,frame_mod = cv2.threshold(frame_mod,thresh,250,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            _,frame_mod = cv2.threshold(frame_mod,thresh,250,cv2.THRESH_BINARY)
    
    aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
    parameters =  aruco.DetectorParameters_create()
    
    #ADVANCED PARAMETERS

    if adv_param == True:
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 21
        parameters.adaptiveThreshWinSizeStep = 8
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.polygonalApproxAccuracyRate = 0.05
        parameters.minCornerDistanceRate = 0.05
        parameters.minDistanceToBorder = 5
        parameters.minMarkerDistanceRate = 0.05
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 30
        parameters.cornerRefinementMinAccuracy = 0.01
        parameters.markerBorderBits = 1
        parameters.perspectiveRemovePixelPerCell = 8
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
        parameters.maxErroneousBitsInBorderRate = 0.04
        parameters.minOtsuStdDev = 3.0
        parameters.errorCorrectionRate = 0.6
    
    corners, ids, _ = aruco.detectMarkers(frame_mod, aruco_dict, parameters=parameters)

    # CENTERS COORDINATE
    mrks_pos = {}
    pos = np.zeros((1,2,len(corners)))

    try:
        if ids.any() != None:
            for i in range(len(corners)):

                C1 = (corners[i][0][0][0], corners[i][0][0][1])
                C2 = (corners[i][0][1][0], corners[i][0][1][1])
                C3 = (corners[i][0][2][0], corners[i][0][2][1])
                C4 = (corners[i][0][3][0], corners[i][0][3][1])
                
                pos[0,0,i] = np.mean([C1[0],C2[0],C3[0],C4[0]])
                pos[0,1,i] = np.mean([C1[1],C2[1],C3[1],C4[1]])
                
                mrks_pos['ID:%i'%(ids[i][0])] = pos[:,:,i]

                if i==0:
                    mrks_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
                mrks_frame = cv2.line(mrks_frame, C1, C2, (0,0,255), 3) 
                mrks_frame = cv2.line(mrks_frame, C2, C3, (0,0,255), 3) 
                mrks_frame = cv2.line(mrks_frame, C3, C4, (0,0,255), 3) 
                mrks_frame = cv2.line(mrks_frame, C4, C1, (0,0,255), 3) 
                
                marks_frame = cv2.putText(mrks_frame, 'ID:%i'%(ids[i]), (int(C1[0])-20, int(C1[1])-20), cv2.FONT_HERSHEY_SIMPLEX,  
                                fontScale = 0.8, color=(0, 255, 0), thickness = 2, lineType=cv2.LINE_AA) 
                        
                P1 = np.array(C1)
                P2 = np.array(C2)
                P3 = np.array(C3)
                P4 = np.array(C4)
                
                dist1 = np.linalg.norm(P1-P2)
                dist2 = np.linalg.norm(P2-P3)
                dist3 = np.linalg.norm(P3-P4)
                dist4 = np.linalg.norm(P4-P1)

                med_mrk = np.mean((dist1,dist2,dist3,dist4))
                std_mrk = np.std((dist1,dist2,dist3,dist4),ddof=1)

                mrks_pos['ID:%i_aux'%(ids[i][0])] = [med_mrk,std_mrk]

        else:
            mrks_frame = frame

    except:

        mrks_frame = frame
     
    return mrks_frame, mrks_pos, ids, frame_mod


def spatial_calibration(mrk_length, mrks_pos, ID):
    """
    Spatial calibration from px to [mm] units
    
    Arguments:
        mrk_length {float} -- marker phisical length in [mm] units
        mrks_pos {array} -- frame by frame marker position matrix
        ID {int} -- id of the marker to be calibrated
    
    Returns:
        relative_disp_mm {array} -- 
        global_center_px -- 
        len_pixel_max -- 
    """ 

    cal_fact = np.zeros((len(mrks_pos),1))
    len_pixel = np.zeros((len(mrks_pos),1))

    j = 0 

    for i in range(len(mrks_pos)):
        try:
            length_pxl = mrks_pos.get('%i'%(i),{}).get('ID:%s_aux'%(ID))[0]
            cal_fact [j] = np.round(mrk_length/length_pxl,3)
            len_pixel [j] = length_pxl
            j+=1
        except:
            pass

    cal_fact = _reconstrution(cal_fact) 

    relative_disp_mm = np.zeros((len(mrks_pos),2))
    global_center_px = np.zeros((len(mrks_pos),2))

    j = 0 

    for i in range(len(mrks_pos)):
        try:
            relative_disp_mm[j] = mrks_pos.get('%i'%(i),{}).get('ID:%s'%ID)*cal_fact[j]
            global_center_px[j] = mrks_pos.get('%i'%(i),{}).get('ID:%s'%ID)
            j+=1
        except:
            pass
        
    len_pixel_max = np.max(len_pixel)

    relative_disp_mm[:,0] = _reconstrution(relative_disp_mm[:,0]) 
    relative_disp_mm[:,1] = _reconstrution(relative_disp_mm[:,1])

    global_center_px[:,0] = _reconstrution(global_center_px[:,0]) 
    global_center_px[:,1] = _reconstrution(global_center_px[:,1])

    disp_mean_value = np.mean(relative_disp_mm,axis=0)

    relative_disp_mm = relative_disp_mm-disp_mean_value

    return relative_disp_mm, global_center_px, len_pixel_max

def _reconstrution(d):
    '''
    Displacement reconstruction using boundary points when the position of the marker is not detected in a frame

    '''

    for i in range(0,d.shape[0]-1):
        if d[i] == 0:
            if i == 0:
                d[i] = d[i+1]
            elif i == d.shape[0]-1:
                d[i] = d[i-1]
            else:
                if d[i+1] != 0:
                    d[i] = np.mean([d[i-1],d[i+1]])
                else:
                    d[i] = d[i-1]
    return d

def plot_disp(disp, fps, ID, save=True):
    '''
    Displacement plotting
    
    Arguments:
        disp {array} -- marker displacement in [mm] units
        fps {float} -- franme rate
        ID {int} -- id of the marker to be plotted
        save {boolean} -- enabling plot save
    '''
    
    time = np.arange(0,(disp.shape[0])/fps,1/fps)

    hf_x = np.abs(np.max(disp[:,0])-np.min(disp[:,0]))/2
    hf_y = np.abs(np.max(disp[:,1])-np.min(disp[:,1]))/2
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time[:-1],disp[:-1,0],color='r',linewidth=0.8)
    #plt.axhline(max_disp_x,linestyle = '--',color = 'C1', zorder = 1)
    #plt.axhline(min_disp_x,linestyle = '--',color = 'C1', zorder = 1)            
    plt.title('ID:%s - Displacement\nx-axis\nhalf-amplitude=%.3f [px]'%(ID,hf_x),fontsize=7)
    #plt.title('%s - Displacement\nx-axis'%(ID),fontsize=8)
 
    plt.xlabel('time [s]',fontsize=7)
    plt.ylabel('[px]',fontsize=7)

    plt.tick_params(axis="x", labelsize=6)
    plt.tick_params(axis="y", labelsize=6)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    plt.subplot(2,1,2)
    plt.plot(time[:-1],disp[:-1,1],color='b',linewidth=0.8)
    #plt.axhline(max_disp_y,linestyle = '--',color = 'C1', zorder = 1)
    #plt.axhline(min_disp_y,linestyle = '--',color = 'C1', zorder = 1)            
    plt.title('y-axis\nhalf-amplitude=%.3f [px]'%(hf_y),fontsize=7)
    #plt.title('y-axis',fontsize=8)

    plt.xlabel('time [s]',fontsize=7)
    plt.ylabel('[px]',fontsize=7)

    plt.tick_params(axis="x", labelsize=6)
    plt.tick_params(axis="y", labelsize=6)
    
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    
    plt.tight_layout()

    if save == True:
        plt.savefig('disp_time_%s.png'%ID, dpi=300)