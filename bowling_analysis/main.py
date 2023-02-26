import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib
from cvzone.PlotModule import LivePlot

mpPose =mp.solutions.pose
pose = mpPose.Pose()
mp_drawing =mp.solutions.drawing_utils

def Detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = pose.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=1)
                             ) 
    
def Landmark_point(name):
    if name == "left_hand":
        list=(11,13,15)
    if name == "right_hand":
        list=(12,14,16)
    

    return list

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


def Detected_And_Draw(frame,name,draw=True,draw_all=True):
    
    ID_list=Landmark_point(name)
    
    p1=int(ID_list[0])
    p2=int(ID_list[1])
    p3=int(ID_list[2])
    
    img , results = Detection(frame)

    if not results.pose_landmarks:
        return None

    lmList = []
    for id,lm in enumerate (results.pose_landmarks.landmark):
        
        h,w,c =img.shape
        cx ,cy = int(lm.x*w),int(lm.y*h)
        lmList.append([id, cx, cy])

    # Get the landmarks
    x1, y1 = lmList[p1][1:]
    a =[x1,y1]
    x2, y2 = lmList[p2][1:]
    b =[x2,y2]
    x3,y3= lmList[p3][1:]
    c =[x3,y3]

    angle = calculate_angle(a,b,c)

    if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    if draw_all:
        bg = cv2.imread("data/bg.jpg")
        bg=cv2.resize(bg,(w,h))

        draw_styled_landmarks(bg, results)
        cv2.putText(bg, str(int(angle)), (x2 + 20, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
                                


    return img,bg,angle

def hconcat_resize(img_list, 
                   interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
      
    # image resizing 
    im_list_resize = [cv2.resize(img,
                       (int(img.shape[1] * h_min / img.shape[0]),
                        h_min), interpolation
                                 = interpolation) 
                      for img in img_list]
      
    # return final image
    return cv2.hconcat(im_list_resize)

cap = cv2.VideoCapture('data/4.mp4')
ptime =0

plot =LivePlot(480,720,[10,190])

while True :
    ret,img = cap.read()
    

    if ret:
        height, width = img.shape[:2]
        #print(height,width)


        frame,bg,angle = Detected_And_Draw(img,'right_hand',draw=True,draw_all=True)
                

        # calcucate the frame rate
        ctime=time.time()
        fps = 1/(ctime-ptime)
        ptime=ctime

        #cv2.putText(final_img,str(int(fps)),(70,80),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
        
        imgPlot =plot.update((angle))
        
        final_result =hconcat_resize([frame,bg,imgPlot])

        cv2.putText(final_result,"FPS : " +str(int(fps)), (int(width/3)+350,30), cv2.FONT_HERSHEY_PLAIN, 2,
                (255, 0, 0),2)
        cv2.putText(final_result, "Bowling hand angle : "+str(int(angle)),(int(width/3)+350,70), cv2.FONT_HERSHEY_COMPLEX, 1,
                (255, 0, 0),2)
        
        cv2.imshow('Result',final_result)
        height, width = final_result.shape[:2]
        #print(height,width)

    else:
        break   

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



