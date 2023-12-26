import cv2 

if __name__=='__main__':
    cap = cv2.VideoCapture('./Hai_Ba.mp4')
    
    if (cap.isOpened()== False):  
        raise

    idx_frame = 1
    while(cap.isOpened()): 
    # Capture each frame 
        ret, frame = cap.read() 
        if ret == True: 
        # Display the resulting frame
            cv2.imwrite("../hai_ba/{}.jpg".format(idx_frame), frame)
        else:
            break
        idx_frame += 1
    cap.release()
        