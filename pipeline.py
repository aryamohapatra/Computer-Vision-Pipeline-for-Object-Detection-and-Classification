# -*- coding: utf-8 -*-
"""
Research Topics in AI - Assignment 2

Authors: David O'Callaghan and Arya Mohapatra

This is the main script for running the pipeline.
Refer to the README for run instructions.
"""
import sys
sys.path.append('keras-yolo3-master')

import cv2
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiprocessing import Process, Queue
from PIL import Image, ImageFont, ImageDraw
from yolo import YOLO 
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score

FRAME_RATE = 30 # frames per second

# To allow user to select the query to run
DO_ANALYSIS = False
QUERY = 'Q3'
try:
    QUERY = sys.argv[1]
except:
    QUERY = 'Q3'
if QUERY == 'Q3':
    DO_ANALYSIS = True

COLOUR_RANGES = {
        'Black':  (np.array([  0,   0,  35]), np.array([255,  30,  65])),
        'Silver': (np.array([  0,   0, 115]), np.array([255,  35, 200])),
        'Red':    (np.array([155,  40,  60]), np.array([190, 255, 200])),
        'White':  (np.array([  0,   0, 210]), np.array([255,  35, 255])),
        'Blue':   (np.array([ 85,  25,  65]), np.array([115, 255, 255]))
        }

results = {} # To store query results from each frame
default_result_row = {'Sedan': {'Black': 0,
                                'Silver': 0,
                                'Red': 0,
                                'White': 0,
                                'Blue': 0},
                      'Hatchback': {'Black': 0,
                                    'Silver': 0,
                                    'Red': 0,
                                    'White': 0,
                                    'Blue': 0},
                      'Total Cars': 0}

def video_streamer(frame_queue):
    print("Started video_streamer()")
    # Read until video is completed
    video_capture = cv2.VideoCapture('./video.mp4')
    while(video_capture.isOpened()):
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret == True:
            # Display the frame
            #cv2.imshow('Frame',frame)
            
            # Add frame to queue
            frame_queue.put(frame)
            
            # Delay by frame rate
            time.sleep(1 / FRAME_RATE)
            
            # Break the loop if "Q" is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else: 
            break
            
    # When everything done, release the video capture object
    video_capture.release()
    # Closes all the frames
    cv2.destroyAllWindows()

def frame_processor(frame_queue):
    print("Started frame_processor()")
    global QUERY
    global DO_ANALYSIS
    
    frame_number = 0
    
    # Instantiate YOLO 
    # Use tiny YOLO weights
    yolo = YOLO(model_path='./keras-yolo3-master/model_data/tiny_yolo.h5',
                anchors_path='./keras-yolo3-master/model_data/tiny_yolo_anchors.txt',
                classes_path='./keras-yolo3-master/model_data/coco_classes.txt') 
    
    # Instantiate car hatchback classifier
    model = keras.models.load_model('./mobilenet_cars.h5')
    
    # to store query statistics
    q1_stats = []
    q2_stats = []
    q3_stats = []
    
    # Font for annotations
    font1 = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=10)
    font2 = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=14)
    
    # Need to store frames for creating the video
    frames_out = []
    
    pipeline_start_time = time.time()
    while True:
        # Get frame from queue
        frame = frame_queue.get()
        
        start_time = time.time()
        frame_number += 1
        print(f'\nFrame {frame_number}')
        video_time = frame_number * (1 / 25) # 25 is the true frame rate of the video
        print(f'Video Time {round(video_time, 2)} s')
        new_result = copy.deepcopy(default_result_row)
        
        if QUERY in ['Q1','Q2','Q3']:
            # ------------------------ STAGE 1 ------------------------
            # Use tiny YOLO to detect car objects
            car_boxes, car_box_dimss = get_car_bounding_boxes(frame, yolo)
            new_result['Total Cars'] = len(car_boxes) # Number of cars in the frame
            q1_stats.append([frame_number, time.time() - start_time])
            # ---------------------------------------------------------
        if QUERY in ['Q2','Q3']:
            # ------------------------ STAGE 2 ------------------------
            car_types = []
            for car_box in car_boxes:
                # Use MobileNet to classify sedan or hatchback
                car_type = get_car_type(car_box, model)
                car_types.append(car_type)
            if len(car_boxes) > 0:
                q2_stats.append([frame_number, time.time() - start_time])
            # ---------------------------------------------------------
        if QUERY in ['Q3']:
            # ------------------------ STAGE 3 ------------------------
            car_colours = []
            for car_box in car_boxes:
                # Use openCV to detect the colour of the cars
                car_colours.append(get_car_colour(car_box))
            if len(car_boxes) > 0:
                q3_stats.append([frame_number, time.time() - start_time])
            image = Image.fromarray(frame)
            # Create annotations for video frame
            for car_box_dims, car_type, car_colour in zip(car_box_dimss, car_types, car_colours):
                draw = ImageDraw.Draw(image)
                if car_type == 'Sedan':
                    box_colour = (251, 14, 14) # Red
                else:
                    box_colour = (14, 38, 251) # Blue
                for i in range(3):
                    # Draw thick box around car
                    draw.rectangle([car_box_dims['left'] + i, car_box_dims['top'] + i, car_box_dims['right'] - i, car_box_dims['bottom'] - i], outline=box_colour)
                # Add label to top left of box
                label = f"{car_colour} {car_type}"
                label_width, label_height = draw.textsize(label, font1)
                draw.rectangle([car_box_dims['left'], car_box_dims['top'], car_box_dims['left']+label_width, car_box_dims['top'] + label_height], fill=box_colour)
                draw.text((car_box_dims['left'],car_box_dims['top']), label, font=font1, fill=(0,0,0,128))
            
            # Add label to top left with count of cars
            draw = ImageDraw.Draw(image)
            label = f"Car Count: {len(car_boxes)}"
            label_width, label_height = draw.textsize(label, font2)
            draw.rectangle([0, 0, label_width, label_height], fill=(0,0,0)) # Black Rectangle
            draw.text((0,0), label, font=font2, fill=(255,255,255,128)) # White text
            frame_annot = np.asarray(image)
            frames_out.append(frame_annot) # Need to store frames for creating the video
            cv2.imshow("Object Detection", frame_annot)
            # ---------------------------------------------------------
        process_time = time.time() - start_time
        print(f'{round(process_time, 3)} s to process ...')
        
        # Info message
        if QUERY == 'Q1':
            for _ in car_boxes:
                print(f'Q1 : Car detected')
        elif QUERY == 'Q2':
            for car_type in car_types:
                print(f'Q2 : Car of type {car_type} detected')
        else:
            for car_type, car_colour in zip(car_types, car_colours):
                print(f'Q3 : Car of type {car_type} and colour {car_colour} detected')
                new_result[car_type][car_colour] += 1
        
            results[frame_number] = copy.deepcopy(new_result)
            
            # Write results to file
            csv_string = f"\n{frame_number}, {new_result['Sedan']['Black']}, " + \
            f"{new_result['Sedan']['Silver']}, {new_result['Sedan']['Red']}, " + \
            f"{new_result['Sedan']['White']}, {new_result['Sedan']['Blue']}, " + \
            f"{new_result['Hatchback']['Black']}, {new_result['Hatchback']['Silver']}, " + \
            f"{new_result['Hatchback']['Red']}, {new_result['Hatchback']['White']}, " + \
            f"{new_result['Hatchback']['Blue']}, {new_result['Total Cars']}"
            with open('./results.csv', 'a') as f:
                f.write(csv_string)
        
        # Break the loop if "Q" is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):    
            break
        # Or if the queue is empty
        elif frame_queue.empty():
            break
        elif cv2.waitKey(25) & 0xFF == ord('1'):
            QUERY = 'Q1'
            DO_ANALYSIS = False
        elif cv2.waitKey(25) & 0xFF == ord('2'):
            QUERY = 'Q2'
            DO_ANALYSIS = False
        elif cv2.waitKey(25) & 0xFF == ord('3'):
            QUERY = 'Q3'
            DO_ANALYSIS = False
    throughput = time.time() - pipeline_start_time
    with open('./output.txt', 'w') as f:
        out_string = f'\nThroughput : {np.round(throughput, 4)} s\n'
        print(out_string)
        f.write(out_string)
    # Save event extraction times
    if QUERY == 'Q1':
        q1_stats = np.array(q1_stats)
        np.save("./q1_stats.npy", q1_stats)
    elif QUERY == 'Q2':
        q2_stats = np.array(q2_stats)
        np.save("./q2_stats.npy", q2_stats)
    else:
        # Save everything for Q3
        q1_stats = np.array(q1_stats)
        q2_stats = np.array(q2_stats)
        q3_stats = np.array(q3_stats)
        np.save("./q1_stats.npy", q1_stats)
        np.save("./q2_stats.npy", q2_stats)
        np.save("./q3_stats.npy", q3_stats)
        
        # Write video
        # Source https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/

        vw1 = cv2.VideoWriter('video_out_5fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (360,288))
        vw2 = cv2.VideoWriter('video_out_30fps.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, (360,288))
        for frame_out in frames_out:
            vw1.write(frame_out)
            vw2.write(frame_out)
        vw1.release()
        vw2.release()
        
def get_car_bounding_boxes(frame, yolo):
    # Convert to an image
    image = Image.fromarray(frame)
    
    # Use tiny YOLO to detect ANY object
    r_image, object_boxes, time_value = yolo.detect_image(image)
    
    # Convert the resulting image to an array
    result = np.asarray(r_image)
    
    # Display the resulting image
    #cv2.imshow("Object Detection", result)
    
    # Iterate through objects detected in the frame
    car_boxes = []
    car_box_dims = []
    for object_box in object_boxes:
        # If the object is classified as a car
        if object_box['class'] == 'car':
            #print('Car detected')
            car_box_dims.append(object_box)
            
            # Store the bounding box dimensions
            top = object_box['top']
            left = object_box['left']
            bottom = object_box['bottom']
            right = object_box['right']
            
            # get the bounding box of the object in the frame
            car_box = frame[top:bottom, left:right] # Pass this to next stage of the pipeline
            car_boxes.append(car_box)
            #cv2.imshow("Bounding box", car_box)
    return car_boxes, car_box_dims

def get_car_type(car_box, model):
    cartype = ''
    car_box_new = Image.fromarray(car_box)
    car_box_new = car_box_new.resize((160,160), Image.ANTIALIAS)
    result = np.asarray(car_box_new)
    result = np.expand_dims(result, axis=0)
    images = np.vstack([result])
    classes = model.predict(images, batch_size=10)
    #print(classes[0])
    if classes[0] > 0.5: # Threshold
        #print( " is a Sedan")
        cartype = 'Sedan'
    else:
        #print(" is a hatch")
        cartype = 'Hatchback'
    return cartype
    
def get_car_colour(car_box):
    # Convert BGR to HSV for thresholding
    hsv = cv2.cvtColor(car_box, cv2.COLOR_BGR2HSV)

    max_colour_pixels = 0
    for colour in COLOUR_RANGES:
        # Threshold the HSV image to get colours and apply masks to frames
        mask = get_mask(car_box, hsv, COLOUR_RANGES[colour])
        
        # Number of pixels that were not filtered by the mask
        colour_pixels = np.sum(mask == 255) / 255
        if colour_pixels > max_colour_pixels:
            car_colour = colour
            max_colour_pixels = colour_pixels        
            
    return car_colour

def get_mask(frame, hsv, colour_range):
    mask = cv2.inRange(hsv, *colour_range)
    #res = cv2.bitwise_and(frame, frame, mask = mask)
    return mask

def compute_results():
    names=['Frame Number', 'Sedan Black', 'Sedan Silver','Sedan Red', 
       'Sedan White', 'Sedan Blue', 'Hatchback Black', 'Hatchback Silver',
       'Hatchback Red', 'Hatchback White', 'Hatchback Blue', 'Total Cars']
    ground_truth = pd.read_excel("./Ground Truth (Assignment 2).xlsx", 
                                 skiprows=1,
                                 index_col=0,
                                 names=names)
    
    predictions = pd.read_csv("./results.csv",
                          skiprows=1,
                          index_col=0,
                          names=names)
    
    f1_scores = []
    # Q1 
    n_frames = len(ground_truth)
    y_true = ground_truth.loc[:,names[-1]].values
    y_pred = predictions.loc[:,names[-1]].values

    tp, fp, fn = get_tp_fp_fn(y_true, y_pred)

    # Calculate F1-score
    f1_scores.append(compute_f1score(tp, fp, fn))

    # Q2
    tp, fp, fn = 0, 0, 0
    true_values = ground_truth[ground_truth['Total Cars'].values != 0]
    pred_values = predictions[ground_truth['Total Cars'].values != 0]

    for cols in (names[1:6], names[6:11]): # Sedan then Hatchback
        y_true = true_values.loc[:,cols].sum(axis=1).values
        y_pred = pred_values.loc[:,cols].sum(axis=1).values
        
        tp_, fp_, fn_ = get_tp_fp_fn(y_true, y_pred)
        tp += tp_
        fp += fp_
        fn += fn_

    # Calculate F1-score
    f1_scores.append(compute_f1score(tp, fp, fn))

    # Q3
    tp, fp, fn = 0, 0, 0
    true_values = ground_truth[ground_truth['Total Cars'].values != 0]
    pred_values = predictions[ground_truth['Total Cars'].values != 0]

    for i in range(1,11): # For each type and colour
        col = names[i]
        y_true = true_values.loc[:,col].values
        y_pred = pred_values.loc[:,col].values
        
        tp_, fp_, fn_ = get_tp_fp_fn(y_true, y_pred)
        tp += tp_
        fp += fp_
        fn += fn_

    # Calculate F1-score
    f1_scores.append(compute_f1score(tp, fp, fn))
        
    plt.bar([f"Q{i}" for i in range(1,4)], (f1_scores))
    plt.title('Query F1 Scores')
    plt.grid(axis='y', alpha=0.5)
    plt.ylim([0,1])
    plt.show()
    
    # Print scores
    with open('./output.txt', 'a') as f:
        f.write('\nF1 Scores\n')
        f.write('---------\n')
        for i, f1 in enumerate(f1_scores):
            f.write(f'Q{i+1} : {round(f1, 3)}\n')

def get_tp_fp_fn(y_true, y_pred):
    tp, fp, fn = 0, 0, 0
    for i in range(len(y_true)):
        # Compute TPs, FPs and FNs
        if y_pred[i] <= y_true[i]:
            tp += y_pred[i]
            fp += 0
            fn += y_true[i] - y_pred[i]
        else:
            tp += y_true[i]
            fp += y_pred[i] - y_true[i]
            fn += 0
    return tp, fp, fn
        
def compute_f1score(tp, fp, fn):
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * (pre * rec) / (pre + rec)

def make_piecewise(stats):
    N = stats.shape[0]
    previous = stats[0,0]
    j = 1
    for _ in range(1, N):
        current = stats[j,0]
        if current != previous + 1:
            # Insert NaN for gaps frame that don't have data
            stats = np.insert(stats, j, np.array([np.nan, np.nan]), axis=0)
            j += 1
        previous = current
        j += 1
    return stats

if __name__=='__main__':
    # Create CSV file to store results
    with open('./results.csv', 'w') as f:
        csv_header = 'Frame Number, Sedan Black, Sedan Silver, Sedan Red, ' + \
        'Sedan White, Sedan Blue, Hatchback Black, Hatchback Silver, ' + \
        'Hatchback Red, Hatchback White, Hatchback Blue, Total Cars'
        f.write(csv_header)
    
    # Create a queue for storing video frames
    frame_queue = Queue()
    
    # Create 2 processes
    p1 = Process(target=frame_processor, args=(frame_queue,))
    p2 = Process(target=video_streamer, args=(frame_queue,))
    
    # Start both processes
    p1.start() #; time.sleep(15) # allow time for yolo to load
    p2.start()
    
    # Wait to finish
    p1.join()
    p2.join()
    
    if QUERY == 'Q3' and DO_ANALYSIS == True:
        # Compute F1-score for each query
        compute_results()
        
        # Plotting
        q1_stats = np.load("./q1_stats.npy")
        q2_stats = np.load("./q2_stats.npy")
        q3_stats = np.load("./q3_stats.npy")
        
        with open('./output.txt', 'a') as f:
            f.write('\nAverage extraction times\n')
            f.write('------------------------\n')
            for i, q_stats in enumerate([q1_stats, q2_stats, q3_stats]):
                average_extraction = np.round(np.mean(q_stats[:,1]), 4)
                std_extraction = np.round(np.std(q_stats[:,1]) , 4)
                f.write(f'Q{i+1} : {average_extraction} +- {std_extraction} s\n')
        
        q1_stats = make_piecewise(q1_stats)
        q2_stats = make_piecewise(q2_stats)
        q3_stats = make_piecewise(q3_stats)

        plt.plot(q1_stats[:,0], q1_stats[:,1], 'g', alpha=1, linewidth=1)
        plt.plot(q2_stats[:,0], q2_stats[:,1], 'b', alpha=0.7, linewidth=1)
        plt.plot(q3_stats[:,0], q3_stats[:,1] + 0.001, 'r', alpha=0.7, linewidth=1)
        plt.legend(['Q1', 'Q2', 'Q3'])
        plt.xlabel('Frame Number')
        plt.ylabel('Time (seconds)')
        plt.title('Query Extraction Time')
        #plt.ylim([0.1,0.26])
        plt.show()
