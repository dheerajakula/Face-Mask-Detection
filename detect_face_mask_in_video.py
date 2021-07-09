from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
#from imutils.video import VideoStream
import face_recognition
import numpy as np
#import argparse
#import imutils
#import time
from csv import reader
import time
import glob
import cv2
import os

def get_final_face_names():

    known_face_names = []
    known_face_encodings = []

    cur_direc = os.getcwd()

    path = os.path.join( cur_direc, 'faces/' )
    list_of_files = [ f for f in glob.glob( path + '*.jpg' ) ]

    for i in range( 0 , len( list_of_files ) ):
        
        img = face_recognition.load_image_file( list_of_files[ i ] )
        
        known_face_encodings.append( face_recognition.face_encodings( img )[ 0 ] )
        
        name = list_of_files[ i ].replace( cur_direc + '/faces/' , '' )
        
        known_face_names.append( name )
        
    cap = cv2.VideoCapture( cur_direc + '/input_video.mp4' )

    model = load_model( cur_direc + '/mask_detector.model' )

    prototxtPath = os.path.sep.join( [ 'face_detector' , "deploy.prototxt" ] )
    weightsPath = os.path.sep.join( [ 'face_detector' , "res10_300x300_ssd_iter_140000.caffemodel" ] )
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    fps = cap.get( cv2.CAP_PROP_FPS )

    frame_count = int( cap.get( cv2.CAP_PROP_FRAME_COUNT ) )
    interval = int( fps )

    final_face_names = set()

    for i in range( 0 , frame_count , 4 ):
        
        if cap.isOpened():
            
            print( i )
            
            ret , frame = cap.read()
            
            if ret == False:
                
                break
            
            input_img = frame
            
            (h, w) = input_img.shape[ 0 : 2 ]

            blob = cv2.dnn.blobFromImage( input_img , 1.0, (300, 300),
            (104.0, 177.0, 123.0) )

            net.setInput(blob)
            detections = net.forward()

            #print( detections )

            box_lst = []
            unmasked_box_lst = []

            for i in range( 0 , detections.shape[ 2 ] ):
        
                confidence = detections[0, 0, i, 2]
        
                if confidence > 0.5:
            
                    box = detections[0, 0, i, 3:7] * np.array( [ w, h, w, h ] )
                    ( startX, startY, endX, endY ) = box.astype("int")
            
                    ( startX, startY ) = ( max( 0 , startX ), max( 0 , startY ) ) 
                    ( endX, endY ) = ( min( w - 1 , endX ), min( h - 1 , endY ) )
            
                    face = input_img[ startY : endY , startX : endX ]
                    face = cv2.cvtColor( face , cv2.COLOR_BGR2RGB )
                    face = cv2.resize( face , (224, 224) )
                    face = img_to_array( face )
                    face = preprocess_input( face )
                    face = np.expand_dims( face , axis = 0 )
            
                    ( mask, withoutMask ) = model.predict(face)[ 0 ]
            
                    label = "Mask" if mask > withoutMask else "No Mask"
            
                    box_loc = ( startX , endY , endX , startY ) 
            
                    box_lst.append( [ box_loc , label , 'Vaccinated' ] )
            
                    if label == "No Mask":
                
                        unmasked_box_lst.append( ( startX , endY , endX , startY ) )
                        
            #vaccine_status_lst = []

            #small_frame = cv2.resize( input_img , (0, 0) , fx = 0.25, fy = 0.25 )
            #rgb_small_frame = input_img[:, :, ::-1]

            #face_locations = face_recognition.face_locations( input_img )

            face_names = []

            face_locations = face_recognition.face_locations( input_img )
            unknown_face_encodings = face_recognition.face_encodings( input_img , face_locations )

            for box , face_encoding in zip( unmasked_box_lst , unknown_face_encodings ):
        
                matches = face_recognition.compare_faces( known_face_encodings, face_encoding )

                name = "Unknown"
        
                face_distances = face_recognition.face_distance( known_face_encodings, face_encoding )
        
                best_match_index = np.argmin( face_distances )
        
                if matches[ best_match_index ]:
            
                    name = known_face_names[ best_match_index ]
            
                    face_names.append( ( box , name ) )

            for face_name in face_names:
        
                #face_name_lst = face_name.split('/')
        
                #face_name = face_name_lst[ -1 ][ 0 : len( face_name_lst[ - 1 ] ) - 4 ]
        
                name_of_person = face_name[ 1 ][ 0 : len( face_name[ 1 ] ) - 4 ]
        
                with open( 'Vaccinated_people.csv' , 'r' ) as read_obj:
            
                    csv_reader = reader( read_obj )
        
                    for row in csv_reader:
        
                        if row[ 0 ] == name_of_person and row[ 1 ] == 'Unvaccinated':
                    
                            final_face_names.add( name_of_person )
                    
    output_file_name = cur_direc + '/output_file.txt'

    f = open( output_file_name , "w" )

    f.write( str( final_face_names ) )
        
    f.close()
    return str(final_face_names)