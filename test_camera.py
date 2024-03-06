import asyncio
import sqlite3
import pickle
import face_recognition
import cv2
import time
import numpy as np
import websockets
import json
from multiprocessing import Process

def process_camera(id, camera_name, websocket):
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()

        # Execute the SQL query
        query = """
            SELECT students.name, students.image_encodings 
            FROM students 
            LEFT JOIN subjects ON students.class_id = subjects.class_id 
            WHERE subjects.id = ? AND students.image_encodings IS NOT NULL;
        """

        cursor.execute(query, (id,))

        # Fetch the results
        results = cursor.fetchall()

        # Close the database connection
        conn.close()

        known_face_encodings = []
        known_face_names = []

        for row in results:
            name, image_encodings = row
            for encoding in pickle.loads(image_encodings):
                known_face_names.append(name)
                known_face_encodings.append(encoding)  

        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(camera_name)
        
        # Initialize some variables
        Thershold = 0.5
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        comparison_interval = 1  # Change this to your desired interval
        last_comparison_time = time.time()

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                raise Exception("Error capturing frame")

            current_time = time.time()
            if current_time - last_comparison_time >= comparison_interval:
                process_this_frame = True
                last_comparison_time = current_time
            else:
                process_this_frame = False

            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # small_frame = frame

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame, 2)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if face_distances[best_match_index] < Thershold:
                            name = known_face_names[best_match_index]
                            
                    face_names.append(name)
        

            process_this_frame = not process_this_frame

            # Send the face names to the client via the queue
            websocket.send(json.dumps(face_names))

            # Display the results (optional)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom), (right, bottom + 35), (255, 255, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom + 27), font, 1.0, (0, 0, 0), 1)

            # Display the resulting image
            cv2.imshow('Face Detection', frame)

            keyCode = cv2.waitKey(1) # wait for input to break the loop
            if cv2.getWindowProperty('Face Detection', cv2.WND_PROP_VISIBLE) <1:
                break

    except Exception as e:
        print(str(e))
        websocket.send({'error': str(e)})

    finally:
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

async def start_server(websocket, path):
    try:
        # Get the JSON data from the WebSocket connection
        async for message in websocket:
            data = json.loads(message)
            if data:
                # Extract the subject ID from the JSON data
                subject_id = data.get('subject_id')
                camera_name = data.get('ip')
                                       
                # Start camera capture in a separate process
                p = Process(target=process_camera, args=(subject_id, camera_name, websocket))
                p.start()



    except websockets.exceptions.ConnectionClosedOK:
        print("Client connection closed")


async def main():
    async with websockets.serve(start_server, "localhost", 8765):
        await asyncio.Future()  # Keep the server running indefinitely


if __name__ == "__main__":
    asyncio.run(main())

'''
Error:

connection handler failed
Traceback (most recent call last):
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\site-packages\websockets\legacy\server.py", line 236, in handler     
    await self.ws_handler(self)
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\site-packages\websockets\legacy\server.py", line 1175, in _ws_handler
    return await cast(
  File "c:/Users/Abdul Rafey/OneDrive/Desktop/electron app/electron-quick-start/python/camera.py", line 31, in start_server
    p.start()
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\process.py", line 121, in start
    self._popen = self._Popen(self)
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\context.py", line 224, in _Popen
    return _default_context.get_context().Process._Popen(process_obj)
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\context.py", line 327, in _Popen
    return Popen(process_obj)
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\popen_spawn_win32.py", line 93, in __init__
    reduction.dump(process_obj, to_child)
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\reduction.py", line 60, in dump
    ForkingPickler(file, protocol).dump(obj)
TypeError: cannot pickle '_asyncio.Future' object
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\spawn.py", line 107, in spawn_main
    new_handle = reduction.duplicate(pipe_handle,
  File "C:\Users\Abdul Rafey\AppData\Local\Programs\Python\Python38\lib\multiprocessing\reduction.py", line 79, in duplicate
    return _winapi.DuplicateHandle(
OSError: [WinError 6] The handle is invalid
'''
