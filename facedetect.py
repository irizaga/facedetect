import cv2
import sys

cascPath = #cv data path containing xml files; apply \\ instead of \ for folders
faceCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cascPath + 'haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    eyes = eyesCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(5, 5),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )	
	
    smile = smileCascade.detectMultiScale(
        gray,
        scaleFactor=1.9,
        minNeighbors=15,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
	
    # Draw a red rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
	# Draw a blue rectangle around the eyes
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
	# Draw a green rectangle around a smile
    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()