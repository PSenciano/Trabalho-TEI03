import face_recognition
import cv2
import numpy
import os

path = "./train/"

known_names = []
known_name_encodings = []

images = os.listdir(path)
for _ in images:
    image = face_recognition.load_image_file(path + _)
    image_path = path + _
    encoding = face_recognition.face_encodings(image)[0]

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print(known_names)

test_image = "./image/source.jpg"
image = cv2.imread(test_image)

face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = face_recognition.face_distance(known_name_encodings, face_encoding)
    best_match = numpy.argmin(face_distances)

    if matches[best_match]:
        name = known_names[best_match]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


cv2.imshow("Result", image)
cv2.imwrite("./result.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
