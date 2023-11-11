from flask import Flask, request, jsonify
from PIL import Image
import cv2
from datetime import datetime
import arabic_reshaper
from detection import PlateDetector
from ocr import PlateReader


app = Flask(__name__)
class lpdr:
    def __init__(self,path):
        self.image_path=path

    def apply_ocr(self):
            image, height, width, channels = self.reader.load_image('./tmp/plate_box.jpg')
            blob, outputs = self.reader.read_plate(image)
            boxes, confidences, class_ids = self.reader.get_boxes(outputs, width, height, threshold=0.3)
            segmented, plate_text = self.reader.draw_labels(boxes, confidences, class_ids, image)
            cv2.imwrite('./tmp/plate_segmented.jpg', segmented)
            return arabic_reshaper.reshape(plate_text)
    def process_image(self):
        self.detector = PlateDetector()
        self.detector.load_model("./weights/detection/yolov3-detection_final.weights", "./weights/detection/yolov3-detection.cfg")

        self.reader = PlateReader()
        self.reader.load_model("./weights/ocr/yolov3-ocr_final.weights", "./weights/ocr/yolov3-ocr.cfg")
        if (self.image_path == ""):
                return
            
        image, height, width, channels = self.detector.load_image(self.image_path)
        blob, outputs = self.detector.detect_plates(image)
        boxes, confidences, class_ids = self.detector.get_boxes(outputs, width, height, threshold=0.3)
        plate_img, LpImg = self.detector.draw_labels(boxes, confidences, class_ids, image)
        if len(LpImg):
            cv2.imwrite('./tmp/car_box.jpg', plate_img)
            cv2.imwrite('./tmp/plate_box.jpg', LpImg[0])        
            return self.apply_ocr()

@app.route('/', methods=['GET']) # le type de requêtes http autorisées.

def home():

     return '''<h1> Plate Detection-OCR API</h1>
            <p>Welcome to the API Please Send your image through a POST to /upload </p>'''

 

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        # Check if an image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400

        uploaded_image = request.files['image']
        image = Image.open(uploaded_image)
        path='./recieved/recieved'+str(datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))+'.jpg'
        #cv2.imwrite(path, image)
        im1 = image.save(path)
        plate=lpdr(path)
        result = plate.process_image()  # Call the function with the image
        #result = path
        print("result: ",result)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
