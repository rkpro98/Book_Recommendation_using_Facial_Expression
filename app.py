from flask import Flask, render_template, Response, url_for, request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_bootstrap import Bootstrap
import numpy as np
import cv2
import cvlib as cv
import csv


model = load_model(r'C:\Users\rkpro\Desktop\test_folder\test_project_folder\model.model')

app = Flask(__name__, static_url_path='/static')
app.config['STATIC_FOLDER'] = 'static'
file= []
def gen_frames():
    webcam = cv2.VideoCapture(0)
    classes = ['angry', 'happy', 'neutral', 'sad']
    
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            face, confidence = cv.detect_face(frame)
            for idx, f in enumerate(face):
                startX, startY = f[0], f[1]
                endX, endY = f[2], f[3]
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                face_crop = np.copy(frame[startY:endY, startX:endX])
                face_crop = cv2.resize(face_crop, (60, 60))
                face_crop = face_crop.astype("float") / 255.0
                face_crop = img_to_array(face_crop)
                face_crop = np.expand_dims(face_crop, axis=0)

                conf = model.predict(face_crop)[0]
                idx = np.argmax(conf)
                label = classes[idx]
                text='nothing'
                

                if(label!=text):
                    file.append(label)
                text=label

                label = "{}".format(label)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
    webcam.release()
    cv2.destroyAllWindows()
books=[]
def read_books_csv():
    books.clear()
    with open('books.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader) # Skip the header row
        for row in reader:
            books.append({
                'title': row[0],
                'author': row[1],
                'ratings': row[2],
                'description': row[3],
                'sentiment': row[4]
            })
    return books

def get_book_by_title(title):
    for book in books:
        if book['title'] == title:
            return book
    return None

@app.route('/books/<emotion>')
def display_books(emotion):
    sentiment_mapping = {
        'happy': 'fear',
        'angry': 'joy',
        'sad': 'surprise',
        'neutral': 'sadness'
    }
    
    
    books = read_books_csv()
    filtered_books = [book for book in books if book['sentiment'] == sentiment_mapping[emotion]]
    return render_template('books.html', emotion=emotion, books=filtered_books)

@app.route('/book_detail/<title>')
def book_detail(title):
    # Retrieve the book object from the books list using the title
    book = get_book_by_title(title)
    if book is None:
        return 'Book not found'
    rating = float(book['ratings'])

    # Render the book_detail.html template with the selected book object
    return render_template('book_detail.html', book=book, rating=rating, file=file)

@app.route('/')
def emotion():
    return render_template('video.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/search', methods=['POST'])
def search():
    #query = request.form['search_query']
    #results = search_database(query)
    return render_template('search_results.html')#, results=results)

#def search_database(query):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT * FROM books WHERE title LIKE ? OR author LIKE ?", ('%'+query+'%', '%'+query+'%'))
    results = c.fetchall()
    conn.close()
    return results

@app.route("/capture")
def capture():
    return render_template('detected.html', file=file)



@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)