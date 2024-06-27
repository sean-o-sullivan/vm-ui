import csv
import os
from flask import Flask, jsonify, request, render_template, url_for, abort, current_app, session
from werkzeug.utils import secure_filename

from embedding import get_stylometric_embedding  # Custom import from your project


app = Flask(__name__)
#app.secret_key= ''
# ------------------- activate verifymeenv to run -------------------------------------- #

# HTML template for the form, with updated styles

#            background-color: #f0f0f0;


app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Use a directory named 'uploads' in the current working directory
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.errorhandler(Exception)
def handle_exception(e):
    current_app.logger.error(f'An error occurred: {e}')
    return jsonify({'error': 'An internal error occurred'}), 500






@app.route('/append', methods=['POST'])
def append_embedding():
    print("starting append method")
    if not request.json or 'csvFile' not in request.json:
        abort(400, description="Invalid data")
    
    # Extract JSON datapy
    data = request.json
    csv_filename = data['csvFile']
    author_id = data['authorId']
    embedding = data['embedding']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

    # Open the CSV file in append mode ('a') and write the new data
    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([author_id, embedding])
        return jsonify({'message': 'Embedding appended successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/append', methods=['POST'])
def append_to_csv():
    # Assuming JSON input with 'name' and 'age' fields
    data = request.json
    file_path = 'data.csv'
    
    # Check if the file exists. If not, create it and write the header
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['name', 'age'])
            writer.writeheader()
    
    # Open the file in append mode and write the new row
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'age'])
        writer.writerow({'name': data['name'], 'age': data['age']})
    
    return {'message': 'Data appended successfully'}


@app.route('/save_embedding', methods=['POST'])
def save_embedding():
    """
    Saves the embedding to a specified CSV file.
    Expects JSON data containing 'csvFile', 'authorId', and 'embedding'.
    """

    if not request.json or 'csvFile' not in request.json:
        abort(400, description="Invalid data")

    data = request.json
    csv_filename = data['csvFile']
    author_id = data['authorId']
    embedding = data['embedding']

    # Construct the full path to the CSV file
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

    # Open the CSV file in append mode ('a') and write the new data
    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([author_id, embedding])
        return jsonify({'message': 'Embedding saved successfully.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    




@app.route('/get_author_ids', methods=['POST'])
def get_author_ids():

    def get_existing_author_ids(csv_file_path):
        print(f'The filepath is: {csv_file_path}')
        author_ids = set()
        try:
            with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                rows = [row for row in reader]
            columns = list(zip(*rows))

            for item in columns[1]:
                author_ids.add(item)  # Assuming the author_id is in the second column

        except FileNotFoundError:
            pass  # If the file doesn't exist, just return an empty set
        print(author_ids)
        return list(author_ids)

    data = request.json
    csv_file_path = data['csvFile']
    print(get_existing_author_ids(csv_file_path))
    author_ids = get_existing_author_ids(csv_file_path)
    print(jsonify(author_ids))
    return jsonify(author_ids)



@app.route('/', methods=['GET', 'POST'])
def embedding():
    if request.method == 'POST':
        mytext = request.form['mytext']
        result = get_stylometric_embedding(mytext)
        return result 

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')




@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    # This example assumes you've correctly set app.config['UPLOAD_FOLDER']
    if 'csvFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Optionally, process the CSV file here or just save its path for later processing
    return jsonify({'message': 'File uploaded successfully', 'filename': filename})


@app.route('/list_csv_files', methods=['GET'])
def list_csv_files():
    csv_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.csv')]
    return jsonify(csv_files)

@app.route('/select_csv', methods=['POST'])
def select_csv():
    selected_csv = request.form['existingCsvFiles']
    session['selected_csv'] = select_csv  #Storing the selected filename in the session
    return f'The selected CSV is {selected_csv}'




#handle uploaded csv

#def allowed_file(fileaname):
#    return '.' in fileaname and \
#        fileaname.rsplit(',',1)
#[1].lower() in {'csv'}


if __name__ == '__main__':
    app.run(debug=True)
