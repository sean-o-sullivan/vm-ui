import csv
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for, abort, current_app, session
from werkzeug.utils import secure_filename
from datetime import datetime 

from embedding import Embedding  
from decision import predict_author

app = Flask(__name__)

# ------------------- activate verifymeenv to run -------------------------------------- #

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Use a directory named 'uploads' in the current working directory
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.errorhandler(Exception)
def handle_exception(e):
    current_app.logger.error(f'An error occurred: {e}')
    return jsonify({'error': 'An internal error occurred'}), 500


@app.route('/append', methods=['POST'])
def append_embedding():
    if not request.json or 'authorId' not in request.json or 'embedding' not in request.json:
        abort(400, description="Invalid data: JSON must contain 'authorId' and 'embedding'.")

    data = request.json
    author_id = data['authorId']
    embedding = data['embedding'].replace('\n', '').replace('\r', '').replace(' ', '').strip('"')
    testname = data['testName']

    # Get current timestamp in your preferred format
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')

    try:
        # Attempt to load the CSV into a DataFrame, or create a new one if the file does not exist
        try:
            df = pd.read_csv(file_path)
            # Determine the next embeddingID by adding 1 to the max currently in use
            next_embedding_id = df['EmbeddingID'].max() + 1
        except FileNotFoundError:
            # Define a new DataFrame with appropriate columns if the file doesn't exist
            df = pd.DataFrame(columns=['EmbeddingID', 'TestName', 'AuthorID', 'Embedding', 'Timestamp'])
            next_embedding_id = 1
        
        # Create a new DataFrame for the row to be appended
        new_row_df = pd.DataFrame({
            'EmbeddingID': [next_embedding_id], 
            'TestName': [testname],
            'AuthorID': [author_id], 
            'Embedding': [embedding], 
            'Timestamp': [timestamp]
        })
        
        # Use pd.concat to append the new row
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Save the updated DataFrame back to the CSV
        df.to_csv(file_path, index=False)
        current_app.logger.info('Embedding appended successfully.')
        return jsonify({'message': 'Embedding appended successfully.'})
    except Exception as e:
        current_app.logger.error(f"Failed to append data: {e}")
        return jsonify({'error': str(e)})




@app.route('/delete_embedding_result/<test_id>', methods=['POST'])
def delete_embedding_result(test_id):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
    try:
        df = pd.read_csv(csv_file_path)
        # Filter out the row to delete
        df = df[df['EmbeddingID'] != int(test_id)]
        df.to_csv(csv_file_path, index=False)
        return jsonify({'message': 'Embedding deleted successfully.'})
    except FileNotFoundError:
        return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/delete_test_result/<test_id>', methods=['POST'])
def delete_test_result(test_id):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    try:
        df = pd.read_csv(csv_file_path)
        # Filter out the row to delete
        df = df[df['TestID'] != int(test_id)]
        df.to_csv(csv_file_path, index=False)
        return jsonify({'message': 'Test result deleted successfully.'})
    except FileNotFoundError:
        return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500





@app.route('/next_test_id', methods=['GET'])
def get_next_test_id():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            next_test_id = int(df['TestID'].max()) + 1
        else:
            next_test_id = 1
    except Exception as e:
        app.logger.error(f"Error accessing results.csv: {e}")
        return jsonify({'error': 'Failed to access results data.'}), 500
    

    print(f'Next Test ID: {next_test_id}')  # Debugging line
    return jsonify({'next_test_id': next_test_id})




@app.route('/append_result', methods=['POST'])
def append_result():
    if not request.json or 'authorId' not in request.json or 'result' not in request.json:
        abort(400, description="Invalid data: JSON must contain 'authorId' and 'result'.")

    data = request.json
    #test_name = data['testname']
    test_name =" "
    author_id = data['authorId']
    result = data['result']

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')

    try:
        try:
            df = pd.read_csv(file_path)
            next_result_id = df['TestID'].max() + 1
        except FileNotFoundError:
            df = pd.DataFrame(columns=['TestID', 'Testname','AuthorID', 'Result', 'Timestamp'])
            next_result_id = 1
        
        new_row_df = pd.DataFrame({
            'TestID': [next_result_id], 
            'Testname': [test_name], 
            'AuthorID': [author_id], 
            'Result': [result], 
            'Timestamp': [timestamp]
        })
        
        # Use pd.concat to append the new row
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Save the updated DataFrame back to the CSV
        df.to_csv(file_path, index=False)
        current_app.logger.info('Embedding appended successfully.')
        return jsonify({'message': 'Embedding appended successfully.'})
    except Exception as e:
        current_app.logger.error(f"Failed to append data: {e}")
        return jsonify({'error': str(e)})
    


@app.route('/count_embeddings/<author_id>', methods=['GET'])
def count_embeddings(author_id):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
    count = 0
    try:
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[2].lower() == author_id.lower():
                    count += 1
    except FileNotFoundError:
        pass  # It's okay if the file doesn't exist yet
    return jsonify({'count': count})



@app.route('/get_author_ids', methods=['GET'])
def get_author_ids():
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
    
    try:
        df = pd.read_csv(csv_file_path)
        author_ids = sorted(df['AuthorID'].unique().tolist())
    except FileNotFoundError:
        author_ids = []

    return jsonify(author_ids)



@app.route('/get_author_results/<author_id>', methods=['GET'])
def get_author_results(author_id):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')
    
    try:
        df = pd.read_csv(csv_file_path)
        # Ensure author_id is of the correct type, e.g., string or int, as needed
        filtered_df = df[df['AuthorID'] == author_id]
        results_list = filtered_df[['TestID','Timestamp', 'Result']].to_dict(orient='records') # we need to consider whether we show the testname when we list the test results.
        return jsonify(results_list)
    except FileNotFoundError:
        return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/get_author_embeddings/<author_id>', methods=['GET'])
def get_author_embeddings(author_id):
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
    
    try:
        df = pd.read_csv(csv_file_path)
        # Ensure author_id is of the correct type, e.g., string or int, as needed
        filtered_df = df[df['AuthorID'] == author_id]
        results_list = filtered_df[['EmbeddingID', 'TestName','Timestamp', 'Embedding']].to_dict(orient='records')
        return jsonify(results_list)
    except FileNotFoundError:
        return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    

@app.route('/', methods=['GET', 'POST'])
def embedding():
    if request.method == 'POST':
        mytext = request.form['mytext'] 
        result = Embedding(mytext, categories=5)  
        return jsonify(result) 
    
    return render_template('index_copy_2.html')


@app.route('/predict_authorship', methods=['GET', 'POST'])
def predict_authorship():
    print('hello, this is the flask method for prediction')
    if request.method == 'POST':
        try:
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
            data = request.get_json() 
            authorID = data['authorId'] 
            check = data['embedding']

            with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  
                rows_list = []

                for row in reader:
                    if row[2] == authorID:
                        cleaned_embedding = row[3].replace('[', '').replace(']', '').replace('"', '').strip()
                        embedding_list = [float(num) for num in cleaned_embedding.split(',')]
                        print(f"Parsed embedding list length: {len(embedding_list)}")
                        rows_list.append(embedding_list)
                        print(np.shape(rows_list))

            print('just exited that previous loop')
            if rows_list:
               # print(f'rows list is: {rows_list}')
                emb_l = pd.DataFrame(rows_list, columns=[i for i in range(58)])
                context = [emb_l.loc[:, idx].mean() for idx in range(58)]
                print('just calling predict author now!')

                context=str(context)
                check=str(check)
                check=check.replace('\n', '').replace('  ', ' ').replace('[ ','[').replace(' ]',']')
                context=context

                #print(f'Check embedding is :{check}')
                #print(f'Context embedding is :{context}')
                result = predict_author(context,check)  
                
                return jsonify({'result': result})
            else:
                return jsonify({"error": "No matching author ID found in the provided file."}), 404
        except Exception as e:
            print(f"Error: {e}")  # For debugging, print the error to the console.
            return jsonify({"error": str(e)}), 500


    return jsonify({"error": "Method not allowed."}), 405


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
#does not work well with books with dialogue. better for essays