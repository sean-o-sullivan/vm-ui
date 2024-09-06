import csv
import os
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for, abort, current_app, session
from werkzeug.utils import secure_filename
from datetime import datetime  
import json
import plotly
import plotly.graph_objects as go

import umap.umap_ as umap



#from embedding import Embedding  
from generation.embedding import generateEmbedding
from eval.evalZ import *

app = Flask(__name__)

# ------------------- activate verifymeenv to run -------------------------------------- #

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')  # Use a directory named 'uploads' in the current working directory
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.errorhandler(Exception)
def handle_exception(e):
    current_app.logger.error(f'An error occurred: {e}')
    return jsonify({'error': 'An internal error occurred'}), 500


# @app.route('/append', methods=['POST'])
# def append_embedding():
#     if not request.json or 'authorId' not in request.json or 'embedding' not in request.json:
#         abort(400, description="Invalid data: JSON must contain 'authorId' and 'embedding'.")

#     data = request.json
#     author_id = data['authorId']
#     embedding = data['embedding'].replace('\n', '').replace('\r', '').replace(' ', '').strip('"')
#     testname = data['testName']

#     # Get current timestamp in your preferred format
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')

#     try:
#         # Attempt to load the CSV into a DataFrame, or create a new one if the file does not exist
#         try:
#             df = pd.read_csv(file_path)
#             # Determine the next embeddingID by adding 1 to the max currently in use
#             next_embedding_id = df['EmbeddingID'].max() + 1
#         except FileNotFoundError:
#             # Define a new DataFrame with appropriate columns if the file doesn't exist
#             df = pd.DataFrame(columns=['EmbeddingID', 'TestName', 'AuthorID', 'Embedding', 'Timestamp'])
#             next_embedding_id = 1
        
#         # Create a new DataFrame for the row to be appended
#         new_row_df = pd.DataFrame({
#             'EmbeddingID': [next_embedding_id], 
#             'TestName': [testname],
#             'AuthorID': [author_id], 
#             'Embedding': [embedding], 
#             'Timestamp': [timestamp]
#         })
        
#         # Use pd.concat to append the new row
#         df = pd.concat([df, new_row_df], ignore_index=True)
        
#         # Save the updated DataFrame back to the CSV
#         df.to_csv(file_path, index=False)
#         current_app.logger.info('Embedding appended successfully.')
#         return jsonify({'message': 'Embedding appended successfully.'})
#     except Exception as e:
#         current_app.logger.error(f"Failed to append data: {e}")
#         return jsonify({'error': str(e)})




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




# @app.route('/append_result', methods=['POST'])
# def append_result():
#     if not request.json or 'authorId' not in request.json or 'result' not in request.json:
#         abort(400, description="Invalid data: JSON must contain 'authorId' and 'result'.")

#     data = request.json
#     #test_name = data['testname']
#     test_name =" "
#     author_id = data['authorId']
#     result = data['result']

#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv')

#     try:
#         # Attempt to load the CSV into a DataFrame, or create a new one if the file does not exist
#         try:
#             df = pd.read_csv(file_path)
#             # Determine the next embeddingID by adding 1 to the max currently in use
#             next_result_id = df['TestID'].max() + 1
#         except FileNotFoundError:
#             # Define a new DataFrame with appropriate columns if the file doesn't exist
#             df = pd.DataFrame(columns=['TestID', 'Testname','AuthorID', 'Result', 'Timestamp'])
#             next_result_id = 1
        
#         # Create a new DataFrame for the row to be appended
#         new_row_df = pd.DataFrame({
#             'TestID': [next_result_id], 
#             'Testname': [test_name], 
#             'AuthorID': [author_id], 
#             'Result': [result], 
#             'Timestamp': [timestamp]
#         })
        
#         # Use pd.concat to append the new row
#         df = pd.concat([df, new_row_df], ignore_index=True)
        
#         # Save the updated DataFrame back to the CSV
#         df.to_csv(file_path, index=False)
#         current_app.logger.info('Embedding appended successfully.')
#         return jsonify({'message': 'Embedding appended successfully.'})
#     except Exception as e:
#         current_app.logger.error(f"Failed to append data: {e}")
#         return jsonify({'error': str(e)})
    




@app.route('/append', methods=['POST'])
def append_embedding():
    if not request.json or 'authorId' not in request.json or 'embedding' not in request.json:
        abort(400, description="Invalid data: JSON must contain 'authorId' and 'embedding'.")
    
    data = request.json
    author_id = data['authorId']
    embedding = data['embedding']
    testname = data['testName']
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:

        embedding_list = [float(x) for x in embedding.split(',')]
        
        latent_embedding = get_latent_embedding(embedding_list)
        
        if latent_embedding is None:
            raise ValueError("Failed to generate latent embedding")
        
        latent_embedding_str = ','.join(map(str, latent_embedding))
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
        
        try:
            df = pd.read_csv(file_path)
            next_embedding_id = df['EmbeddingID'].max() + 1
        except FileNotFoundError:
            df = pd.DataFrame(columns=['EmbeddingID', 'TestName', 'AuthorID', 'Embedding', 'Timestamp', 'LatentEmbedding'])
            next_embedding_id = 1
        
        new_row_df = pd.DataFrame({
            'EmbeddingID': [next_embedding_id],
            'TestName': [testname],
            'AuthorID': [author_id],
            'Embedding': [embedding],
            'Timestamp': [timestamp],
            'LatentEmbedding': [latent_embedding_str]
        })
        
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.to_csv(file_path, index=False)
        
        current_app.logger.info('Embedding appended successfully.')
        return jsonify({'message': 'Embedding appended successfully.'})
    except ValueError as ve:
        current_app.logger.error(f"Failed to process embedding: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to append data: {e}")
        return jsonify({'error': str(e)}), 500
    




import traceback
import torch
import torch.nn.functional as F




import traceback
import torch
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latent_embedding(embedding):
    global global_model
    print("Entering get_latent_embedding function")
    print(f"Received embedding type: {type(embedding)}")
    print(f"Received embedding length: {len(embedding)}")
    print(f"First few elements of embedding: {embedding[:50]}")  # Reduced to first 50 elements for brevity

    try:
        # Load the model if it hasn't been loaded yet
        if global_model is None:
            print("Global model is None, loading model...")
            input_size = 112
            hidden_size = 256
            current_dir = os.path.dirname(os.path.abspath(__file__))
            print(f"Current directory: {current_dir}")
            model_path = "/Users/sean/Desktop/app_Z/eval/BnG_10_best_transformer_siamese_model.pth"
            print(f"Model path: {model_path}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            global_model, _ = load_model(model_path, input_size, hidden_size)
            print("Model loaded successfully")
            print(f"Global model type: {type(global_model)}")
        else:
            print("Global model already loaded")

        print("Converting embedding to tensor")
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        print(f"Embedding tensor shape: {embedding_tensor.shape}")
        print(f"Embedding tensor device: {embedding_tensor.device}")

        print("Getting latent embedding")
        with torch.no_grad():
            # Use the encoder directly from the Siamese network
            raw_embedding, latent_embedding = global_model.encoder(embedding_tensor)
            print(f"Raw embedding shape: {raw_embedding.shape}")
            print(f"Latent embedding shape: {latent_embedding.shape}")

        print("Converting latent embedding to list")
        result = latent_embedding.cpu().numpy().tolist()[0]  # Get the first (and only) item in the batch
        print(f"Result type: {type(result)}")
        print(f"Result length: {len(result)}")
        print(f"First few elements of result: {result[:5]}")

        return result
    except Exception as e:
        print(f"Error in get_latent_embedding: {str(e)}")
        print("Exception type:", type(e))
        print("Exception args:", e.args)
        print("Traceback:")
        traceback.print_exc()
        return None

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
    
    

# @app.route('/', methods=['GET', 'POST'])
# def embedding():
#     if request.method == 'POST':
#         mytext = request.form['mytext']  # This matches the 'name' attribute of your <textarea>
#         result = generateEmbedding(mytext)  # Ensure this function is implemented to process the text
#         print(f'the result is! : {result}')
#         return jsonify(result)  # Assuming the result is JSON serializable
    
#     return render_template('index_copy_2.html')





# @app.route('/', methods=['GET', 'POST'])
# def embedding():
#     if request.method == 'POST':
#         mytext = request.form['mytext']  # Get the input text
#         cleaned_text = mytext.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
#         print(f"the text is: {mytext}")
#         result = generateEmbedding(cleaned_text)  # Generate embedding with cleaned text
#         print(f'The result is: {result}')
#         return jsonify(result)  # Assuming the result is JSON serializable
    
#     return render_template('index_copy_2.html')





@app.route('/', methods=['GET', 'POST'])
def embedding():
    if request.method == 'POST':
        try:
            mytext = request.form['mytext']
            cleaned_text = mytext.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
            result = generateEmbedding(cleaned_text)

            if isinstance(result, list) and all(isinstance(x, (int, float)) for x in result):
                return jsonify({'embedding': result})
            else:
                raise ValueError("Invalid embedding format")
        except Exception as e:
            current_app.logger.error(f"Error generating embedding: {e}")
            return jsonify({'error': 'An internal error occurred'}), 500
    return render_template('index_copy_2.html')





# @app.route('/predict_authorship', methods=['GET', 'POST'])
# def predict_authorship():
#     print('hello, this is the flask method for prediction')
#     if request.method == 'POST':
#         try:
#             csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
#             data = request.get_json()  # Use get_json() to parse JSON data
#             authorID = data['authorId']  # Note the change to 'authorId' to match JavaScript
#             check = data['embedding']
            
#             print(f"Received authorID: {authorID}")
#             print(f"Received check embedding: {check}")
            
#             with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
#                 reader = csv.reader(csvfile)
#                 next(reader, None)
#                 rows_list = []
#                 for row in reader:
#                     if row[2] == authorID:
#                         cleaned_embedding = row[3].replace('[', '').replace(']', '').replace('"', '').strip()
#                         embedding_list = [float(num) for num in cleaned_embedding.split(',')]
#                         print(f"Parsed embedding list length: {len(embedding_list)}")
#                         rows_list.append(embedding_list)
                
#                 print(np.shape(rows_list))
#                 print('just exited that previous loop')
                
#                 if rows_list:
#                     emb_l = pd.DataFrame(rows_list, columns=[i for i in range(112)])  # Changed to 112
#                     context = [emb_l.loc[:, idx].mean() for idx in range(112)]  # Changed to 112
#                     print('just calling predict author now!')
#                     print(f"Context embedding: {context}")
#                     print(f"Check embedding: {check}")
                    
#                     # Convert string representations to lists of floats
#                     if isinstance(context, str):
#                         context = eval(context)  # Be cautious with eval, use only with trusted input
#                     if isinstance(check, str):
#                         check = eval(check)  # Be cautious with eval, use only with trusted input
                    
#                     # predict_author now receives lists of floats
#                     is_same_author, distance, confidence = predict_author(context, check)
                    
#                     print(f"Same author: {is_same_author}")
#                     print(f"Distance: {distance:.4f}")
#                     print(f"Confidence: {confidence:.4f}")
                    
#                     return jsonify({
#                         'is_same_author': bool(is_same_author),
#                         'distance': float(distance),
#                         'confidence': float(confidence)
#                     })
#                 else:
#                     return jsonify({"error": "No matching author ID found in the provided file."}), 404
#         except Exception as e:
#             print(f"Error: {e}")  # For debugging, print the error to the console.
#             return jsonify({"error": str(e)}), 500
    
#     return jsonify({"error": "Method not allowed."}), 405





@app.route('/predict_authorship', methods=['GET', 'POST'])
def predict_authorship():
    print('hello, this is the flask method for prediction')
    if request.method == 'POST':
        try:
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embeddings.csv')
            data = request.get_json()
            authorID = data['authorId']
            check = data['embedding']
            
            print(f"Received authorID: {authorID}")
            print(f"Received check embedding: {check}")
            
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
                    emb_l = pd.DataFrame(rows_list, columns=[i for i in range(112)])
                    context = [emb_l.loc[:, idx].mean() for idx in range(112)]
                    print('just calling predict author now!')
                    print(f"Context embedding: {context}")
                    print(f"Check embedding: {check}")
                    

                    if isinstance(context, str):
                        context = eval(context)
                    # if isinstance(check, str):
                    #     check = eval(check)

                    print(type(check))


                    dictionary = json.loads(check)


                    value = dictionary.get("embedding")  # Extract value for the key 'name'
                    print(value)
                    check = value


                    # Updated predict_author call to receive the new return values
                    is_same_author, distance, confidence, norm_context, norm_check = predict_author(context, check)
                    
                    print(f"Same author: {is_same_author}")
                    print(f"Distance: {distance:.4f}")
                    print(f"Confidence: {confidence:.4f}")
                    print(f"\n\n")
                    print(f"Context: {norm_context}")
                    print(f"Check: {norm_check}")                    
                    return jsonify({
                        'is_same_author': bool(is_same_author),
                        'distance': float(distance),
                        'confidence': float(confidence),
                        'latent context_embedding': norm_context,
                        'latent check_embedding': norm_check
                    })
                else:
                    return jsonify({"error": "No matching author ID found in the provided file."}), 404
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed."}), 405




@app.route('/about')
def about():
    return render_template('about.html')







def generate_umap_plot(embeddings, authors):
    # Reduce dimensions to 3D using UMAP
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    # Get unique authors
    unique_authors = list(set(authors))

    # Define color palette for the authors
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    color_dict = {author: color_palette[i % len(color_palette)] for i, author in enumerate(unique_authors)}

    # 3D scatter plot
    fig = go.Figure()

    for author in unique_authors:
        author_mask = np.array(authors) == author
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[author_mask, 0],
            y=embeddings_3d[author_mask, 1],
            z=embeddings_3d[author_mask, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=color_dict[author],
                opacity=0.8
            ),
            text=[f"Author: {author}" for _ in range(sum(author_mask))],
            hoverinfo='text',  
            showlegend=False   # Disable legend for this trace
        ))
    fig.update_layout(
        title="",  
        margin=dict(l=0, r=0, t=0, b=0),  
        scene=dict(
            xaxis_title="",  
            yaxis_title="",
            zaxis_title="",
            xaxis=dict(
                showticklabels=True,  
                showline=True,
                zeroline=False,  
                backgroundcolor='rgba(0,0,0,0)',  
                gridcolor='rgba(0,0,0,0)',  
                tickcolor='black', 
                showspikes=False  # No spikes on hover
            ),
            yaxis=dict(
                showticklabels=True,
                showline=True,
                zeroline=False,
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(0,0,0,0)',
                tickcolor='black',
                showspikes=False
            ),
            zaxis=dict(
                showticklabels=True,
                showline=True,
                zeroline=False,
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(0,0,0,0)',
                tickcolor='black',
                showspikes=False
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Make the paper (entire plot area) background transparent
        plot_bgcolor='rgba(0,0,0,0)', 
        hovermode='closest'
    )


    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)




@app.route('/update_plot')
def update_plot():
    scatter_plot = generate_umap_plot(embeddings, authors)
    return jsonify(scatter_plot)


if __name__ == '__main__':
    app.run(debug=True)
#does not work well with books with dialogue. better for essays, for future iterations: must have filtering mechanism or text suitability evaluation