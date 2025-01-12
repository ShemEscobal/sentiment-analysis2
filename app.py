# app.py
import os
from flask import Flask, request, render_template, send_file, session
import pandas as pd
from fb_classifier import classify_emotions
from werkzeug.middleware.proxy_fix import ProxyFix

def create_app():
    app = Flask(__name__)
    
    # Add ProxyFix middleware for running behind a proxy
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    
    # Use environment variable for secret key with a fallback
    app.secret_key = os.environ.get('SECRET_KEY', 'iotlab_2023')
    
    # Set upload folder using absolute path
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    # Ensure the upload folder exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return 'No file part', 400
            file = request.files['file']
            if file.filename == '':
                return 'No selected file', 400
            if file and file.filename.endswith('.xlsx'):
                filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filename)
                
                # Read the uploaded Excel file
                df = pd.read_excel(filename)
                
                # Get the columns for selection
                columns = df.columns.tolist()
                
                # Store the filepath in the session
                session['filepath'] = filename
                return render_template('select_column.html', columns=columns)
            
            return 'Invalid file format', 400
        except Exception as e:
            app.logger.error(f"Error in upload: {str(e)}")
            return 'An error occurred while processing the file', 500

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            selected_column = request.form.get('column')
            if not selected_column:
                return 'No column selected', 400
                
            filepath = session.get('filepath')
            if not filepath:
                return 'No file found', 400
            
            if not os.path.exists(filepath):
                return 'File not found', 404
            
            # Read the uploaded Excel file
            df = pd.read_excel(filepath)
            
            # Classify emotions
            classified_df = classify_emotions(df, selected_column)
            
            # Prepare predictions
            predictions = classified_df['predicted_emotion'].tolist()
            
            # Count labels
            label_counts = {}
            if predictions:
                for label in predictions:
                    label_counts[label] = label_counts.get(label, 0) + 1
            
            total_responses = len(predictions)  # Calculate total number of responses
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                app.logger.warning(f"Failed to remove temporary file: {str(e)}")
            
            return render_template('report.html', label_counts=label_counts, total_responses=total_responses)
            
        except Exception as e:
            app.logger.error(f"Error in predict: {str(e)}")
            return 'An error occurred while processing the prediction', 500

    # Add error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=67895, debug=False)
