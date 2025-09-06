import os
from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        features = df[['Latitude', 'Longitude']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        df['KMeans'] = kmeans_labels
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        db_labels = dbscan.fit_predict(X_scaled)
        df['DBSCAN'] = db_labels
        agglo = AgglomerativeClustering(n_clusters=5)
        agglo_labels = agglo.fit_predict(X_scaled)
        df['Agglo'] = agglo_labels
        def combine_clusters(row):
            if row['DBSCAN'] == -1:
                return -1  # Noise from DBSCAN gets labeled -1
            # If KMeans and Agglo agree, keep that label
            if row['KMeans'] == row['Agglo']:
                return row['KMeans']
            # Otherwise, combine into a composite label
            return f"K{row['KMeans']}_A{row['Agglo']}_D{row['DBSCAN']}"
        df['FinalCluster'] = df['KMeans']
        if 'Primary Type' in df.columns:
            cluster_to_crimetype = {}
            for cluster_id in sorted(df['FinalCluster'].unique()):
                common_type = df[df['FinalCluster'] == cluster_id]['Primary Type'].mode()
                cluster_to_crimetype[cluster_id] = common_type.iloc[0] if not common_type.empty else f"Cluster {cluster_id}"

            df['Cluster_Label'] = df['FinalCluster'].map(cluster_to_crimetype)
            hue_column = 'Cluster_Label'
        else:
            hue_column = 'FinalCluster'
        result_path = os.path.join(UPLOAD_FOLDER, 'clustered_' + file.filename)
        df.to_csv(result_path, index=False)
        plt.figure(figsize=(14, 10))
        sns.scatterplot(data=df, x='Longitude', y='Latitude', hue=hue_column, palette='tab10', s=80, edgecolor='w')
        plt.title('Crime Clustering with KMeans and Refined Labeling', fontsize=16)
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        legend = plt.legend(title='Cluster Label', title_fontsize='13', fontsize='11', loc='upper right')
        legend.get_frame().set_edgecolor('black')
        plot_path = os.path.join('static', 'cluster_plot.png')
        os.makedirs('static', exist_ok=True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return render_template('result.html', csv_file=url_for('download', filename=result_path), plot_file=plot_path)
    return "Invalid file format. Please upload a CSV file."
@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
