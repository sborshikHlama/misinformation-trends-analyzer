# 📰 Fake News Style Detector

A Streamlit web application that uses BERTopic to analyze news text and detect if it's written in a fake news style. The model identifies topics and provides probability scores based on training data from 21,000 fake news articles (2016-2018).

## 🚀 Features

- **Topic Detection**: Identifies which topic cluster the news text belongs to
- **Probability Scoring**: Shows confidence level of the classification
- **Keyword Extraction**: Displays top 10 keywords associated with detected topics
- **User-Friendly Interface**: Simple text input with instant analysis

## 📋 Prerequisites

- Python 3.12+
- Docker (optional, for containerized deployment)

## 🛠️ Installation

### Option 1: Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
streamlit run app.py
```

5. **Open in browser**

```
http://localhost:8501
```

### Option 2: Docker Installation

1. **Build the Docker image**

```bash
docker build --platform linux/amd64 -t fake-news-detector:latest .
```

2. **Run the container**

```bash
docker run -p 8501:8501 fake-news-detector:latest
```

3. **Open in browser**

```
http://localhost:8501
```

### Option 3: Docker Compose

1. **Start the application**

```bash
docker-compose up -d
```

2. **Stop the application**

```bash
docker-compose down
```

## 📦 Requirements

```
streamlit
uvicorn[standard]
bertopic
sentence-transformers
transformers
torch==2.4.1
scikit-learn==1.5.2
scipy==1.11.4
joblib==1.4.2
tqdm==4.66.5
pandas==2.2.3
plotly==5.24.1
pydantic==2.9.2
numpy>=2.0,<2.3
```

## 🎯 Usage

1. Open the application in your browser
2. Enter news text in the text area
3. Click "Analyze" button
4. View results:
   - **Topic #**: Assigned topic cluster number
   - **Probability**: Confidence score of the classification
   - **Keywords**: Top 10 relevant keywords for the topic

### Example Input

```
Trump just said something shocking about Obama!
```

## 🏗️ Project Structure

```
fake-news-detector/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── README.md            # Project documentation
```

## 🔧 Configuration

The application uses a pre-trained BERTopic model hosted on Hugging Face:

- Model: `evstiars/bertopic-fake-news-detector`
- Training data: 21,000 fake news articles from 2016-2018

## 🐛 Troubleshooting

### scipy.sparse.diags Error

If you encounter `ValueError: Different number of diagonals and offsets`, the application includes a monkey patch that automatically fixes this compatibility issue between different scipy versions.

### Model Loading Issues

- Ensure you have a stable internet connection for downloading the model from Hugging Face
- The first run may take longer as the model is downloaded and cached

### Memory Issues

The model requires significant memory. Recommended minimum:

- RAM: 4GB+
- Disk space: 2GB+ for model cache

## 🌐 Deployment

### Deploy on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy!

### Deploy on Heroku

```bash
heroku create your-app-name
git push heroku main
heroku open
```

### Deploy with Docker on Cloud Platforms

The Docker image is compatible with:

- AWS ECS/Fargate
- Google Cloud Run
- Azure Container Instances
- DigitalOcean App Platform

Example for Google Cloud Run:

```bash
# Build for linux/amd64
docker build --platform linux/amd64 -t gcr.io/PROJECT_ID/fake-news-detector .

# Push to Google Container Registry
docker push gcr.io/PROJECT_ID/fake-news-detector

# Deploy to Cloud Run
gcloud run deploy fake-news-detector \
  --image gcr.io/PROJECT_ID/fake-news-detector \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 📊 Model Information

- **Model Type**: BERTopic (Bidirectional Encoder Representations from Transformers for Topic Modeling)
- **Training Dataset**: 21,000 fake news articles (2016-2018)
- **Embedding Model**: Sentence Transformers
- **Clustering**: HDBSCAN with UMAP dimensionality reduction

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BERTopic library by Maarten Grootendorst
- Hugging Face for model hosting
- Streamlit for the web framework
