# 🎬 AIOMovie

> Search a movie and get AI-powered recommendations using Neural Collaborative Filtering (NCF) — a deep learning model trained on 100k+ MovieLens ratings that learns hidden user and movie preferences, without needing to know anything about the movie's content.

---

## Live Demo

Try it now on Hugging Face Spaces:
**[https://richardalcala-aiomoviedemo.hf.space](https://richardalcala-aiomoviedemo.hf.space)**

---

## What is AIOMovie?

AIOMovie is a movie recommendation system powered by **Neural Collaborative Filtering (NCF)**, a deep learning framework that learns latent representations of users and movies from historical rating data.

Given a movie title, the system recommends the 10 most similar movies by computing **cosine similarity** between movie embeddings learned during training — no genre tags, no plot descriptions, no content metadata needed.

The app lets you select and compare 3 NCF model variants simultaneously:

| Model | Architecture | Description |
|---|---|---|
| **GMF** | Generalized Matrix Factorization | Learns linear interactions via element-wise product of embeddings |
| **MLP** | Multi-Layer Perceptron | Learns non-linear interactions via concatenation + dense layers |
| **NeuMF** | Neural Matrix Factorization | Combines GMF + MLP for best of both worlds |

---

## Project Structure

```
AIOMovie/
├── data/
│   ├── links.csv              # MovieLens movie → TMDB/IMDB ID mapping
│   ├── movies.csv             # Movie titles and genres
│   ├── ratings.csv            # User ratings (100k+ records)
│   └── tags.csv               # User-generated tags
├── models/
│   └── ncf/
│       ├── aiomovie_ncf_gmf.keras     # Trained GMF model
│       ├── aiomovie_ncf_mlp.keras     # Trained MLP model
│       └── aiomovie_ncf_neumf.keras   # Trained NeuMF model
├── notebooks/
│   └── AIO_Movie_Core_Training.ipynb  # Model training (Google Colab)
├── app.py                     # Gradio app (entry point)
├── requirements.txt           # Python dependencies
└── README.md
```

**GitHub Repository:** [github.com/richard-alcala-code/AIOMovie](https://github.com/richard-alcala-code/AIOMovie)

---

## Training Notebook

The models were trained in Google Colab (GPU T4):

- **Colab:** [Open in Google Colab](https://colab.research.google.com/drive/1M7Kjo6P5Wzf3wE7oAxrbmCF2ynWrO4bY?usp=sharing)
- **GitHub:** [AIO_Movie_Core_Training.ipynb](https://github.com/richard-alcala-code/AIOMovie/blob/develop/notebooks/AIO_Movie_Core_Training.ipynb)

### Training details

- **Dataset:** MovieLens ml-latest-small (~100,836 ratings)
- **Split:** 80% train / 20% validation
- **Embedding size:** 50
- **Batch size:** 64 | **Epochs:** 5
- **Optimizer:** Adam | **Loss:** MSE

### Results

| Model | Validation Loss (MSE) |
|---|---|
| GMF | 7.3887 |
| MLP | 7.3378 |
| **NeuMF** | **7.3377** |

---

## References

- **Dataset:** [MovieLens — GroupLens Research](https://grouplens.org/datasets/movielens/)
- **Movie metadata & posters:** [TMDB API](https://developer.themoviedb.org/docs/getting-started)
- **Paper:** He et al. (2017) — [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)

---

## Run Locally

### 1. Prerequisites

- Python **3.12.13** (recommended — newer versions may have compatibility issues with TensorFlow)
- A [TMDB API account](https://www.themoviedb.org/signup) to get your Bearer Token
- (Optional) A [Hugging Face account](https://huggingface.co/join) to deploy your own Space

### 2. Clone the repository

```bash
git clone https://github.com/richard-alcala-code/AIOMovie.git
cd AIOMovie
```

### 3. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate        # macOS/Linux
env\Scripts\activate           # Windows
```

### 4. Install dependencies

Sample Content of `requirements.txt`:

```
gradio
tensorflow
pandas
numpy
requests
scikit-learn
```

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the root directory:

```bash
TMDB_TOKEN=your_tmdb_bearer_token_here
```

To get your TMDB Bearer Token:
1. Create an account at [https://www.themoviedb.org/signup](https://www.themoviedb.org/signup)
2. Go to **Settings → API**
3. Request an API key and copy the **Bearer Token (Read Access Token)**

### 6. Run the app

```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## Deploy to Hugging Face Spaces

### 1. Create a Hugging Face account

Sign up at [https://huggingface.co/join](https://huggingface.co/join)

### 2. Create a new Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Gradio** as the SDK
3. Set visibility to **Public** or **Private**

### 3. Add your TMDB token as a Secret

1. Go to your Space → **Settings → Variables and Secrets**
2. Add a new secret: `TMDB_TOKEN` = your Bearer Token

### 4. Push your code

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git push space main
```

Your app will be live at `https://YOUR_USERNAME-YOUR_SPACE_NAME.hf.space`

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Data processing | Pandas, NumPy, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| External API | TMDB API |
| Deployment | Hugging Face Spaces + Gradio |
| Training environment | Google Colab (GPU T4) |
