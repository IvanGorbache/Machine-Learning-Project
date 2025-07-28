"""
Suicide Detection Analysis - Comprehensive Machine Learning Pipeline

This script performs:
1. Data loading and preprocessing
2. Feature engineering (TF-IDF and Universal Sentence Encoder)
3. Model training and evaluation (SVM, Random Forest, Logistic Regression, Neural Network)
4. Demographic analysis (age and gender distributions)
5. Clustering analysis
6. Visualization (word clouds, confusion matrices, etc.)
7. Gender-enhanced modeling
"""

# Import libraries
import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             fbeta_score, recall_score, precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack, csr_matrix
from wordcloud import WordCloud
import gc

# Set matplotlib backend
matplotlib.use('TkAgg')


# ======================
# Utility Functions
# ======================

def preprocess_text(text, for_wordcloud=False):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text = re.sub(r'u\/\w+', '', text)  # Remove user mentions
    text = re.sub(r'r\/\w+', '', text)  # Remove subreddit mentions
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'\bfiller\b', '', text)  # Remove frequent non-suicide word

    if not for_wordcloud:
        text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation/symbols

    text = re.sub(r'\s+', ' ', text).strip()  # Remove excess whitespace
    return text


def detect_gender(text):
    """Extract gender information from text"""
    text = str(text).lower()
    male_keywords = r'\b(m|male|man|boy|he|him|his|guy|dude|gentleman|mr|sir|manly|masculine|boyish|masculinity|trans man|ftm|transmale|amab|assigned male)\b'
    female_keywords = r'\b(f|female|woman|girl|she|her|hers|lady|miss|ms|mrs|ma\'am|feminine|womanly|girly|femininity|trans woman|mtf|transfemale|afab|assigned female)\b'

    if re.search(male_keywords, text):
        return 'Male'
    elif re.search(female_keywords, text):
        return 'Female'
    return 'Unknown'


def extract_age(text):
    """Extract age information from text"""
    text = str(text).lower()

    # School year mappings
    school_year_to_age = {
        'freshman': 14,
        'sophomore': 15,
        'junior': 16,
        'senior': 17
    }
    for term, age in school_year_to_age.items():
        if term in text:
            return age

    # Various age patterns
    contextual_patterns = [
        r'(?:^|\s)(?:i am|i\'m|im|me)\s*(\d{2})\s*(?:yo|y/o|years? old|yrs?)?\b',
        r'(?:^|\s)(?:turn(?:ing)?|turned)\s*(\d{2})\b',
        r'(?:^|\s)age(?:d)?\s*(\d{2})\b',
        r'\b(\d{2})\s*(?:yo|y/o|yrs?|years? old)\b',
        r'\b(?:f|female|m|male)\s*(\d{2})\b',
        r'\b(\d{2})\s*(?:f|female|m|male)\b',
        r'(?:^|\s)(?:just)?\s*(?:hit|reached|celebrated|turning)\s*(\d{2})\b',
        r'(\d{2})\s*(?:birthday|b-day|bday)\b',
        r'\b(?:i\'ll|i will|i\'m going to)\s*(?:be|turn)\s*(\d{2})\b',
        r'\b(\d{2})(?=\s*in\s*(?:school|grade))'
    ]

    for pattern in contextual_patterns:
        match = re.search(pattern, text)
        if match:
            age = int(match.group(1))
            if 10 <= age <= 100:
                return age

    # Additional context checks
    for match in re.finditer(r'(?<!\d)(\d{2})(?!\d)', text):
        age = int(match.group(1))
        if 10 <= age <= 100:
            start, end = match.span()
            context_window = text[max(0, start - 25):min(len(text), end + 25)]
            if any(word in context_window for word in [
                'year', 'years', 'old', 'age', 'born', 'since', 'birthday',
                'school', 'teen', 'grade', 'turn', 'turned', 'i am', 'i\'m', 'me',
                'class', 'middle', 'high', 'elementary', 'freshman', 'sophomore',
                'junior', 'senior', 'vote', 'permit', 'license', 'puberty',
                'college', 'driving', 'prom', 'graduated', 'graduate', 'next year', 'becoming'
            ]):
                return age

    return np.nan


def embed_text_batched(text_series, embed, batch_size=512):
    """Embed text in batches to manage memory"""
    embeddings = []
    for start in range(0, len(text_series), batch_size):
        batch = text_series.iloc[start:start + batch_size].tolist()
        embedded_batch = embed(batch).numpy()
        embeddings.append(embedded_batch)
        del embedded_batch
        gc.collect()
    return np.vstack(embeddings)


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name=""):
    """Evaluate model performance with different thresholds"""
    model.fit(X_train, y_train)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # For neural networks
        y_proba = model.predict(X_test).ravel()

    for threshold in [0.5, 0.3]:
        y_pred = (y_proba > threshold).astype(int)
        print(f"\n{model_name} Evaluation (Threshold = {threshold}):")
        print(classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide']))

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)

        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F2 Score: {f2:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
                    xticklabels=['non-suicide', 'suicide'],
                    yticklabels=['non-suicide', 'suicide'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{model_name} Confusion Matrix (Threshold = {threshold})")
        plt.tight_layout()
        plt.show()


def build_dense_model(input_shape):
    """Build a dense neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def perform_clustering(X, df_subset, method_name):
    """Perform clustering and visualization"""
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_reduced = svd.fit_transform(X)

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_reduced)

    print("\nPosts per Cluster:")
    print(pd.Series(clusters).value_counts().sort_index())

    sil_score = silhouette_score(X_reduced, clusters)
    print(f"\nSilhouette Score (higher = better separation): {sil_score:.4f}")

    print("\nClass distribution within each cluster:")
    print(pd.crosstab(clusters, df_subset['class'], normalize='index'))

    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200,
                n_iter_without_progress=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_reduced)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='tab10', s=50)
    plt.title(f"t-SNE Visualization Colored by KMeans Cluster ({method_name})")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df_subset['class'], palette='Set2', s=50)
    plt.title(f"t-SNE Visualization Colored by True Class Label ({method_name})")
    plt.legend(title="Class")
    plt.tight_layout()
    plt.show()

    print("\nCluster Purity (majority class % per cluster):")
    crosstab = pd.crosstab(clusters, df_subset['class'])
    purity = crosstab.max(axis=1) / crosstab.sum(axis=1)
    for cluster_id, score in purity.items():
        print(f"Cluster {cluster_id}: {score:.2%} pure")


# ======================
# Main Analysis Pipeline
# ======================

def main():
    print("Starting Suicide Detection Analysis Pipeline...")

    # 1. Data Loading and Preprocessing
    print("\nStep 1: Loading and preprocessing data...")
    df = pd.read_csv('Suicide_Detection.csv', engine='python', nrows=100000)
    df = df.dropna(subset=['text', 'class'])
    df['label'] = df['class'].map({'non-suicide': 0, 'suicide': 1})
    df['clean_text'] = df['text'].apply(lambda x: preprocess_text(x, for_wordcloud=False))
    df['gender'] = df['text'].apply(detect_gender)
    df['age'] = df['text'].apply(extract_age)

    # Create age groups
    bins = [10, 13, 18, 25, 30, 40, 50, 65, 100]
    labels = ['10-12', '13-17', '18-24', '25-29', '30-39', '40-49', '50-64', '65+']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )

    # 2. Feature Engineering
    print("\nStep 2: Feature engineering...")
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

    # Universal Sentence Encoder
    print("Loading Universal Sentence Encoder...")
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print("Encoding text with USE...")
    X_train_use = embed_text_batched(X_train_text, embed, batch_size=256)
    X_test_use = embed_text_batched(X_test_text, embed, batch_size=256)

    # 3. Model Training and Evaluation
    print("\nStep 3: Model training and evaluation...")

    # Initialize models
    base_svm = LinearSVC(class_weight='balanced', random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=2)
    rf = RandomForestClassifier(n_estimators=20, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    logreg = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)

    # Evaluate models
    print("\nEvaluating SVM with TF-IDF...")
    evaluate_model(svm, X_train_tfidf, X_test_tfidf, y_train, y_test, "SVM (TF-IDF)")

    print("\nEvaluating SVM with USE...")
    evaluate_model(svm, X_train_use, X_test_use, y_train, y_test, "SVM (USE)")

    print("\nEvaluating Random Forest with TF-IDF...")
    evaluate_model(rf, X_train_tfidf, X_test_tfidf, y_train, y_test, "Random Forest (TF-IDF)")

    print("\nEvaluating Random Forest with USE...")
    evaluate_model(rf, X_train_use, X_test_use, y_train, y_test, "Random Forest (USE)")

    print("\nEvaluating Logistic Regression with TF-IDF...")
    evaluate_model(logreg, X_train_tfidf, X_test_tfidf, y_train, y_test, "Logistic Regression (TF-IDF)")

    print("\nEvaluating Logistic Regression with USE...")
    evaluate_model(logreg, X_train_use, X_test_use, y_train, y_test, "Logistic Regression (USE)")

    # Neural Network
    print("\nTraining Neural Network with TF-IDF...")
    X_train_tfidf_dense = X_train_tfidf.toarray()
    X_test_tfidf_dense = X_test_tfidf.toarray()

    dense_model = build_dense_model((X_train_tfidf_dense.shape[1],))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = dense_model.fit(
        X_train_tfidf_dense, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.01,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate NN with confusion matrix
    print("\nEvaluating Neural Network with TF-IDF...")
    evaluate_model(dense_model, X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test, "Neural Network (TF-IDF)")

    print("\nTraining Neural Network with USE...")
    dense_model_use = build_dense_model((X_train_use.shape[1],))
    history = dense_model_use.fit(
        X_train_use, y_train,
        epochs=20,
        batch_size=64,
        validation_split=0.01,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate NN with confusion matrix
    print("\nEvaluating Neural Network with USE...")
    evaluate_model(dense_model_use, X_train_use, X_test_use, y_train, y_test, "Neural Network (USE)")

    # 4. Demographic Analysis
    print("\nStep 4: Demographic analysis...")

    # Age group distribution
    groups = {
        'Overall': df,
        'Suicide': df[df['class'] == 'suicide'],
        'Non-suicide': df[df['class'] == 'non-suicide']
    }

    for group_name, group_df in groups.items():
        plt.figure(figsize=(10, 5))
        ax = sns.countplot(data=group_df, x='age_group', order=labels, palette='magma')

        total = len(group_df)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 2,
                    f'{height / total:.1%}',
                    ha='center', color='black', fontsize=11, fontweight='bold')

        plt.title(f'Age Group Distribution - {group_name}', fontsize=14, pad=15)
        plt.xlabel('Age Group')
        plt.ylabel('Post Count')
        plt.xticks(rotation=45)
        plt.ylim(0, max([p.get_height() for p in ax.patches]) * 1.15)
        plt.tight_layout()
        plt.show()

    # Gender distribution
    for title, group_df in groups.items():
        gender_counts = group_df['gender'].value_counts()
        percentages = (gender_counts / len(group_df)) * 100

        plt.figure(figsize=(7, 5))
        bars = plt.bar(percentages.index, percentages.values,
                       color=['#3498db', '#e74c3c'], width=0.6)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height - 5,
                     f'{height:.1f}%',
                     ha='center', color='white', fontweight='bold', fontsize=12)

        plt.title(f'Gender Distribution - {title}', fontsize=14, pad=15)
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 100)
        plt.gca().spines[['top', 'right']].set_visible(False)

        # Add annotation box
        plt.annotate(f'Total posts: {len(group_df):,}\n'
                     f'Male: {gender_counts.get("Male", 0):,}\n'
                     f'Female: {gender_counts.get("Female", 0):,}',
                     xy=(0.95, 0.95), xycoords='axes fraction',
                     ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.8))

        plt.tight_layout()
        plt.show()

    # 5. Clustering Analysis
    print("\nStep 5: Clustering analysis...")
    print("Performing clustering with TF-IDF features...")
    perform_clustering(X_train_tfidf, df.loc[X_train_text.index], "TF-IDF")

    print("\nPerforming clustering with USE features...")
    perform_clustering(X_train_use, df.loc[X_train_text.index], "USE")

    # 6. Word Clouds
    print("\nStep 6: Generating word clouds...")
    # Preprocess text specifically for word clouds (preserving punctuation)
    df['text_for_wordcloud'] = df['text'].apply(lambda x: preprocess_text(x, for_wordcloud=True))

    suicide_text = ' '.join(df[df['class'] == 'suicide']['text_for_wordcloud'].dropna().tolist())
    non_suicide_text = ' '.join(df[df['class'] == 'non-suicide']['text_for_wordcloud'].dropna().tolist())

    suicide_wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(suicide_text)
    non_suicide_wc = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(
        non_suicide_text)

    # Display suicide word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(suicide_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Suicide Posts')
    plt.show()

    # Display non-suicide word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(non_suicide_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Non-Suicide Posts')
    plt.show()

    # 7. Gender-Enhanced Models
    print("\nStep 7: Gender-enhanced models...")

    # Reload data specifically for gender analysis to maximize examples
    print("Reloading data for gender-enhanced analysis...")
    gender_df = pd.read_csv('Suicide_Detection.csv', engine='python')
    gender_df = gender_df.dropna(subset=['text', 'class'])
    gender_df['label'] = gender_df['class'].map({'non-suicide': 0, 'suicide': 1})
    gender_df['clean_text'] = gender_df['text'].apply(preprocess_text)
    gender_df['gender'] = gender_df['text'].apply(detect_gender)

    # Filter to only keep posts with identifiable gender
    gender_df = gender_df[gender_df['gender'] != 'Unknown']
    print(f"Number of posts with identifiable gender: {len(gender_df)}")
    print(gender_df['gender'].value_counts())

    # Prepare gender-enhanced features
    gender_encoded = pd.get_dummies(gender_df['gender'], prefix='gender')

    # TF-IDF + Gender
    gender_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X_gender_tfidf = gender_tfidf.fit_transform(gender_df['clean_text'])
    X_gender_meta = np.hstack([gender_encoded.values])
    X_gender_combined = hstack([X_gender_tfidf, csr_matrix(X_gender_meta)])
    y_gender = gender_df['label'].values

    X_train_gender, X_test_gender, y_train_gender, y_test_gender = train_test_split(
        X_gender_combined, y_gender, test_size=0.2, stratify=y_gender, random_state=42
    )

    # Train SVM
    print("\nEvaluating SVM with TF-IDF + Gender...")
    evaluate_model(svm, X_train_gender, X_test_gender, y_train_gender, y_test_gender, "SVM (TF-IDF + Gender)")

    # USE + Gender
    X_train_gender_text, X_test_gender_text, y_train_gender, y_test_gender = train_test_split(
        gender_df['text'], gender_df['label'], test_size=0.2, stratify=gender_df['label'], random_state=42
    )

    X_train_gender_use = embed_text_batched(X_train_gender_text, embed, batch_size=256)
    X_test_gender_use = embed_text_batched(X_test_gender_text, embed, batch_size=256)

    X_train_gender_meta = np.hstack([
        pd.get_dummies(gender_df.loc[X_train_gender_text.index]['gender'], prefix='gender').values
    ])
    X_test_gender_meta = np.hstack([
        pd.get_dummies(gender_df.loc[X_test_gender_text.index]['gender'], prefix='gender').values
    ])

    X_train_gender_final = np.hstack([X_train_gender_use, X_train_gender_meta])
    X_test_gender_final = np.hstack([X_test_gender_use, X_test_gender_meta])

    print("\nEvaluating SVM with USE + Gender...")
    evaluate_model(svm, X_train_gender_final, X_test_gender_final,
                   y_train_gender, y_test_gender, "SVM (USE + Gender)")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
