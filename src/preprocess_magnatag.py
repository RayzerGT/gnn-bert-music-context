"""Preprocess MagnaTagATune audio and metadata for the GNN-BERT music context project."""

import shutil
import zipfile
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd

from audio_features import extract_chroma, extract_log_mel, normalize_feature, segment_audio
from graph_builder import build_chord_transition_graph, build_segment_similarity_graph


def read_clip_info(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / 'MagnaTagATune' / 'clip_info_final.csv'
    df = pd.read_csv(path, sep='\t', quotechar='"', engine='python', dtype=str, keep_default_na=False)
    df['clip_id'] = df['clip_id'].astype(str).str.strip()
    return df


def read_annotations(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / 'MagnaTagATune' / 'annotations_final.csv'
    df = pd.read_csv(path, sep='\t', quotechar='"', engine='python', dtype=str, keep_default_na=False)
    df['clip_id'] = df['clip_id'].astype(str).str.strip()
    return df


def read_comparisons(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / 'MagnaTagATune' / 'comparisons_final.csv'
    return pd.read_csv(path, sep='\t', quotechar='"', engine='python', dtype=str, keep_default_na=False)


def extract_split_zip(raw_dir: Path, archive_name: str = 'mp3.zip', num_parts: int = 3) -> bool:
    """Concatenate split zip parts and extract the archive if needed."""
    archive_dir = raw_dir / 'MagnaTagATune'
    parts = [archive_dir / f'{archive_name}.{i:03d}' for i in range(1, num_parts + 1)]
    if not all(part.exists() for part in parts):
        return False

    zip_path = archive_dir / archive_name
    if not zip_path.exists():
        print(f'Concatenating split archive into {zip_path}')
        with zip_path.open('wb') as out_file:
            for part in parts:
                with part.open('rb') as in_file:
                    shutil.copyfileobj(in_file, out_file)

    print(f'Extracting audio files from {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(archive_dir)
    return True


def build_tag_matrix(annotations: pd.DataFrame) -> pd.DataFrame:
    text_cols = [c for c in annotations.columns if c not in {'clip_id', 'mp3_path'}]
    tags = annotations.set_index('clip_id')[text_cols]
    tags = tags.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    return tags


def build_labels(annotations: pd.DataFrame, clip_info: pd.DataFrame) -> pd.DataFrame:
    tags = build_tag_matrix(annotations)
    clips = clip_info[['clip_id', 'mp3_path']].set_index('clip_id')
    return clips.join(tags, how='inner')


def get_audio_path(raw_dir: Path, mp3_path: str) -> Path:
    return raw_dir / 'MagnaTagATune' / mp3_path


def load_audio(audio_path: Path, sr: int = 22050) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    return audio, sr


def extract_features(audio: np.ndarray, sr: int, feature_type: str = 'log_mel') -> np.ndarray:
    if feature_type == 'log_mel':
        features = extract_log_mel(audio, sr=sr)
    elif feature_type == 'chroma':
        features = extract_chroma(audio, sr=sr)
    else:
        raise ValueError(f'Unsupported feature type: {feature_type}')
    return normalize_feature(features)


def preprocess_magnatag(raw_dir: Path, processed_dir: Path, feature_type: str = 'log_mel', segment_length: int = 216, hop_length: int = 216):
    processed_dir.mkdir(parents=True, exist_ok=True)
    clip_info = read_clip_info(raw_dir)
    annotations = read_annotations(raw_dir)
    metadata = build_labels(annotations, clip_info)

    if not any((raw_dir / 'MagnaTagATune').glob('**/*.mp3')):
        extracted = extract_split_zip(raw_dir)
        if not extracted:
            raise FileNotFoundError('No extracted MP3 files found and split archive extraction failed.')

    output_rows = []
    for clip_id, row in metadata.iterrows():
        mp3_path = row['mp3_path']
        audio_path = get_audio_path(raw_dir, mp3_path)
        if not audio_path.exists():
            print(f'SKIP missing audio: {audio_path}')
            continue

        audio, sr = load_audio(audio_path)
        features = extract_features(audio, sr, feature_type=feature_type)
        segments = segment_audio(features, segment_length=segment_length, hop_length=hop_length)

        chroma = extract_chroma(audio, sr=sr)
        chroma = normalize_feature(chroma)
        chord_node_features, chord_edge_index, chord_edge_weight = build_chord_transition_graph(chroma)
        segment_similarity_edge_index, segment_similarity_edge_weight = build_segment_similarity_graph(segments)

        track_dir = processed_dir / 'magnatag' / str(clip_id)
        track_dir.mkdir(parents=True, exist_ok=True)
        np.save(track_dir / 'features.npy', features)
        np.save(track_dir / 'segments.npy', segments)
        np.save(track_dir / 'chroma.npy', chroma)
        np.save(track_dir / 'chord_node_features.npy', chord_node_features)
        np.save(track_dir / 'chord_edge_index.npy', chord_edge_index)
        np.save(track_dir / 'chord_edge_weight.npy', chord_edge_weight)
        np.save(track_dir / 'segment_similarity_edge_index.npy', segment_similarity_edge_index)
        np.save(track_dir / 'segment_similarity_edge_weight.npy', segment_similarity_edge_weight)

        output_rows.append({
            'clip_id': clip_id,
            'mp3_path': mp3_path,
            'feature_type': feature_type,
            'segment_length': segment_length,
            'hop_length': hop_length,
            'num_segments': segments.shape[0],
            'num_chord_nodes': chord_node_features.shape[0],
            'num_chord_edges': chord_edge_index.shape[1],
            'num_segment_similarity_edges': segment_similarity_edge_index.shape[1],
            **{f'tag_{tag}': row[tag] for tag in annotations.columns if tag != 'clip_id'},
        })

    output_path = processed_dir / 'magnatag_metadata.csv'
    pd.DataFrame(output_rows).to_csv(output_path, index=False)
    print(f'Wrote processed metadata to {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess MagnaTagATune for GNN-BERT project.')
    parser.add_argument('--raw-dir', type=str, default='data/raw', help='Raw dataset root directory')
    parser.add_argument('--processed-dir', type=str, default='data/processed', help='Processed output directory')
    parser.add_argument('--feature-type', type=str, choices=['log_mel', 'chroma'], default='log_mel', help='Audio feature type')
    parser.add_argument('--segment-length', type=int, default=216, help='Frame length for segments')
    parser.add_argument('--hop-length', type=int, default=216, help='Frame hop length for segments')
    args = parser.parse_args()

    preprocess_magnatag(Path(args.raw_dir), Path(args.processed_dir), feature_type=args.feature_type, segment_length=args.segment_length, hop_length=args.hop_length)
